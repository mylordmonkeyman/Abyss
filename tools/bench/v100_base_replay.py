#!/usr/bin/env python3
"""Fixed-token LOGP replay of the apparent v9 short-prompt difference.

Uses unmodified cached upstream executables. LOGP includes host logits transfer
and CPU log-softmax; elapsed times are not normal generation throughput.
"""
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).parent))
from v100_base_compare import ROOT, PACK, BASES, Sampler, fixed_env, read, numa_bytes
import p0

EXPECTED = {'historical':'5a75dda569d4d5a18041b4825416741da19fe1b578f4d7aa9ae575699fb8531f',
            'current':'5f11a611b284af9bb0df85d5e049e41bbbe7e02b81eb8176abce28fc6426ca89'}
BINARIES = {'historical':Path('/actions-runner/abyss-base-builds/47001f31/strata'),
            'current':Path('/actions-runner/abyss-base-builds/7c893ac1/strata')}
FIXTURE = Path(__file__).parent/'fixtures/v100-replay-600.tokens'


def parse_lp(line, expected):
    fields=line.split()
    if len(fields)!=4 or fields[0]!='LP' or int(fields[2])!=expected or fields[3] not in ('0','1'):
        raise ValueError('invalid fixed-token reply: '+line)
    value=float(fields[1])
    if not math.isfinite(value): raise ValueError('nonfinite replay likelihood')
    return {'sum_log_probability':value,'scored_tokens':expected,'all_greedy':fields[3]=='1'}


def arm(binary, out, ids):
    out.mkdir(parents=True,exist_ok=False)
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    argv=[str(binary),'--glm-pack',str(PACK),'--serve','--max-context','12288',
          '--max-new','128','--prefill','auto','--temperature','0','--eos-ids','2147483647']
    p0.write_json(out/'invocation.json',{'argv':argv,'environment':{k:v for k,v in env.items()
                  if k.startswith('STRATA_')},'snapshot_reuse_prevented':'alternate LOGP contexts 2 and 600',
                  'cpu_affinity':sorted(os.sched_getaffinity(0))})
    q=queue.Queue();records=[]
    with Sampler(out) as monitor,(out/'engine.log').open('w') as log:
        proc=subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                              text=True,bufsize=1)
        monitor.pid=proc.pid
        def reader():
            with (out/'wire.log').open('w') as f:
                for line in proc.stdout:
                    f.write(line);f.flush();q.put((line.strip(),time.monotonic()))
            q.put((None,time.monotonic()))
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        def until(prefix,timeout=1200):
            end=time.monotonic()+timeout
            while True:
                line,stamp=q.get(timeout=max(.1,end-time.monotonic()))
                if line is None or line.startswith('ERR '): raise RuntimeError(str(line))
                if line.startswith(prefix): return line,stamp
        def host_state():
            return {'cpu_stat':read('/proc/stat'),'loadavg':read('/proc/loadavg'),
                    'numa_mapped_bytes':numa_bytes(read(f'/proc/{proc.pid}/numa_maps'))}
        try:
            until('READY ')
            for repetition in range(6):
                # LOGP caches/restores an equal context independently of GEN's
                # prefix-reuse control. Different explicit contexts force reset
                # and prefill on every measured call, with fixed token bytes.
                proc.stdin.write('LOGP 2 '+','.join(map(str,ids[:3]))+'\n');proc.stdin.flush()
                separator,_=until('LP ');parse_lp(separator,1)
                before=host_state();start=time.monotonic()
                proc.stdin.write('LOGP 600 '+','.join(map(str,ids))+'\n');proc.stdin.flush()
                line,end=until('LP ')
                result=parse_lp(line,128)
                result.update(repetition=repetition,warmup=repetition==0,
                              begin_monotonic_s=start,end_monotonic_s=end,
                              replay_wire_ms=1000*(end-start),host_before=before,host_after=host_state())
                records.append(result);p0.write_json(out/'requests.json',records)
                print(json.dumps({'arm':out.name,**{k:v for k,v in result.items() if not k.startswith('host_')}}),flush=True)
            proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=120)
            if proc.returncode: raise RuntimeError(f'engine exit {proc.returncode}')
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
            thread.join(timeout=5)
    return records


def main():
    from model_bringup import NAMES, SIZES
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=False)
    for name,size in zip(NAMES,SIZES):
        if (Path('/models')/name).stat().st_size!=size: raise ValueError('wrong mounted model size')
    pack=json.loads((PACK/'v100_pack_complete.json').read_text())
    if pack['packer_sha256']!=p0.sha256(ROOT/'tools/iq_pack.py'): raise ValueError('pack provenance changed')
    provenance=json.loads(FIXTURE.with_suffix('.json').read_text())
    if p0.sha256(FIXTURE)!=provenance['input_sha256']: raise ValueError('replay fixture changed')
    ids=list(map(int,FIXTURE.read_text().strip().split(',')))
    if len(ids)!=728: raise ValueError('expected 600+128 fixed tokens')
    for name,binary in BINARIES.items():
        if p0.sha256(binary)!=EXPECTED[name]: raise ValueError('cached binary identity mismatch: '+name)
    p0.write_json(out/'toolchain_manifest.json',p0.manifest(os.environ.get('ABYSS_GPU','0')))
    report={'schema_version':1,'base_shas':BASES,'binary_sha256':EXPECTED,'fixture':provenance,'pack':pack,
            'model_integrity':'same read-only size-checked pack; full model SHA256 unverified',
            'scope':'LOGP host-logit/log-softmax replay wall time, not generated-token throughput',
            'order':['historical','current','current','historical','historical','current'],
            'arms':[],'status':'running','production_change':False}
    p0.write_json(out/'replay.json',report)
    for i,name in enumerate(report['order']):
        records=arm(BINARIES[name],out/f'{i}-{name}',ids)
        report['arms'].append({'index':i,'base':name,'requests':records,
                              'warm_median_ms':statistics.median(r['replay_wire_ms'] for r in records[1:])})
        p0.write_json(out/'replay.json',report)
    report['paired_median_changes_percent']=[]
    for i in range(0,6,2):
        pair={a['base']:a for a in report['arms'][i:i+2]}
        report['paired_median_changes_percent'].append(100*(pair['current']['warm_median_ms']/
                                                          pair['historical']['warm_median_ms']-1))
    report['status']='complete; review per-request clocks, placement, CPU load and likelihood before base admission'
    p0.write_json(out/'replay.json',report)
    print(json.dumps({'status':report['status'],'paired_median_changes_percent':report['paired_median_changes_percent']}))


if __name__=='__main__': main()
