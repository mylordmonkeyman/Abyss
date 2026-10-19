#!/usr/bin/env python3
"""Checkpointed P0 matrix on immutable upstream binaries; no optimized dispatch."""
from __future__ import annotations
import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
from v100_base_compare import BASES, PACK, ROOT, Sampler, fixtures, parse_done, read, score
from v100_base_replay import BINARIES, EXPECTED
import p0
from model_baseline import fixed_env, normalize_logprobs

ORDER=['historical','current','current','historical','historical','current']
CAMPAIGN=Path('/actions-runner/abyss-p0-v9/campaign-1')


class Monitor(Sampler):
    def snapshot(self):
        s=super().snapshot()
        s.update(host_cpu_stat=read('/proc/stat'),loadavg=read('/proc/loadavg'),
                 cpu_frequency_khz={str(c):read(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq')
                                    for c in (0,16)})
        return s

    def loop(self):
        with gzip.open(self.out/'telemetry.jsonl.gz','wt') as f:
            while not self.stop.is_set():
                f.write(json.dumps(self.snapshot())+'\n');f.flush();self.stop.wait(10)


def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('MemAvailable missing; cannot safely launch long context')


def model_identity():
    from model_bringup import NAMES
    files=[Path('/models')/name for name in NAMES]
    signature={p.name:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                           inode=p.stat().st_ino,device=p.stat().st_dev) for p in files}
    cache=CAMPAIGN.parent/'model-sha256.json'
    if cache.exists():
        result=json.loads(cache.read_text())
        if result['stat_signature']==signature:
            if any(v is False for v in result.get('matches_local_SHA256SUMS',{}).values()):
                raise ValueError('cached model checksum mismatch remains unresolved')
            return result
    hashes={}
    for path in files:
        print('Hashing model shard once:',path.name,flush=True)
        hashes[path.name]=p0.sha256(path)
    result=dict(stat_signature=signature,sha256=hashes,computed_utc=p0.now(),
                cache_policy='reused only with unchanged read-only mount stat signature')
    sums=Path('/models/SHA256SUMS')
    if sums.is_file():
        expected={}
        for line in sums.read_text().splitlines():
            fields=line.split()
            if len(fields)==2:expected[fields[1].lstrip('*')]=fields[0]
        result['local_SHA256SUMS_sha256']=p0.sha256(sums)
        result['matches_local_SHA256SUMS']={name:(expected[name]==value if name in expected else None) for name,value in hashes.items()}
        if any(name in expected and expected[name]!=value for name,value in hashes.items()):
            p0.write_json(cache,result);raise ValueError('model SHA256SUMS mismatch')
    p0.write_json(cache,result)
    return result


def paired(rows):
    groups={}
    for arm in rows:
        for row in arm['requests']:
            if row['request']==1:
                groups.setdefault((arm['index']//2,row['case'],row['output_tokens']),{})[arm['base']]=row
    result=[]
    for (pair,case,count),arms in groups.items():
        if set(arms)!={'historical','current'}:continue
        a,b=arms['historical'],arms['current']
        result.append(dict(pair=pair,case=case,output_tokens=count,**{
            k+'_change_percent':100*(b[k]/a[k]-1) for k in ('prefill_ms','decode_ms','ttft_ms')},
            generated_ids_equal=a['output_ids_sha256']==b['output_ids_sha256']))
    return result


def arm(binary,out,cases,env,context,prefill):
    out.mkdir(parents=True,exist_ok=False)
    argv=[str(binary),'--glm-pack',str(PACK),'--serve','--max-context',str(context),
          '--max-new','1024','--prefill',str(prefill),'--temperature','0','--eos-ids','2147483647']
    p0.write_json(out/'invocation.json',dict(argv=argv,environment={k:v for k,v in env.items()
        if k.startswith(('STRATA_','ABYSS_V100_'))},cpu_affinity=sorted(os.sched_getaffinity(0)),
        cache_state='new process; OS cache uncontrolled; subsequent cases inherit expert residency'))
    messages=queue.Queue();records=[];started=time.monotonic();aborted=threading.Event();stop=threading.Event()
    with Monitor(out) as monitor,(out/'engine.log').open('w') as log:
        proc=subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
        monitor.pid=proc.pid
        def reader():
            with (out/'wire.log').open('w') as f:
                for line in proc.stdout:
                    f.write(line);f.flush();messages.put((line.rstrip(),time.monotonic()))
            messages.put((None,time.monotonic()))
        def guard():
            while not stop.wait(2):
                if memory_available()<8*1024**3:
                    aborted.set();proc.kill();return
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        watcher=threading.Thread(target=guard,daemon=True);watcher.start()
        def until(prefix,timeout=7200):
            deadline=time.monotonic()+timeout;wire=[];first=None
            while True:
                line,stamp=messages.get(timeout=max(.1,deadline-time.monotonic()))
                if line is None or line.startswith('ERR '):
                    raise RuntimeError('memory safety stop' if aborted.is_set() else f'engine stopped: {line}')
                wire.append(line)
                if line.startswith('T ') and first is None:first=stamp
                if line.startswith(prefix):return line,wire,first
        try:
            until('READY ',1200)
            p0.write_json(out/'startup.json',dict(ready_wall_s=time.monotonic()-started))
            p0.write_json(out/'ready_placement.json',dict(status=read(f'/proc/{proc.pid}/status'),
                numa_maps=read(f'/proc/{proc.pid}/numa_maps'),smaps_rollup=read(f'/proc/{proc.pid}/smaps_rollup')))
            for name,ids,count in cases:
                for repetition in (0,1):
                    before=dict(io=read(f'/proc/{proc.pid}/io'),stat=read(f'/proc/{proc.pid}/stat'),
                                vmstat=read('/proc/vmstat'),loadavg=read('/proc/loadavg'))
                    begin=time.monotonic()
                    proc.stdin.write(f'GEN {count} temperature=0 '+','.join(map(str,ids))+'\n');proc.stdin.flush()
                    done,wire,first=until('DONE ')
                    row=parse_done(done,wire,len(ids),count)
                    row.update(case=name,request=repetition,begin_monotonic_s=begin,end_monotonic_s=time.monotonic(),
                        ttft_ms=(first-begin)*1000 if first else None,wire_wall_s=time.monotonic()-begin,
                        cache_state='process-first expert residency' if not records else 'carry-over expert residency; no prefix reuse',
                        host_before=before,host_after=dict(io=read(f'/proc/{proc.pid}/io'),stat=read(f'/proc/{proc.pid}/stat'),
                        vmstat=read('/proc/vmstat'),loadavg=read('/proc/loadavg')))
                    records.append(row);p0.write_json(out/'requests.json',records)
                    print(json.dumps(dict(arm=out.name,case=name,request=repetition,prompt_tokens=len(ids),
                        output_tokens=count,prefill_ms=row['prefill_ms'],decode_ms=row['decode_ms'])),flush=True)
            proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=120)
            if proc.returncode:raise RuntimeError(f'engine exit {proc.returncode}')
        finally:
            stop.set()
            if proc.poll() is None:proc.kill();proc.wait()
            thread.join(timeout=5);watcher.join(timeout=5)
    p0.write_json(out/'completion.json',dict(status='pass',requests=len(records),returncode=proc.returncode))
    return records


def quality(out,teacher,vocab,env):
    out.mkdir(exist_ok=True)
    # Keep raw distributions locally for resumability; only summaries/hashes are
    # copied to the bounded artifact. Reusing them never reloads the model.
    arrays={};identities={}
    for label,base,lane in [('historical-0','historical','28'),('current-0','current','28'),
                           ('current-1','current','28'),('historical-1','historical','28'),
                           ('gpu-only-0','historical','0'),('gpu-only-1','historical','0')]:
        directory=out/label;dump=out/(label+'.f32')
        if not (directory/'complete.json').exists():
            if directory.exists():raise RuntimeError(f'incomplete quality case preserved: {directory}; investigate before retry')
            score(BINARIES[base],directory,teacher,dump,dict(env,STRATA_GLM_CPU_LANE=lane))
            if dump.stat().st_size!=32*vocab*4:raise ValueError('wrong quality dump bytes')
            p0.write_json(directory/'complete.json',dict(sha256=p0.sha256(dump),bytes=dump.stat().st_size))
        identity=json.loads((directory/'complete.json').read_text())
        if p0.sha256(dump)!=identity['sha256']:raise ValueError('cached quality dump identity changed')
        arrays[label]=normalize_logprobs(np.fromfile(dump,dtype='<f4').reshape(32,vocab));identities[label]=identity
    comparisons=[]
    for a,b in [('historical-0','historical-1'),('current-0','current-1'),
                ('historical-0','current-0'),('historical-1','current-1'),
                ('gpu-only-0','gpu-only-1'),('historical-0','gpu-only-0')]:
        ref,test=arrays[a],arrays[b];kl=(np.exp(ref)*(ref-test)).sum(axis=1)
        comparisons.append(dict(reference=a,candidate=b,mean_kl=float(kl.mean()),max_kl=float(kl.max()),
            p99_kl=float(np.quantile(kl,.99)),top1_matches=int((ref.argmax(1)==test.argmax(1)).sum()),
            rows=32,within_exact_math_screen=bool(kl.mean()<=.0005),positions=[dict(
            predicted_token_index_zero_based=2041+i,kl=float(k)) for i,k in enumerate(kl)]))
    result=dict(scope='same-pack F32 baseline repeatability and CPU-lane isolation, not an optimization quality pass',
        model_identity_reference='../identity.json full shard SHA256',dumps=identities,comparisons=comparisons)
    p0.write_json(out/'quality.json',result)
    return result


def main():
    from model_bringup import NAMES, SIZES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['quality','extended','long','chunks','all'],default='all')
    selection=parser.parse_args().phase
    CAMPAIGN.mkdir(parents=True,exist_ok=True);out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(exist_ok=True,parents=True)
    harness=p0.sha256(Path(__file__))
    for name,binary in BINARIES.items():
        if p0.sha256(binary)!=EXPECTED[name]:raise ValueError('immutable upstream binary changed: '+name)
    for name,size in zip(NAMES,SIZES):
        if (Path('/models')/name).stat().st_size!=size:raise ValueError('model shard size mismatch')
    pack=json.loads((PACK/'v100_pack_complete.json').read_text())
    if pack['packer_sha256']!=p0.sha256(ROOT/'tools/iq_pack.py'):raise ValueError('pack provenance changed')
    model=model_identity()
    p0.write_json(out/'model-sha256.json',model)
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    gg=GGUFFile(Path('/models')/NAMES[0]);tok=Tokenizer.from_dir(PACK/'tokenizer')
    template=ChatTemplate(PACK/'tokenizer/chat_template.jinja',
        bos_token=gg.metadata['tokenizer.ggml.tokens'][gg.metadata['tokenizer.ggml.bos_token_id']])
    fixture_dir=CAMPAIGN/'fixtures';fixture_dir.mkdir(exist_ok=True)
    short,teacher=fixtures(fixture_dir,tok,template)
    text=''.join(f'Record {i}: compare memory placement and expert routing. '
        f'Español: explica {i%97}; 中文：比较计算和存储。 '
        f'def route_{i}(x): return sum(v*v for v in x) + {i%31}\n' for i in range(30000))
    rendered=template.render([{'role':'user','content':text+'\nExplain the evidence carefully in a detailed report.'}],enable_thinking=False)
    all_ids=tok.encode(rendered,parse_special=True);long={}
    for length in (32768,65536,131072):
        if len(all_ids)<length:raise ValueError('long fixture too short')
        ids=all_ids[:length-128]+all_ids[-128:];long[f'mixed-{length}']=ids
        (fixture_dir/f'mixed-{length}.tokens').write_text(','.join(map(str,ids))+'\n')
    identity=dict(base_shas=BASES,binary_sha256=EXPECTED,pack=pack,
        model_sha256=model['sha256'],model_integrity='full shard SHA256 computed before this campaign',
        fixtures={p.name:dict(bytes=p.stat().st_size,sha256=p0.sha256(p)) for p in fixture_dir.glob('*.tokens')})
    if (CAMPAIGN/'identity.json').exists() and json.loads((CAMPAIGN/'identity.json').read_text())!=identity:
        raise ValueError('campaign identity changed; preserve existing campaign and start a new version')
    p0.write_json(CAMPAIGN/'identity.json',identity)
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    if memory_available()<16*1024**3:raise RuntimeError('insufficient initial host memory safety margin')
    p0.write_json(out/'toolchain_manifest.json',p0.manifest(os.environ.get('ABYSS_GPU','0')))
    phases=[('extended',12288,'auto',[(n,ids,count) for n,ids in short.items() for count in (128,512,1024)]),
            ('long',135168,'auto',[(n,ids,1024) for n,ids in long.items()])]
    phases.extend((f'chunk-{chunk}',12288,str(chunk),[('mixed-8192',short['mixed-8192'],128)])
                  for chunk in (512,1024,4096))
    report=dict(schema_version=1,status='running',harness_sha256=harness,selected_base='historical',
                arms=[],chunk_128_disposition='unsupported: source chooses chunks on a 256-token grid; not relabeled as 128',
                cache_scope='process-first versus warmed requests; OS cache uncontrolled; no global cache drop')
    p0.write_json(CAMPAIGN/'progress.json',dict(active_phase=selection,status='running',harness_sha256=harness))
    try:
        if selection in ('quality','all'):
            report['quality']=quality(CAMPAIGN/'quality',teacher,len(gg.metadata['tokenizer.ggml.tokens']),env)
        for phase,context,prefill,cases in phases:
            if selection!='all' and selection!=phase and not (selection=='chunks' and phase.startswith('chunk-')):continue
            phase_rows=[]
            for index,name in enumerate(ORDER):
                directory=CAMPAIGN/f'{phase}-{index}-{name}'
                if (directory/'completion.json').exists():
                    rows=json.loads((directory/'requests.json').read_text())
                    if len(rows)!=2*len(cases):raise ValueError('completed arm request count changed')
                else:
                    if directory.exists():raise RuntimeError(f'incomplete arm preserved: {directory}; fix before retry')
                    if memory_available()<16*1024**3:raise RuntimeError('host safety margin below 16 GiB before arm')
                    rows=arm(BINARIES[name],directory,cases,env,context,prefill)
                item=dict(phase=phase,index=index,base=name,requests=rows)
                phase_rows.append(item);report['arms'].append(item)
                p0.write_json(CAMPAIGN/('progress-'+selection+'.json'),report)
                p0.write_json(CAMPAIGN/'progress.json',dict(active_phase=selection,status=report['status'],harness_sha256=harness))
            p0.write_json(CAMPAIGN/(phase+'-paired.json'),paired(phase_rows))
        report['status']='matrix complete; dispatch, unique memory and acceptance review still required'
    except Exception as e:
        report.update(status='stopped at first failure; no automatic unchanged retry',error=str(e));raise
    finally:
        p0.write_json(CAMPAIGN/('progress-'+selection+'.json'),report)
        p0.write_json(CAMPAIGN/'progress.json',dict(active_phase=selection,status=report['status'],harness_sha256=harness))
        # Never copy temporary full-vocabulary distributions into the artifact.
        for p in CAMPAIGN.rglob('*'):
            if p.is_file() and p.suffix!='.f32':
                target=out/'campaign'/p.relative_to(CAMPAIGN);target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(p.read_bytes())
        total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
        if total>64*1024*1024:raise ValueError('phase evidence exceeds 64 MiB cap; upload refused')
        if os.environ.get('GITHUB_ENV'):
            with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P0_'+selection.upper()+'_CAP=true\n')


if __name__=='__main__':main()
