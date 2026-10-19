#!/usr/bin/env python3
"""v9 P0a: bounded, unmodified-SHA, same-pack V100 baseline comparison.

Does not rebase, repack, hash the 97-GB model, drop system caches, change clocks,
or infer that virtual mappings/pinned/RSS/cgroup charges are additive.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/v100'), str(ROOT / 'tools'), str(ROOT)]
import p0
from model_baseline import PACK, fixed_env, normalize_logprobs

BASES = {'historical': '47001f31cbfba14bcbf4c083e3af07c50707ec0c',
         'current': '7c893ac17e61ec45c934b3278356807fc4626c3c'}


def read(path):
    try:
        return Path(path).read_text(errors='replace')
    except OSError:
        return None


def kb_fields(text):
    return {k: int(v) * 1024 for k,v in re.findall(r'^(?:Node \d+ )?(\w+):\s+(\d+) kB$', text or '', re.M)}


def numa_bytes(text):
    totals = collections.Counter()
    for line in (text or '').splitlines():
        page = re.search(r'kernelpagesize_kB=(\d+)', line)
        size = int(page[1])*1024 if page else 4096
        for node,pages in re.findall(r'\bN(\d+)=(\d+)', line):
            totals[node] += int(pages) * size
    return dict(totals)


class Sampler:
    def __init__(self, out):
        self.out, self.pid = out, None
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def snapshot(self):
        sample = {'utc': p0.now(), 'monotonic_s': time.monotonic(),
                  'host_memory_bytes': kb_fields(read('/proc/meminfo')),
                  'vmstat': read('/proc/vmstat'), 'pressure_memory': read('/proc/pressure/memory'),
                  'cgroup_memory_current': read('/sys/fs/cgroup/memory.current'),
                  'cgroup_memory_events': read('/sys/fs/cgroup/memory.events'),
                  'cgroup_memory_stat': read('/sys/fs/cgroup/memory.stat'),
                  'nodes': {p.parent.name: kb_fields(read(p))
                            for p in Path('/sys/devices/system/node').glob('node*/meminfo')}}
        if self.pid:
            sample.update(process_status=kb_fields(read(f'/proc/{self.pid}/status')),
                          smaps_rollup=kb_fields(read(f'/proc/{self.pid}/smaps_rollup')),
                          process_io=read(f'/proc/{self.pid}/io'),
                          numa_mapped_bytes=numa_bytes(read(f'/proc/{self.pid}/numa_maps')),
                          process_stat=read(f'/proc/{self.pid}/stat'))
        raw = p0.command(['nvidia-smi','-i',os.environ.get('ABYSS_GPU','0'),'-q','-x'])
        sample['gpu_xml'] = raw.get('stdout')
        try:
            sample['gpu'] = p0.xml_sample(raw['stdout'])
        except (KeyError, ValueError) as e:
            sample['gpu'] = {'error': str(e)}
        self.samples.append(sample)
        return sample

    def loop(self):
        with (self.out / 'telemetry.jsonl').open('w') as f:
            while not self.stop.is_set():
                f.write(json.dumps(self.snapshot())+'\n'); f.flush()
                self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set(); self.thread.join()
        gpu = [x['gpu'] for x in self.samples]
        p0.write_json(self.out / 'thermal.json', p0.thermal_assessment(gpu))
        peak = lambda part,key: max((s.get(part,{}).get(key,0) for s in self.samples), default=0)
        summary = {'peak_rss_bytes': peak('process_status','VmRSS'),
                   'peak_pss_bytes': peak('smaps_rollup','Pss'),
                   'peak_anonymous_bytes': peak('smaps_rollup','Anonymous'),
                   'peak_anon_huge_bytes': peak('smaps_rollup','AnonHugePages'),
                   'peak_vmlck_bytes_not_cuda_pin_accounting': peak('process_status','VmLck'),
                   'minimum_host_memavailable_bytes': min((s['host_memory_bytes'].get('MemAvailable',0)
                                                          for s in self.samples), default=0),
                   'accounting': 'RSS/PSS/anonymous/THP and cgroup charge overlap; never add them.',
                   'samples': len(self.samples)}
        p0.write_json(self.out / 'memory_summary.json', summary)


def parse_done(done, wire, expected_prompt, generated_count):
    fields = done.split()
    generated = [int(x.split()[1]) for x in wire if x.startswith('T ')]
    if (len(fields) < 9 or int(fields[1]) != generated_count or int(fields[2]) != expected_prompt
            or int(fields[8]) != 0 or len(generated) != generated_count):
        raise ValueError(f'wrong lengths or prefix reuse: {done}')
    return {'prompt_tokens':expected_prompt,'output_tokens':generated_count,
            'prefill_ms':float(fields[3]),'decode_ms':float(fields[4]),'done':done,
            'prefill_tokens_s':expected_prompt*1000/float(fields[3]),
            'decode_tokens_s':generated_count*1000/float(fields[4]),
            'output_ids_sha256':hashlib.sha256(','.join(map(str,generated)).encode()).hexdigest(),
            'output_ids': generated}


def bench(binary, out, fixtures, env):
    out.mkdir(parents=True, exist_ok=False)
    argv=[str(binary),'--glm-pack',str(PACK),'--serve','--max-context','12288',
          '--max-new','128','--prefill','auto','--temperature','0','--eos-ids','2147483647']
    records, messages = [], queue.Queue()
    started=time.monotonic()
    p0.write_json(out/'invocation.json',{'argv':argv,'environment':{k:v for k,v in env.items()
                   if k.startswith(('STRATA_','ABYSS_'))},'cpu_affinity':sorted(os.sched_getaffinity(0)),
                   'cache_state':'new process per arm, OS file cache uncontrolled; no global cache drop'})
    with Sampler(out) as sampler, (out/'engine.log').open('w') as log:
        proc=subprocess.Popen(argv,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                              text=True,bufsize=1)
        sampler.pid=proc.pid
        def reader():
            with (out/'wire.log').open('w') as f:
                for line in proc.stdout:
                    f.write(line); f.flush(); messages.put((line.rstrip(),time.monotonic()))
            messages.put((None,time.monotonic()))
        thread=threading.Thread(target=reader,daemon=True);thread.start()
        def until(prefix, timeout=1200):
            deadline=time.monotonic()+timeout; lines=[]; first_token=None
            while True:
                line,stamp=messages.get(timeout=max(0.1,deadline-time.monotonic()))
                if line is None or line.startswith('ERR '):
                    raise RuntimeError(f'engine stopped at {prefix}: {line}; inspect {out}/engine.log')
                lines.append(line)
                if line.startswith('T ') and first_token is None: first_token=stamp
                if line.startswith(prefix): return line,lines,first_token
        try:
            until('READY ')
            p0.write_json(out/'startup.json',{'ready_wall_s':time.monotonic()-started})
            p0.write_json(out/'ready_placement.json',{
                'status':read(f'/proc/{proc.pid}/status'),'numa_maps':read(f'/proc/{proc.pid}/numa_maps'),
                'smaps_rollup':read(f'/proc/{proc.pid}/smaps_rollup'),
                'threads':{t.name:read(t/'status') for t in Path(f'/proc/{proc.pid}/task').glob('*')}})
            for case,ids in fixtures.items():
                for request in range(2):
                    t0=time.monotonic()
                    proc.stdin.write('GEN 128 temperature=0 '+','.join(map(str,ids))+'\n');proc.stdin.flush()
                    done,wire,first_token=until('DONE ')
                    row=parse_done(done,wire,len(ids),128)
                    row.update(case=case,request=request,wire_wall_s=time.monotonic()-t0,
                               ttft_ms=(first_token-t0)*1000 if first_token is not None else None,
                               cache_state='first prompt of process' if not records else
                                           'same-process warmed tiers; prefix reuse disabled')
                    records.append(row);p0.write_json(out/'requests.json',records)
                    print(json.dumps({'arm':out.name,**{k:v for k,v in row.items() if k!='output_ids'}}),flush=True)
            proc.stdin.write('QUIT\n');proc.stdin.flush();proc.wait(timeout=120)
            if proc.returncode: raise RuntimeError(f'engine exit {proc.returncode}')
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
            thread.join(timeout=5)
    p0.write_json(out/'completion.json',{'returncode':proc.returncode,'wall_s_including_load':time.monotonic()-started})
    return records


def score(binary, out, tokens, dump, env):
    out.mkdir(parents=True,exist_ok=False)
    context, rows = 2040,32
    score_env=dict(env,STRATA_GLM_SCORE=str(context),STRATA_GLM_SCORE_DUMP=str(dump),STRATA_GLM_SCORE_PREFILL='1')
    argv=[str(binary),'--glm-pack',str(PACK),'--tokens-file',str(tokens),'--max-new','1',
          '--max-context','4096','--prefill','auto','--temperature','0']
    p0.write_json(out/'invocation.json',{'argv':argv,'environment':{k:v for k,v in score_env.items() if k.startswith('STRATA_')},
                                      'context':context,'scored_rows':rows})
    with Sampler(out) as sampler,(out/'engine.log').open('w') as log:
        proc=subprocess.Popen(argv,env=score_env,stdout=log,stderr=subprocess.STDOUT)
        sampler.pid=proc.pid
        try: rc=proc.wait(timeout=1800)
        finally:
            if proc.poll() is None: proc.kill();proc.wait()
    if rc: raise RuntimeError(f'score exit {rc}: {out}')


def fixtures(out, tok, template):
    texts={
        'prose-600': ''.join(f'Paragraph {i}: A researcher compares memory placement, computation, and data movement. '
                           'Explain which observations support each conclusion and distinguish latency from throughput.\n'
                           for i in range(1000)),
        'code-4096': ''.join(f'def compute_{i}(values):\n    # case {i}; preserve empty inputs\n'
                            f'    return sum((x + {i%19}) ** 2 for x in values)\n\n' for i in range(1000)),
        'mixed-8192': ''.join(f'Entry {i}: English: compare {i%97} and {(i*7)%101}. '
                             'Español: explica el resultado. 中文：比较内存和计算。 '
                             'Français: justifiez chaque étape. 日本語：手順を説明してください。\n'
                             for i in range(1000))}
    result={}
    for name,text in texts.items():
        length=int(name.split('-')[-1])
        rendered=template.render([{'role':'user','content':text+'\nExplain the evidence in a detailed report.'}],
                                 enable_thinking=False)
        all_ids=tok.encode(rendered,parse_special=True)
        if len(all_ids)<length: raise ValueError('fixture too short')
        ids=all_ids[:length-128]+all_ids[-128:]
        (out/(name+'.tokens')).write_text(','.join(map(str,ids))+'\n')
        result[name]=ids
    teacher=[0]+tok.encode(texts['code-4096']+texts['mixed-8192'])[:2072]
    if len(teacher)!=2073: raise ValueError('teacher fixture length')
    tf=out/'teacher.tokens';tf.write_text(','.join(map(str,teacher))+'\n')
    p0.write_json(out/'fixtures.json',{p.name:dict(bytes=p.stat().st_size,sha256=p0.sha256(p))
                                      for p in out.glob('*.tokens')})
    return result,tf


def compare_dumps(reference, candidate):
    ref, test = normalize_logprobs(reference),normalize_logprobs(candidate)
    if ref.shape != test.shape: raise ValueError('mismatched distributions')
    kl=(np.exp(ref)*(ref-test)).sum(axis=1)
    return dict(mean_kl=float(kl.mean()),p99_kl=float(np.quantile(kl,.99)),max_kl=float(kl.max()),
                top1_matches=int((ref.argmax(axis=1)==test.argmax(axis=1)).sum()),rows=len(kl),
                positions=[{'predicted_token_index_zero_based':2041+i,'kl':float(k),
                            'same_top1':bool(ref[i].argmax()==test[i].argmax())} for i,k in enumerate(kl)],
                exact_math_review_limit=0.0005,within_incremental_screen=bool(kl.mean()<=.0005),
                interpretation='Base comparison, not external-reference KL, task quality or complete base admission')


def main():
    from model_bringup import NAMES, SIZES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical',type=Path,required=True)
    parser.add_argument('--current',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();out=a.output.absolute();out.mkdir(parents=True,exist_ok=False)
    engines={'historical':a.historical.absolute(),'current':a.current.absolute()}
    for n,size in zip(NAMES,SIZES):
        if (Path('/models')/n).stat().st_size!=size: raise ValueError('wrong model sizes')
    signature=json.loads((PACK/'v100_pack_complete.json').read_text())
    if signature['packer_sha256']!=p0.sha256(ROOT/'tools/iq_pack.py'): raise ValueError('pack changed')
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    gg=GGUFFile(Path('/models')/NAMES[0]);tok=Tokenizer.from_dir(PACK/'tokenizer')
    template=ChatTemplate(PACK/'tokenizer/chat_template.jinja',
                          bos_token=gg.metadata['tokenizer.ggml.tokens'][gg.metadata['tokenizer.ggml.bos_token_id']])
    cases,teacher=fixtures(out,tok,template)
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    report={'schema_version':1,'specification':'9.0','base_shas':BASES,
            'binary_sha256':{k:p0.sha256(v) for k,v in engines.items()},'pack':signature,
            'model_integrity':'exact sizes, same read-only shards/pack; SHA256 unverified',
            'harness_commit':p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
            'settings':{k:v for k,v in env.items() if k.startswith('STRATA_')},
            'arms':[],'selected_base':None,'status':'running; no base selected'}
    p0.write_json(out/'bases.json',report)
    # A/B, B/A, A/B: three new-process samples per arm, two requests per fixture.
    for index,name in enumerate(['historical','current','current','historical','historical','current']):
        arm=out/f'{index}-{name}'
        try:
            rows=bench(engines[name],arm,cases,env)
            report['arms'].append({'index':index,'base':name,'status':'pass','requests':rows})
        except Exception as e:
            report['arms'].append({'index':index,'base':name,'status':'failed','error':str(e)})
            report['status']='first runtime failure; remaining campaign stopped'
            p0.write_json(out/'bases.json',report);raise
        p0.write_json(out/'bases.json',report)
    vocab=len(gg.metadata['tokenizer.ggml.tokens'])
    with tempfile.TemporaryDirectory(prefix='abyss-v9-logits-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        raw=[];hashes={}
        for name in BASES:
            dump=Path(tmp)/(name+'.f32');score(engines[name],out/('score-'+name),teacher,dump,env)
            if dump.stat().st_size!=32*vocab*4: raise ValueError('wrong logit dump length')
            hashes[name]={'bytes':dump.stat().st_size,'sha256':p0.sha256(dump)}
            raw.append(np.fromfile(dump,dtype='<f4').reshape(32,vocab))
        report['teacher_forced']=compare_dumps(*raw);report['teacher_forced']['dumps']=hashes
    report['status']='initial matrix complete; extended P0/placement/thermal review required before base selection'
    report['unresolved']=['Full memory component/pin ledger incomplete; samples are not additive',
                          'Cold means process start only; OS file cache uncontrolled',
                          '128 output tokens measured; 512/1024 and filled 32K+ remain',
                          'Single teacher-forced corpus, no complete task/reuse/rollback gate',
                          'Actual per-layer dispatch counters and clean sustained matching remain',
                          'QPI/hardware counters unavailable; no fabricated traffic/occupancy']
    p0.write_json(out/'bases.json',report)
    print(json.dumps({'status':report['status'],'teacher_forced':report['teacher_forced']}),flush=True)


if __name__=='__main__': main()
