#!/usr/bin/env python3
"""Restore the missing exact-128K upstream/F32 repeatability control."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling'))
import p1_f32_triage as triage
q=triage.quality
np=triage.np
ORDER=(('upstream-0','upstream'),('working-0','working'),('working-1','working'),('upstream-1','upstream'))

def environment(dump):
    env=q.fixed_env()
    env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1',ABYSS_V100_IDX='f32',ABYSS_V100_PROFILE='1',
               STRATA_GLM_SCORE='131039',STRATA_GLM_SCORE_DUMP=str(dump),STRATA_GLM_SCORE_PREFILL='1')
    return env

def score(binary,out,tokens,dump):
    out.mkdir();env=environment(dump)
    argv=[str(binary),'--glm-pack',str(q.PACK),'--tokens-file',str(tokens),'--max-new','1',
          '--max-context','135168','--prefill','auto','--temperature','0']
    q.p0.write_json(out/'invocation.json',dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith(('STRATA_','ABYSS_V100_'))}))
    if q.campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB before baseline arm')
    started=time.monotonic()
    try:
        with q.campaign.Monitor(out) as monitor,(out/'engine.log').open('w') as log:
            child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT);monitor.pid=child.pid
            try:
                while child.poll() is None:
                    if q.campaign.memory_available()<8*1024**3:raise RuntimeError('own-child baseline stop below 8 GiB')
                    if time.monotonic()-started>1800:raise TimeoutError('baseline arm exceeded 1800 s')
                    try:child.wait(timeout=.25)
                    except subprocess.TimeoutExpired:pass
                if child.returncode:raise RuntimeError('baseline process failed; inspect preserved engine log')
            finally:
                if child.poll() is None:child.kill();child.wait()
    finally:
        if (out/'engine.log').exists():q.compress(out/'engine.log')
    return dict(binary_sha256=q.p0.sha256(binary),diagnostic_wall_s_including_load=time.monotonic()-started,
                dispatch_expectation='Original upstream has no P1 dispatch instrumentation; legacy STRATA_DS4_IDX=1 forces F32 in both binaries')

def main():
    out=Path(os.environ['EVIDENCE']).absolute()/'upstream-repeat';out.mkdir(parents=True,exist_ok=False)
    binaries=dict(upstream=q.campaign.BINARIES['current'],working=ROOT/'build-v100-diagnostics/strata')
    expected=dict(upstream=q.campaign.EXPECTED['current'],working=triage.BINARY_SHA)
    for name,binary in binaries.items():
        if q.p0.sha256(binary)!=expected[name]:raise ValueError('immutable/qualified binary changed: '+name)
    model=q.campaign.model_identity();identity=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if identity['model_sha256']!=model['sha256']:raise ValueError('frozen model identity changed')
    if identity['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    q.p0.write_json(out/'model-identity.json',model)
    run=os.environ['GITHUB_RUN_ID'];attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not run.isdecimal() or not attempt.isdecimal():raise ValueError('invalid run identity')
    retained=Path('/actions-runner/abyss-p1-v9/upstream-repeat')/(run+'-'+attempt)
    retained.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(retained).free<256*1024**2:raise RuntimeError('insufficient diagnostic disk margin')
    ids=q.fixture('mixed-131072')
    if len(ids)!=131072:raise ValueError('wrong frozen fixture')
    tokens=retained/'tokens';tokens.write_text(','.join(map(str,ids)))
    previous=json.loads((ROOT/'docs/v100/p1-128k-failure-2026-10-20.json').read_text())['quality']
    if q.p0.sha256(tokens)!=previous['fixture']['sha256']:raise ValueError('input differs from failed 128K gate')
    sizes={d['bytes'] for d in previous['dumps'].values()}
    if len(sizes)!=1:raise ValueError('inconsistent original ABI')
    expected_size=sizes.pop()
    if expected_size<=0 or expected_size%(32*4) or expected_size*4>64*1024**2:raise ValueError('unexpected original dump ABI')
    state=dict(schema=1,status='running; baseline characterization',source_run=run,
        source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        upstream_source_sha='7c893ac17e61ec45c934b3278356807fc4626c3c',binary_sha256=expected,
        geometry=q.GEOMETRIES['128k'],fixture_sha256=q.p0.sha256(tokens),retained_directory=str(retained),
        order=[label for label,_ in ORDER],arms={},dumps={},comparisons=[],
        scope='Exact failed 128K geometry, CPU lane 28 and fixed settings. Original selected upstream and working F32, '
        'interleaved upstream/working/working/upstream. No candidate arithmetic. Characterizes baseline repeat and '
        'port drift, not known-correct answers or optimization admission. Do not subtract baseline KL or relax thresholds. '
        'Raw logits retained runner-side under 64 MiB cap; wall times include load and diagnostics, not speed evidence. '
        'Model integrity is against the local manifest, not independent publisher authenticity; hardware counters/QPI remain unavailable/unmeasured.')
    def checkpoint():
        q.p0.write_json(out/'repeat.json',state);q.p0.write_json(retained/'checkpoint.json',state)
    checkpoint();arrays={}
    try:
        for label,base in ORDER:
            print('Starting baseline arm',label,flush=True);dump=retained/(label+'.f32')
            state['arms'][label]=score(binaries[base],out/label,tokens,dump)
            if dump.stat().st_size!=expected_size:raise ValueError('original/working distribution ABI changed')
            state['dumps'][label]=dict(bytes=expected_size,sha256=q.p0.sha256(dump),path=str(dump))
            arrays[label]=np.fromfile(dump,dtype='<f4').reshape(32,-1)
            # Preserve upstream repeat as soon as its second arm completes, plus
            # matched port comparisons. Every result is checkpointed separately.
            for a,b in [('upstream-0','upstream-1'),('working-0','working-1'),('upstream-0','working-0'),('upstream-1','working-1')]:
                if a in arrays and b in arrays and not any(r['reference']==a and r['candidate']==b for r in state['comparisons']):
                    result=triage.comparison(arrays[a],arrays[b],131039)
                    result.update(reference=a,candidate=b,passes_written_mean_kl_limit=result['mean_kl']<=.0005)
                    state['comparisons'].append(result)
            checkpoint();print('Completed baseline arm',label,flush=True)
        state['status']='complete; baseline characterized; P1 admission remains separate'
        checkpoint()
        # Characterization completes even when upstream itself exceeds the gate.
        # Explicit per-comparison dispositions prevent a green collection job
        # from being mistaken for P1 acceptance.
    except BaseException as e:
        state.update(status='incomplete; collection failed',error=str(e));checkpoint();raise

if __name__=='__main__':main()
