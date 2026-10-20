#!/usr/bin/env python3
"""Same-binary long-context candidate/repeat screen; no model speed claim."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools')]
import numpy as np
import p0
import v100_p0_campaign as campaign
from p0_accounting import fixture,compress
from model_baseline import fixed_env,normalize_logprobs,PACK

# First candidate chunk and full 128K accumulation use the same frozen corpus.
GEOMETRIES={
    '100k':dict(score_start=100353,token_count=100386,max_context=102400,candidate_positions=[98304]),
    '128k':dict(score_start=131039,token_count=131072,max_context=135168,candidate_positions=list(range(98304,120001,2048))),
}

def arm(binary,out,tokens,dump,mode,geometry,overrides=None,cwd=None):
    out.mkdir()
    env=fixed_env();env.update(ABYSS_V100_IDX=mode,STRATA_DS4_ATTN='1',
        STRATA_GLM_SCORE=str(geometry['score_start']),STRATA_GLM_SCORE_DUMP=str(dump),
        STRATA_GLM_SCORE_PREFILL='1',ABYSS_V100_PROFILE='1')
    env.update(overrides or {})
    if mode=='f32':env['STRATA_DS4_IDX']='1'
    argv=[str(binary),'--glm-pack',str(PACK),'--tokens-file',str(tokens),
          '--max-new','1','--max-context',str(geometry['max_context']),'--prefill','auto','--temperature','0']
    p0.write_json(out/'invocation.json',dict(argv=argv,environment={k:v for k,v in env.items()
        if k.startswith(('STRATA_','ABYSS_V100_'))}))
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB before quality arm')
    started=time.monotonic()
    with campaign.Monitor(out) as monitor,(out/'engine.log').open('w') as log:
        child=subprocess.Popen(argv,env=env,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('own-child quality stop below 8 GiB')
                if time.monotonic()-started>1800:raise TimeoutError('quality arm exceeded 1800 s')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('quality process failed; inspect engine log')
        finally:
            if child.poll() is None:child.kill();child.wait()
    text=(out/'engine.log').read_text()
    records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in text.splitlines() if line.startswith('p1_indexer ')]
    candidate=[r for r in records if r.get('scorer') in ('cublas','volta')]
    if mode in ('cublas','volta'):
        expected={(str(layer),str(pos)) for layer in range(2,43,2) for pos in geometry['candidate_positions']}
        if len(candidate)!=len(expected) or {(r['layer'],r['p0']) for r in candidate}!=expected or any(r['T']!='2048' or r['scorer']!=mode for r in candidate):
            raise ValueError('candidate did not run at every intended real-model layer/chunk')
    elif candidate:raise ValueError('F32 arm unexpectedly dispatched a candidate')
    if not records:raise ValueError('missing actual scorer dispatch records')
    compress(out/'engine.log')
    result=dict(mode=mode,dispatch_records=records,candidate_calls=len(candidate),
                diagnostic_wall_s_including_load=time.monotonic()-started)
    p0.write_json(out/'dispatch.json',result)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--geometry',choices=GEOMETRIES,default='100k')
    args=ap.parse_args();geometry=GEOMETRIES[args.geometry]
    out=Path(os.environ['EVIDENCE']).absolute()/'model-quality';out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build-v100-diagnostics/strata'
    model=campaign.model_identity();identity=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if identity['model_sha256']!=model['sha256']:raise ValueError('frozen model identity changed')
    if identity['pack']!=json.loads((PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    p0.write_json(out/'model-identity.json',model)
    ids=fixture('mixed-131072')[:geometry['token_count']]
    if len(ids)!=geometry['token_count']:raise ValueError('frozen fixture too short')
    arrays={};dumps={};arms={}
    with tempfile.TemporaryDirectory(prefix='abyss-p1-integrated-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tokens=Path(tmp)/'tokens';tokens.write_text(','.join(map(str,ids)))
        fixture_sha=p0.sha256(tokens)
        for label,mode in [('f32-0','f32'),('cublas-0','cublas'),('cublas-1','cublas'),('f32-1','f32')]:
            dump=Path(tmp)/(label+'.f32')
            print('Starting quality arm',label,args.geometry,flush=True)
            arms[label]=arm(binary,out/label,tokens,dump,mode,geometry)
            print('Completed quality arm',label,flush=True)
            size=dump.stat().st_size
            if size<=0 or size%(32*4):raise ValueError('unexpected 32-position distribution ABI')
            vocab=size//(32*4)
            arrays[label]=normalize_logprobs(np.fromfile(dump,dtype='<f4').reshape(32,vocab))
            dumps[label]=dict(bytes=size,sha256=p0.sha256(dump),vocab=vocab)
        if len({v['bytes'] for v in dumps.values()})!=1:raise ValueError('distribution sizes differ')
        comparisons=[]
        for a,b in [('f32-0','f32-1'),('cublas-0','cublas-1'),('f32-0','cublas-0'),('f32-1','cublas-1')]:
            ref,test=arrays[a],arrays[b];kl=(np.exp(ref)*(ref-test)).sum(axis=1)
            if not np.isfinite(kl).all():raise ValueError('nonfinite KL')
            comparisons.append(dict(reference=a,candidate=b,mean_kl=float(kl.mean()),max_kl=float(kl.max()),
                top1_matches=int((ref.argmax(1)==test.argmax(1)).sum()),rows=32,
                positions=[dict(predicted_token_index_zero_based=geometry['score_start']+1+i,kl=float(k)) for i,k in enumerate(kl)]))
        result=dict(schema=1,source_sha=p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
            source_run=os.environ.get('GITHUB_RUN_ID'),binary_sha256=p0.sha256(binary),
            geometry=geometry,fixture=dict(parent='mixed-131072.tokens',prefix_tokens=geometry['token_count'],sha256=fixture_sha),
            arms=arms,dumps=dumps,comparisons=comparisons,scope='Same-pack same-binary explicit experimental '
            'cuBLAS long-context downstream logits and repeatability screen; profiles/dispatch logging enabled. '
            'No whole-model speed, direct downstream-attention or full P1 acceptance claim.')
        p0.write_json(out/'quality.json',result)
        if any(x['mean_kl']>.0005 for x in comparisons):raise ValueError('P1 mean KL gate exceeded')

if __name__=='__main__':main()
