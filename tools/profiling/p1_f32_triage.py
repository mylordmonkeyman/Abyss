#!/usr/bin/env python3
"""Diagnostic CPU-lane-off repeat at the FAILED 128K geometry; never admission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling')]
import p1_integrated_quality as quality
import numpy as np

BINARY_SHA='fc7a26eca2da31c1a63819b3b4a6440616cbedd59d775c4f70dc8cc51c8c2392'

def comparison(a,b,start):
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=32:
        raise ValueError('triage requires identical 32-position distributions')
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite logits')
    a=quality.normalize_logprobs(a);b=quality.normalize_logprobs(b)
    kl=(np.exp(a)*(a-b)).sum(axis=1)
    if not np.isfinite(kl).all():raise ValueError('nonfinite KL')
    return dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()),top1_matches=int((a.argmax(1)==b.argmax(1)).sum()),
        rows=32,positions=[dict(predicted_token_index_zero_based=start+1+i,kl=float(k)) for i,k in enumerate(kl)])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate',action='store_true',help='also compare cuBLAS twice under this controlled CPU-lane-off setting')
    args=ap.parse_args()
    out=Path(os.environ['EVIDENCE']).absolute()/'f32-triage';out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build-v100-diagnostics/strata'
    if quality.p0.sha256(binary)!=BINARY_SHA:raise ValueError('component-qualified binary changed')
    geometry=quality.GEOMETRIES['128k']
    model=quality.campaign.model_identity()
    identity=json.loads((quality.campaign.CAMPAIGN/'identity.json').read_text())
    if identity['model_sha256']!=model['sha256']:raise ValueError('frozen model changed')
    if identity['pack']!=json.loads((quality.PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    quality.p0.write_json(out/'model-identity.json',model)
    # Each run owns a new, bounded directory; no deletion of previous evidence.
    run=os.environ['GITHUB_RUN_ID'];attempt=os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not run.isdecimal() or not attempt.isdecimal():raise ValueError('invalid run identity')
    retained=Path('/actions-runner/abyss-p1-v9/triage')/(run+'-'+attempt)
    retained.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(retained).free<256*1024**2:raise RuntimeError('insufficient diagnostic disk margin')
    ids=quality.fixture('mixed-131072')
    if len(ids)!=geometry['token_count']:raise ValueError('wrong frozen fixture size')
    tokens=retained/'tokens';tokens.write_text(','.join(map(str,ids)))
    state=dict(schema=1,status='running; diagnostic only',source_sha=quality.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        source_run=run,binary_sha256=BINARY_SHA,geometry=geometry,fixture_sha256=quality.p0.sha256(tokens),
        retained_directory=str(retained),changed_variable={'STRATA_GLM_CPU_LANE':'0'},arms={},dumps={},
        scope='Two F32 repeats at the original failed partial-tail geometry with CPU expert lane disabled. '
        'This isolates a scheduling/arithmetic axis; even a pass does not repair or qualify the default CPU-lane-28 configuration. '
        'No automatic promotion, default-setting acceptance, throughput or root-cause claim. Raw logits stay runner-side (64 MiB total cap).',
        candidate_control=args.candidate)
    def checkpoint():
        quality.p0.write_json(out/'triage.json',state)
        quality.p0.write_json(retained/'checkpoint.json',state)
    checkpoint()
    original_env=quality.fixed_env
    def diagnostic_env():
        env=original_env();env['STRATA_GLM_CPU_LANE']='0';return env
    # Reuse the existing own-child memory guard, telemetry and dispatch checker.
    # Only this process's diagnostic environment adapter changes, never engine code.
    quality.fixed_env=diagnostic_env
    arrays={};comparisons=[]
    try:
        order=[('f32-0','f32'),('cublas-0','cublas'),('cublas-1','cublas'),('f32-1','f32')] if args.candidate else [('f32-0','f32'),('f32-1','f32')]
        for label,mode in order:
            dump=retained/(label+'.f32')
            print('Starting CPU-lane-off triage',label,flush=True)
            state['arms'][label]=quality.arm(binary,out/label,tokens,dump,mode,geometry)
            size=dump.stat().st_size
            if size<=0 or size%(32*4) or sum(p.stat().st_size for p in retained.glob('*.f32'))>64*1024**2:
                raise ValueError('unexpected distribution ABI or retained logit cap')
            state['dumps'][label]=dict(bytes=size,sha256=quality.p0.sha256(dump),path=str(dump))
            arrays[label]=np.fromfile(dump,dtype='<f4').reshape(32,-1)
            checkpoint()
        pairs=[('f32-0','f32-1')]
        if args.candidate:pairs += [('cublas-0','cublas-1'),('f32-0','cublas-0'),('f32-1','cublas-1')]
        for a,b in pairs:
            result=comparison(arrays[a],arrays[b],geometry['score_start'])
            result.update(reference=a,candidate=b)
            comparisons.append(result)
        state['comparisons']=comparisons
        state['comparison']=comparisons[0] # backwards-compatible baseline repeat
        passed=all(r['mean_kl']<=.0005 for r in comparisons)
        state['status']='diagnostic control '+('PASS' if passed else 'FAIL')+'; original P1 gate remains blocked'
        checkpoint()
        if not passed:raise ValueError('CPU-lane-off diagnostic exceeds original KL threshold')
    except BaseException as e:
        state['status']='diagnostic failed; original P1 gate remains blocked'
        state['error']=str(e);checkpoint();raise
    finally:quality.fixed_env=original_env

if __name__=='__main__':main()
