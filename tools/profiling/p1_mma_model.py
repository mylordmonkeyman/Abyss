#!/usr/bin/env python3
"""Actual scaled-MMA model dispatch, live capture and deterministic KL repeats."""
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import p1_integrated_quality as q
from p1_f32_triage import comparison
from p1_live_capture import compare_capture

ORDER=(('f32-0','f32'),('volta-0','volta'),('volta-1','volta'),('f32-1','f32'))


def main():
    out=Path(os.environ['EVIDENCE']).absolute()/'model';out.mkdir(parents=True,exist_ok=False)
    binary=q.ROOT/'build-v100-p1-mma-model-refined/strata';geometry=q.GEOMETRIES['128k']
    model=q.campaign.model_identity();frozen=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=frozen['model_sha256'] or frozen['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):
        raise ValueError('frozen model/pack changed')
    run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not all(s.isdecimal() for s in run.split('-')):raise ValueError('invalid model run identity')
    cache=Path('/actions-runner/abyss-p1-v9/mma-model')/run;cache.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(cache).free<256*1024**2:raise RuntimeError('model evidence disk margin below 256 MiB')
    ids=q.fixture('mixed-131072')
    if len(ids)!=131072:raise ValueError('wrong frozen fixture length')
    tokens=cache/'tokens';tokens.write_text(','.join(map(str,ids)))
    state=dict(schema=1,status='running; experimental scaled MMA',source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        binary_sha256=q.p0.sha256(binary),fixture_sha256=q.p0.sha256(tokens),model_identity=model,
        geometry=geometry,cpu_lane=0,scorer_implementation='prepared-mma-empirical-cutoff-f32-batched-select',
        order=[label for label,_ in ORDER],arms={},dumps={},comparisons=[],
        scope='Same binary, original F32 versus explicit scaled eight-query MMA884 with empirical cutoff-band operational-F32 '
        'rescore and two batched unchanged selectors, CPU lane 0 to characterize deterministic repeats. '
        'All 231 eligible layer/chunk calls required; first candidate captures late-chunk same-input scores/selection/attention '
        'and five query rows per C4 layer, then restores candidate IDs before continuation. '
        'Diagnostic synchronization excludes these arms from speed claims. Mean KL limit remains 0.0005. '
        'Default CPU-lane-28 quality, task corpus, release and clean performance qualification remain separate. '
        'Model verification is against local manifest; publisher authenticity and hardware-counter/QPI limitations remain unchanged.')
    def checkpoint():
        q.p0.write_json(out/'quality.json',state);q.p0.write_json(cache/'checkpoint.json',state)
    checkpoint();arrays={}
    try:
        for label,mode in ORDER:
            dump=cache/(label+'.f32');case=out/label
            print('Starting scaled MMA model arm',label,flush=True)
            state['arms'][label]=q.arm(binary,case,tokens,dump,mode,geometry,
                overrides=dict(STRATA_GLM_CPU_LANE='0',STRATA_V100_MMA_CAPTURE='1' if label=='volta-0' else '0',
                    ABYSS_V100_CAPTURE_P0='118784'),cwd=case)
            if mode=='volta' and any(r.get('refinement')!='empirical-band-f32' for r in state['arms'][label]['dispatch_records'] if r['scorer']=='volta'):
                raise ValueError('actual model did not execute the qualified rescore implementation')
            size=dump.stat().st_size
            if size<=0 or size%(32*4) or sum(p.stat().st_size for p in cache.glob('*.f32'))>64*1024**2:
                raise ValueError('unexpected model logit ABI or retained cap')
            state['dumps'][label]=dict(bytes=size,sha256=q.p0.sha256(dump),path=str(dump))
            arrays[label]=np.fromfile(dump,dtype='<f4').reshape(32,-1)
            capture=case/'p1-indexer-capture'
            if label=='volta-0':
                direct=compare_capture(capture,118784)
                direct['scope']=direct['scope'].replace('cuBLAS path','scaled MMA884 path')
                ranges=[]
                for layer in range(2,43,2):
                    item=json.loads((capture/f'layer-{layer}'/'query-range.json').read_text())
                    if item['sampled_values']!=5*64*128 or item['sampled_heads']!=5*64:
                        raise ValueError('live query range sample population mismatch')
                    ranges.append(dict(layer=layer,**item))
                state['query_ranges']=ranges
                state['direct_summary']=dict(samples=len(direct['samples']),
                    maximum_score_relative_l2=max(r['score_relative_l2'] for r in direct['samples']),
                    maximum_attention_relative_l2=max(r['attention_relative_l2'] for r in direct['samples']),
                    full_selection=direct.get('full_selection',[]))
                q.p0.write_json(out/'direct.json',direct)
            elif capture.exists():raise ValueError('unexpected duplicate model capture')
            for a,b in [('f32-0','f32-1'),('volta-0','volta-1'),('f32-0','volta-0'),('f32-1','volta-1')]:
                if a in arrays and b in arrays and not any(r['reference']==a and r['candidate']==b for r in state['comparisons']):
                    state['comparisons'].append(dict(reference=a,candidate=b,**comparison(arrays[a],arrays[b],geometry['score_start'])))
            checkpoint();print('Completed scaled MMA model arm',label,flush=True)
        if len({v['bytes'] for v in state['dumps'].values()})!=1:raise ValueError('model distribution sizes differ')
        state['mean_kl_gate_pass']=all(r['mean_kl']<=.0005 for r in state['comparisons'])
        state['attention_investigation']=state['direct_summary']['maximum_attention_relative_l2']>1e-4
        state['status']='complete; inspect model qualification'
        checkpoint()
        if not state['mean_kl_gate_pass']:raise ValueError('scaled MMA full-128K mean KL gate failed')
        if state['attention_investigation']:raise ValueError('scaled MMA attention exceeds investigation screen')
    except BaseException as error:
        state['error']=str(error)
        if not state['status'].startswith('complete;'):state['status']='incomplete; inspect cause before retry'
        checkpoint();raise


if __name__=='__main__':main()
