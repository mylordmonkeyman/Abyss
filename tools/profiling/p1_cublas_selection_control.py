#!/usr/bin/env python3
"""cuBLAS computes scores; diagnostic reference selection intervenes before attention."""
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
import p1_integrated_quality as q
from p1_f32_triage import comparison

ORDER=(('f32-0','f32',''),('reference-order','cublas','reference-order'),
       ('reference-membership','cublas','reference-membership'),('f32-1','f32',''))


def validate_controls(records,kind,geometry):
    expected={(str(layer),str(pos)) for layer in range(2,43,2) for pos in geometry['candidate_positions']}
    if len(records)!=len(expected) or {(r['layer'],r['p0']) for r in records}!=expected:
        raise ValueError('incomplete reference selection intervention')
    for row in records:
        if row['kind']!=kind or int(row['T'])!=2048 or int(row['rows'])!=2048:
            raise ValueError('wrong reference selection control/geometry')
        ordered,member,overlap=(int(row[k]) for k in ('ordered_equal_rows','membership_equal_rows','overlap'))
        if not 0<=ordered<=member<=2048 or not 0<=overlap<=2048*512:
            raise ValueError('invalid selection intervention accounting')


def main():
    out=Path(os.environ['EVIDENCE']).absolute()/'control';out.mkdir(parents=True,exist_ok=False)
    binary=q.ROOT/'build-v100-p1-cublas-control/strata';geometry=q.GEOMETRIES['128k']
    model=q.campaign.model_identity();frozen=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=frozen['model_sha256'] or frozen['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):
        raise ValueError('frozen model/pack changed')
    run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not all(s.isdecimal() for s in run.split('-')):raise ValueError('invalid run identity')
    cache=Path('/actions-runner/abyss-p1-v9/cublas-selection-control')/run;cache.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(cache).free<256*1024**2:raise RuntimeError('evidence disk margin below 256 MiB')
    ids=q.fixture('mixed-131072')
    if len(ids)!=131072:raise ValueError('wrong frozen fixture')
    tokens=cache/'tokens';tokens.write_text(','.join(map(str,ids)))
    state=dict(schema=1,status='running diagnostic intervention',source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        binary_sha256=q.p0.sha256(binary),model_identity=model,fixture_sha256=q.p0.sha256(tokens),geometry=geometry,
        cpu_lane=0,order=[label for label,_,_ in ORDER],arms={},dumps={},comparisons=[],
        scope='Original cuBLAS scoring retained; reference-order replaces membership and ordering before attention. '
        'Reference-membership substitutes F32 membership ordered by the original cuBLAS scores, ties by row ID. '
        'All 231 eligible calls intervened; same binary/pack/fixture, CPU lane 0, F32 repeats bracket controls. '
        'Selection counters compare original cuBLAS and F32 on each control trajectory, not an unchanged standalone '
        'cuBLAS trajectory. Extra F32 work/copies/host statistics synchronization are diagnostic only and excluded '
        'from performance. No candidate admission, production default or KL threshold change; model authenticity '
        'is limited to the local manifest, counters unavailable and QPI unmeasured.')
    def checkpoint():q.p0.write_json(out/'control.json',state);q.p0.write_json(cache/'checkpoint.json',state)
    checkpoint();arrays={}
    try:
        for label,mode,kind in ORDER:
            print('Starting cuBLAS selection intervention',label,flush=True)
            dump=cache/(label+'.f32');case=out/label
            arm=q.arm(binary,case,tokens,dump,mode,geometry,overrides=dict(STRATA_GLM_CPU_LANE='0',
                STRATA_V100_CUBLAS_CAPTURE='0',STRATA_V100_MMA_CAPTURE='0',STRATA_V100_CUBLAS_SELECTION_CONTROL=kind))
            with gzip.open(case/'engine.log.gz','rt') as stream:log=stream.read()
            records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_cublas_selection_control ')]
            if kind:validate_controls(records,kind,geometry)
            elif records:raise ValueError('F32 arm unexpectedly intervened')
            arm['selection_controls']=records;state['arms'][label]=arm
            size=dump.stat().st_size
            if size<=0 or size%(32*4) or sum(p.stat().st_size for p in cache.glob('*.f32'))>64*1024**2:
                raise ValueError('retained logit ABI/cap mismatch')
            state['dumps'][label]=dict(bytes=size,sha256=q.p0.sha256(dump),path=str(dump))
            arrays[label]=np.fromfile(dump,dtype='<f4').reshape(32,-1)
            checkpoint()
        for a,b in (('f32-0','f32-1'),('f32-0','reference-order'),('f32-0','reference-membership')):
            state['comparisons'].append(dict(reference=a,candidate=b,**comparison(arrays[a],arrays[b],geometry['score_start'])))
        if len({item['bytes'] for item in state['dumps'].values()})!=1:raise ValueError('model logit populations differ')
        state['f32_repeat_gate_pass']=state['comparisons'][0]['mean_kl']<=.0005
        state['reference_order_gate_pass']=state['comparisons'][1]['mean_kl']<=.0005
        state['membership_gate_pass']=state['comparisons'][2]['mean_kl']<=.0005
        state['status']='complete diagnostic intervention; inspect causality and ordering';checkpoint()
        if not state['f32_repeat_gate_pass'] or not state['reference_order_gate_pass']:
            raise ValueError('reference-order intervention or F32 repeat failed; inspect before correction trials')
    except BaseException as error:
        state['error']=str(error)
        if not state['status'].startswith('complete'):state['status']='incomplete; inspect cause before retry'
        checkpoint();raise


if __name__=='__main__':main()
