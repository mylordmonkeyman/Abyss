#!/usr/bin/env python3
"""Digest-check upstream and CPU-lane-off controls, preserving failed dispositions."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def summarize(archive,digest,run,kind):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('control artifact digest mismatch')
    prefix='upstream-repeat' if kind=='upstream' else 'f32-triage'
    with zipfile.ZipFile(archive) as z:
        def read(name):return json.loads(z.read(prefix+'/'+name))
        result=read('repeat.json' if kind=='upstream' else 'triage.json')
        source=z.read('source-sha.txt').decode().strip()
        if result['source_sha']!=source or str(result['source_run'])!=str(run):raise ValueError('source/run mismatch')
        if result['geometry']['score_start']!=131039 or result['geometry']['token_count']!=131072:raise ValueError('wrong failed geometry')
        labels={'upstream-0','upstream-1','working-0','working-1'} if kind=='upstream' else {'f32-0','f32-1','cublas-0','cublas-1'}
        if set(result['arms'])!=labels or set(result['dumps'])!=labels:raise ValueError('incomplete control population')
        if len({d['bytes'] for d in result['dumps'].values()})!=1:raise ValueError('dump ABI mismatch')
        pairs={('upstream-0','upstream-1'),('working-0','working-1'),('upstream-0','working-0'),('upstream-1','working-1')} if kind=='upstream' else {('f32-0','f32-1'),('cublas-0','cublas-1'),('f32-0','cublas-0'),('f32-1','cublas-1')}
        if len(result['comparisons'])!=4 or {(r['reference'],r['candidate']) for r in result['comparisons']}!=pairs:raise ValueError('comparison population mismatch')
        for row in result['comparisons']:
            if row['rows']!=32 or [p['predicted_token_index_zero_based'] for p in row['positions']]!=list(range(131040,131072)):raise ValueError('wrong scored positions')
            if not 0<=row['mean_kl']<=row['max_kl'] or not 0<=row['top1_matches']<=32:raise ValueError('invalid comparison metrics')
            row['passes_written_mean_kl_limit']=row['mean_kl']<=.0005
        telemetry={}
        for arm in labels:
            env=read(arm+'/invocation.json')['environment']
            if env['STRATA_GLM_CPU_LANE']!=('28' if kind=='upstream' else '0'):raise ValueError('wrong CPU lane control')
            if env['ABYSS_V100_IDX']!=('cublas' if arm.startswith('cublas') else 'f32'):raise ValueError('wrong scorer')
            if not arm.startswith('cublas') and env.get('STRATA_DS4_IDX')!='1':raise ValueError('missing forced-F32 override')
            if kind!='upstream':
                rows=result['arms'][arm].pop('dispatch_records')
                candidates=[r for r in rows if r['scorer']=='cublas']
                expected={(str(layer),str(pos)) for layer in range(2,43,2) for pos in range(98304,120001,2048)} if arm.startswith('cublas') else set()
                if len(candidates)!=len(expected) or {(r['layer'],r['p0']) for r in candidates}!=expected or any(r['T']!='2048' for r in candidates):raise ValueError('wrong actual dispatch coverage')
                result['arms'][arm]['dispatch_summary']=dict(records=len(rows),candidate_calls=len(candidates))
            memory=read(arm+'/memory_summary.json')
            if memory['samples']<=0:raise ValueError('missing telemetry')
            telemetry[arm]=dict(memory=memory,thermal=read(arm+'/thermal.json'))
        return dict(schema=1,run=run,source_sha=source,artifact_zip_sha256=digest,kind=kind,control=result,telemetry=telemetry,
            disposition='Characterization only; collection success is not quality acceptance. Original upstream repeats exceed the written limit at CPU lane 28. '
            'Under CPU lane 0 each scorer repeats bitwise, while F32/candidate drift independently exceeds the limit. Neither baseline subtraction nor a threshold waiver is justified. '
            'No production arithmetic/default or P1 admission change; raw logits remain on the existing runner. Local manifest integrity is not publisher authenticity; counters/QPI remain unavailable/unmeasured.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',type=int,required=True);p.add_argument('--kind',choices=('upstream','candidate'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run,a.kind),indent=2)+'\n')
