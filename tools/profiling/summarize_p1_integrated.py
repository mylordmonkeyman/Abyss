#!/usr/bin/env python3
"""Digest-checked long-context P1 runtime evidence; no automatic promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from summarize_p1_batched import summarize as component


def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('integration artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):return json.loads(z.read('model-quality/'+name))
        quality=read('quality.json');source=z.read('source-sha.txt').decode().strip()
        if quality['source_sha']!=source or str(quality['source_run'])!=str(run):raise ValueError('quality/source run mismatch')
        if quality['fixture']['prefix_tokens']!=100386:raise ValueError('wrong long-context fixture prefix')
        required={'f32-0','f32-1','cublas-0','cublas-1'}
        if set(quality['arms'])!=required or set(quality['dumps'])!=required:raise ValueError('incomplete quality/repeat population')
        if len({(d['bytes'],d['vocab']) for d in quality['dumps'].values()})!=1:raise ValueError('mismatched distribution ABI')
        dispatch={};memory={};thermal={}
        candidate_layers=[]
        for arm in sorted(required):
            rows=quality['arms'][arm]['dispatch_records']
            inv=read(arm+'/invocation.json');env=inv['environment']
            if env['ABYSS_V100_IDX']!=arm.split('-')[0]:raise ValueError('wrong runtime mode')
            if (env.get('STRATA_DS4_IDX')=='1')!=(arm.startswith('f32')):raise ValueError('legacy override mismatch')
            candidates=[r for r in rows if r['scorer']=='cublas']
            if any(r['p0']!='98304' or r['T']!='2048' or r['heads']!='64' or r['dim']!='128'
                   or r['dynamic_shared']!='0' or r['borrowed_scratch_bytes']!='33816576' for r in candidates):
                raise ValueError('unexpected actual candidate geometry or scratch')
            if bool(candidates)!=arm.startswith('cublas'):raise ValueError('missing/forbidden candidate calls')
            layers={r['layer'] for r in rows if r['p0']=='98304' and r['T']=='2048'}
            if arm.startswith('cublas'):
                if {r['layer'] for r in candidates}!=layers:raise ValueError('partial intended dispatch')
                candidate_layers.append(layers)
            dispatch[arm]=dict(records=len(rows),candidate_calls=len(candidates),layers_at_target=sorted(layers,key=int))
            memory[arm]=read(arm+'/memory_summary.json');thermal[arm]=read(arm+'/thermal.json')
            if memory[arm]['samples']<=0:raise ValueError('sustained telemetry missing')
        if candidate_layers[0]!=candidate_layers[1]:raise ValueError('candidate repeated layer population changed')
        expected={('f32-0','f32-1'),('cublas-0','cublas-1'),('f32-0','cublas-0'),('f32-1','cublas-1')}
        comparisons=quality['comparisons']
        if len(comparisons)!=4 or {(r['reference'],r['candidate']) for r in comparisons}!=expected:raise ValueError('comparison population changed')
        for row in comparisons:
            if row['rows']!=32 or not row['mean_kl']<=.0005:raise ValueError('bounded incremental KL gate failed')
            if [p['predicted_token_index_zero_based'] for p in row['positions']]!=list(range(100354,100386)):
                raise ValueError('scored position population changed')
        return dict(schema=1,source_sha=source,run=run,artifact_zip_sha256=digest,
            component=component(archive,digest,run),quality=quality,dispatch=dispatch,memory=memory,thermal=thermal,
            model_identity=read('model-identity.json'),
            promotion='None. auto remains upstream; explicit experimental mode only.',
            remaining=['direct downstream-attention/real-score boundary diagnostics',
                       'unprofiled paired whole-model performance and short-context regression',
                       'batched MMA correctness/key reuse candidate',
                       'full boundary/device/platform acceptance'],
            scope='Shared runtime scorer has component/graph/sanitizer evidence and a bounded long-context '
                  'same-pack same-binary logit screen. Diagnostic phase/wall timings are not ordinary model performance. '
                  'Local manifest integrity does not establish independent publisher authenticity. '
                  'Hardware counters/QPI traffic remain unavailable/unmeasured.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',required=True,type=int)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
