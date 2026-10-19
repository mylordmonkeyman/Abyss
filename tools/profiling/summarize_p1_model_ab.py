#!/usr/bin/env python3
"""Digest-verified same-binary model A/B; preserve timing and decode limits."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import zipfile
ORDER=['f32','cublas','cublas','f32','f32','cublas']
def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('model artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        r=json.loads(z.read('model-ab.json'));arms=r['arms']
        if len(arms)!=6 or [a['mode'] for a in arms]!=ORDER or [a['index'] for a in arms]!=list(range(6)):
            raise ValueError('process order/population changed')
        pairs=[];dispatch=[]
        for a in arms:
            label=str(a['index'])+'-'+a['mode'];rows=a['requests']
            if len(rows)!=4 or {(x['case'],x['request'],x['output_tokens']) for x in rows}!={('mixed-8192',j,128) for j in (0,1)}|{('mixed-131072',j,16) for j in (0,1)}:
                raise ValueError('request population changed')
            invocation=json.loads(z.read(label+'/invocation.json'));env=invocation['environment']
            if env['ABYSS_V100_IDX']!=a['mode'] or (env.get('STRATA_DS4_IDX')=='1')!=(a['mode']=='f32'):
                raise ValueError('explicit runtime controls changed')
            if 'ABYSS_V100_PROFILE' in env or 'STRATA_GLM_PREFILL_PROF' in env:raise ValueError('profiled throughput arm refused')
            log=gzip.decompress(z.read(label+'/engine.log.gz')).decode()
            records=[dict(re.findall(r'(\w+)=([^\s]+)',x)) for x in log.splitlines() if x.startswith('p1_indexer ')]
            candidate=[x for x in records if x['scorer']=='cublas']
            if len(candidate)!=(462 if a['mode']=='cublas' else 0):raise ValueError('actual candidate call population changed')
            if any(x['T']!='2048' or not 98304<=int(x['p0'])<=120000 for x in candidate):raise ValueError('candidate outside experimental regime')
            if not records or any(x['scorer']=='cublas' and int(x['p0'])<98304 for x in records):raise ValueError('short-path dispatch changed')
            if a['thermal']['samples']<=0 or a['memory']['samples']<=0:raise ValueError('sustained telemetry missing')
            dispatch.append(dict(arm=label,records=len(records),candidate_calls=len(candidate)))
        for pair in range(3):
            group={a['mode']:a for a in arms[2*pair:2*pair+2]}
            for case in ('mixed-8192','mixed-131072'):
                f,c=(next(x for x in group[m]['requests'] if x['case']==case and x['request']==1) for m in ('f32','cublas'))
                pairs.append(dict(pair=pair,case=case,**{k+'_change_percent':100*(c[k]/f[k]-1)
                    for k in ('prefill_ms','decode_ms','ttft_ms')},generated_ids_equal=f['output_ids_sha256']==c['output_ids_sha256']))
        if pairs!=r['pairs']:raise ValueError('stored pairs disagree with recomputation')
        medians={case:{k:statistics.median(x[k] for x in pairs if x['case']==case)
            for k in ('prefill_ms_change_percent','decode_ms_change_percent','ttft_ms_change_percent')}
            for case in ('mixed-8192','mixed-131072')}
        return dict(schema=1,run=run,source_sha=z.read('source-sha.txt').decode().strip(),artifact_zip_sha256=digest,
            signature=r['signature'],arms=arms,dispatch=dispatch,pairs=pairs,medians=medians,
            disposition='Keep explicit experimental control; auto unchanged pending remaining P1 acceptance.',
            scope='Observed 128K prefill change across three process pairs; 8K fallback is a regression control. '
            'Greedy output IDs differ in five of six primary case-pairs, including every short fallback pair. '
            'Decode is unchanged by this prefill-only candidate; short output counts, differing trajectories and '
            'carry-over expert state prevent attributing decode timing differences to a decode optimization. '
            'No sampled thermal flags; observed power limits and OS cache/tier state prevent claiming perfectly '
            'matched clock/cache conditions. Model identity verifies a local manifest, not publisher authenticity. '
            'Hardware counters/QPI traffic remain unavailable/unmeasured.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
