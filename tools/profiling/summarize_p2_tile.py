#!/usr/bin/env python3
"""Validate complete isolated tile evidence and six matched timing pairs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

def timing_summary(rows):
    groups={}
    for row in rows:
        if row.get('timing')!='attention_tile':continue
        key=tuple(row[k] for k in ('ratio','p0','T','key_tile','warps'))
        groups.setdefault(key,[]).append(row)
    expected={(r,131071,t,k,w) for r in (0,4,128) for t in (256,2048) for k in (16,32) for w in (4,8)}
    if set(groups)!=expected:raise ValueError('timing shape population changed')
    result=[]
    for key,pairs in sorted(groups.items()):
        pairs=sorted(pairs,key=lambda r:r['pair'])
        if len(pairs)!=6 or [r['pair'] for r in pairs]!=list(range(6)):raise ValueError('six unique pairs required')
        if any(not math.isfinite(r[k]) or r[k]<=0 for r in pairs for k in ('f32_ms','candidate_ms')):raise ValueError('invalid elapsed time')
        changes=[100*(r['candidate_ms']/r['f32_ms']-1) for r in pairs]
        result.append(dict(zip(('ratio','p0','T','key_tile','warps'),key),pairs=pairs,
            paired_elapsed_changes_pct=changes,median_paired_elapsed_change_pct=statistics.median(changes),
            candidate_faster_pairs=sum(x<0 for x in changes)))
    return result

def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('tile artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('tile artifact cap exceeded')
        def rows(name):return [json.loads(l) for l in z.read(name).decode().splitlines() if l.startswith('{')]
        parity=rows('parity.jsonl');cases=[r for r in parity if r.get('case')=='attention_tile']
        expected_cases={(r,p,t,m,k,w) for k in (16,32) for w in (4,8) for r,p,t,m in
            ((0,0,1,0),(0,0,33,1),(0,127,7,2),(0,129,8,3),(4,2,7,0),(4,2047,8,0),(128,126,7,0),(128,131071,1,0))}
        if len(cases)!=32 or {tuple(r[k] for k in ('ratio','p0','T','mode','key_tile','warps')) for r in cases}!=expected_cases:
            raise ValueError('parity population changed')
        timed=rows('timing/pairs.jsonl');checks=[r for r in timed if r.get('case')=='attention_tile']
        if len(checks)!=24:raise ValueError('missing timed numerical checks')
        for r in cases+checks:
            if not all(math.isfinite(r[k]) for k in ('relative_l2','oracle_relative_l2','max_abs')):raise ValueError('nonfinite tile metric')
            if not r.get('pass') or r['relative_l2']>=1e-5 or r['oracle_relative_l2']>=1e-5 or r['max_abs']>=.001:
                raise ValueError('tile numerical screen failed')
            if r['local_bytes']!=0 or r['active_ctas_per_sm']<=0:raise ValueError('tile spills/resource failure')
        if not parity[-1].get('pass') or not timed[-1].get('pass'):raise ValueError('incomplete tile suite')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in z.read(tool+'.log').decode():raise ValueError('missing sanitizer success')
        source=z.read('source-sha.txt').decode().strip()
        resources={}
        for r in cases+checks:
            key=f"{r['key_tile']}-keys-{r['warps']}-warps"
            values={k:r[k] for k in ('shared_bytes','registers','local_bytes','active_ctas_per_sm')}
            if key in resources and resources[key]!=values:raise ValueError('compiled resources changed')
            resources[key]=values
        return dict(schema=1,run=run,source_sha=source,artifact_zip_sha256=digest,
            binary_sha256=z.read('binary-sha256.txt').decode().split()[0],parity_cases=cases,timed_numerical_cases=checks,
            resources=resources,timings=timing_summary(timed),memcheck_errors=0,synccheck_errors=0,
            memory=json.loads(z.read('timing/memory_summary.json')),thermal=json.loads(z.read('timing/thermal.json')),
            scope='Synthetic isolated latent attention with valid 64-D identity RoPE, actual F32 attend_batch comparator, '
            'same inputs and six alternating pairs per shape. Resource occupancy is not hardware-counter occupancy. '
            'No full-model speed, learned projection, general model-RoPE, quality or production-dispatch admission. '
            'Counters/QPI remain unavailable/unmeasured; existing model-authenticity limitations remain unchanged.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
