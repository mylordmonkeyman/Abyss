#!/usr/bin/env python3
"""Digest-checked isolated MMA batch evidence; no production admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

def rows(text):return [json.loads(line) for line in text.splitlines() if line.startswith('{')]
def check_numerical(record):
    metrics=('relative_l2_vs_f32','max_abs_vs_f32','sampled_max_abs_vs_fp64')
    if not record['pass'] or any(not math.isfinite(record[k]) or record[k]<0 for k in metrics):
        raise ValueError('invalid MMA numerical record')
    scaled=record.get('scaled',False)
    if scaled or record['mode']!=2:
        if not record['nominal_relative_pass'] or record['relative_l2_vs_f32']>=1e-5:
            raise ValueError('nominal MMA relative gate failed')
    if scaled:
        value=record['sampled_relative_l2_vs_fp64']
        if not math.isfinite(value) or value<0 or value>=1e-5:raise ValueError('scaled FP64 relative gate failed')

def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('MMA artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('MMA artifact cap exceeded')
        def read(name):return z.read(name).decode()
        parity=rows(read('mma-batch-parity.jsonl'));bench=rows(read('mma-batch-benchmark.jsonl'))
        revision=parity[0].get('matrix_revision')
        if revision not in (1,2,3) or bench[0].get('matrix_revision')!=revision:raise ValueError('unsupported MMA matrix revision')
        variants=[(False,1),(False,8)]+([(True,1),(True,8)] if revision>=2 else [])
        end={'suite':'isolated_mma_batch_indexer','pass':True}
        if parity[-1]!=end or bench[-1]!=end:raise ValueError('incomplete MMA suite')
        cases=[r for r in parity if r.get('case')=='mma_batch_indexer']
        expected={(s,q,T,p,0) for s,q in variants for T in (1,3,8,9,33) for p in (2040,2052,8192)}
        expected|={(s,q,9,2052,m) for s,q in variants for m in (1,2,3,4)}
        if revision>=2:expected|={(True,q,9,2052,m) for q in (1,8) for m in (5,6,7)}
        if revision==3:expected|={(True,q,9,2052,m) for q in (1,8) for m in (8,9)}
        if len(cases)!=len(expected) or {(r.get('scaled',False),r['query_tile'],r['T'],r['p0'],r['mode']) for r in cases}!=expected:
            raise ValueError('MMA diagnostic population changed')
        timed_variants=[(True,8)] if revision==3 else variants
        shapes={(s,q,T,p) for s,q in timed_variants for T in (256,1024,4096) for p in (2048,8192,32768,120000)}
        shapes|={(s,q,2048,p) for s,q in timed_variants for p in (65536,98304,120000)}
        numerical=[r for r in bench if r.get('case')=='mma_batch_indexer']
        if len(numerical)!=len(shapes) or {(r.get('scaled',False),r['query_tile'],r['T'],r['p0']) for r in numerical}!=shapes:
            raise ValueError('MMA timed numerical population changed')
        for record in cases+numerical:check_numerical(record)
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in read('mma-batch-'+tool+'.log'):raise ValueError('MMA sanitizer failure/missing')
        if 'mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32' not in read('mma-batch-sm70.ptx') or 'HMMA' not in read('mma-batch-sm70.sass'):
            raise ValueError('MMA instruction evidence missing')
        resources=[r for r in parity if r.get('case')=='resources']
        if len(resources)!=len(variants) or {(r.get('scaled',False),r['query_tile']) for r in resources}!=set(variants):raise ValueError('missing resource population')
        timing=[r for r in bench if r.get('case')=='mma_batch_stage_timing']
        if len(timing)!=len(shapes)*6 or {(r.get('scaled',False),r['query_tile'],r['T'],r['p0'],r['pair']) for r in timing}!={(*s,i) for s in shapes for i in range(6)}:
            raise ValueError('incomplete MMA timing pairs')
        groups=[]
        for scaled,q,T,p in sorted(shapes):
            pairs=sorted((r for r in timing if (r.get('scaled',False),r['query_tile'],r['T'],r['p0'])==(scaled,q,T,p)),key=lambda r:r['pair'])
            if any(not math.isfinite(r[k]) or r[k]<=0 for r in pairs for k in ('f32_ms','candidate_ms')):raise ValueError('invalid timings')
            delta=[100*(r['candidate_ms']/r['f32_ms']-1) for r in pairs]
            groups.append(dict(scaled=scaled,query_tile=q,T=T,p0=p,pairs=pairs,elapsed_change_percent=delta,
                median_elapsed_change_percent=statistics.median(delta),all_six_candidate_faster=all(x<0 for x in delta)))
        return dict(schema=2,matrix_revision=revision,run=run,artifact_zip_sha256=digest,source_sha=read('source-sha.txt').strip(),
            resources=resources,component_cases=cases,timed_numerical_cases=numerical,timings=groups,
            tiny_query_nominal_failures=[r for r in cases if not r['nominal_relative_pass']],
            thermal=json.loads(read('thermal.json')),memory=json.loads(read('memory_summary.json')),
            scope='Isolated one-query and eight-query key-reuse MMA884 with hi/lo chains; revision 2 adds per-head power-of-two scaling; unchanged selector. '
            'Stage times include T single-token selectors versus one batched baseline selection. '
            'Absolute tiny-query diagnostic bounds do not qualify relative precision. No runtime/model promotion, '
            'whole-model speedup or hardware-counter/DRAM/QPI traffic claim.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',required=True,type=int)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
