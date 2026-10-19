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
def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('MMA artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):return z.read(name).decode()
        parity=rows(read('mma-batch-parity.jsonl'));bench=rows(read('mma-batch-benchmark.jsonl'))
        end={'suite':'isolated_mma_batch_indexer','pass':True}
        if parity[-1]!=end or bench[-1]!=end:raise ValueError('incomplete MMA suite')
        cases=[r for r in parity if r.get('case')=='mma_batch_indexer']
        expected={(q,T,p,0) for q in (1,8) for T in (1,3,8,9,33) for p in (2040,2052,8192)}
        expected|={(q,9,2052,m) for q in (1,8) for m in (1,2,3,4)}
        if len(cases)!=len(expected) or {(r['query_tile'],r['T'],r['p0'],r['mode']) for r in cases}!=expected:
            raise ValueError('MMA diagnostic population changed')
        shapes={(q,T,p) for q in (1,8) for T in (256,1024,4096) for p in (2048,8192,32768,120000)}
        shapes|={(q,2048,p) for q in (1,8) for p in (65536,98304,120000)}
        numerical=[r for r in bench if r.get('case')=='mma_batch_indexer']
        if len(numerical)!=len(shapes) or {(r['query_tile'],r['T'],r['p0']) for r in numerical}!=shapes:
            raise ValueError('MMA timed numerical population changed')
        if not all(r['pass'] for r in cases+numerical):raise ValueError('diagnostic bounds failed')
        if any(not r['nominal_relative_pass'] for r in cases+numerical if r['mode']!=2):raise ValueError('nominal case failed')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in read('mma-batch-'+tool+'.log'):raise ValueError('MMA sanitizer failure/missing')
        if 'mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32' not in read('mma-batch-sm70.ptx') or 'HMMA' not in read('mma-batch-sm70.sass'):
            raise ValueError('MMA instruction evidence missing')
        resources=[r for r in parity if r.get('case')=='resources']
        if len(resources)!=2 or {r['query_tile'] for r in resources}!={1,8}:raise ValueError('missing resource population')
        timing=[r for r in bench if r.get('case')=='mma_batch_stage_timing']
        if len(timing)!=len(shapes)*6 or {(r['query_tile'],r['T'],r['p0'],r['pair']) for r in timing}!={(*s,i) for s in shapes for i in range(6)}:
            raise ValueError('incomplete MMA timing pairs')
        groups=[]
        for q,T,p in sorted(shapes):
            pairs=sorted((r for r in timing if (r['query_tile'],r['T'],r['p0'])==(q,T,p)),key=lambda r:r['pair'])
            if any(not math.isfinite(r[k]) or r[k]<=0 for r in pairs for k in ('f32_ms','candidate_ms')):raise ValueError('invalid timings')
            delta=[100*(r['candidate_ms']/r['f32_ms']-1) for r in pairs]
            groups.append(dict(query_tile=q,T=T,p0=p,pairs=pairs,elapsed_change_percent=delta,
                median_elapsed_change_percent=statistics.median(delta),all_six_candidate_faster=all(x<0 for x in delta)))
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=read('source-sha.txt').strip(),
            resources=resources,component_cases=cases,timed_numerical_cases=numerical,timings=groups,
            tiny_query_nominal_failures=[r for r in cases if not r['nominal_relative_pass']],
            thermal=json.loads(read('thermal.json')),memory=json.loads(read('memory_summary.json')),
            scope='Isolated one-query and eight-query key-reuse MMA884 with hi/lo chains; unchanged selector. '
            'Stage times include T single-token selectors versus one batched baseline selection. '
            'Absolute tiny-query diagnostic bounds do not qualify relative precision. No runtime/model promotion, '
            'whole-model speedup or hardware-counter/DRAM/QPI traffic claim.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',required=True,type=int)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
