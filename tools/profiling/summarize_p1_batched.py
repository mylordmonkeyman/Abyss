#!/usr/bin/env python3
"""Digest-verified P1e component evidence, not production admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def rows(text):return [json.loads(x) for x in text.splitlines() if x.startswith('{')]


def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('batched artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):return z.read(name).decode()
        suite={'suite':'isolated_cublas_indexer','pass':True}
        parity=rows(read('cublas-parity.jsonl'));bench=rows(read('cublas-stage-benchmark.jsonl'))
        if not parity or not bench or parity[-1]!=suite or bench[-1]!=suite:raise ValueError('batched suite incomplete')
        cases=[r for r in parity if r.get('case')=='cublas_indexer']
        boundary=parity[0].get('boundary_p0',2048)
        if boundary not in (2048,2052):raise ValueError('unknown boundary fixture')
        expected={(T,p,0) for T in (1,3,8,33) for p in (2040,8192)}|{(3,boundary,m) for m in (1,2,3)}
        if len(cases)!=len(expected) or {(r['T'],r['p0'],r['mode']) for r in cases}!=expected:
            raise ValueError('component shape population mismatch')
        shapes={(T,p) for T in (1,2,3,8,16,32,256) for p in (2048,8192,32768,125000)}
        shapes|={(T,p) for T in (1024,4096) for p in (2048,8192,32768,120000)}
        revision=parity[0].get('matrix_revision',1)
        if revision not in (1,2):raise ValueError('unknown timed matrix revision')
        if revision==2:
            shapes|={(T,p) for T in (256,1024,4096) for p in (65536,98304)}
            shapes|={(2048,p) for p in (2048,8192,32768,65536,98304,120000)}
        numerical=[r for r in bench if r.get('case')=='cublas_indexer']
        if len(numerical)!=len(shapes) or {(r['T'],r['p0']) for r in numerical}!=shapes:
            raise ValueError('timed numerical population mismatch')
        if not all(r['pass'] for r in cases+numerical):raise ValueError('component parity failure')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in read('cublas-'+tool+'.log'):raise ValueError('sanitizer result missing')
        timings=[r for r in bench if r.get('case')=='cublas_stage_timing']
        expected_pairs={(T,p,i) for T,p in shapes for i in range(6)}
        if len(timings)!=len(expected_pairs) or {(r['T'],r['p0'],r['pair']) for r in timings}!=expected_pairs:
            raise ValueError('six complete timing pairs per shape required')
        groups=[]
        for T,p in sorted(shapes):
            pairs=sorted((r for r in timings if (r['T'],r['p0'])==(T,p)),key=lambda r:r['pair'])
            if any(not math.isfinite(r[k]) or r[k]<=0 for r in pairs for k in ('f32_ms','candidate_ms')):
                raise ValueError('invalid event timing')
            delta=[100*(r['candidate_ms']/r['f32_ms']-1) for r in pairs]
            groups.append(dict(T=T,p0=p,pairs=pairs,elapsed_change_percent=delta,
                median_elapsed_change_percent=statistics.median(delta),all_six_candidate_faster=all(x<0 for x in delta)))
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=read('source-sha.txt').strip(),
            cublas=parity[0],component_cases=cases,timed_numerical_cases=numerical,timings=groups,
            thermal=json.loads(read('thermal.json')),memory=json.loads(read('memory_summary.json')),
            production_candidate_enabled=False,scope='Synthetic FP32 SGEMM scorer, unchanged upstream selection. '
            'Stage times include key conversion, GEMM, epilogue and T single-token selector launches; '
            'baseline includes one batched selector launch. No tensor-core, model KL, whole-model speedup '
            'or production dispatch claim. Common outputs and opaque library allocations are outside explicit scratch.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path)
    ap.add_argument('--sha256',required=True);ap.add_argument('--run',required=True,type=int)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')


if __name__=='__main__':main()
