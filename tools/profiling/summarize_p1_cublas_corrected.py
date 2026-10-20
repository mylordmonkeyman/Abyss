#!/usr/bin/env python3
"""Validate bounded isolated cuBLAS correction evidence, never model admission."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

VARIANTS=('original','batched-warp','batched-sequential','full-flagged','band-flagged','band-all','band-all-sequential')
def rows(text):return [json.loads(line) for line in text.splitlines() if line.startswith('{')]
def key(r):return r['variant'],r['T'],r['p0'],r['mode']
def validate_population(records,expected):
    numerical=[r for r in records if r.get('case')=='cublas_indexer']
    corrections=[r for r in records if r.get('case')=='correction']
    if len(numerical)!=len(expected) or {key(r) for r in numerical}!=expected or len(corrections)!=len(expected) or {key(r) for r in corrections}!=expected:
        raise ValueError('incomplete cuBLAS numerical/correction population')
    for r in numerical:
        if not r['pass'] or any(not math.isfinite(r[k]) or r[k]<0 for k in ('relative_l2_vs_f32','max_abs_vs_f32','sampled_max_abs_vs_fp64')):
            raise ValueError('invalid cuBLAS numerical record')
        if r['relative_l2_vs_f32']>=1e-5 or r['sampled_max_abs_vs_fp64']>=1e-4:raise ValueError('cuBLAS numerical gate failed')
        T,p=r['T'],r['p0']
        if r['selection_count']!=sum(min(512,(p+t+1)//4) for t in range(T)) or not 0<=r['selection_overlap']<=r['selection_count']:
            raise ValueError('invalid cuBLAS selection accounting')
    for r in corrections:
        T,p=r['T'],r['p0'];corrected=r['variant'] in VARIANTS[3:]
        if not r['pass'] or not 0<=r['flagged_tokens']<=T or not 0<=r['corrected_tokens']<=T or not 0<=r['rescored_rows']<=sum((p+t+1)//4 for t in range(T)):
            raise ValueError('invalid cuBLAS correction accounting')
        if not 0<=r['unflagged_order_errors']<=T-r['corrected_tokens'] or not 0<=r['unflagged_membership_errors']<=r['unflagged_order_errors'] or (corrected and r['unflagged_membership_errors']!=0) or r['exact_corrected_scores_and_order']!=corrected:
            raise ValueError('invalid cuBLAS correction contract')
        if r['variant'] in VARIANTS[5:] and r['corrected_tokens']!=T:raise ValueError('all-token correction incomplete')
    return numerical,corrections

def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('correction ZIP digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('correction artifact too large')
        def read(name):return z.read(name).decode()
        parity=rows(read('cublas-correction-parity.jsonl'));bench=rows(read('cublas-correction-benchmark.jsonl'))
        for records in (parity,bench):
            if records[0]['matrix_revision']!=1 or records[-1]!={'suite':'isolated_cublas_corrections','pass':True}:raise ValueError('incomplete correction matrix')
        cases={(v,T,p,0) for v in VARIANTS for T in (1,3,8,9,33) for p in (2040,2052,8192)}
        cases|={(v,3,2052,m) for v in VARIANTS for m in (1,2,3,4)}
        cases|={(v,3,40000,1) for v in VARIANTS}
        shapes={(v,T,p,0) for v in VARIANTS for T in (256,1024,4096) for p in (2048,8192,32768,120000)}
        shapes|={(v,2048,p,0) for v in VARIANTS for p in (65536,98304,120000)}
        numerical,corrections=validate_population(parity,cases)
        for r in corrections:
            if r['variant'] in VARIANTS[3:] and r['p0']==40000 and r['mode']==1 and (r['corrected_tokens']!=3 or r['rescored_rows']!=30000):
                raise ValueError('overflow/full-reference fallback was not exercised')
        timed,timed_corrections=validate_population(bench,shapes)
        timings=[r for r in bench if r.get('case')=='cublas_stage_timing']
        expected={(*s[:3],i) for s in shapes for i in range(6)}
        if len(timings)!=630 or {(r['variant'],r['T'],r['p0'],r['pair']) for r in timings}!=expected:raise ValueError('incomplete correction timing population')
        groups=[]
        for v,T,p,_ in sorted(shapes):
            pairs=sorted((r for r in timings if (r['variant'],r['T'],r['p0'])==(v,T,p)),key=lambda r:r['pair'])
            if any(not math.isfinite(r[k]) or r[k]<=0 for r in pairs for k in ('f32_ms','candidate_ms')):raise ValueError('invalid stage timing')
            delta=[100*(r['candidate_ms']/r['f32_ms']-1) for r in pairs]
            groups.append(dict(variant=v,T=T,p0=p,pairs=pairs,elapsed_change_percent=delta,median_elapsed_change_percent=statistics.median(delta)))
        resources=[r for r in parity if r.get('case')=='resources']
        expected_resources={(v,k) for v in VARIANTS[3:] for k in ('detector','rescore')}|{(v,'sequential-epilogue') for v in (VARIANTS[2],VARIANTS[6])}
        if len(resources)!=10 or {(r['variant'],r['kernel']) for r in resources}!=expected_resources:raise ValueError('missing correction resources')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in read('cublas-correction-'+tool+'.log'):raise ValueError('missing correction sanitizer pass')
        if 'sm_70' not in read('object-images.txt') or not read('cublas-correction-sm70.sass').strip():raise ValueError('missing SM70 code evidence')
        return dict(schema=1,run=run,source_sha=read('source-sha.txt').strip(),artifact_zip_sha256=digest,
            component_cases=numerical,corrections=corrections,timed_numerical_cases=timed,timed_corrections=timed_corrections,
            resources=resources,timings=groups,causality_provenance=json.loads(read('causality-provenance.json')),
            thermal=json.loads(read('thermal.json')),memory=json.loads(read('memory_summary.json')),
            scope='Isolated components only. Empirical 32eps error envelope is not a global bound. Corrected selected scores and ordering are checked literally; flagged-only policies may leave unflagged ordering differences. '
            'All GPU detection, compaction, overflow/full rescore and selectors are included in paired stage times. Component scratch allocations are not runtime unique-memory claims. '
            'Variants are separately paired with F32, not directly interleaved with each other. No full-model quality, whole-model speed, default admission, counters or QPI claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',required=True,type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
