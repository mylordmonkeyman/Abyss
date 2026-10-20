#!/usr/bin/env python3
"""Bounded isolated cuBLAS correction matrix; no model or production dispatch."""
import os
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/v100')]
import v100_p0_campaign as campaign
from model_baseline import fixed_env
from p1_cublas_selection_control import validate_controls
from p1_integrated_quality import GEOMETRIES

def require_control(control):
    if control['source_sha']!='2e1de6f02cdaa689d537101e3b3b3ef2ce92d09e' or control['status']!='complete diagnostic intervention; inspect causality and ordering':
        raise ValueError('completed pinned cuBLAS causality control required')
    labels={'f32-0','f32-1','reference-order','reference-membership'}
    if control['geometry']!=GEOMETRIES['128k'] or control['cpu_lane']!=0 or set(control['dumps'])!=labels or set(control['arms'])!=labels:
        raise ValueError('wrong causality geometry or arm population')
    sizes={d['bytes'] for d in control['dumps'].values()}
    if len(sizes)!=1 or next(iter(sizes))!=16547840:
        raise ValueError('wrong causality logit population')
    expected={('f32-0','f32-1'),('f32-0','reference-order'),('f32-0','reference-membership')}
    comparisons=control['comparisons']
    if len(comparisons)!=3 or {(r['reference'],r['candidate']) for r in comparisons}!=expected:
        raise ValueError('incomplete causality comparisons')
    for row in comparisons:
        values=[p['kl'] for p in row['positions']]
        if len(values)!=32 or [p['predicted_token_index_zero_based'] for p in row['positions']]!=list(range(131040,131072)):
            raise ValueError('wrong control scored positions')
        if any(not math.isfinite(v) or v < -1e-12 for v in values) or not math.isclose(row['mean_kl'],math.fsum(values)/32,rel_tol=1e-10,abs_tol=1e-15):
            raise ValueError('invalid control KL ledger')
        if row['rows']!=32 or not math.isclose(row['max_kl'],max(values),rel_tol=1e-10,abs_tol=1e-15) or not 0<=row['top1_matches']<=32:
            raise ValueError('invalid control aggregate')
        if row['candidate']!='reference-membership' and row['mean_kl']>.0005:
            raise ValueError('F32 repeat or reference-order causality gate failed')
    if control['dumps']['f32-0']['sha256']!=control['dumps']['f32-1']['sha256']:
        raise ValueError('causality requires deterministic F32 repeats')
    for kind in ('reference-order','reference-membership'):
        validate_controls(control['arms'][kind]['selection_controls'],kind,control['geometry'])

class Monitor(campaign.Monitor):
    loop=campaign.Sampler.loop

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build-v100-p1-cublas-corrections/test_dsv4_cublas_corrected')
    parser.add_argument('--control-checkpoint',type=Path,required=True)
    parser.add_argument('--check-control-only',action='store_true')
    args=parser.parse_args()
    require_control(json.loads(args.control_checkpoint.read_text()))
    if args.check_control_only:return
    out=Path(os.environ['EVIDENCE'])
    campaign.p0.write_json(out/'causality-provenance.json',dict(checkpoint=str(args.control_checkpoint),
        sha256=campaign.p0.sha256(args.control_checkpoint)))
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
    env=fixed_env();env['STRATA_DS4_IDX']='1'
    with Monitor(out) as monitor,(out/'cublas-correction-benchmark.jsonl').open('w') as log:
        child=subprocess.Popen([str(args.binary.resolve()),'--bench'],
            env=env,stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid;deadline=time.monotonic()+3600
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('own-child memory stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('cuBLAS correction matrix exceeded 3600 s')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('cuBLAS correction matrix failed; inspect preserved JSONL')
        finally:
            if child.poll() is None:child.kill();child.wait()
if __name__=='__main__':main()
