#!/usr/bin/env python3
"""Bounded isolated batched MMA matrix; no model or production dispatch."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/v100')]
import v100_p0_campaign as campaign
from model_baseline import fixed_env
class Monitor(campaign.Monitor):
    loop=campaign.Sampler.loop

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,default=ROOT/'build-v100-diagnostics/test_dsv4_mma_batched')
    args=parser.parse_args()
    out=Path(os.environ['EVIDENCE'])
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
    env=fixed_env();env['STRATA_DS4_IDX']='1'
    with Monitor(out) as monitor,(out/'mma-batch-benchmark.jsonl').open('w') as log:
        child=subprocess.Popen([str(args.binary.resolve()),'--bench'],
            env=env,stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid;deadline=time.monotonic()+2400
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('own-child memory stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('batched MMA matrix exceeded 2400 s')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('batched MMA matrix failed; inspect preserved JSONL')
        finally:
            if child.poll() is None:child.kill();child.wait()
if __name__=='__main__':main()
