#!/usr/bin/env python3
"""Run the isolated batched scorer with bounded telemetry and own-child safety."""
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/v100')]
import v100_p0_campaign as campaign
from model_baseline import fixed_env


class BenchMonitor(campaign.Monitor):
    # The component matrix is shorter than a model arm: use the existing 2 s sampler.
    loop=campaign.Sampler.loop


def main():
    out=Path(os.environ['EVIDENCE'])
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
    binary=ROOT/'build-v100-diagnostics/test_dsv4_cublas_indexer'
    env=fixed_env();env['STRATA_DS4_IDX']='1'
    with BenchMonitor(out) as monitor,(out/'cublas-stage-benchmark.jsonl').open('w') as log:
        child=subprocess.Popen([str(binary),'--bench'],env=env,stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid;deadline=time.monotonic()+1800
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('benchmark safety stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('benchmark exceeded 1800 seconds')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('batched candidate failed; inspect preserved JSONL')
        finally:
            if child.poll() is None:child.kill();child.wait()


if __name__=='__main__':main()
