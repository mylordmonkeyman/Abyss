#!/usr/bin/env python3
"""Bounded synthetic attention timing with own-child memory guard and telemetry."""
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools')]
import v100_p0_campaign as campaign

def main():
    out=Path(os.environ['EVIDENCE'])/'timing';out.mkdir(parents=True,exist_ok=False)
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
    binary=ROOT/'build-v100-p2-feasibility/test_dsv4_attention_tile'
    with campaign.Monitor(out) as monitor,(out/'pairs.jsonl').open('w') as log:
        child=subprocess.Popen([str(binary),'--bench'],stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid;deadline=time.monotonic()+1200
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('own-child attention stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('attention benchmark exceeded 1200 seconds')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('attention benchmark failed; inspect preserved output')
        finally:
            if child.poll() is None:child.kill();child.wait()
if __name__=='__main__':main()
