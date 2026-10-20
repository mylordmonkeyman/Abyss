#!/usr/bin/env python3
"""Read-only toolchain/device inventory; missing SDK is never a build pass."""
import json
import os
from pathlib import Path
import shutil
import subprocess

def command(argv):
    try:
        p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        return dict(argv=argv,returncode=p.returncode,output=p.stdout)
    except (OSError,subprocess.TimeoutExpired) as e:return dict(argv=argv,error=str(e))

def main():
    out=Path(os.environ['EVIDENCE']);out.mkdir(parents=True,exist_ok=True)
    hip=shutil.which('hipcc')
    if not hip and Path('/opt/rocm/bin/hipcc').is_file():hip='/opt/rocm/bin/hipcc'
    sdk=bool(hip)
    result=dict(schema=1,visible_nvidia_devices=command(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader']),
        hip_compiler=command([hip,'--version']) if sdk else None,
        hip_disposition='SDK found; compile gate must actually run' if sdk else 'Unavailable: no hipcc on PATH or /opt/rocm/bin/hipcc; no HIP build/runtime claim',
        scope='Toolchain inventory and visible devices only. A CUDA policy unit test is not other-device inference. '
              'SM80 compilation must be recorded separately from SM80 runtime, which needs supported hardware. '
              'No toolkit installs, container churn, privilege changes or hardware-counter/QPI probes.')
    (out/'platform-probe.json').write_text(json.dumps(result,indent=2)+'\n')
    with open(os.environ['GITHUB_ENV'],'a') as f:
        f.write('P1_HIP_SDK='+('true' if sdk else 'false')+'\n')
        if sdk:f.write('P1_ROCM_PREFIX='+str(Path(hip).resolve().parent.parent)+'\n')
if __name__=='__main__':main()
