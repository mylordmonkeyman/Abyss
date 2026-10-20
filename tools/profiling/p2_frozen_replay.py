#!/usr/bin/env python3
"""Revalidate selected-base latent math using immutable P0 tensors, without loading a model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
CACHE=Path('/actions-runner/abyss-v100-stage-captures/frozen-2048-v1')

def main():
    out=Path(os.environ['EVIDENCE']);out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((CACHE/'identity.json').read_text())
    total=0
    for name,expected in identity['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe frozen capture member')
        path=CACHE/'tensors'/relative
        if path.is_symlink() or not path.is_file():raise ValueError('missing/linked frozen tensor')
        total+=path.stat().st_size
        if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:
            raise ValueError('immutable stage tensor changed: '+name)
    if total>32*1024**2:raise ValueError('frozen capture exceeds original cap')
    for ratio in (0,4,128):
        if not (CACHE/f'tensors/ratio-{ratio}/complete').is_file():raise ValueError('missing frozen ratio')
    binary=ROOT/'build-v100-p2-feasibility/test_dsv4_mla_latent_parity'
    env=dict(os.environ,STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    for label,prefix in [('replay',[]),('memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','1']),
                         ('synccheck',['compute-sanitizer','--tool','synccheck','--error-exitcode','1'])]:
        with (out/('frozen-'+label+'.log')).open('w') as log:
            subprocess.run(prefix+[str(binary),str(CACHE/'tensors')],env=env,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=600)
    rows=[json.loads(line) for line in (out/'frozen-replay.log').read_text().splitlines() if line.startswith('{')]
    if len([r for r in rows if 'real_stage' in r])!=33 or not all(r.get('pass') for r in rows):
        raise ValueError('frozen stage parity population/failure')
    for tool in ('memcheck','synccheck'):
        if 'ERROR SUMMARY: 0 errors' not in (out/('frozen-'+tool+'.log')).read_text():raise ValueError('missing sanitizer success')
    result=dict(schema=1,source_run=os.environ.get('GITHUB_RUN_ID'),
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),frozen_identity=identity,
        frozen_bytes=total,results=rows,memcheck_errors=0,synccheck_errors=0,
        scope='Selected-base P2 latent/gather/QK/softmax/PV/inverse-RoPE/projection revalidation on preserved P0 activation bytes. '
        'No model load, repack, tensor recapture, production arithmetic, fused attention or model quality/performance admission. '
        'Historical capture provenance remains historical; this does not convert its model-integrity claim into publisher authenticity. '
        'Hardware counters and QPI traffic remain unavailable/unmeasured.')
    (out/'frozen-replay.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
