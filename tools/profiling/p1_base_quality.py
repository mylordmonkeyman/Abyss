#!/usr/bin/env python3
"""Bounded ported-base F32 comparison, reusing verified immutable P0 distributions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools'),str(ROOT)]
import numpy as np
import p0
import v100_p0_campaign as campaign
from p0_accounting import fixture,compress
from model_baseline import fixed_env,normalize_logprobs,PACK


def score(binary,directory,tokens,dump):
    directory.mkdir()
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1',
        STRATA_GLM_SCORE='2040',STRATA_GLM_SCORE_DUMP=str(dump),STRATA_GLM_SCORE_PREFILL='1')
    argv=[str(binary),'--glm-pack',str(PACK),'--tokens-file',str(tokens),'--max-new','1',
          '--max-context','4096','--prefill','auto','--temperature','0']
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB before quality arm')
    p0.write_json(directory/'invocation.json',dict(argv=argv,environment={k:v for k,v in env.items() if k.startswith('STRATA_')}))
    with campaign.Monitor(directory) as monitor,(directory/'engine.log').open('w') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        monitor.pid=child.pid;deadline=time.monotonic()+1800
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('quality arm memory stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('quality arm exceeded 1800 seconds')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('quality process failed; inspect preserved log')
        finally:
            if child.poll() is None:child.kill();child.wait()
    compress(directory/'engine.log')


def main():
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=False)
    binary=ROOT/'build-v100-diagnostics/strata'
    current=campaign.BINARIES['current']
    if p0.sha256(current)!=campaign.EXPECTED['current']:raise ValueError('immutable v1.3.31 binary changed')
    model=campaign.model_identity();p0.write_json(out/'model-identity.json',model)
    identity=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if identity['model_sha256']!=model['sha256']:raise ValueError('frozen model identity mismatch')
    if identity['pack']!=json.loads((PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    fixture('teacher') # Verify the frozen teacher file; never rerun a newer tokenizer/packer.
    teacher=campaign.CAMPAIGN/'fixtures/teacher.tokens'
    prior=json.loads((ROOT/'docs/v100/p0-quality-2026-10-19.json').read_text())
    sizes={x['bytes'] for x in prior['dumps'].values()}
    if len(sizes)!=1:raise ValueError('inconsistent frozen quality ABI')
    count=sizes.pop();vocab=count//(32*4)
    if count!=32*4*vocab:raise ValueError('invalid frozen quality shape')
    arrays={};dumps={}
    with tempfile.TemporaryDirectory(prefix='abyss-p1-quality-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        for label,base in [('reference-0','current'),('working-0','working'),('working-1','working'),('reference-1','current')]:
            frozen=campaign.CAMPAIGN/'quality'/('current-'+label[-1]+'.f32')
            expected=prior['dumps']['current-'+label[-1]]
            reuse=base=='current' and frozen.exists()
            if reuse:
                if frozen.stat().st_size!=expected['bytes'] or p0.sha256(frozen)!=expected['sha256']:
                    raise ValueError('frozen reference distribution changed')
                dump=frozen
            else:
                dump=Path(tmp)/(label+'.f32');score(binary if base=='working' else current,out/label,teacher,dump)
            if dump.stat().st_size!=count:raise ValueError('teacher-forced dump shape mismatch')
            raw=np.fromfile(dump,dtype='<f4').reshape(32,vocab)
            arrays[label]=normalize_logprobs(raw)
            dumps[label]=dict(sha256=p0.sha256(dump),bytes=count,reused_frozen_reference=reuse,
                binary_sha256=p0.sha256(binary if base=='working' else current))
        comparisons=[]
        for a,b in [('working-0','working-1'),('reference-0','working-0'),('reference-1','working-1')]:
            ref,test=arrays[a],arrays[b];kl=(np.exp(ref)*(ref-test)).sum(axis=1)
            comparisons.append(dict(reference=a,candidate=b,mean_kl=float(kl.mean()),
                p99_kl=float(np.quantile(kl,.99)),max_kl=float(kl.max()),
                top1_matches=int((ref.argmax(1)==test.argmax(1)).sum()),rows=32,
                positions=[dict(predicted_token_index_zero_based=2041+i,kl=float(k)) for i,k in enumerate(kl)]))
        result=dict(schema=1,source_sha=p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
            source_run=os.environ.get('GITHUB_RUN_ID'),fixture_sha256=p0.sha256(teacher),
            dumps=dumps,comparisons=comparisons,scope='Same-pack selected-base F32 port/repeatability screen, '
            '32 positions across 2048; frozen references are not interleaved timing arms. '
            'No candidate scorer runs, no model-performance or P1 arithmetic acceptance claim.')
        p0.write_json(out/'quality.json',result)
        if any(r['mean_kl']>.0005 or r['top1_matches']!=32 for r in comparisons):
            raise ValueError('ported F32 comparison exceeds bounded screen; investigate before integration')


def cap():
    out=Path(os.environ['EVIDENCE'])
    if not out.exists():return
    files=list(out.rglob('*'))
    if any(p.is_symlink() for p in files):raise ValueError('quality artifact symlink refused')
    if sum(p.stat().st_size for p in files if p.is_file())>64*1024**2:raise ValueError('quality artifact exceeds 64 MiB')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P1_QUALITY_CAP=true\n')


if __name__=='__main__':
    try:main()
    finally:cap()
