#!/usr/bin/env python3
"""Bounded real-model activation capture/replay; no throughput or model KL claim."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

import p0
from model_baseline import ENGINE, PACK, fixed_env
from model_bringup import NAMES, SIZES


def main():
    out=Path(os.environ['EVIDENCE']).absolute()
    for name,size in zip(NAMES,SIZES):
        if (Path('/models')/name).stat().st_size!=size:
            raise ValueError('mounted model size mismatch')
    pack=json.loads((PACK/'v100_pack_complete.json').read_text())
    if pack['packer_sha256']!=p0.sha256(p0.ROOT/'tools/iq_pack.py'):
        raise ValueError('pack provenance mismatch')
    fixture=p0.ROOT/'tools/bench/fixtures/v100-replay-600.tokens'
    original=list(map(int,fixture.read_text().strip().split(',')))
    if len(original)!=728: raise ValueError('unexpected frozen fixture')
    # Repetition supplies a deterministic 2400-token diagnostic history, making
    # position 2048 reachable in batched prefill. Not a semantic-quality corpus.
    ids=original[:600]*4
    tokens=out/'capture.tokens';tokens.write_text(','.join(map(str,ids))+'\n')
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1',STRATA_GLM_PREFILL_SUB='256')
    argv=[str(ENGINE),'--glm-pack',str(PACK),'--tokens-file',str(tokens),
          '--max-context','4096','--max-new','1','--temperature','0','--prefill','auto',
          '--eos-ids','2147483647']
    report=dict(schema_version=1,mode='compile-only STRATA_V100_CAPTURE_STAGES',
                model_integrity='read-only size-checked model; full SHA256 unverified',
                pack=pack,engine_sha256=p0.sha256(ENGINE),fixture_sha256=p0.sha256(fixture),
                tokens_sha256=p0.sha256(tokens),prompt_tokens=len(ids),target_position=2048,
                scope='real projected/prepared activations; no new inference dispatch',
                performance_claim=False,full_model_delta_kl_gate_complete=False,
                actual_output_weights_replayed=False,argv=argv,
                environment={k:v for k,v in env.items() if k.startswith(('STRATA_','ABYSS_V100_'))})
    p0.write_json(out/'capture_manifest.json',report)
    p0.write_json(out/'toolchain_manifest.json',p0.manifest(os.environ.get('ABYSS_GPU','0')))
    root=out/'v100-stage-capture'
    # Freeze one immutable activation campaign, even if its candidate replay
    # fails. Arithmetic debugging must not repeatedly load the complete model.
    cache=Path('/actions-runner/abyss-v100-stage-captures/frozen-2048-v1')
    model_loaded=False
    if not cache.exists():
        prior=p0.ROOT/'results/ci-v100-stages-39179744335-1/v100-stage-capture'
        if all((prior/f'ratio-{r}/complete').is_file() for r in (0,4,128)):
            source=prior
            origin=json.loads((prior.parent/'capture_manifest.json').read_text())
            source_commit=(prior.parent/'build-sha.txt').read_text().strip()
        else:
            with (out/'capture-engine.log').open('w') as log:
                subprocess.run(argv,env=env,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
            model_loaded=True
            source=root;origin=report;source_commit=(out/'build-sha.txt').read_text().strip()
        inventory={str(p.relative_to(source)):dict(bytes=p.stat().st_size,sha256=p0.sha256(p))
                   for p in source.rglob('*') if p.is_file()}
        if sum(x['bytes'] for x in inventory.values())>32*1024*1024:
            raise ValueError('source activation campaign exceeds hard cap')
        cache.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='freeze-',dir=cache.parent) as temporary:
            stage=Path(temporary)/'ready';stage.mkdir()
            shutil.copytree(source,stage/'tensors')
            p0.write_json(stage/'identity.json',dict(source_commit=source_commit,
                          engine_sha256=origin['engine_sha256'],tokens_sha256=origin['tokens_sha256'],
                          pack=origin['pack'],files=inventory))
            stage.rename(cache)
    identity=json.loads((cache/'identity.json').read_text())
    if identity['tokens_sha256']!=report['tokens_sha256'] or identity['pack']!=pack:
        raise ValueError('frozen campaign fixture or pack mismatch')
    for name,expected in identity['files'].items():
        path=cache/'tensors'/name
        if path.stat().st_size!=expected['bytes'] or p0.sha256(path)!=expected['sha256']:
            raise ValueError('frozen activation bytes changed: '+name)
    if not root.exists():shutil.copytree(cache/'tensors',root)
    report.update(frozen_campaign=identity,model_reload_for_replay=model_loaded)
    p0.write_json(out/'capture_manifest.json',report)
    snapshots={}
    for ratio in (0,4,128):
        directory=root/f'ratio-{ratio}'
        if not (directory/'complete').is_file(): raise ValueError('missing completed ratio capture')
        snapshots[str(ratio)]={p.name:dict(bytes=p.stat().st_size,sha256=p0.sha256(p))
                               for p in sorted(directory.iterdir()) if p.is_file()}
    capture_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if capture_bytes>32*1024*1024: raise ValueError('capture exceeds 32 MiB cap')
    executable=p0.ROOT/'build-v100-diagnostics/test_dsv4_mla_latent_parity'
    for name,prefix in [('real-stage-replay.jsonl',[]),
                        ('real-stage-memcheck.log',['compute-sanitizer','--tool','memcheck','--error-exitcode','1']),
                        ('real-stage-synccheck.log',['compute-sanitizer','--tool','synccheck','--error-exitcode','1'])]:
        with (out/name).open('w') as log:
            subprocess.run(prefix+[str(executable),str(root)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    rows=[json.loads(line) for line in (out/'real-stage-replay.jsonl').read_text().splitlines()]
    if len([x for x in rows if 'real_stage' in x])!=33 or not all(x.get('pass') for x in rows):
        raise ValueError('real-stage replay count or result mismatch')
    for name in ('real-stage-memcheck.log','real-stage-synccheck.log'):
        if 'ERROR SUMMARY: 0 errors' not in (out/name).read_text():
            raise ValueError('missing zero-error sanitizer summary')
    report.update(status='captured and replayed all three ratios',capture_bytes=capture_bytes,
                  snapshots=snapshots,stage_results=rows,memcheck_errors=0,synccheck_errors=0)
    p0.write_json(out/'capture_manifest.json',report)
    print(json.dumps({k:report[k] for k in ('status','capture_bytes','memcheck_errors','synccheck_errors')}))


if __name__=='__main__':main()
