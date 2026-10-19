#!/usr/bin/env python3
"""P0 recovery: one long case per process/job, chunk pairs, then long timelines.

Separate campaign preserves interrupted data and bounds loss to one unit. Long
cases start fresh processes consistently for both bases; do not splice these
with the earlier carry-over-residency arms for a causal comparison.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'profiling'))
import v100_p0_campaign as campaign
from p0_accounting import fixture
from model_baseline import fixed_env
import p0

CACHE=Path('/actions-runner/abyss-p0-v9/completion-2')


def plan(unit):
    if 0<=unit<18:
        length=(32768,65536,131072)[unit//6];index=unit%6;base=campaign.ORDER[index]
        return dict(kind='long',length=length,index=index,bases=[base],context=135168,prefill='auto',output=1024)
    if 18<=unit<21:
        chunk=(512,1024,4096)[unit-18]
        return dict(kind='chunk',length=8192,index=unit-18,bases=campaign.ORDER,context=12288,prefill=str(chunk),output=128)
    if unit in (21,22):return dict(kind='trace',length=(32768,131072)[unit-21])
    raise ValueError('unit outside frozen plan')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--unit',type=int,required=True);args=ap.parse_args()
    task=plan(args.unit);out=Path(os.environ['EVIDENCE']).absolute()
    if task['kind']=='trace':
        subprocess.run([sys.executable,str(campaign.ROOT/'tools/profiling/p0_long_trace.py'),
                        '--tokens',str(task['length'])],check=True)
        return
    out.mkdir(parents=True,exist_ok=False);CACHE.mkdir(parents=True,exist_ok=True)
    model=campaign.model_identity();p0.write_json(out/'model-identity.json',model)
    frozen=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    pack=json.loads((campaign.PACK/'v100_pack_complete.json').read_text())
    if pack!=frozen['pack'] or pack['packer_sha256']!=p0.sha256(campaign.ROOT/'tools/iq_pack.py'):
        raise ValueError('pack identity changed')
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    identity=dict(version=2,original_identity=frozen,control_environment={k:v for k,v in env.items() if k.startswith('STRATA_')},
                  long_cache_scope='one case per new process, both first/excluded and second/warm request retained')
    if (CACHE/'identity.json').exists() and json.loads((CACHE/'identity.json').read_text())!=identity:
        raise ValueError('recovery identity changed; preserve campaign and investigate')
    p0.write_json(CACHE/'identity.json',identity);p0.write_json(out/'identity.json',identity)
    name='mixed-'+str(task['length'])
    ids=fixture(name);arms=[]
    try:
        for j,base in enumerate(task['bases']):
            binary=campaign.BINARIES[base]
            if p0.sha256(binary)!=campaign.EXPECTED[base]:raise ValueError('immutable binary changed')
            index=task['index'] if task['kind']=='long' else j
            directory=CACHE/f"{task['kind']}-{task['length']}-{task['prefill']}-{index}-{base}"
            if (directory/'completion.json').exists():
                rows=json.loads((directory/'requests.json').read_text())
                if len(rows)!=2 or any(r['prompt_tokens']!=len(ids) or r['output_tokens']!=task['output'] for r in rows):
                    raise ValueError('completed unit shape mismatch')
            else:
                if directory.exists():raise RuntimeError('incomplete unit preserved; inspect before any retry: '+str(directory))
                if campaign.memory_available()<16*1024**3:raise RuntimeError('host memory safety margin below 16 GiB')
                rows=campaign.arm(binary,directory,[(name,ids,task['output'])],env,task['context'],task['prefill'])
            arms.append(dict(index=index,base=base,requests=rows))
            shutil.copytree(directory,out/directory.name)
            p0.write_json(out/'unit.json',dict(unit=args.unit,task=task,arms=arms,
                harness_sha256=p0.sha256(Path(__file__)),paired=campaign.paired(arms),
                interpretation='New recovery campaign; long cases have consistent fresh-process expert tiers. '
                'Do not merge timing strata with the original carry-over-residency campaign.'))
    finally:
        # Preserve the current incomplete unit's bounded evidence as well.
        if 'directory' in locals() and directory.exists() and not (out/directory.name).exists():
            shutil.copytree(directory,out/directory.name)


def cap():
    out=Path(os.environ['EVIDENCE']).absolute()
    if not out.exists():return
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>64*1024**2:raise ValueError('unit evidence exceeds 64 MiB; retained on runner, upload refused')
    p0.write_json(out/'upload-cap.json',dict(bytes_before_manifest=total,limit_bytes=64*1024**2))
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P0_REMAINING_CAP=true\n')

if __name__=='__main__':
    try:main()
    finally:cap()
