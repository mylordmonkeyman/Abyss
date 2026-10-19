#!/usr/bin/env python3
"""Checkpointed same-binary P1 model A/B. Requires committed quality evidence."""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools')]
import p0
import v100_p0_campaign as campaign
from p0_accounting import fixture,compress
from model_baseline import fixed_env,PACK
ORDER=['f32','cublas','cublas','f32','f32','cublas']

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--quality',required=True,type=Path);args=ap.parse_args()
    quality=json.loads(args.quality.read_text())
    binary=ROOT/'build-v100-diagnostics/strata';binary_sha=p0.sha256(binary)
    if quality['quality']['binary_sha256']!=binary_sha:raise ValueError('model A/B binary differs from qualified binary')
    if any(r['mean_kl']>.0005 for r in quality['quality']['comparisons']):raise ValueError('quality gate did not pass')
    identity=campaign.model_identity()
    if identity!=quality['model_identity']:raise ValueError('model identity differs from quality evidence')
    frozen=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if frozen['pack']!=json.loads((PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    cases=[('mixed-8192',fixture('mixed-8192'),128),('mixed-131072',fixture('mixed-131072'),16)]
    checkpoint=Path('/actions-runner/abyss-p1-v9/model-ab')/binary_sha
    checkpoint.mkdir(parents=True,exist_ok=True)
    signature=dict(binary_sha256=binary_sha,quality_evidence_sha256=p0.sha256(args.quality),
        model_identity=identity,case_fixture_sha256={n:p0.sha256(campaign.CAMPAIGN/'fixtures'/(n+'.tokens')) for n,_,_ in cases},
        order=ORDER,requests_per_case=2,context=135168)
    manifest=checkpoint/'identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=signature:raise ValueError('checkpoint identity changed')
    p0.write_json(manifest,signature)
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=True)
    report=dict(schema=1,signature=signature,arms=[],pairs=[],scope='Three process pairs, A/B B/A A/B; '
        'same binary/pack/fixtures, 8K short and 128K long, two uncached requests per case. '
        'Primary comparison uses request 1; warm-up request 0 retained. OS cache uncontrolled; no cache drops. '
        'No profiler or hardware counters; actual scorer logs and sustained telemetry enabled.')
    for i,mode in enumerate(ORDER):
        label=f'{i}-{mode}';directory=checkpoint/label
        env=fixed_env();env.update(ABYSS_V100_IDX=mode,STRATA_DS4_ATTN='1')
        if mode=='f32':env['STRATA_DS4_IDX']='1'
        if directory.exists():
            if not (directory/'completion.json').exists():raise ValueError('incomplete prior arm; inspect before retry: '+str(directory))
            completion=json.loads((directory/'completion.json').read_text())
            if completion['returncode']!=0:raise ValueError('prior arm failed')
            rows=json.loads((directory/'requests.json').read_text())
        else:
            if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
            rows=campaign.arm(binary,directory,cases,env,135168,'auto')
            compress(directory/'engine.log')
        if len(rows)!=4 or {(r['case'],r['request']) for r in rows}!={(n,j) for n,_,_ in cases for j in (0,1)}:
            raise ValueError('arm request population changed')
        with gzip.open(directory/'engine.log.gz','rt') as f:log=f.read()
        records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_indexer ')]
        calls=sum(r.get('scorer')=='cublas' for r in records)
        if bool(calls)!=(mode=='cublas'):raise ValueError('actual runtime scorer disagrees with mode')
        if not records:raise ValueError('dispatch records missing')
        copy=out/label
        if copy.exists():raise ValueError('refuse existing evidence directory')
        shutil.copytree(directory,copy)
        report['arms'].append(dict(index=i,mode=mode,requests=rows,candidate_calls=calls,
            thermal=json.loads((directory/'thermal.json').read_text()),memory=json.loads((directory/'memory_summary.json').read_text())))
        p0.write_json(out/'model-ab.json',report)
    for i in range(3):
        arms={a['mode']:a for a in report['arms'][i*2:i*2+2]}
        for case,_,_ in cases:
            a,b=(next(r for r in arms[m]['requests'] if r['case']==case and r['request']==1) for m in ('f32','cublas'))
            report['pairs'].append(dict(pair=i,case=case,**{k+'_change_percent':100*(b[k]/a[k]-1)
                for k in ('prefill_ms','decode_ms','ttft_ms')},generated_ids_equal=a['output_ids_sha256']==b['output_ids_sha256']))
    report['medians']={case:{key:statistics.median(r[key] for r in report['pairs'] if r['case']==case)
        for key in ('prefill_ms_change_percent','decode_ms_change_percent','ttft_ms_change_percent')} for case,_,_ in cases}
    p0.write_json(out/'model-ab.json',report)
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>64*1024**2:raise ValueError('model A/B artifact exceeds cap')

if __name__=='__main__':main()
