#!/usr/bin/env python3
"""Check model ledger and independently recompute paired MMA capture tensors."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import re
import tempfile
import zipfile
import numpy as np
from p1_live_capture import compare_capture
from summarize_p1_mma_batched import check_numerical,rows

PAIRS={('f32-0','f32-1'),('volta-0','volta-1'),('f32-0','volta-0'),('f32-1','volta-1')}


def check_comparisons(comparisons):
    if len(comparisons)!=4 or {(r['reference'],r['candidate']) for r in comparisons}!=PAIRS:
        raise ValueError('incomplete model comparison population')
    for row in comparisons:
        positions=row['positions'];values=[p['kl'] for p in positions]
        if len(positions)!=32 or [p['predicted_token_index_zero_based'] for p in positions]!=list(range(131040,131072)):
            raise ValueError('wrong model scored positions')
        if any(not math.isfinite(v) or v < -1e-12 for v in values):raise ValueError('invalid model KL')
        if not np.isclose(row['mean_kl'],np.mean(values),rtol=1e-10,atol=1e-15) or not np.isclose(row['max_kl'],max(values),rtol=1e-10,atol=1e-15):
            raise ValueError('model KL aggregate mismatch')
        if row['rows']!=32 or not 0<=row['top1_matches']<=32:raise ValueError('invalid model output population')
    return all(r['mean_kl']<=.0005 for r in comparisons)


def summarize(archive,digest,run,diagnostic_failure=False):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('MMA model artifact digest mismatch')
    with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory(prefix='mma-model-recheck-') as tmp:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('MMA model artifact cap exceeded')
        def read(name):return json.loads(z.read(name))
        quality=read('model/quality.json');source=z.read('source-sha.txt').decode().strip()
        if quality['status']!='complete; inspect model qualification':raise ValueError('incomplete model campaign')
        if quality['source_sha']!=source or quality['binary_sha256']!=z.read('binary-sha256.txt').decode().split()[0]:
            raise ValueError('MMA model source/binary identities disagree')
        order=['f32-0','volta-0','volta-1','f32-1']
        if quality['order']!=order or set(quality['arms'])!=set(order) or set(quality['dumps'])!=set(order):
            raise ValueError('incomplete model arms/dumps')
        gate=check_comparisons(quality['comparisons'])
        if gate!=quality['mean_kl_gate_pass']:raise ValueError('model gate disposition mismatch')
        if not gate and not diagnostic_failure:raise ValueError('MMA model quality failed')
        telemetry={};expected={(str(layer),str(p)) for layer in range(2,43,2) for p in range(98304,120001,2048)}
        for label in order:
            prefix='model/'+label+'/';mode='volta' if label.startswith('volta') else 'f32'
            inv=read(prefix+'invocation.json');env=inv['environment']
            if env['ABYSS_V100_IDX']!=mode or env['STRATA_GLM_CPU_LANE']!='0' or env['STRATA_GLM_SCORE']!='131039':
                raise ValueError('MMA model control environment mismatch')
            if (env.get('STRATA_DS4_IDX')=='1')!=(mode=='f32'):raise ValueError('model legacy F32 override mismatch')
            if env['STRATA_V100_MMA_CAPTURE']!=('1' if label=='volta-0' else '0'):raise ValueError('model capture population mismatch')
            log=gzip.decompress(z.read(prefix+'engine.log.gz')).decode()
            dispatch=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_indexer ')]
            if dispatch!=quality['arms'][label]['dispatch_records']:raise ValueError('model dispatch checkpoint mismatch')
            candidates=[r for r in dispatch if r['scorer']!='f32']
            if mode=='volta':
                if len(candidates)!=231 or {(r['layer'],r['p0']) for r in candidates}!=expected or any(r['T']!='2048' or r['scorer']!='volta' for r in candidates):
                    raise ValueError('MMA model candidate dispatch population mismatch')
            elif candidates:raise ValueError('candidate executed in F32 control')
            telemetry[label]=dict(thermal=read(prefix+'thermal.json'),memory=read(prefix+'memory_summary.json'))
        sizes={r['bytes'] for r in quality['dumps'].values()}
        if len(sizes)!=1 or not 0<sum(r['bytes'] for r in quality['dumps'].values())<=64*1024**2:
            raise ValueError('model raw logit population/cap mismatch')
        for label,dump in quality['dumps'].items():
            if dump['bytes']%(32*4) or not re.fullmatch('[0-9a-f]{64}',dump['sha256']) or not re.fullmatch(
                r'/actions-runner/abyss-p1-v9/mma-model/'+str(run)+r'-\d+/'+label+r'\.f32',dump['path']):
                raise ValueError('model logit run identity mismatch')
        root=Path(tmp);prefix='model/volta-0/p1-indexer-capture/'
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe model artifact member')
            if info.filename.startswith(prefix) and not info.is_dir():
                target=root/info.filename[len(prefix):];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
        direct=compare_capture(root,118784);recorded=read('model/direct.json')
        if direct['files']!=recorded['files'] or direct.get('full_selection')!=recorded.get('full_selection'):
            raise ValueError('MMA raw capture identity/selection changed')
        if len(direct['samples'])!=105 or len(recorded['samples'])!=105:raise ValueError('MMA capture sample population changed')
        for a,b in zip(direct['samples'],recorded['samples']):
            if set(a)!=set(b):raise ValueError('MMA capture schema mismatch')
            for key,value in a.items():
                if isinstance(value,float):
                    if not np.isclose(value,b[key],rtol=1e-10,atol=1e-20):raise ValueError('MMA captured metric mismatch')
                elif value!=b[key]:raise ValueError('MMA captured metric mismatch')
        direct['scope']=direct['scope'].replace('cuBLAS path','prepared scaled MMA884 path')
        ranges=[]
        for layer in range(2,43,2):
            item=json.loads((root/f'layer-{layer}'/'query-range.json').read_text())
            if item['sampled_values']!=40960 or item['sampled_heads']!=320 or not 0<=item['zero_values']<=40960:
                raise ValueError('live query sample population mismatch')
            if any(not math.isfinite(item[k]) or item[k]<0 for k in ('minimum_nonzero_abs','maximum_abs')):
                raise ValueError('invalid live query range')
            ranges.append(dict(layer=layer,**item))
        if ranges!=quality['query_ranges']:raise ValueError('live query range ledger mismatch')
        attention=max(r['attention_relative_l2'] for r in direct['samples'])>1e-4
        if attention!=quality['attention_investigation']:raise ValueError('attention disposition mismatch')
        parity=rows(z.read('parity.jsonl').decode());cases=[r for r in parity if r.get('case')=='mma_batch_indexer']
        if len(cases)!=110 or parity[-1]!={'suite':'isolated_mma_batch_indexer','pass':True}:raise ValueError('MMA shared component population failed')
        for record in cases:check_numerical(record)
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in z.read(tool+'.log').decode():raise ValueError('MMA model sanitizer evidence missing')
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=source,quality=quality,
            direct=direct,telemetry=telemetry,component_cases=len(cases),
            passes_written_mean_kl_limit=gate,
            passes_bounded_model_screen=gate and not attention,
            tail_metrics=[dict(reference=r['reference'],candidate=r['candidate'],
                p99_kl=float(np.quantile([p['kl'] for p in r['positions']],.99))) for r in quality['comparisons']],
            f32_repeats_bitwise=quality['dumps']['f32-0']['sha256']==quality['dumps']['f32-1']['sha256'],
            mma_repeats_bitwise=quality['dumps']['volta-0']['sha256']==quality['dumps']['volta-1']['sha256'],
            scope='Raw paired score/ID/attention capture independently recomputed; model KL aggregates checked against '
            'the worker per-position ledger. Raw logits remain hash-identified on the runner, not uploaded here. '
            'CPU-lane-0 diagnostic repeats do not qualify default CPU-lane-28, task corpus or clean performance. '
            'Local-manifest/publisher-authenticity and unavailable counter/QPI limitations remain unchanged.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path)
    ap.add_argument('--sha256',required=True);ap.add_argument('--run',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--diagnostic-failure',action='store_true')
    a=ap.parse_args();a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run,a.diagnostic_failure),indent=2)+'\n')
