#!/usr/bin/env python3
"""Gate clean MMA timing on model evidence and instrumentation-off logit parity."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import p1_integrated_quality as q
from summarize_p1_mma_model import check_comparisons

ORDER=['f32','volta','volta','f32','f32','volta']


def qualified(evidence):
    quality=evidence['quality']
    if quality['status']!='complete; inspect model qualification' or not re.fullmatch('[0-9a-f]{40}',quality['source_sha']):
        raise ValueError('completed source-identified quality evidence required')
    if not check_comparisons(quality['comparisons']) or quality['attention_investigation']:
        raise ValueError('model quality or direct attention screen failed')
    if not evidence['passes_bounded_model_screen'] or not evidence['f32_repeats_bitwise'] or not evidence['mma_repeats_bitwise']:
        raise ValueError('clean bridge requires passing deterministic model repeats')
    for mode in ('f32','volta'):
        if quality['dumps'][mode+'-0']['sha256']!=quality['dumps'][mode+'-1']['sha256']:
            raise ValueError('clean bridge requires identical retained repeat hashes')
    if quality['cpu_lane']!=0 or quality['geometry']!=q.GEOMETRIES['128k']:
        raise ValueError('clean campaign requires the qualified CPU-lane-0 full-128K geometry')
    return quality


def clean_cache(path):
    values={}
    for line in path.read_text().splitlines():
        if not line.startswith(('#','//')) and '=' in line and ':' in line.split('=',1)[0]:
            key,value=line.split('=',1);values[key.split(':',1)[0]]=value
    for name in ('STRATA_V100_CAPTURE_INDEXER','STRATA_V100_CAPTURE_STAGES','STRATA_V100_CAPTURE_COST'):
        if values.get(name)!='OFF':raise ValueError('clean timing requires '+name+'=OFF')
    if values.get('CMAKE_BUILD_TYPE')!='Release' or values.get('CMAKE_CUDA_ARCHITECTURES')!='70':
        raise ValueError('clean build must be Release / SM70')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quality',type=Path,required=True)
    args=ap.parse_args();evidence=json.loads(args.quality.read_text());quality=qualified(evidence)
    build=q.ROOT/'build-v100-p1-mma-clean';binary=build/'strata';clean_cache(build/'CMakeCache.txt')
    if q.campaign.model_identity()!=quality['model_identity']:raise ValueError('qualified model identity changed')
    frozen=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if frozen['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen pack changed')
    # Building without capture must not quietly change the arithmetic we qualified.
    source=q.p0.command(['git','diff','--exit-code',quality['source_sha'],'--','src','include','CMakeLists.txt','CMakePresets.json'])
    if source['returncode']!=0:raise ValueError('runtime sources differ from qualified model campaign')
    run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not all(part.isdecimal() for part in run.split('-')):raise ValueError('invalid run identity')
    cache=Path('/actions-runner/abyss-p1-v9/mma-clean')/run;cache.mkdir(parents=True,exist_ok=False)
    out=Path(os.environ['EVIDENCE']).absolute()/'clean';out.mkdir(parents=True,exist_ok=False)
    tokens=cache/'tokens';tokens.write_text(','.join(map(str,q.fixture('mixed-131072'))))
    if q.p0.sha256(tokens)!=quality['fixture_sha256']:raise ValueError('qualified fixture changed')
    if shutil.disk_usage(cache).free<256*1024**2:raise RuntimeError('evidence disk margin below 256 MiB')
    state=dict(schema=1,status='running clean bridge',quality_evidence_sha256=q.p0.sha256(args.quality),
        qualified_run=evidence['run'],qualified_binary_sha256=quality['binary_sha256'],binary_sha256=q.p0.sha256(binary),
        source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),cpu_lane=0,bridge={},arms=[],pairs=[],
        scope='Instrumentation-off binary, same runtime sources/model/pack/fixtures. Two CPU-lane-0 full-128K '
        'logit hashes must reproduce qualified F32 and MMA repeats before any clean timing. Three interleaved '
        'process pairs use two uncached requests each at 8K/128 outputs and 128K/16 outputs; request 0 excluded '
        'from primary comparison but retained. CPU lane 0 only, not default CPU lane 28 qualification. '
        'OS cache uncontrolled; no global cache changes. Counters unavailable and QPI unmeasured; local model '
        'manifest does not establish publisher authenticity. Performance requires review of per-pair deltas/telemetry.')
    def checkpoint():
        q.p0.write_json(out/'clean-ab.json',state);q.p0.write_json(cache/'checkpoint.json',state)
    checkpoint()
    try:
        for mode in ('f32','volta'):
            dump=cache/(mode+'.f32')
            state['bridge'][mode]=q.arm(binary,out/('bridge-'+mode),tokens,dump,mode,q.GEOMETRIES['128k'],
                overrides=dict(STRATA_GLM_CPU_LANE='0',STRATA_V100_MMA_CAPTURE='0'))
            sha=q.p0.sha256(dump);expected=quality['dumps'][mode+'-0']
            state['bridge'][mode].update(bytes=dump.stat().st_size,sha256=sha)
            checkpoint()
            if dump.stat().st_size!=expected['bytes'] or sha!=expected['sha256']:
                raise ValueError('clean '+mode+' logits differ from qualified capture build; inspect before timing')
        cases=[('mixed-8192',q.fixture('mixed-8192'),128),('mixed-131072',q.fixture('mixed-131072'),16)]
        state.update(status='running clean timing',order=ORDER,requests_per_case=2,
            fixture_sha256={name:q.p0.sha256(q.campaign.CAMPAIGN/'fixtures'/(name+'.tokens')) for name,_,_ in cases})
        checkpoint()
        for index,mode in enumerate(ORDER):
            directory=out/(str(index)+'-'+mode);env=q.fixed_env()
            env.update(ABYSS_V100_IDX=mode,STRATA_DS4_ATTN='1',STRATA_GLM_CPU_LANE='0')
            if mode=='f32':env['STRATA_DS4_IDX']='1'
            if q.campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
            requests=q.campaign.arm(binary,directory,cases,env,135168,'auto');q.compress(directory/'engine.log')
            with gzip.open(directory/'engine.log.gz','rt') as stream:log=stream.read()
            dispatch=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_indexer ')]
            candidate=[r for r in dispatch if r['scorer']!='f32']
            if not dispatch or len(candidate)!=(462 if mode=='volta' else 0) or any(r['scorer']!='volta' or r['T']!='2048' for r in candidate):
                raise ValueError('clean actual scorer population disagrees with qualified regime')
            if len(requests)!=4:raise ValueError('incomplete clean request population')
            state['arms'].append(dict(index=index,mode=mode,requests=requests,candidate_calls=len(candidate),
                thermal=json.loads((directory/'thermal.json').read_text()),memory=json.loads((directory/'memory_summary.json').read_text())))
            if index%2:
                pair={arm['mode']:arm for arm in state['arms'][-2:]}
                for name,_,_ in cases:
                    a,b=(next(r for r in pair[m]['requests'] if r['case']==name and r['request']==1) for m in ('f32','volta'))
                    state['pairs'].append(dict(pair=index//2,case=name,
                        generated_ids_equal=a['output_ids_sha256']==b['output_ids_sha256'],
                        **{key+'_change_percent':100*(b[key]/a[key]-1) for key in ('prefill_ms','decode_ms','ttft_ms')}))
            checkpoint()
        state['medians']={name:{key:statistics.median(row[key] for row in state['pairs'] if row['case']==name)
            for key in ('prefill_ms_change_percent','decode_ms_change_percent','ttft_ms_change_percent')} for name,_,_ in cases}
        state['status']='complete; review sustained paired performance';checkpoint()
    except BaseException as error:
        state.update(status='incomplete; inspect cause before retry',error=str(error));checkpoint();raise
    if sum(path.stat().st_size for path in out.rglob('*') if path.is_file())>64*1024**2:
        raise ValueError('clean campaign artifact exceeds 64 MiB')


if __name__=='__main__':main()
