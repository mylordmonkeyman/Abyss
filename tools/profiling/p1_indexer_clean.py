#!/usr/bin/env python3
"""Clean logit bridge followed by four balanced, checkpointed model timing blocks."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import p1_integrated_quality as q
from p1_indexer_head_to_head import ORDER as QUALITY_ORDER,POLICY
from p1_mma_clean_ab import clean_cache
from summarize_p1_head_to_head import check_quality,MODES,LABELS

# Williams design: each mode occupies each process position once, and every
# ordered predecessor pair occurs once across the four blocks.
ORDERS=(('f32','cublas','volta','corrected'),('cublas','corrected','f32','volta'),
        ('corrected','volta','cublas','f32'),('volta','f32','corrected','cublas'))
QUALITY=q.ROOT/'docs/v100/p1-indexer-model-head-to-head-2026-10-20.json'
BUILD=q.ROOT/'build-v100-p1-indexer-clean'
CONTEXT=135168

def qualified(evidence):
    if not evidence['passes_bounded_corrected_and_mma_screen'] or set(evidence['geometries'])!=set(q.GEOMETRIES):
        raise ValueError('both verified model fixtures required')
    for name,record in evidence['geometries'].items():
        state=dict(record['quality'],source_sha=evidence['source_sha'],binary_sha256=evidence['binary_sha256'],
            arms={k:{} for k in LABELS},order=list(QUALITY_ORDER),
            qualified_mean_kl_gate_pass=record['passes_corrected_and_mma_mean_kl_limit'],
            original_cublas_mean_kl_gate_pass=record['original_cublas_passes_mean_kl_limit'])
        passed,_,_=check_quality(state,name,evidence['source_sha'],evidence['binary_sha256'],evidence['run'])
        if not passed:raise ValueError('corrected/MMA model quality failed')
        for mode in ('corrected','volta'):
            direct=record['direct'][mode]
            if (direct['samples'],direct['selection_rows'],direct['ordered_identical'],direct['membership_changed'],
                direct['exact_selected_score_samples'],direct['maximum_attention_relative_l2'])!=(105,43008,43008,0,105,0):
                raise ValueError('exact model capture qualification required')
    return evidence

def environment(mode,profile=False):
    env=q.fixed_env();env.update(ABYSS_V100_IDX='cublas' if mode=='corrected' else mode,
        ABYSS_V100_CUBLAS_CORRECTION=POLICY if mode=='corrected' else 'none',
        ABYSS_V100_PROFILE='1' if profile else '0',STRATA_DS4_ATTN='1',STRATA_GLM_CPU_LANE='0',
        STRATA_V100_CUBLAS_CAPTURE='0',STRATA_V100_MMA_CAPTURE='0',STRATA_V100_CUBLAS_SELECTION_CONTROL='')
    if mode=='f32':env['STRATA_DS4_IDX']='1'
    return env

def cache_path():
    run=os.environ['GITHUB_RUN_ID']
    if not run.isdecimal():raise ValueError('invalid clean campaign run identity')
    # Stable across retries of failed jobs. Completed work is reused only after
    # identity checks; an incomplete child directory is never retried silently.
    return Path('/actions-runner/abyss-p1-v9/indexer-clean')/run

def source_check(evidence):
    result=q.p0.command(['git','diff','--exit-code',evidence['source_sha'],'--','src','include','CMakeLists.txt','CMakePresets.json'])
    if result['returncode']!=0:raise ValueError('runtime source differs from qualified model or could not be checked')
    current=q.campaign.model_identity()
    if any(current.get(k)!=evidence['model_identity'].get(k) for k in
           ('sha256','stat_signature','local_SHA256SUMS_sha256','matches_local_SHA256SUMS')):
        raise ValueError('qualified model identity changed')
    frozen=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if frozen['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen model pack changed')

def cases():
    long=q.fixture('mixed-131072')
    return [('mixed-8192',q.fixture('mixed-8192'),128),
            ('mixed-100386',long[:100386],32),('mixed-131072',long,32)]

def case_identities():
    return {name:dict(tokens=len(ids),output_tokens=count,canonical_tokens_sha256=hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest())
        for name,ids,count in cases()}

def libraries(binary):
    result=q.p0.command(['ldd',str(binary)])
    if result['returncode']!=0 or 'not found' in result['stdout']:raise ValueError('clean binary runtime libraries unresolved')
    paths={Path(p).resolve() for p in re.findall(r'/[^\s()]+',result['stdout'])}
    return {str(p):q.p0.sha256(p) for p in sorted(paths)}

def validate_dispatch(records,mode,geometries,repetitions):
    if not records:raise ValueError('missing clean scorer dispatch')
    candidates=[r for r in records if r['scorer']!='f32']
    expected=Counter()
    engine_mode='cublas' if mode=='corrected' else mode
    refinement=POLICY if mode=='corrected' else 'empirical-band-f32' if mode=='volta' else 'none'
    if mode!='f32':
        for geometry in geometries:
            for layer in range(2,43,2):
                for pos in geometry['candidate_positions']:expected[str(layer),str(pos),'2048',engine_mode,refinement]+=repetitions
    actual=Counter((r['layer'],r['p0'],r['T'],r['scorer'],r.get('refinement')) for r in candidates)
    if actual!=expected:raise ValueError('clean candidate dispatch population changed')
    return len(candidates)

def manifest(cache,evidence):
    state=json.loads((cache/'manifest.json').read_text())
    clean_cache(cache/'CMakeCache.txt')
    if state['quality_evidence_sha256']!=q.p0.sha256(QUALITY) or state['binary_sha256']!=q.p0.sha256(cache/'strata') or state['case_identities']!=case_identities():
        raise ValueError('clean checkpoint binary/evidence/fixtures changed')
    if state['qualified_source_sha']!=evidence['source_sha'] or state['qualified_run']!=evidence['run']:
        raise ValueError('clean checkpoint qualified source changed')
    if state['orders']!=[list(r) for r in ORDERS] or state['gpu_uuid']!=os.environ.get('CUDA_VISIBLE_DEVICES') or state['context']!=CONTEXT:
        raise ValueError('clean checkpoint order or GPU changed')
    if state['runtime_libraries']!=libraries(cache/'strata'):raise ValueError('clean runtime library identity changed')
    return state

def bridge(cache,evidence):
    if not cache.exists():
        if shutil.disk_usage(cache.parent.parent).free<2*1024**3:raise RuntimeError('clean evidence disk margin below2GiB')
        clean_cache(BUILD/'CMakeCache.txt');cache.mkdir(parents=True,exist_ok=False)
        shutil.copy2(BUILD/'strata',cache/'strata');(cache/'strata').chmod(0o555)
        shutil.copy2(BUILD/'CMakeCache.txt',cache/'CMakeCache.txt')
        q.p0.write_json(cache/'manifest.json',dict(schema=1,qualified_source_sha=evidence['source_sha'],qualified_run=evidence['run'],
            source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),quality_evidence_sha256=q.p0.sha256(QUALITY),
            binary_sha256=q.p0.sha256(cache/'strata'),case_identities=case_identities(),orders=ORDERS,
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),context=CONTEXT,runtime_libraries=libraries(cache/'strata')))
    identity=manifest(cache,evidence)
    out=cache/'bridge';out.mkdir(exist_ok=True);raw=cache/'raw';raw.mkdir(exist_ok=True)
    checkpoint=out/'bridge.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(status='running clean bridge',manifest=identity,arms={})
    for name,geometry in q.GEOMETRIES.items():
        tokens=raw/(name+'.tokens')
        tokens.write_text(','.join(map(str,q.fixture('mixed-131072')[:geometry['token_count']])))
        quality=evidence['geometries'][name]['quality']
        if q.p0.sha256(tokens)!=quality['fixture_sha256']:raise ValueError('qualified teacher-forced fixture changed')
        for mode in MODES:
            label=name+'-'+mode;dump=raw/(label+'.f32');directory=out/label
            expected=quality['dumps'][mode+'-0']
            if label not in state['arms']:
                if directory.exists():raise RuntimeError('incomplete bridge arm preserved; inspect before retry: '+label)
                print('Starting clean logit bridge',label,flush=True)
                # The fixed allocation context is also used by timing at all
                # prompt lengths. Hash parity certifies this layout as well.
                arm=q.arm(cache/'strata',directory,tokens,dump,'cublas' if mode=='corrected' else mode,
                    dict(geometry,max_context=CONTEXT),overrides=environment(mode,True))
                validate_dispatch(arm['dispatch_records'],mode,[geometry],1)
                arm.update(bytes=dump.stat().st_size,sha256=q.p0.sha256(dump))
                state['arms'][label]=arm;q.p0.write_json(checkpoint,state)
            arm=state['arms'][label]
            if dump.stat().st_size!=expected['bytes'] or q.p0.sha256(dump)!=expected['sha256'] or arm['sha256']!=expected['sha256']:
                raise ValueError('clean build changed qualified logits: '+label)
            if sum(p.stat().st_size for p in raw.glob('*.f32'))>128*1024**2:raise ValueError('clean raw logit cap exceeded')
    state['status']='complete; eight clean logit hashes match model qualification';q.p0.write_json(checkpoint,state)

def block(cache,evidence,index):
    identity=manifest(cache,evidence)
    bridge_state=json.loads((cache/'bridge/bridge.json').read_text())
    if bridge_state['status']!='complete; eight clean logit hashes match model qualification' or len(bridge_state['arms'])!=8:
        raise ValueError('all eight clean logit bridge arms required before timing')
    out=cache/('block-'+str(index));out.mkdir(exist_ok=True);checkpoint=out/'timing.json'
    state=json.loads(checkpoint.read_text()) if checkpoint.exists() else dict(schema=1,status='running clean timing block',
        manifest=identity,block=index,order=ORDERS[index],arms=[],comparisons=[],
        scope='CPU0, same immutable clean binary, pack and canonical fixtures. Two no-prefix-reuse requests per case; primary comparison uses request1, request0 retained. '
        'Both requests retain carry-over expert residency and OS cache is uncontrolled. Original cuBLAS is an unqualified quality comparator; differing generated trajectories confound decode. '
        'No global cache/page-placement/privilege changes. RSS/PSS/cgroup/pin and borrowed scratch are overlapping views, never additive. Local manifest authenticity/counter/QPI limitations remain.')
    if state['status']=='complete; review block performance':return
    if tuple(state['order'])!=ORDERS[index] or [a['mode'] for a in state['arms']]!=list(ORDERS[index][:len(state['arms'])]):
        raise ValueError('timing checkpoint arm order changed')
    for mode in ORDERS[index]:
        if any(a['mode']==mode for a in state['arms']):continue
        directory=out/mode
        if directory.exists():raise RuntimeError('incomplete timing arm preserved; inspect before retry: '+mode)
        if q.campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below16GiB')
        print('Starting clean timing block',index,mode,flush=True)
        requests=q.campaign.arm(cache/'strata',directory,cases(),environment(mode),CONTEXT,'auto')
        q.compress(directory/'engine.log')
        with gzip.open(directory/'engine.log.gz','rt') as stream:log=stream.read()
        if 'p1_cublas_correction ' in log or 'p1_cublas_selection_control ' in log or (directory/'p1-indexer-capture').exists():
            raise ValueError('diagnostic correction/capture work appeared in clean timing')
        records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_indexer ')]
        calls=validate_dispatch(records,mode,list(q.GEOMETRIES.values()),2)
        if len(requests)!=6 or {(r['case'],r['request']) for r in requests}!={(c,i) for c,_,_ in cases() for i in (0,1)}:
            raise ValueError('incomplete clean request population')
        if any(not isinstance(r.get(k),(int,float)) or not math.isfinite(r[k]) or r[k]<=0
               for r in requests for k in ('prefill_ms','decode_ms','ttft_ms')):
            raise ValueError('invalid clean timing metrics')
        state['arms'].append(dict(mode=mode,requests=requests,candidate_calls=calls,
            thermal=json.loads((directory/'thermal.json').read_text()),memory=json.loads((directory/'memory_summary.json').read_text())))
        q.p0.write_json(checkpoint,state)
    for reference,candidate in [('f32',m) for m in MODES[1:]]+[('corrected','volta')]:
        pair={a['mode']:a for a in state['arms']}
        for name,_,_ in cases():
            a,b=(next(r for r in pair[m]['requests'] if r['case']==name and r['request']==1) for m in (reference,candidate))
            state['comparisons'].append(dict(reference=reference,candidate=candidate,case=name,
                generated_ids_equal=a['output_ids_sha256']==b['output_ids_sha256'],
                **{k+'_change_percent':100*(b[k]/a[k]-1) for k in ('prefill_ms','decode_ms','ttft_ms')}))
    state['status']='complete; review block performance';q.p0.write_json(checkpoint,state)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge',action='store_true');parser.add_argument('--block',type=int,choices=range(4))
    args=parser.parse_args()
    if args.bridge==(args.block is not None):raise ValueError('choose exactly bridge or block')
    evidence=qualified(json.loads(QUALITY.read_text()));source_check(evidence);cache=cache_path()
    part='bridge' if args.bridge else 'block-'+str(args.block)
    try:
        if args.bridge:bridge(cache,evidence)
        else:block(cache,evidence,args.block)
    finally:
        directory=cache/part
        if directory.exists():
            target=Path(os.environ['EVIDENCE'])/part;shutil.copytree(directory,target,dirs_exist_ok=True)
            if sum(p.stat().st_size for p in target.rglob('*') if p.is_file())>64*1024**2:raise ValueError('clean artifact cap exceeded')

if __name__=='__main__':main()
