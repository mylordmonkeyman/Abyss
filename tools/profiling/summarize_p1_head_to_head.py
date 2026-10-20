#!/usr/bin/env python3
"""Check four-indexer model ledgers and independently re-read six raw captures."""
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
from p1_indexer_head_to_head import ORDER,POLICY,direct_capture,validate_corrections
from p1_integrated_quality import GEOMETRIES
from summarize_p1_cublas_corrected import VARIANTS,rows,validate_population

MODES=('f32','cublas','corrected','volta')
LABELS={m+'-'+str(i) for m in MODES for i in (0,1)}
PAIRS={('f32-0','f32-1')}|{(m+'-0',m+'-1') for m in MODES[1:]}
PAIRS|={(f'f32-{i}',m+'-'+str(i)) for i in (0,1) for m in MODES[1:]}

def check_quality(state,name,source,binary,run):
    geometry=GEOMETRIES[name]
    if state['status']!='complete diagnostic model comparison' or state['source_sha']!=source or state['binary_sha256']!=binary:
        raise ValueError('incomplete model campaign or source/binary mismatch')
    if state['geometry']!=geometry or state['cpu_lane']!=0 or state['correction_policy']!=POLICY or state['order']!=list(ORDER):
        raise ValueError('wrong model fixture/configuration/order')
    if set(state['arms'])!=LABELS or set(state['dumps'])!=LABELS:raise ValueError('incomplete model arm/logit population')
    for label,dump in state['dumps'].items():
        if dump['bytes']!=16547840 or not re.fullmatch('[a-f0-9]{64}',dump['sha256']) or not re.fullmatch(
            r'/actions-runner/abyss-p1-v9/indexer-head-to-head/'+str(run)+r'-\d+/'+name+'/'+label+r'\.f32',dump['path']):
            raise ValueError('invalid retained model logit identity')
    for mode in MODES:
        if state['dumps'][mode+'-0']['sha256']!=state['dumps'][mode+'-1']['sha256']:
            raise ValueError('non-deterministic model repeat')
    comparisons=state['comparisons']
    if len(comparisons)!=10 or {(r['reference'],r['candidate']) for r in comparisons}!=PAIRS:
        raise ValueError('incomplete model comparison population')
    tails=[]
    for r in comparisons:
        values=[p['kl'] for p in r['positions']]
        if len(values)!=32 or [p['predicted_token_index_zero_based'] for p in r['positions']]!=list(range(geometry['score_start']+1,geometry['score_start']+33)):
            raise ValueError('wrong teacher-forced scored positions')
        if any(not math.isfinite(v) or v< -1e-12 for v in values) or not math.isclose(r['mean_kl'],math.fsum(values)/32,rel_tol=1e-10,abs_tol=1e-15) or not math.isclose(r['max_kl'],max(values),rel_tol=1e-10,abs_tol=1e-15):
            raise ValueError('invalid KL or aggregate ledger')
        if r['rows']!=32 or not 0<=r['top1_matches']<=32:raise ValueError('invalid model output population')
        if state['dumps'][r['reference']]['sha256']==state['dumps'][r['candidate']]['sha256'] and (any(v!=0 for v in values) or r['top1_matches']!=32):
            raise ValueError('identical raw logits disagree with comparison ledger')
        tails.append(dict(reference=r['reference'],candidate=r['candidate'],mean_kl=r['mean_kl'],worst_kl=r['max_kl'],
            p99_kl=float(np.quantile(values,.99)),top1_matches=r['top1_matches'],passes_mean_kl_limit=r['mean_kl']<=.0005))
    is_original_cross=lambda r:r['reference'].startswith('f32') and r['candidate'].startswith('cublas')
    passed=all(r['mean_kl']<=.0005 for r in comparisons if not is_original_cross(r))
    original_pass=all(r['mean_kl']<=.0005 for r in comparisons if is_original_cross(r))
    if state['qualified_mean_kl_gate_pass']!=passed or state['original_cublas_mean_kl_gate_pass']!=original_pass:
        raise ValueError('model gate disposition mismatch')
    return passed,original_pass,tails

def same(a,b):
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-20)
    return a==b

def checked_zip(path,digest):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('artifact ZIP digest mismatch')
    z=zipfile.ZipFile(path)
    try:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('artifact cap exceeded')
        names=[]
        for i in z.infolist():
            p=PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe artifact path')
            names.append(i.filename)
        if len(names)!=len(set(names)):raise ValueError('duplicate artifact member')
    except BaseException:z.close();raise
    return z

def summarize(archive,digest,run,captures):
    expected={(g,m) for g in GEOMETRIES for m in MODES[1:]}
    if len(captures)!=6 or {(g,m) for g,m,_,_ in captures}!=expected:raise ValueError('exactly six capture artifacts required')
    capture_map={(g,m):(Path(p),sha) for g,m,p,sha in captures}
    with checked_zip(archive,digest) as z:
        source=z.read('source-sha.txt').decode().strip();binary=z.read('binary-sha256.txt').decode().split()[0]
        if not re.fullmatch('[a-f0-9]{40}',source) or not re.fullmatch('[a-f0-9]{64}',binary):raise ValueError('invalid build identity')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in z.read('alias-'+tool+'.log').decode():raise ValueError('alias sanitizer evidence incomplete')
        parity=rows(z.read('alias-parity.jsonl').decode())
        if parity[0].get('borrowed_aliases') is not True or parity[-1]!={'suite':'isolated_cublas_corrections','pass':True}:
            raise ValueError('missing actual borrowed-alias component suite')
        cases={(v,T,p,0) for v in VARIANTS for T in (1,3,8,9,33) for p in (2040,2052,8192)}
        cases|={(v,3,2052,m) for v in VARIANTS for m in (1,2,3,4)}|{(v,3,40000,1) for v in VARIANTS}
        validate_population(parity,cases)
        result=dict(schema=1,run=run,source_sha=source,binary_sha256=binary,artifact_zip_sha256=digest,
            alias_component_cases=140,geometries={},passes_bounded_corrected_and_mma_screen=True)
        identity=None
        for name,geometry in GEOMETRIES.items():
            state=json.loads(z.read('model/'+name+'/quality.json'))
            passed,original_pass,tails=check_quality(state,name,source,binary,run)
            if identity is not None and state['model_identity']!=identity:raise ValueError('model identity differs between fixtures')
            identity=state['model_identity'];telemetry={};dispatch={};corrections={};direct={}
            for label in sorted(LABELS):
                mode=label.rsplit('-',1)[0];prefix='model/'+name+'/'+label+'/'
                invocation=json.loads(z.read(prefix+'invocation.json'));env=invocation['environment']
                engine_mode='cublas' if mode=='corrected' else mode
                capture=mode!='f32' and label.endswith('-0')
                if env['ABYSS_V100_IDX']!=engine_mode or env['STRATA_GLM_CPU_LANE']!='0' or env['STRATA_GLM_SCORE']!=str(geometry['score_start']):raise ValueError('model control environment mismatch')
                if (env.get('STRATA_DS4_IDX')=='1')!=(mode=='f32') or env['ABYSS_V100_CUBLAS_CORRECTION']!=(POLICY if mode=='corrected' else 'none'):
                    raise ValueError('model correction/F32 override mismatch')
                if env.get('STRATA_V100_CUBLAS_SELECTION_CONTROL') or any(env[k]!=('1' if capture else '0') for k in ('STRATA_V100_CUBLAS_CAPTURE','STRATA_V100_MMA_CAPTURE')):
                    raise ValueError('unexpected selection intervention or capture control')
                if env['STRATA_GLM_SCORE_DUMP']!=state['dumps'][label]['path'] or env['STRATA_GLM_SCORE_PREFILL']!='1' or env['STRATA_DS4_ATTN']!='1' or env['ABYSS_V100_CAPTURE_P0']!=str(geometry['candidate_positions'][-1]):
                    raise ValueError('wrong model distribution dump or live-capture geometry')
                argv=invocation['argv']
                if argv[argv.index('--max-context')+1]!=str(geometry['max_context']) or argv[argv.index('--tokens-file')+1]!=str(Path(state['dumps'][label]['path']).parent/'tokens'):
                    raise ValueError('wrong model context or frozen token file')
                log=gzip.decompress(z.read(prefix+'engine.log.gz')).decode()
                records=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_indexer ')]
                if not records or records!=state['arms'][label]['dispatch_records']:raise ValueError('dispatch checkpoint disagrees with engine log')
                candidate=[r for r in records if r['scorer']!='f32']
                expected_calls={(str(layer),str(p)) for layer in range(2,43,2) for p in geometry['candidate_positions']} if mode!='f32' else set()
                refinement=POLICY if mode=='corrected' else 'empirical-band-f32' if mode=='volta' else 'none'
                if len(candidate)!=len(expected_calls) or {(r['layer'],r['p0']) for r in candidate}!=expected_calls or any(r['T']!='2048' or r['scorer']!=engine_mode or r['refinement']!=refinement for r in candidate):raise ValueError('actual model candidate coverage changed')
                counts=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_cublas_correction ')]
                if counts!=state['arms'][label]['corrections']:raise ValueError('correction checkpoint disagrees with engine log')
                if mode=='corrected':validate_corrections(counts,geometry,POLICY)
                elif counts:raise ValueError('unexpected correction in comparator')
                dispatch[label]=dict(records=len(records),candidate_calls=len(candidate),canonical_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())
                if counts:
                    totals={k:sum(int(r[k]) for r in counts) for k in ('T','flagged_tokens','corrected_tokens','full_tokens','rescored_rows')}
                    totals['flagged_fraction']=totals['flagged_tokens']/totals['T'];totals['corrected_fraction']=totals['corrected_tokens']/totals['T']
                    corrections[label]=totals
                telemetry[label]={k:json.loads(z.read(prefix+k+'.json')) for k in ('thermal','memory_summary')}
                if telemetry[label]['memory_summary']['samples']<=0:raise ValueError('missing model telemetry')
                if capture:
                    path,sha=capture_map[name,mode]
                    with checked_zip(path,sha) as raw,tempfile.TemporaryDirectory(prefix='p1-four-indexer-') as tmp:
                        raw.extractall(tmp) # all paths, duplicate names, symlinks and total bytes checked above
                        actual=direct_capture(Path(tmp),geometry['candidate_positions'][-1],mode)
                    recorded=state['arms'][label]['capture']
                    if not same(actual,recorded):raise ValueError('raw capture and model ledger disagree')
                    samples=actual['direct']['samples'];full=actual['direct']['full_selection']
                    direct[mode]=dict(artifact_zip_sha256=sha,samples=len(samples),ordered_identical=sum(r['ordered_identical'] for r in full),
                        selection_rows=sum(r['rows'] for r in full),membership_changed=sum(r['membership_changed'] for r in full),
                        exact_selected_score_samples=actual['exact_selected_score_samples'],maximum_score_relative_l2=max(r['score_relative_l2'] for r in samples),
                        maximum_attention_relative_l2=max(r['attention_relative_l2'] for r in samples),
                        query_ranges=actual['query_ranges'],full_selection=full,
                        raw_file_ledger_canonical_sha256=hashlib.sha256(json.dumps(actual['direct']['files'],sort_keys=True).encode()).hexdigest())
            result['geometries'][name]=dict(quality=dict(status=state['status'],geometry=geometry,cpu_lane=0,correction_policy=POLICY,
                fixture_sha256=state['fixture_sha256'],dumps=state['dumps'],comparisons=state['comparisons']),tail_metrics=tails,
                passes_corrected_and_mma_mean_kl_limit=passed,original_cublas_passes_mean_kl_limit=original_pass,
                dispatch=dispatch,corrections=corrections,direct=direct,telemetry=telemetry)
            result['passes_bounded_corrected_and_mma_screen'] &= passed
        result['model_identity']=identity
        result['scope']='Six digest-checked raw captures re-read; model per-position KL aggregates, tails, dispatch/correction logs, repeat hashes, aliases and sanitizers checked. '
        result['scope']+='Raw logits remain hash-identified on the runner. Original cuBLAS is separately reported and never silently admitted. CPU0 screens do not qualify default CPU28/tasks/platforms or clean performance. '
        result['scope']+='Empirical envelope is not a global bound; local manifest does not prove publisher authenticity; counters unavailable/QPI unmeasured.'
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',type=int,required=True);p.add_argument('--capture',nargs=4,action='append',required=True,metavar=('GEOMETRY','MODE','ZIP','SHA256'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run,a.capture),indent=2)+'\n')
