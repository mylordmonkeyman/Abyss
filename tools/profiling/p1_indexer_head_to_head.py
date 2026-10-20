#!/usr/bin/env python3
"""Same-binary F32/cuBLAS/corrected-cuBLAS/MMA deterministic model screen."""
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import numpy as np
import p1_integrated_quality as q
from p1_f32_triage import comparison
from p1_live_capture import compare_capture

ORDER=('f32','cublas','corrected','volta','volta','corrected','cublas','f32')
POLICY='band-all' # Admission still requires recorded component and model evidence.

def validate_corrections(records,geometry,policy):
    expected={(str(layer),str(p)) for layer in range(2,43,2) for p in geometry['candidate_positions']}
    if len(records)!=len(expected) or {(r['layer'],r['p0']) for r in records}!=expected:
        raise ValueError('incomplete actual-model correction population')
    for r in records:
        if r['policy']!=policy or int(r['T'])!=2048 or int(r['corrected_tokens'])!=2048 or not 0<=int(r['flagged_tokens'])<=2048:
            raise ValueError('wrong actual-model correction policy/accounting')
        if not 0<=int(r['full_tokens'])<=2048 or not 512*2048<=int(r['rescored_rows'])<=sum((int(r['p0'])+t+1)//4 for t in range(2048)):
            raise ValueError('invalid actual-model rescore accounting')

def direct_capture(directory,target,mode):
    direct=compare_capture(directory,target)
    direct['scope']+=' Comparator='+mode+'.'
    selected_exact=0;ranges=[]
    for layer in range(2,43,2):
        root=directory/f'layer-{layer}'
        ranges.append(dict(layer=layer,**json.loads((root/'query-range.json').read_text())))
        for t in (0,255,256,1023,2047):
            a=np.fromfile(root/f't-{t}-candidate-score.f32',dtype='<f4')
            b=np.fromfile(root/f't-{t}-reference-score.f32',dtype='<f4')
            ids=np.fromfile(root/f't-{t}-candidate-ids.i32',dtype='<i4')
            selected_exact+=int(np.array_equal(a[ids],b[ids]))
    if len(direct.get('full_selection',[]))!=21:raise ValueError('missing all-row selection comparison')
    if any(r['sampled_values']!=40960 or r['sampled_heads']!=320 for r in ranges):raise ValueError('query range population changed')
    if mode in ('corrected','volta'):
        if selected_exact!=105 or any(r['ordered_identical']!=2048 for r in direct['full_selection']) or any(r['attention_max_abs']!=0 for r in direct['samples']):
            raise ValueError('corrected/MMA actual-model exact selection/attention contract failed')
    return dict(direct=direct,query_ranges=ranges,exact_selected_score_samples=selected_exact)

def run_geometry(binary,out,cache,name,model):
    geometry=q.GEOMETRIES[name];out.mkdir();cache.mkdir()
    tokens=cache/'tokens';ids=q.fixture('mixed-131072')[:geometry['token_count']]
    if len(ids)!=geometry['token_count']:raise ValueError('frozen fixture length mismatch')
    tokens.write_text(','.join(map(str,ids)))
    state=dict(schema=1,status='running model head-to-head',source_sha=q.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        geometry=geometry,cpu_lane=0,correction_policy=POLICY,order=list(ORDER),binary_sha256=q.p0.sha256(binary),
        fixture_sha256=q.p0.sha256(tokens),model_identity=model,arms={},dumps={},comparisons=[],
        scope='Same binary/pack/frozen fixture, CPU0, balanced eight-arm order. First repeat of each candidate captures live selections/scores/attention. '
        'Original cuBLAS quality failure is measured as comparator; corrected and MMA admission screens remain unchanged at meanKL<=0.0005. '
        'All host counter copies and capture synchronization are diagnostic; these arms cannot measure speed. Raw logits retained at <=256MiB total across both fixtures. '
        'CPU28/task/platform/clean-performance gates remain. Local manifest verification does not prove publisher authenticity; counters unavailable/QPI unmeasured.')
    def checkpoint():q.p0.write_json(out/'quality.json',state);q.p0.write_json(cache/'checkpoint.json',state)
    arrays={};seen={};checkpoint()
    for mode in ORDER:
        repeat=seen.get(mode,0);seen[mode]=repeat+1;label=mode+'-'+str(repeat)
        engine_mode='cublas' if mode=='corrected' else mode
        dump=cache/(label+'.f32');case=out/label;capture=mode!='f32' and repeat==0
        print('Starting head-to-head',name,label,flush=True)
        arm=q.arm(binary,case,tokens,dump,engine_mode,geometry,overrides=dict(STRATA_GLM_CPU_LANE='0',
            ABYSS_V100_CUBLAS_CORRECTION=POLICY if mode=='corrected' else 'none',
            STRATA_V100_CUBLAS_SELECTION_CONTROL='',STRATA_V100_CUBLAS_CAPTURE='1' if capture else '0',
            STRATA_V100_MMA_CAPTURE='1' if capture else '0',ABYSS_V100_CAPTURE_P0=str(geometry['candidate_positions'][-1])),cwd=case)
        candidates=[r for r in arm['dispatch_records'] if r['scorer'] in ('cublas','volta')]
        expected_refinement=POLICY if mode=='corrected' else 'empirical-band-f32' if mode=='volta' else 'none'
        if any(r.get('refinement')!=expected_refinement for r in candidates):raise ValueError('wrong actual-model correction dispatch')
        with gzip.open(case/'engine.log.gz','rt') as stream:log=stream.read()
        corrections=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.splitlines() if line.startswith('p1_cublas_correction ')]
        if mode=='corrected':validate_corrections(corrections,geometry,POLICY)
        elif corrections:raise ValueError('unexpected correction work in comparator')
        arm['corrections']=corrections;state['arms'][label]=arm
        size=dump.stat().st_size
        if size!=16547840 or sum(p.stat().st_size for p in cache.parent.rglob('*.f32'))>256*1024**2:raise ValueError('raw logit ABI/cap mismatch')
        arrays[label]=np.fromfile(dump,dtype='<f4').reshape(32,-1)
        state['dumps'][label]=dict(bytes=size,sha256=q.p0.sha256(dump),path=str(dump))
        if capture:
            state['arms'][label]['capture']=direct_capture(case/'p1-indexer-capture',geometry['candidate_positions'][-1],mode)
            # Capture gets its own bounded artifact, leaving summaries and logs small.
            target=Path(os.environ['EVIDENCE']).absolute()/'captures'/name/mode
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(case/'p1-indexer-capture'),str(target))
        pairs=[('f32-0','f32-1')]+[(m+'-0',m+'-1') for m in ('cublas','corrected','volta')]
        pairs += [(f'f32-{i}',m+'-'+str(i)) for i in (0,1) for m in ('cublas','corrected','volta')]
        for a,b in pairs:
            if a in arrays and b in arrays and not any(r['reference']==a and r['candidate']==b for r in state['comparisons']):
                state['comparisons'].append(dict(reference=a,candidate=b,**comparison(arrays[a],arrays[b],geometry['score_start'])))
        checkpoint()
    if len(state['comparisons'])!=10:raise ValueError('incomplete quality comparison population')
    for mode in ('f32','cublas','corrected','volta'):
        if state['dumps'][mode+'-0']['sha256']!=state['dumps'][mode+'-1']['sha256']:raise ValueError('non-deterministic CPU0 repeat: '+mode)
    state['qualified_mean_kl_gate_pass']=all(r['mean_kl']<=.0005 for r in state['comparisons'] if not (r['reference'].startswith('f32') and r['candidate'].startswith('cublas')))
    state['original_cublas_mean_kl_gate_pass']=all(r['mean_kl']<=.0005 for r in state['comparisons'] if r['reference'].startswith('f32') and r['candidate'].startswith('cublas'))
    state['status']='complete diagnostic model comparison';checkpoint()
    if not state['qualified_mean_kl_gate_pass']:raise ValueError('corrected/MMA or F32-repeat KL gate failed')

def main():
    root=Path(os.environ['EVIDENCE']).absolute();out=root/'model';out.mkdir()
    run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not all(s.isdecimal() for s in run.split('-')):raise ValueError('invalid run identity')
    cache=Path('/actions-runner/abyss-p1-v9/indexer-head-to-head')/run;cache.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(cache).free<1024**3:raise RuntimeError('raw-evidence disk margin below1GiB')
    model=q.campaign.model_identity();frozen=json.loads((q.campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=frozen['model_sha256'] or frozen['pack']!=json.loads((q.PACK/'v100_pack_complete.json').read_text()):raise ValueError('frozen model/pack changed')
    binary=q.ROOT/'build-v100-p1-indexer-head-to-head/strata'
    for geometry in ('100k','128k'):run_geometry(binary,out/geometry,cache/geometry,geometry,model)

if __name__=='__main__':main()
