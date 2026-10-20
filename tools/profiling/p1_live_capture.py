#!/usr/bin/env python3
"""Bounded same-input score/selection/attention capture and long-context KL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools')]
import numpy as np
import p0
import v100_p0_campaign as campaign
from p0_accounting import fixture,compress
from model_baseline import fixed_env,normalize_logprobs,PACK
SAMPLES=[0,255,256,1023,2047]

def compare_capture(directory,target_position=98304):
    layers=sorted(directory.glob('layer-*'),key=lambda p:int(p.name.split('-')[1]))
    if [int(p.name.split('-')[1]) for p in layers]!=list(range(2,43,2)):raise ValueError('capture layer population changed')
    results=[];files={};full_selection=[]
    for layer in layers:
        meta=(layer/'meta.txt').read_text().splitlines()
        if meta[0]!='ABYSS_P1_LIVE_1':raise ValueError('capture ABI changed')
        il,pos,T,hd,ld=map(int,meta[1].split())
        if (pos,T,hd)!=(target_position,2048,32768) or list(map(int,meta[2].split()))!=SAMPLES:
            raise ValueError('capture geometry changed')
        if not (layer/'complete').exists() or (layer/'selection-restored').read_text()!='1048576 int32 values; bitwise verified\n':
            raise ValueError('capture or candidate restoration incomplete')
        full=layer/'full-selection.tsv'
        if full.exists():
            rows=[list(map(int,line.split())) for line in full.read_text().splitlines()[1:]]
            if len(rows)!=2048 or [r[0] for r in rows]!=list(range(2048)) or any(len(r)!=3 or r[1] not in (0,1) or not 0<=r[2]<=512 or (r[1] and r[2]!=512) for r in rows):
                raise ValueError('full selection population changed')
            full_selection.append(dict(layer=il,rows=2048,ordered_identical=sum(r[1] for r in rows),
                membership_changed=sum(r[2]!=512 for r in rows),minimum_overlap=min(r[2] for r in rows),
                membership_changes=[dict(token_offset=r[0],overlap=r[2]) for r in rows if r[2]!=512]))
        for t in SAMPLES:
            n=(pos+t+1)//4
            def array(kind,stage,dtype,count):
                path=layer/f't-{t}-{kind}-{stage}'
                value=np.fromfile(path,dtype=dtype)
                if len(value)!=count or not np.isfinite(value).all():raise ValueError('capture shape/nonfinite failure')
                return value
            c=array('candidate','score.f32','<f4',n).astype(np.float64)
            r=array('reference','score.f32','<f4',n).astype(np.float64)
            ci=array('candidate','ids.i32','<i4',512);ri=array('reference','ids.i32','<i4',512)
            for scores,ids in ((c,ci),(r,ri)):
                if len(set(map(int,ids)))!=512 or np.any(ids<0) or np.any(ids>=n):raise ValueError('selection range/duplicate/future failure')
                expected=np.lexsort((np.arange(n),-scores))[:512]
                if not np.array_equal(ids,expected):raise ValueError('unchanged selector ordering/tie failure')
            error=np.abs(c-r);max_error=float(error.max());rel=float(np.linalg.norm(c-r)/max(np.linalg.norm(r),1e-300))
            if rel>1e-5:raise ValueError('ordinary same-input scorer relative error exceeds component target')
            meaningful=int(np.count_nonzero(r[ci]+2*max_error<r[ri[-1]]))
            if meaningful:raise ValueError('selection disagreement outside measured score-error envelope')
            c_order=np.lexsort((np.arange(n),-c));r_order=np.lexsort((np.arange(n),-r))
            ca=array('candidate','attention.f32','<f4',hd).astype(np.float64)
            ra=array('reference','attention.f32','<f4',hd).astype(np.float64)
            ae=np.abs(ca-ra);attention_relative=float(np.linalg.norm(ca-ra)/max(np.linalg.norm(ra),1e-300))
            identical=bool(np.array_equal(ci,ri))
            if full.exists() and (rows[t][1]!=int(identical) or rows[t][2]!=len(set(map(int,ci))&set(map(int,ri)))):
                raise ValueError('full selection report disagrees with raw sampled IDs')
            if identical and not np.array_equal(ca,ra):raise ValueError('same selection gave different attention; investigate identical-input replay')
            results.append(dict(layer=il,token_offset=t,absolute_position=pos+t,visible=n,
                score_relative_l2=rel,score_max_abs=max_error,score_p99_abs=float(np.quantile(error,.99)),
                reference_margin512_513=float(r[r_order[511]]-r[r_order[512]]),
                candidate_margin512_513=float(c[c_order[511]]-c[c_order[512]]),
                overlap=int(len(set(map(int,ci))&set(map(int,ri)))),selected=512,
                meaningful_disagreements=meaningful,selection_identical=identical,
                attention_relative_l2=attention_relative,attention_max_abs=float(ae.max()),
                attention_p99_abs=float(np.quantile(ae,.99)),attention_above_1e4_investigation=attention_relative>1e-4))
        for path in layer.iterdir():
            files[str(path.relative_to(directory))]=dict(bytes=path.stat().st_size,sha256=p0.sha256(path))
    if full_selection and len(full_selection)!=21:raise ValueError('partial full-selection layer population')
    return dict(schema=1,samples=results,files=files,**(dict(full_selection=full_selection) if full_selection else {}),scope='Same live inputs, actual T=2048 cuBLAS path, '
        'unchanged F32 baseline and selectors; baseline attention before bitwise restoration of all candidate IDs. '
        'Direct attention differences are conditioned on selection/margin and reviewed alongside model KL. '
        'Diagnostic captures include extra kernels/synchronization and cannot measure throughput.')


def launch(binary,out,tokens,dump,mode,geometry,cpu_lane,target_position):
    out.mkdir();env=fixed_env();env.update(ABYSS_V100_IDX=mode,STRATA_DS4_ATTN='1',
        STRATA_GLM_SCORE=str(geometry['score_start']),STRATA_GLM_SCORE_DUMP=str(dump),STRATA_GLM_SCORE_PREFILL='1',
        STRATA_GLM_CPU_LANE=str(cpu_lane),ABYSS_V100_CAPTURE_P0=str(target_position))
    if mode=='f32':env['STRATA_DS4_IDX']='1'
    argv=[str(binary),'--glm-pack',str(PACK),'--tokens-file',str(tokens),'--max-new','1',
        '--max-context',str(geometry['max_context']),'--prefill','auto','--temperature','0']
    p0.write_json(out/'invocation.json',dict(argv=argv,cwd=str(out),environment={k:v for k,v in env.items() if k.startswith(('STRATA_','ABYSS_'))}))
    if campaign.memory_available()<16*1024**3:raise RuntimeError('host margin below 16 GiB')
    with campaign.Monitor(out) as monitor,(out/'engine.log').open('w') as log:
        child=subprocess.Popen(argv,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT);monitor.pid=child.pid
        deadline=time.monotonic()+1800
        try:
            while child.poll() is None:
                if campaign.memory_available()<8*1024**3:raise RuntimeError('own-child capture stop below 8 GiB')
                if time.monotonic()>deadline:raise TimeoutError('capture arm exceeded 1800 seconds')
                try:child.wait(timeout=.25)
                except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError('capture arm failed; inspect engine log')
        finally:
            if child.poll() is None:child.kill();child.wait()
    compress(out/'engine.log')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture-directory',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--geometry',choices=('100k','128k'),default='100k')
    ap.add_argument('--cpu-lane',type=int,choices=(0,28),default=28)
    ap.add_argument('--capture-position',type=int,default=98304)
    args=ap.parse_args()
    if args.capture_position<98304 or args.capture_position>120000 or args.capture_position%2048:
        ap.error('capture-position must be one eligible 2048-aligned chunk')
    from p1_integrated_quality import GEOMETRIES
    geometry=GEOMETRIES[args.geometry]
    if args.capture_position not in geometry['candidate_positions']:ap.error('capture position outside fixture geometry')
    if args.capture_directory:
        args.output.write_text(json.dumps(compare_capture(args.capture_directory,args.capture_position),indent=2)+'\n');return
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=True)
    model=campaign.model_identity();identity=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=identity['model_sha256'] or identity['pack']!=json.loads((PACK/'v100_pack_complete.json').read_text()):
        raise ValueError('frozen model/pack identity changed')
    binary=ROOT/'build-v100-p1-capture/strata';arrays={};dumps={}
    with tempfile.TemporaryDirectory(prefix='abyss-p1-capture-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tokens=Path(tmp)/'tokens';tokens.write_text(','.join(map(str,fixture('mixed-131072')[:geometry['token_count']])))
        for mode in ('f32','cublas'):
            dump=Path(tmp)/(mode+'.f32');launch(binary,out/mode,tokens,dump,mode,geometry,args.cpu_lane,args.capture_position)
            count=dump.stat().st_size
            if count<=0 or count%(32*4):raise ValueError('32-row logit ABI changed')
            arrays[mode]=normalize_logprobs(np.fromfile(dump,dtype='<f4').reshape(32,-1))
            dumps[mode]=dict(bytes=count,sha256=p0.sha256(dump))
        if (out/'f32/p1-indexer-capture').exists():raise ValueError('F32 arm unexpectedly captured a candidate')
        direct=compare_capture(out/'cublas/p1-indexer-capture',args.capture_position);p0.write_json(out/'direct.json',direct)
        ref,test=arrays['f32'],arrays['cublas'];kl=(np.exp(ref)*(ref-test)).sum(axis=1)
        quality=dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()),top1_matches=int((ref.argmax(1)==test.argmax(1)).sum()),
            positions=[dict(predicted_token_index_zero_based=geometry['score_start']+1+i,kl=float(k)) for i,k in enumerate(kl)])
        p0.write_json(out/'capture-result.json',dict(schema=1,source_sha=p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
            run=os.environ.get('GITHUB_RUN_ID'),binary_sha256=p0.sha256(binary),model_identity=model,
            fixture_sha256=p0.sha256(tokens),geometry=geometry,cpu_lane=args.cpu_lane,capture_position=args.capture_position,
            dumps=dumps,quality=quality,direct=direct))
        if not np.isfinite(kl).all() or quality['mean_kl']>.0005:raise ValueError('capture-build model KL gate failed')
        if any(x['attention_above_1e4_investigation'] for x in direct['samples']):
            raise ValueError('direct attention exceeds 1e-4 investigation threshold; review margins/KL before admission')
if __name__=='__main__':main()
