#!/usr/bin/env python3
"""Calibrate empirical GPU-guard candidates on captured cuBLAS/F32 score rows."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def probe(reference,candidate,factor):
    if reference.ndim!=1 or candidate.shape!=reference.shape or len(reference)<513 or not np.isfinite(reference).all() or not np.isfinite(candidate).all() or not np.isfinite(factor) or factor<=0:
        raise ValueError('expected finite same-sized score rows with at least 513 entries and positive envelope')
    n=len(reference);order=lambda x:np.lexsort((np.arange(n),-x))[:512]
    ref=order(reference);old=order(candidate)
    cut=candidate[old[-1]];margin=float(cut-np.partition(candidate,n-513)[n-513])
    envelope=float(np.max(np.abs(candidate)))*factor;flag=margin<=2*envelope
    band=candidate>=float(cut)-2*envelope
    results={}
    for policy in ('full-flagged','band-flagged','band-all'):
        mask=np.ones(n,dtype=bool) if policy=='full-flagged' else band.copy()
        if policy!='band-all' and not flag:mask[:]=False
        corrected=candidate.copy();corrected[mask]=reference[mask];ids=order(corrected)
        results[policy]=dict(scored_rows=int(mask.sum()),membership_equal=set(map(int,ids))==set(map(int,ref)),
            ordered_equal=bool(np.array_equal(ids,ref)))
    return dict(visible=n,envelope=envelope,maximum_error=float(np.max(np.abs(candidate-reference))),
        margin512_513=margin,flagged=bool(flag),
        original_membership_equal=set(map(int,old))==set(map(int,ref)),original_ordered_equal=bool(np.array_equal(old,ref)),
        corrections=results)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--output',required=True,type=Path);p.add_argument('--compact',action='store_true');a=p.parse_args()
    if hashlib.sha256(a.archive.read_bytes()).hexdigest()!=a.sha256:raise ValueError('capture digest mismatch')
    with zipfile.ZipFile(a.archive) as z:
        populations={}
        for name,factor in [('4eps',4*2**-23),('8eps',8*2**-23),('16eps',16*2**-23),('32eps',32*2**-23),('64eps',64*2**-23),('1e-4',1e-4)]:
            rows=[]
            for layer in range(2,43,2):
                for token in (0,255,256,1023,2047):
                    prefix=f'cublas/p1-indexer-capture/layer-{layer}/t-{token}-'
                    reference=np.frombuffer(z.read(prefix+'reference-score.f32'),dtype='<f4').astype(np.float64)
                    candidate=np.frombuffer(z.read(prefix+'candidate-score.f32'),dtype='<f4').astype(np.float64)
                    if len(reference)!=len(candidate) or not np.isfinite(reference).all() or not np.isfinite(candidate).all():raise ValueError('invalid score population')
                    rows.append(dict(layer=layer,token_offset=token,**probe(reference,candidate,factor)))
            populations[name]=dict(factor=factor,rows=rows,flagged=sum(r['flagged'] for r in rows),
                maximum_error_over_envelope=max(r['maximum_error']/max(r['envelope'],1e-300) for r in rows),
                unflagged_membership_errors=sum(not r['flagged'] and not r['original_membership_equal'] for r in rows),
                policies={policy:dict(ordered_errors=sum(not r['corrections'][policy]['ordered_equal'] for r in rows),
                    membership_errors=sum(not r['corrections'][policy]['membership_equal'] for r in rows),
                    scored_rows=sum(r['corrections'][policy]['scored_rows'] for r in rows)) for policy in ('full-flagged','band-flagged','band-all')})
    result=dict(schema=1,artifact_zip_sha256=a.sha256,samples=105,populations=populations,
        scope='Offline calibration on 105 late same-input cuBLAS/F32 score rows. Error envelopes are empirical '
        'max-absolute-score-relative guards, not certified bounds. Counterfactual replacement uses captured '
        'reference scores, not a tested GPU correction kernel or changed model trajectory. No model admission, '
        'KL subtraction or threshold relaxation; flags must be validated on additional inputs and in model tests.')
    if a.compact:
        for population in result['populations'].values():population.pop('rows')
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
