#!/usr/bin/env python3
"""Offline cutoff-rescore feasibility on captured rows; no model validation."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from summarize_p1_mma_model import summarize


def replay(reference,candidate):
    n=len(reference);order=lambda x:np.lexsort((np.arange(n),-x))[:512]
    selected=order(candidate);cut=candidate[selected[-1]]
    band=float(np.max(np.abs(candidate)))*1e-4
    mask=candidate>=float(cut)-band
    revised=candidate.copy();revised[mask]=reference[mask]
    return dict(visible=n,refined_rows=int(mask.sum()),band=band,
        maximum_observed_error=float(np.max(np.abs(candidate-reference))),
        baseline_top512_covered=bool(mask[order(reference)].all()),
        refined_order_identical=bool(np.array_equal(order(revised),order(reference))),
        original_order_identical=bool(np.array_equal(selected,order(reference))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',required=True,type=int)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=summarize(a.archive,a.sha256,a.run,diagnostic_failure=True)
    with zipfile.ZipFile(a.archive) as z:
        rows=[]
        for sample in result['direct']['samples']:
            prefix=f"model/volta-0/p1-indexer-capture/layer-{sample['layer']}/t-{sample['token_offset']}-"
            ref=np.frombuffer(z.read(prefix+'reference-score.f32'),dtype='<f4').astype(np.float64)
            candidate=np.frombuffer(z.read(prefix+'candidate-score.f32'),dtype='<f4').astype(np.float64)
            rows.append(dict(layer=sample['layer'],token_offset=sample['token_offset'],**replay(ref,candidate)))
    # Keep the numerical evidence, not megabytes of duplicated dispatch strings.
    for arm in result['quality']['arms'].values():
        records=arm.pop('dispatch_records');arm['dispatch_records_count']=len(records)
        arm['dispatch_records_canonical_sha256']=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    files=result['direct'].pop('files')
    result['direct']['file_ledger_canonical_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    result['cutoff_replay']=dict(relative_band=1e-4,samples=rows,
        all_sampled_reference_selections_covered=all(r['baseline_top512_covered'] for r in rows),
        all_sampled_refined_orders_identical=all(r['refined_order_identical'] for r in rows),
        scope='Counterfactual replacement with captured original F32 scores on 105 same-input rows. '
        'The max-absolute-score-relative band is an empirical research guard, not a certified universal '
        'error bound. Unobserved rows, changed model trajectories, actual rescore arithmetic and speed '
        'are not validated by this replay. No KL adjustment or threshold waiver.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
