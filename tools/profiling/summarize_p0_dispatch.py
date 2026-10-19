#!/usr/bin/env python3
"""Compact actual-branch evidence; keep repeat-quantization bounds explicit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(ledger):
    if ledger['summary']['dropped_calls'] or ledger['unresolved_decode_service']:
        raise ValueError('incomplete dispatch evidence')
    totals = defaultdict(lambda: dict(calls=0,rows=0,logical_weight_bytes=0))
    quantized = 0
    distinct_upper = 0
    for r in ledger['host_records']:
        key=(r['phase'],r['type'],r['nt'] if r['nt']<=8 else '9+',r['path'])
        for metric in ('calls','rows','logical_weight_bytes'):
            totals[key][metric]+=r[metric]
        if r['phase']==1 and r['path']=='mmq_quantize:q8_1_gather':
            quantized+=r['rows']
        if r['phase']==1 and r['path'] in ('mmq_glu_selected','mmq_glu_not_selected'):
            # Identical chunk anchors from repeated requests aggregate in one
            # record. Count every invocation; never dedupe the requests here.
            distinct_upper+=r['chunk_tokens']*r['calls']
    return dict(schema=1,source_run=39194149121,artifact_id=12162360960,
        artifact_zip_sha256='37bb947a3d96b5dff59328796bb713d5ff02292896e3c78d28ea442a268df926',
        capture=ledger['summary'],unresolved_decode_service=ledger['unresolved_decode_service'],
        by_layer_type_token_group=ledger['by_layer_type_token_group'],
        totals=[dict(phase=k[0],type=k[1],nt=k[2],path=k[3],**v)
                for k,v in sorted(totals.items(),key=lambda item:tuple(map(str,item[0])))],
        input_quantization=dict(actual_routed_input_rows=quantized,
            distinct_layer_chunk_input_rows_upper_bound=distinct_upper,
            repeated_input_rows_lower_bound=max(0,quantized-distinct_upper),
            method='Each full-prompt layer/chunk invocation has at most chunk_tokens distinct input rows. '
            'Account for repeated requests via the GLU-decision call count. Routed MMQ input rows above '
            'this bound prove repeated quantization. Exact per-call/per-token distinct counts were not captured.'),
        scope='Working-fork host dispatch capture, 600/8192 prompts, two requests each, 16 outputs. '
              'Phase 0 setup/calibration; phase 1 prefill; phase 2 decode. T2-8 here are expert prefill '
              'batches, not an enabled speculative verifier. No timings/DRAM bandwidth inferred from counters.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('ledger',type=Path)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(summarize(json.loads(args.ledger.read_text())),indent=2)+'\n')


if __name__=='__main__':main()
