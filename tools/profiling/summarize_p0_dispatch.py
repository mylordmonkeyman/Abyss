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
    exact_calls = []; layer_distinct = []; gather_calls = 0; layer_calls = 0
    for r in ledger['host_records']:
        key=(r['phase'],r['type'],r['nt'] if r['nt']<=8 else '9+',r['path'])
        for metric in ('calls','rows','logical_weight_bytes'):
            totals[key][metric]+=r[metric]
        if r['phase']==1 and r['path']=='mmq_quantize:q8_1_gather':
            quantized+=r['rows']
            gather_calls+=r['calls']
        if r['phase']==1 and r['path'].startswith('mmq_input_distinct:'):
            distinct=int(r['path'].split(':')[1])
            if not 0<distinct<=r['nt'] or r['rows']!=distinct*r['calls']:
                raise ValueError('invalid distinct-row histogram')
            exact_calls.append(dict(layer=r['layer'],position=r['position'],type=r['type'],
                                    chunk_tokens=r['chunk_tokens'],input_rows=r['nt'],
                                    distinct_rows=distinct,calls=r['calls']))
        if r['phase']==1 and r['path']=='mmq_input_layer_distinct':
            if not 0<=r['rows']<=r['chunk_tokens']*r['calls']:
                raise ValueError('invalid layer distinct-row union')
            layer_distinct.append(r)
        if r['phase']==1 and r['path'] in ('mmq_glu_selected','mmq_glu_not_selected'):
            # Identical chunk anchors from repeated requests aggregate in one
            # record. Count every invocation; never dedupe the requests here.
            distinct_upper+=r['chunk_tokens']*r['calls']
            layer_calls+=r['calls']
    exact=None
    if exact_calls or layer_distinct:
        if (sum(r['calls'] for r in exact_calls)!=gather_calls or
            sum(r['input_rows']*r['calls'] for r in exact_calls)!=quantized or
            sum(r['calls'] for r in layer_distinct)!=layer_calls):
            raise ValueError('incomplete exact input-quantization coverage')
        per_call=sum(r['distinct_rows']*r['calls'] for r in exact_calls)
        per_layer=sum(r['rows'] for r in layer_distinct)
        if not per_layer<=per_call<=quantized:
            raise ValueError('distinct-row union exceeds per-call count')
        exact=dict(per_call_histogram=exact_calls,distinct_rows_summed_per_call=per_call,
                   distinct_rows_summed_per_layer_invocation=per_layer,
                   repeated_rows_within_calls=quantized-per_call,
                   repeated_rows_across_calls=per_call-per_layer,
                   total_repeated_input_rows=quantized-per_layer,
                   method='Actual sorted GPU gather IDs copied to the host in a diagnostic run; '
                          'each layer invocation has a separate union. Histogram preserves '
                          'per-call (input rows, distinct rows) counts; no chronology inferred.')
    return dict(schema=2,
        capture=ledger['summary'],unresolved_decode_service=ledger['unresolved_decode_service'],
        by_layer_type_token_group=ledger['by_layer_type_token_group'],
        totals=[dict(phase=k[0],type=k[1],nt=k[2],path=k[3],**v)
                for k,v in sorted(totals.items(),key=lambda item:tuple(map(str,item[0])))],
        input_quantization=dict(actual_routed_input_rows=quantized,
            exact=exact,
            distinct_layer_chunk_input_rows_upper_bound=distinct_upper,
            repeated_input_rows_lower_bound=max(0,quantized-distinct_upper),
            method='Each full-prompt layer/chunk invocation has at most chunk_tokens distinct input rows. '
            'Account for repeated requests via the GLU-decision call count. Routed MMQ input rows above '
            'this bound prove repeated quantization. See exact when gather IDs were captured.'),
        scope='Working-fork host dispatch capture, 600/8192 prompts, two requests each, 16 outputs. '
              'Phase 0 setup/calibration; phase 1 prefill; phase 2 decode. T2-8 here are expert prefill '
              'batches, not an enabled speculative verifier. No timings/DRAM bandwidth inferred from counters.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('ledger',type=Path)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(summarize(json.loads(args.ledger.read_text())),indent=2)+'\n')


if __name__=='__main__':main()
