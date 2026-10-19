#!/usr/bin/env python3
"""Summarize a digest-verified isolated P1 artifact; never admits model dispatch."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v100'))
import p0


def records(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith('{')]


def summarize(archive, digest, run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:
        raise ValueError('P1 artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):
            return z.read(name).decode()
        test=records(read('test_dsv4_volta_indexer.jsonl'))
        if not test or test[-1]!={'suite':'isolated_volta_indexer','pass':True}:
            raise ValueError('isolated scorer suite incomplete')
        cases=[r for r in test if r.get('case')=='volta_indexer']
        expected={(w,n,0) for w in (4,8) for n in (1,7,31,129,512,513,1024,8192,31250)}
        expected|={(w,513,m) for w in (4,8) for m in (1,2,3,4)}
        if len(cases)!=len(expected) or {(r['warps'],r['visible'],r['mode']) for r in cases}!=expected:
            raise ValueError('scorer shape/mode population incomplete or duplicated')
        if not all(r['pass'] for r in cases):raise ValueError('scorer case failed')
        for target in ('test_mma884_fragment_contract','test_dsv4_volta_indexer','dsv4_fallback_parity'):
            for tool in ('memcheck','synccheck'):
                if 'ERROR SUMMARY: 0 errors' not in read(f'{target}-{tool}.log'):
                    raise ValueError('missing clean sanitizer result')
        ptx=read('indexer-sm70.ptx');sass=read('indexer-sm70.sass')
        if 'mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32' not in ptx or 'HMMA' not in sass:
            raise ValueError('missing emitted MMA884/HMMA evidence')
        rows=records(read('scorer-benchmark.jsonl'))
        if not rows or rows[-1]!={'suite':'isolated_volta_indexer','pass':True}:
            raise ValueError('benchmark suite incomplete')
        timings=[r for r in rows if r.get('case')=='isolated_timing']
        expected_t={(w,n,p) for w in (4,8) for n in (1,7,31,129,512,513,1024,8192,31250) for p in range(6)}
        if len(timings)!=len(expected_t) or {(r['warps'],r['visible'],r['pair']) for r in timings}!=expected_t:
            raise ValueError('six complete timing pairs per shape required')
        groups=[]
        for w,n in sorted({(r['warps'],r['visible']) for r in timings}):
            pair=sorted((r for r in timings if (r['warps'],r['visible'])==(w,n)),key=lambda r:r['pair'])
            if any(not math.isfinite(r[k]) or r[k]<=0 for r in pair for k in ('f32_us','volta_us','select_us')):
                raise ValueError('invalid timing')
            deltas=[100*(r['volta_us']/r['f32_us']-1) for r in pair]
            groups.append(dict(warps=w,visible=n,pairs=pair,score_elapsed_change_percent=deltas,
                median_score_elapsed_change_percent=statistics.median(deltas)))
        samples=[p0.xml_sample(x) for x in re.findall(r'<nvidia_smi_log>.*?</nvidia_smi_log>',read('scorer-telemetry.xml'),re.S)]
        return dict(schema=1,run=run,artifact_zip_sha256=digest,
            source_sha=read('source-sha.txt').strip(),component_cases=cases,timings=groups,
            sampled_telemetry=samples,thermal=p0.thermal_assessment(samples),
            memcheck_synccheck='zero errors for all three executables',
            scope='Isolated synthetic scorer and unchanged selector on selected-base branch; '
                  'event timings include launch overhead, exclude model query preparation, routing and attention. '
                  'No model KL, end-to-end performance, batched key reuse or production dispatch acceptance.',
            production_candidate_enabled=False)


def base_audit(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('base artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):return json.loads(z.read(name))
        allocation=read('working-fork-allocation-ledger.json')
        dispatch=read('dispatch-summary.json');routes=read('route-ledger.json')
        if not allocation['capture_complete'] or allocation['failures'] or allocation['ledger_errors']:
            raise ValueError('incomplete allocation capture')
        if dispatch['capture']['dropped_calls'] or dispatch['unresolved_decode_service'] or routes['capture']['dropped']:
            raise ValueError('incomplete route/dispatch capture')
        if read('working-fork/completion.json')!={'status':'pass','requests':4,'returncode':0}:
            raise ValueError('full-model process incomplete')
        totals={}
        for r in dispatch['totals']:
            if r['path'].startswith('mmq_input'):continue
            totals[r['path']]=totals.get(r['path'],0)+r['calls']
        quant=dict(dispatch['input_quantization'])
        quant['exact']=dict(quant['exact'])
        histogram=quant['exact'].pop('per_call_histogram')
        quant['exact']['per_call_histogram_cells']=len(histogram)
        quant['exact']['per_call_histogram_canonical_sha256']=hashlib.sha256(
            json.dumps(histogram,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return dict(schema=1,run=run,artifact_zip_sha256=digest,
            binary_sha256=dispatch['binary_sha256'],dispatch_capture=dispatch['capture'],
            unresolved_decode_service=dispatch['unresolved_decode_service'],dispatch_calls=totals,
            input_quantization=quant,route_capture=routes['capture'],
            route_ledger_json_sha256=hashlib.sha256(z.read('route-ledger.json')).hexdigest(),
            prefill_route_totals=routes['prefill_route_totals'],decode_route_totals=routes['decode_route_totals'],
            allocation_events=allocation['event_count'],allocation_peak=allocation['peak'],
            allocation_capture_complete=True,memory=read('working-fork/memory_summary.json'),
            thermal=read('working-fork/thermal.json'),model_identity=read('model-identity.json'),
            scope='Ported v1.3.31 branch, 600/8192 prompts, two requests each, 16 outputs. '
                  'Diagnostic allocation/dispatch/route audit only; perturbation excluded from throughput. '
                  'Context and reservations differ from immutable long-context memory budgets; no regression inferred. '
                  'Long-context P0b and branch quality revalidation remain separate.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('archive',type=Path);ap.add_argument('--sha256',required=True)
    ap.add_argument('--run',required=True,type=int);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();result=summarize(a.archive,a.sha256,a.run)
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
