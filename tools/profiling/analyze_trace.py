#!/usr/bin/env python3
"""Source-tagged timing ledger from a bounded Nsight SQLite export (not hardware counters)."""
import argparse
import collections
import json
from pathlib import Path
import sqlite3


def union_ns(intervals, lo, hi):
    clipped = sorted((max(a,lo),min(b,hi)) for a,b in intervals if b>lo and a<hi)
    total, end = 0, lo
    for a,b in clipped:
        if b>end:
            total += b-max(a,end)
            end=b
    return total


def group(name):
    if 'mul_mat_q' in name: return 'quantized routed-expert MMQ'
    if 'moe_cpu_wait' in name: return 'CPU expert dependency wait on GPU'
    if 'moe_fetch' in name: return 'RAM-tier fetch kernel'
    if 'moe_gate_up' in name or 'moe_down' in name: return 'token-path routed-expert arithmetic'
    if 'idx_score' in name or 'idx_select' in name: return 'indexer scoring/selection'
    if 'attend_' in name: return 'DeepSeek attention'
    if 'mv_kernel' in name: return 'dense/shared/projection GEMV (roles combined)'
    if 'gemm' in name or 'cutlass::' in name: return 'dense/shared/projection GEMM (roles combined)'
    return 'other kernels'


def analyze(database, done, prompt_tokens=2048):
    db=sqlite3.connect(database)
    db.row_factory=sqlite3.Row
    kernels=[dict(r) for r in db.execute('SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                                       'JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
    copies=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start')]
    maxima=[k for k in kernels if '::argmax_kernel(' in k['name']]
    fields=done.split()
    if not maxima or len(maxima)!=int(fields[1]):
        raise ValueError('expected one argmax per token-path forward, including final prompt token')
    split=maxima[0]['end']
    end=maxima[-1]['end']
    prefill_start=max(0,split-round(float(fields[3])*1e6))
    drift_ms=abs((end-split)/1e6-float(fields[4]))
    if drift_ms>1.0:
        raise ValueError('argmax boundary does not match reported decode phase within 1 ms')
    report={'scope':f'one warmed {prompt_tokens}-token prefill + {int(fields[1])} generated tokens; tracing perturbs timings',
            'timing_kind':'CUPTI activity timestamps; no SM/HBM/L2 hardware-counter claim',
            'phase_boundary':'end of first argmax (final prompt forward); cross-checked against DONE decode time',
            'decode_phase_alignment_error_ms':drift_ms, 'done':done,
            'qsa_kernel_launches':sum('qsa_' in k['name'] for k in kernels),
            'mxfp4_down_token_path_launches':sum('moe_down_kernel<(int)39>' in k['name'] for k in kernels),
            'mxfp4_mmq_launches':sum('mul_mat_q<(ggml_type)39,' in k['name'] for k in kernels),
            'fetch_kernel_launches':dict(collections.Counter(k['name'] for k in kernels if 'moe_fetch' in k['name'])),
            'f32_indexer_symbols':sorted({k['name'] for k in kernels if 'idx_score' in k['name']}),
            'f32_attention_symbols':sorted({k['name'] for k in kernels if 'attend_' in k['name']}),
            'phases':{}}
    main_stream=maxima[0].get('streamId')
    if main_stream is None or any(k.get('streamId')!=main_stream for k in maxima):
        raise ValueError('argmax main compute stream is missing or ambiguous')
    report['main_compute_stream_id']=main_stream
    for phase,lo,hi in [('prefill',prefill_start,split),('decode',split,end)]:
        categories=collections.Counter(); names=collections.defaultdict(lambda:{'calls':0,'ns':0,'registers_per_thread':set(),'blocks':set()})
        for k in kernels:
            duration=max(0,min(k['end'],hi)-max(k['start'],lo))
            if not duration: continue
            categories[group(k['name'])]+=duration
            row=names[k['name']];row['calls']+=1;row['ns']+=duration
            row['registers_per_thread'].add(k['registersPerThread'])
            row['blocks'].add((k['blockX'],k['blockY'],k['blockZ']))
        busy=union_ns([(k['start'],k['end']) for k in kernels+copies],lo,hi)
        compute_busy=union_ns([(k['start'],k['end']) for k in kernels],lo,hi)
        summed=sum(categories.values())
        # A CUDA stream's activity is ordered; this is its measured wall-lane
        # budget, not a causal DAG reconstruction from summed GPU kernels.
        lane=[k for k in kernels if k.get('streamId')==main_stream and k['end']>lo and k['start']<hi]
        lane_categories=collections.Counter()
        lane_sum=0
        for k in lane:
            duration=max(0,min(k['end'],hi)-max(k['start'],lo))
            lane_categories[group(k['name'])]+=duration;lane_sum+=duration
        lane_union=union_ns([(k['start'],k['end']) for k in lane],lo,hi)
        if lane_sum-lane_union>1000:
            raise ValueError('main-stream intervals overlap: cannot form a disjoint lane budget')
        ranked=[dict(name=n,wall_lane_ms=v/1e6,phase_wall_percent=100*v/max(1,hi-lo))
                for n,v in lane_categories.most_common()]
        critical_path_proxy=dict(main_stream_id=main_stream,
            measured_main_stream_kernel_union_ms=lane_union/1e6,
            unattributed_main_stream_gap_ms=(hi-lo-lane_union)/1e6,
            contributors=ranked,top_three=ranked[:3],
            interpretation='Serialized main compute-stream kernel residence and unassigned gaps. '
            'Not a complete causal CPU/CUDA dependency graph. Copy streams and CPU service may overlap. '
            'Wait kernel residence measures an exposed device dependency, not standalone CPU work. '
            'Unassigned gaps require API/CPU/I/O correlation before attributing a cause.')
        report['phases'][phase]={'begin_trace_ns':lo,'end_trace_ns':hi,'wall_ms':(hi-lo)/1e6,
          'critical_path_proxy':critical_path_proxy,
          'kernel_sum_ms':summed/1e6,'compute_active_union_ms':compute_busy/1e6,
          'compute_or_explicit_copy_active_union_ms':busy/1e6,
          'no_traced_kernel_or_explicit_copy_ms':(hi-lo-busy)/1e6,
          'categories':[{'name':n,'kernel_ms':v/1e6,'kernel_time_percent':100*v/max(1,summed)} for n,v in categories.most_common()],
          'top_kernel_symbols':[{'name':n,'calls':v['calls'],'kernel_ms':v['ns']/1e6,
                                 'registers_per_thread':sorted(v['registers_per_thread']),
                                 'block_shapes':sorted(v['blocks'])}
                                for n,v in sorted(names.items(),key=lambda nv:nv[1]['ns'],reverse=True)[:15]],
          'interpretation':'Kernel sums can overlap; idle intervals are not assigned to CPU, I/O or launch causes without correlation. '
                           'GPU zero-copy reads are not explicit memcpy activities. Static launch resources are not achieved occupancy/spill counters.'}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('database');p.add_argument('capture_windows');p.add_argument('output')
    p.add_argument('--prompt-tokens',type=int,default=2048)
    a=p.parse_args()
    done=json.loads(Path(a.capture_windows).read_text())[0]['done']
    Path(a.output).write_text(json.dumps(analyze(a.database,done,a.prompt_tokens),indent=2)+'\n')
