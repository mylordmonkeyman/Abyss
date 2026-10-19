#!/usr/bin/env python3
"""Summarize completed P0 arms and active-window telemetry without causal claims."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics


def numbers(text):
    return {k:int(v) for k,v in re.findall(r'^(\w+):?\s+(\d+)\s*$',text or '',re.M)}


def numeric(value):
    m=re.search(r'-?\d+(?:\.\d+)?',str(value))
    return float(m[0]) if m else None


def span(values):
    v=[x for x in values if x is not None]
    return dict(min=min(v),median=statistics.median(v),max=max(v),samples=len(v)) if v else None


def active_samples(samples,rows):
    return [s for s in samples if any(r['begin_monotonic_s']<=s['monotonic_s']<=r['end_monotonic_s'] for r in rows)]


def phase_windows(rows, phase):
    result=[]
    for r in rows:
        if r.get('ttft_ms') is None:continue
        first=r['begin_monotonic_s']+r['ttft_ms']/1000
        if not r['begin_monotonic_s']<=first<=r['end_monotonic_s']:
            raise ValueError('first token timestamp outside request')
        lo,hi=(r['begin_monotonic_s'],first) if phase=='prefill_to_first_token' else (first,r['end_monotonic_s'])
        result.append(dict(begin_monotonic_s=lo,end_monotonic_s=hi))
    return result


def clock_samples(active):
    return dict(samples=len(active),
        sm_clock_mhz=span(numeric(s.get('gpu',{}).get('sm_clock')) for s in active),
        memory_clock_mhz=span(numeric(s.get('gpu',{}).get('memory_clock')) for s in active),
        power_w=span(numeric(s.get('gpu',{}).get('power_draw')) for s in active),
        temperature_c=span(numeric(s.get('gpu',{}).get('temperature')) for s in active),
        thermal_reason_samples=sum(s.get('gpu',{}).get('thermal_flag') is True for s in active),
        power_cap_samples=sum(s.get('gpu',{}).get('power_cap_active') is True for s in active))


def summarize(root,phase):
    arms=[];grouped=defaultdict(list)
    for directory in sorted(root.glob(phase+'-[0-9]*-*')):
        if not (directory/'completion.json').is_file():continue
        index=int(directory.name.split('-')[-2]);base=directory.name.split('-')[-1]
        rows=json.loads((directory/'requests.json').read_text())
        path=directory/'telemetry.jsonl.gz'
        with gzip.open(path,'rt') as f:samples=[json.loads(line) for line in f]
        active=active_samples(samples,rows)
        telemetry=dict(active_samples=len(active),sample_interval_s=10,
            sm_clock_mhz=span(numeric(s.get('gpu',{}).get('sm_clock')) for s in active),
            memory_clock_mhz=span(numeric(s.get('gpu',{}).get('memory_clock')) for s in active),
            power_w=span(numeric(s.get('gpu',{}).get('power_draw')) for s in active),
            temperature_c=span(numeric(s.get('gpu',{}).get('temperature')) for s in active),
            thermal_reason_samples=sum(s.get('gpu',{}).get('thermal_flag') is True for s in active),
            power_cap_samples=sum(s.get('gpu',{}).get('power_cap_active') is True for s in active),
            min_host_memavailable_bytes=min((s['host_memory_bytes']['MemAvailable'] for s in active),default=None),
            numa_mapped_bytes=[s.get('numa_mapped_bytes') for s in active],
            interpretation='Samples restricted to measured request wall windows, excluding startup. '
            'Ten-second sampling can miss short phases. No unsampled clock/thermal guarantee.')
        telemetry['warm_wall_phases']={phase:clock_samples(active_samples(samples,phase_windows(
            [r for r in rows if r['request']==1],phase)))
            for phase in ('prefill_to_first_token','decode_after_first_token')}
        telemetry['phase_boundary_scope']='Host first-token receipt, not CUDA event timing; excludes first request warmup.'
        cases=[]
        for r in rows:
            before=r.get('host_before',{});after=r.get('host_after',{})
            io0=numbers(before.get('io'));io1=numbers(after.get('io'))
            vm0=numbers(before.get('vmstat'));vm1=numbers(after.get('vmstat'))
            cases.append(dict(case=r['case'],request=r['request'],output_tokens=r['output_tokens'],
                prefill_tokens_s=r['prefill_tokens_s'],decode_tokens_s=r['decode_tokens_s'],ttft_ms=r['ttft_ms'],
                active_samples=len(active_samples(samples,[r])),
                process_io_delta={k:io1[k]-io0[k] for k in io0.keys()&io1.keys()},
                global_vm_delta={k:vm1[k]-vm0[k] for k in ('pgmajfault','pgpgin','pgpgout','pswpin','pswpout')
                                 if k in vm0 and k in vm1}))
            if r['request']==1:grouped[(r['case'],r['output_tokens'],base)].append(r)
        arms.append(dict(index=index,base=base,telemetry=telemetry,cases=cases))
    aggregate=[]
    for (case,count,base),rows in sorted(grouped.items()):
        aggregate.append(dict(case=case,output_tokens=count,base=base,warm_repeats=len(rows),
            prefill_tokens_s=span(r['prefill_tokens_s'] for r in rows),
            decode_tokens_s=span(r['decode_tokens_s'] for r in rows),
            ttft_ms=span(r['ttft_ms'] for r in rows)))
    return dict(schema=1,phase=phase,complete_arms=arms,warm_summary=aggregate,
        interpretation='Request 0 is excluded from warm summaries. Process order and output trajectories matter. '
        'Clock/power/NUMA distributions are observations, not proof of matched conditions or causal speedup.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);ap.add_argument('--phase',required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(summarize(args.root,args.phase),indent=2)+'\n')

if __name__=='__main__':main()
