#!/usr/bin/env python3
"""Summarize v9 base artifacts without conflating startup, warm or memory views."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


def summarize(root):
    p0 = root / 'p0'
    campaign = json.loads((p0/'bases.json').read_text())
    summary = {k:v for k,v in campaign.items() if k!='arms'}
    summary['paired_warm'] = []
    pairs = {}
    memory = []
    arm_summaries=[]
    for arm in campaign['arms']:
        index, name = arm['index'], arm['base']
        directory=p0/f'{index}-{name}'
        if arm['status']!='pass':
            arm_summaries.append(arm)
            continue
        requests=arm['requests']
        log=(directory/'engine.log').read_text(errors='replace')
        mem=json.loads((directory/'memory_summary.json').read_text())
        thermal=json.loads((directory/'thermal.json').read_text())
        startup=json.loads((directory/'startup.json').read_text())
        used=[];samples=[]
        for line in (directory/'telemetry.jsonl').read_text().splitlines():
            s=json.loads(line);samples.append(s)
            if s.get('gpu_xml'):
                try:
                    val=ET.fromstring(s['gpu_xml']).findtext('gpu/fb_memory_usage/used')
                    if val and val.endswith(' MiB'): used.append(float(val[:-4])*1024**2)
                except ET.ParseError:
                    pass
        # Explicit engine tier allocations are distinct from RSS/cgroup totals.
        tier=re.search(r'RAM tier ([\d.]+) GB pinned',log)
        pool=re.search(r'expert pool ([\d.]+) GB',log)
        dense=re.search(r'dense weights ([\d.]+)',log)
        item={'index':index,'base':name,'startup':startup,'memory':mem,'thermal':thermal,
              'peak_sampled_gpu_used_bytes':max(used,default=None),
              'engine_ram_tier_gib_rounded':float(tier[1]) if tier else None,
              'engine_expert_pool_gib_rounded':float(pool[1]) if pool else None,
              'engine_dense_gib_rounded':float(dense[1]) if dense else None,
              'startup_and_route_log':[s for s in log.splitlines() if any(t in s for t in
                  ('RAM tier','CPU lane:','expert pool','glm stat:','MADV_HUGEPAGE','chunks of'))],
              'warm_requests':[{k:v for k,v in r.items() if k!='output_ids'}
                               for r in requests if r['request']==1]}
        arm_summaries.append(item)
        for r in requests:
            if r['request']==1: pairs.setdefault((index//2,r['case']),{})[name]=r
        # Every worksheet category exists even when instrumentation is partial.
        measurements=[
            ('model_disk','see tensor_inventory.csv and fixed shard sizes; not physical residency',None,'source metadata'),
            ('gpu_total_used', 'CUPTI not required; sampled NVML used includes overlapping allocations',max(used,default=None),'sampled'),
            ('gpu_dense', 'rounded engine allocation; subcomponent of GPU total',float(dense[1])*1024**3 if dense else None,'engine log'),
            ('gpu_expert_pool','rounded engine allocation; subcomponent of GPU total',float(pool[1])*1024**3 if pool else None,'engine log'),
            ('host_expert_pinned','rounded engine tier; aliases overlap process and cgroup accounting',float(tier[1])*1024**3 if tier else None,'engine log'),
            ('host_other_pinned','token embedding, staging and mailbox allocation ledger incomplete',None,'unmeasured remainder'),
            ('process_rss','physical resident view; includes pinned/model/heap/file pages',mem['peak_rss_bytes'],'sampled'),
            ('process_pss','proportional resident view; overlaps RSS',mem['peak_pss_bytes'],'sampled'),
            ('host_unpinned_unique','requires allocator/page-type separation from registered pages',None,'not separately measured'),
            ('file_page_cache_working_set','host and cgroup file/cache samples available; unique model working set unknown',None,'partial'),
            ('os_docker_other_workloads','host/cgroup pressure sampled; unique OS/other-workload attribution unknown',None,'partial'),
            ('free_safety_floor','minimum sampled system MemAvailable, not container limit minus RSS',mem['minimum_host_memavailable_bytes'],'sampled'),
            ('socket0_socket1','ready_placement.json and per-sample NUMA maps; QPI traffic unmeasured',None,'placement measured; traffic unavailable'),
            ('thp','AnonHugePages is a subcomponent of resident anonymous pages',mem['peak_anon_huge_bytes'],'sampled'),
            ('ssd_fetch','per-process /proc/io and faults plus engine route stats; no system cache drop',None,'raw samples retained'),
            ('unique_memory_budget_total','not computed by adding overlapping RSS, pinned, cache and cgroup views',None,'incomplete unique component ledger'),
        ]
        for category,note,value,evidence in measurements:
            memory.append(dict(base=name,arm=index,category=category,bytes=value,
                               decimal_gb=value/1e9 if value is not None else None,
                               binary_gib=value/1024**3 if value is not None else None,
                               evidence=evidence,note=note))
    for (pair,case), arms in sorted(pairs.items()):
        if set(arms)!= {'historical','current'}: continue
        old,new=arms['historical'],arms['current']
        row={'pair':pair,'case':case}
        for field in ('prefill_ms','decode_ms','ttft_ms','wire_wall_s'):
            row['historical_'+field]=old[field];row['current_'+field]=new[field]
            row[field+'_change_percent']=100*(new[field]/old[field]-1)
        row['generated_ids_equal']=old['output_ids_sha256']==new['output_ids_sha256']
        summary['paired_warm'].append(row)
    summary['warm_medians']=[]
    cases=sorted({r['case'] for r in summary['paired_warm']})
    for case in cases:
        rows=[r for r in summary['paired_warm'] if r['case']==case]
        summary['warm_medians'].append({'case':case,'paired_samples':len(rows),**{
            field:statistics.median(r[field] for r in rows)
            for field in ('prefill_ms_change_percent','decode_ms_change_percent','ttft_ms_change_percent')}})
    summary['arms']=arm_summaries
    summary['interpretation']=('Negative paired latency change favors current. Three process pairs, initial screen only; '
                              'no base selection from medians alone. Greedy-string equality is diagnostic, '
                              'not a substitute for fixed-token logits and existing repeat variability.')
    return summary,memory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact_directory',type=Path)
    parser.add_argument('output_directory',type=Path)
    a=parser.parse_args()
    a.output_directory.mkdir(parents=True,exist_ok=True)
    summary,memory=summarize(a.artifact_directory)
    (a.output_directory/'bases-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (a.output_directory/'memory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['base','arm','category','bytes','decimal_gb','binary_gib','evidence','note'])
        w.writeheader();w.writerows(memory)
    print(json.dumps({'status':summary['status'],'warm_medians':summary['warm_medians'],
                      'teacher_forced':summary.get('teacher_forced'),'selected_base':summary['selected_base']}))


if __name__=='__main__': main()
