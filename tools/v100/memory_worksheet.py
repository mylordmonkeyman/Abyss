#!/usr/bin/env python3
"""Disjoint host meminfo worksheet plus explicitly overlapping process views.

Consumes one contemporaneous sample, not independently selected metric peaks.
Unassigned bytes remain a residual rather than an invented allocation category.
"""
import argparse
import json
from pathlib import Path


def kb_fields(text):
    result={}
    for line in text.splitlines():
        key,sep,value=line.partition(':')
        fields=value.split()
        if sep and len(fields)==2 and fields[1]=='kB':result[key]=int(fields[0])*1024
    return result


def worksheet(mem, process=None):
    required=('MemTotal','MemFree','AnonPages','Shmem','Cached','Buffers','Slab','KernelStack','PageTables')
    missing=set(required)-mem.keys()
    if missing:raise ValueError('missing meminfo fields: '+','.join(sorted(missing)))
    if mem['Cached']<mem['Shmem']:raise ValueError('Cached smaller than Shmem: inconsistent sample')
    parts=dict(free=mem['MemFree'],anonymous=mem['AnonPages'],shared_tmpfs=mem['Shmem'],
               file_cache_excluding_shmem=mem['Cached']-mem['Shmem'],buffers=mem['Buffers'],
               slab=mem['Slab'],kernel_stacks=mem['KernelStack'],page_tables=mem['PageTables'])
    # Do not add Active/Inactive, Unevictable, Mlocked, THP, SReclaimable or
    # MemAvailable: they overlap categories above. HugeTLB is left in residual.
    residual=mem['MemTotal']-sum(parts.values())
    if residual<0:raise ValueError('negative residual: sample cannot form this worksheet')
    parts['unassigned_kernel_driver_reserved_and_other']=residual
    return dict(schema=1,host_disjoint_bytes=parts,host_total_bytes=mem['MemTotal'],
                headroom_bytes=mem.get('MemAvailable'),
                swap=dict(total_bytes=mem.get('SwapTotal'),free_bytes=mem.get('SwapFree')),
                overlapping_host_views={k:mem[k] for k in ('Mlocked','Unevictable','AnonHugePages',
                    'ShmemHugePages','SReclaimable','Active','Inactive') if k in mem},
                overlapping_process_view_bytes=process or {},
                interpretation='Host categories sum to MemTotal by an explicit unassigned residual. '
                'Process RSS/PSS/Anonymous/file mappings, pinned CUDA ranges, THP and MemAvailable '
                'are overlapping views; never add them to these host categories. '
                'CUDA registration is not measured by Mlocked/VmLck. '
                'A registered virtual-page union does not prove physical residency or uniqueness '
                'across different virtual aliases. RSS includes shared file cache. '
                'This worksheet alone does not attribute all physical bytes to the application.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    # Reads have a small temporal skew; preserve both raw inputs and timestamps.
    import time
    start=time.monotonic_ns();mem=Path('/proc/meminfo').read_text()
    proc=Path(f'/proc/{args.pid}/smaps_rollup').read_text();end=time.monotonic_ns()
    result=worksheet(kb_fields(mem),kb_fields(proc))
    result.update(start_monotonic_ns=start,end_monotonic_ns=end,raw_meminfo=mem,raw_smaps_rollup=proc)
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
