#!/usr/bin/env python3
"""Reconstruct observed live CUDA API ranges; never equate these with physical RSS.

Only synchronous APIs used in the current engine source are observed. Driver,
context, module, vendor-internal allocations and noninterposable calls are outside
coverage. Page union is a virtual-page footprint, not measured resident pages.
"""
import argparse
import json
from pathlib import Path


def union_bytes(ranges, page_size=1):
    spans=sorted((a//page_size*page_size, (a+n+page_size-1)//page_size*page_size)
                 for a,n in ranges if n)
    total=0; end=0
    for a,b in spans:
        total+=max(0,b-max(a,end)); end=max(end,b)
    return total


def reconstruct(events, page_size=4096):
    if page_size<=0 or page_size & (page_size-1):
        raise ValueError('page_size must be a positive power of two')
    live={'device':{},'host_allocated':{},'host_registered':{}}
    aliases={}; peak={'device':0,'host_unique_virtual':0,'host_virtual_page_footprint':0}
    snapshots=[]; failures=[]; errors=[]; pid=None; expected=1; ended=False; allocations=0
    alloc={'cudaMalloc':'device','cudaMallocHost':'host_allocated',
           'cudaHostAlloc':'host_allocated','cudaHostRegister':'host_registered'}
    free={'cudaFree':'device','cudaFreeHost':'host_allocated','cudaHostUnregister':'host_registered'}
    for e in events:
        if pid is None: pid=e['pid']
        if e['pid']!=pid or e['seq']!=expected:
            raise ValueError('mixed processes, appended sessions or missing/reordered events')
        expected+=1
        api=e['api']; ptr=e['ptr']
        if ended: raise ValueError('events after observer_end')
        if api=='observer_end': ended=True; continue
        if e['result']!=0:
            failures.append(e); continue
        if api in alloc:
            allocations+=1
            space=live[alloc[api]]
            if ptr in space: errors.append(dict(seq=e['seq'],error='duplicate live allocation',api=api,ptr=ptr))
            space[ptr]=e['bytes']
        elif api in free:
            space=live[free[api]]
            if ptr not in space and ptr:
                errors.append(dict(seq=e['seq'],error='unmatched free/unregister',api=api,ptr=ptr))
            space.pop(ptr,None)
            if api!='cudaFree': aliases.pop(ptr,None)
        elif api=='cudaHostGetDevicePointer': aliases[ptr]=e['alias']
        else: raise ValueError('unsupported observer event '+api)
        host=list(live['host_allocated'].items())+list(live['host_registered'].items())
        now=dict(device=sum(live['device'].values()),host_unique_virtual=union_bytes(host),
                 host_virtual_page_footprint=union_bytes(host,page_size))
        for k,v in now.items(): peak[k]=max(peak[k],v)
        snapshots.append(dict(seq=e['seq'],monotonic_ns=e['monotonic_ns'],**now))
    if not allocations: raise ValueError('no events: CUDA interposition coverage is unverified')
    return dict(schema=1,pid=pid,event_count=expected-1,capture_complete=ended,page_size=page_size,peak=peak,
                live_ranges={k:[dict(address=a,bytes=n) for a,n in v.items()] for k,v in live.items()},
                device_aliases=[dict(host=a,device=b) for a,b in aliases.items()],
                failures=failures,ledger_errors=errors,snapshots=snapshots,
                interpretation='Requested live API bytes and virtual page union only; not physical residency. '
                'Device aliases add zero host bytes. Not a complete CUDA/host physical budget. '
                'Validate dynamic symbol bindings and compare device bytes with contemporaneous NVML. '
                'Profile timing is excluded from throughput comparisons.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('events',type=Path)
    ap.add_argument('--page-size',type=int,default=4096);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result=reconstruct((json.loads(line) for line in args.events.read_text().splitlines()),args.page_size)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['ledger_errors']: raise SystemExit('Invalid allocation lifecycle; inspect ledger_errors')

if __name__=='__main__':main()
