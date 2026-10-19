#!/usr/bin/env python3
"""Read-only recovery evidence for an interrupted P0 campaign; never launches an engine."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v100'))
import p0


def main():
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=False)
    cache=Path('/actions-runner/abyss-p0-v9/campaign-1')
    records=[];total=0
    for path in sorted(cache.rglob('*')):
        if not path.is_file() or path.suffix=='.f32':continue
        rel=path.relative_to(cache)
        # Already-uploaded quality/extended arms need not be copied again.
        if len(rel.parts)>1 and not rel.parts[0].startswith(('long-','chunk-')):continue
        if path.suffix not in ('.json','.gz','.log'):continue
        data=path.read_bytes() if path.stat().st_size<=4*1024**2 else None
        if data is None and path.suffix=='.log':
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-256*1024));data=f.read()
        if data is None or total+len(data)>24*1024**2:
            records.append(dict(path=str(rel),bytes=path.stat().st_size,copied=False));continue
        target=out/'checkpoint'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        total+=len(data);records.append(dict(path=str(rel),bytes=path.stat().st_size,copied=True,
            partial_tail=len(data)!=path.stat().st_size,sha256_of_copy=p0.sha256(target)))
    p0.write_json(out/'checkpoint-inventory.json',records)
    processes=[]
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():continue
        try:
            exe=str((d/'exe').resolve(strict=True))
            if Path(exe).name!='strata':continue
            processes.append(dict(pid=int(d.name),executable=exe,
                cmdline=(d/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'),
                status=(d/'status').read_text(),stat=(d/'stat').read_text(),
                io=(d/'io').read_text(),smaps_rollup=(d/'smaps_rollup').read_text()))
        except OSError:continue
    p0.write_json(out/'engine-processes.json',processes)
    system={}
    for path in ['/proc/meminfo','/proc/vmstat','/proc/loadavg','/proc/pressure/memory',
                 '/sys/fs/cgroup/memory.events','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.stat']:
        try:system[path]=Path(path).read_text()
        except OSError as e:system[path]=dict(unavailable=str(e))
    p0.write_json(out/'system.json',system)
    p0.write_json(out/'gpu.json',p0.command(['nvidia-smi','-q','-x'],timeout=30))
    p0.write_json(out/'device-processes.json',p0.command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],timeout=30))
    p0.write_json(out/'scope.json',dict(utc=p0.now(),action='read-only inspection; no engine launch, kill, reset or retry',
        checkpoint_bytes=total,remaining='Review preserved partial arms and system evidence before deciding recovery.'))
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>32*1024**2:
        raise ValueError('inspection upload exceeds cap')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P0_INSPECT_CAP=true\n')

if __name__=='__main__':main()
