#!/usr/bin/env python3
"""Separate P0 memory/dispatch diagnostic; never contributes throughput samples."""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools'),str(ROOT)]
import v100_p0_campaign as campaign
from allocation_ledger import reconstruct
from dispatch_ledger import parse as parse_dispatch
from route_ledger import parse as parse_routes
from memory_worksheet import worksheet,kb_fields
from model_baseline import fixed_env
from summarize_p0_dispatch import summarize as summarize_dispatch
import p0


class AccountingMonitor(campaign.Monitor):
    def snapshot(self):
        s=super().snapshot()
        if self.pid:
            try:
                mem=Path('/proc/meminfo').read_text()
                smaps=Path(f'/proc/{self.pid}/smaps_rollup').read_text()
                s['memory_worksheet']=worksheet(kb_fields(mem),kb_fields(smaps))
                s['raw_meminfo']=mem;s['raw_smaps_rollup']=smaps
                s['process_maps']=Path(f'/proc/{self.pid}/maps').read_text()
                s['process_numa_maps']=Path(f'/proc/{self.pid}/numa_maps').read_text()
            except (OSError,ProcessLookupError):pass # process can exit between reads
        return s


def fixture(name):
    path=campaign.CAMPAIGN/'fixtures'/(name+'.tokens')
    identity=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if p0.sha256(path)!=identity['fixtures'][path.name]['sha256']:
        raise ValueError('frozen fixture identity changed')
    return list(map(int,path.read_text().strip().split(',')))


def compress(path):
    with path.open('rb') as src,gzip.open(str(path)+'.gz','wb') as dst:
        import shutil
        shutil.copyfileobj(src,dst)
    path.unlink()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['memory','dispatch'],required=True)
    args=ap.parse_args();os.environ['P0_ACCOUNTING_PHASE']=args.phase
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=False)
    p0.write_json(out/'model-identity.json',campaign.model_identity())
    observer=out/'cuda_alloc_observer.so'
    subprocess.run(['c++','-shared','-fPIC','-std=c++17','-O2','-pthread',
        str(ROOT/'tools/v100/cuda_alloc_observer.cpp'),'-ldl','-o',str(observer)],check=True,timeout=60)
    p0.write_json(out/'observer.json',dict(source_sha256=p0.sha256(ROOT/'tools/v100/cuda_alloc_observer.cpp'),
        binary_sha256=p0.sha256(observer),scope='opt-in diagnostic, timing excluded; dynamic CUDA runtime only'))
    # No need to distribute the observer binary; the source and build identity suffice.
    campaign.Monitor=AccountingMonitor
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    if args.phase=='memory':
        binaries=campaign.BINARIES
        cases=[(n,fixture(n),16) for n in ('mixed-32768','mixed-131072')]
        context=135168
    else:
        binaries={'working-fork':ROOT/'build-v100-diagnostics/strata'}
        cases=[(n,fixture(n),16) for n in ('prose-600','mixed-8192')]
        context=12288
        env.update(ABYSS_V100_DISPATCH='1',ABYSS_V100_PROFILE='1')
    report=dict(scope='accounting/profile diagnostic; no throughput acceptance',phase=args.phase,
        model_identity_sha256=p0.sha256(out/'model-identity.json'),arms=[])
    p0.write_json(out/'host-build-metadata.json',dict(lscpu=p0.command(['lscpu']),
        compiler=p0.command(['c++','--version']),
        thp={k:campaign.read('/sys/kernel/mm/transparent_hugepage/'+k) for k in ('enabled','defrag')},
        affinity=sorted(os.sched_getaffinity(0)),manifest=p0.manifest(os.environ.get('ABYSS_GPU','0'))))
    for base,binary in binaries.items():
        if campaign.memory_available()<16*1024**3:raise RuntimeError('host memory safety margin below 16 GiB')
        binary_hash=p0.sha256(binary)
        if base in campaign.EXPECTED and binary_hash!=campaign.EXPECTED[base]:
            raise ValueError('immutable upstream binary changed')
        symbols=p0.command(['readelf','--dyn-syms','--wide',str(binary)])
        (out/(base+'-dynamic-symbols.txt')).write_text(symbols.get('stdout','')+symbols.get('stderr',''))
        if symbols['returncode'] or not re.search(r'\bUND\s+cudaMalloc(?:@|\s|$)',symbols.get('stdout','')):
            raise RuntimeError('cudaMalloc is not dynamically imported; interposition coverage unverified')
        directory=out/base;allocation_log=out/(base+'-allocations.jsonl')
        child_env=dict(env,LD_PRELOAD=str(observer),ABYSS_CUDA_ALLOC_LOG=str(allocation_log))
        rows=campaign.arm(binary,directory,cases,child_env,context,'auto')
        ledger=reconstruct((json.loads(line) for line in allocation_log.read_text().splitlines()),os.sysconf('SC_PAGE_SIZE'))
        p0.write_json(out/(base+'-allocation-ledger.json'),ledger)
        if not ledger['capture_complete'] or ledger['ledger_errors']:
            raise ValueError('allocation capture incomplete or inconsistent; preserve before investigation')
        if args.phase=='dispatch':
            dispatch=parse_dispatch((directory/'engine.log').read_text(errors='replace'))
            p0.write_json(out/'dispatch-ledger.json',dispatch)
            if dispatch['unresolved_decode_service']:raise ValueError('unresolved decode dispatch positions')
            summary=summarize_dispatch(dispatch)
            if summary['input_quantization']['exact'] is None:
                raise ValueError('exact MMQ input capture missing; preserve diagnostic evidence')
            summary['source_run']=os.environ.get('GITHUB_RUN_ID')
            summary['binary_sha256']=binary_hash
            p0.write_json(out/'dispatch-summary.json',summary)
            routes=parse_routes((directory/'engine.log').read_text(errors='replace'),dispatch)
            p0.write_json(out/'route-ledger.json',routes)
        report['arms'].append(dict(base=base,binary_sha256=binary_hash,requests=rows,
            allocation_events=ledger['event_count'],allocation_peak=ledger['peak']))
        p0.write_json(out/'accounting.json',report)
        compress(allocation_log);compress(directory/'engine.log')
    observer.unlink()
    report['status']='collection complete; physical reconciliation, ISA review and decisions still required'
    p0.write_json(out/'accounting.json',report)


def cap():
    out=Path(os.environ['EVIDENCE']).absolute()
    phase=os.environ.get('P0_ACCOUNTING_PHASE')
    if not out.exists() or not phase:return
    # Hash but never upload binary tools. Also remove one left by a failed arm.
    observer=out/'cuda_alloc_observer.so'
    if observer.exists():observer.unlink()
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>64*1024*1024:raise ValueError('accounting evidence exceeds 64 MiB; upload refused, retained on runner')
    p0.write_json(out/'upload-cap.json',dict(bytes_before_manifest=total,limit_bytes=64*1024*1024))
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P0_ACCOUNTING_'+os.environ['P0_ACCOUNTING_PHASE'].upper()+'_CAP=true\n')

if __name__=='__main__':
    try:main()
    finally:cap()
