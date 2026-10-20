#!/usr/bin/env python3
"""Recompute bounded observer-control evidence from a digest-checked artifact."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import zipfile
from p3_observer_control import contrasts,ORDER
from v100_base_replay import parse_lp


def ledger(mode,text):
    found=re.findall(r'V100_COST_END ranges=(\d+) dropped=(\d+)',text)
    if mode=='plain':
        if found:raise ValueError('compiled-out arm has observer ledger')
    elif len(found)!=1 or not 0<int(found[0][0])<=32768 or int(found[0][1]):
        raise ValueError('missing/truncated observer ledger')
    return [list(row) for row in found]


def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('control artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('control artifact cap exceeded')
        def read(name):return json.loads(z.read(name))
        data=read('control.json');arms=data['arms'];recomputed=contrasts(arms)
        if data['context_tokens']!=8192 or data['scored_tokens']!=32:raise ValueError('control geometry mismatch')
        if data['status']!='complete; overhead screening only' or data['order']!=list(ORDER):
            raise ValueError('incomplete control collection')
        if recomputed!=data['contrasts']:raise ValueError('control contrast mismatch')
        if data['source_sha']!=z.read('source-sha.txt').decode().strip():raise ValueError('control source identity mismatch')
        for name,manifest in (('plain','control'),('observer','cost')):
            if data['binary_sha256'][name]!=z.read(f'binary-{manifest}-sha256.txt').decode().split()[0]:
                raise ValueError('control binary identity mismatch')
        result=[];environment=None
        for i,arm in enumerate(arms):
            mode=arm['mode'];prefix=f'{i}-{mode}/';inv=read(prefix+'invocation.json')
            expected=data['binary_sha256']['plain' if mode=='plain' else 'observer']
            if inv['mode']!=mode or inv['binary_sha256']!=expected:raise ValueError('arm binary/mode mismatch')
            if environment is None:environment=inv['environment']
            if inv['environment']!=environment:raise ValueError('control environments differ')
            if environment.get('STRATA_DS4_IDX')!='1' or environment.get('STRATA_DS4_ATTN')!='1':
                raise ValueError('control does not force F32')
            parsed=[parse_lp(line,32) for line in z.read(prefix+'wire.log').decode().splitlines()
                if line.startswith('LP ') and line.split()[2]=='32']
            if len(parsed)!=4 or any(any(p[k]!=row[k] for k in p) for p,row in zip(parsed,arm['requests'])):
                raise ValueError('control wire replies disagree')
            observed=ledger(mode,gzip.decompress(z.read(prefix+'engine.log.gz')).decode())
            if observed!=arm['observer_ledger']:raise ValueError('control ledger mismatch')
            if read(prefix+'requests.json')!=arm['requests'] or read(prefix+'result.json')!=arm:
                raise ValueError('arm checkpoint mismatch')
            item=dict(index=i,mode=mode,binary_sha256=expected,observer_ledger=observed,
                requests=[{k:v for k,v in row.items() if not k.startswith('host_')} for row in arm['requests']],
                memory=read(prefix+'memory_summary.json'),thermal=read(prefix+'thermal.json'))
            if mode=='trace':
                commands=read(prefix+'profiler-commands.json')
                for action in ('start','stop'):
                    matching=[r for r in commands if r['argv'][1]==action]
                    if len(matching)!=1 or matching[0]['returncode']:raise ValueError('incomplete profiler session')
                raw=arm['raw_report']
                if not 0<raw['bytes']<=2*1024**3 or raw['uploaded'] or not re.fullmatch('[0-9a-f]{64}',raw['sha256']):
                    raise ValueError('invalid retained trace identity')
                item['raw_report']=raw
            result.append(item)
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=data['source_sha'],
            binary_sha256=data['binary_sha256'],source_fixture=data['source_fixture'],
            model_identity=data['model_identity'],
            pack_identity_sha256=hashlib.sha256(json.dumps(data['pack'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            source_fixture_sha256=data['source_fixture_sha256'],fixed_token_sha256=data['fixed_token_sha256'],
            context_tokens=data['context_tokens'],scored_tokens=data['scored_tokens'],environment=environment,
            arms=result,contrasts=recomputed,scope=data['scope'],performance_claim_admitted=False,
            observer_contrasts_change_sign=min(r['observer_vs_plain_pct'] for r in recomputed)<0<max(r['observer_vs_plain_pct'] for r in recomputed),
            trace_slower_in_both_order_halves=all(r['trace_vs_observer_pct']>0 for r in recomputed),
            disposition='These descriptive fixed-token LOGP controls neither isolate causal overhead nor admit adaptive allocation.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path)
    ap.add_argument('--sha256',required=True);ap.add_argument('--run',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(summarize(args.archive,args.sha256,args.run),indent=2)+'\n')
