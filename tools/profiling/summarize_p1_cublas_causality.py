#!/usr/bin/env python3
"""Digest-check the cuBLAS selection intervention; no performance admission."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from p1_cublas_corrected_bench import require_control

def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('control ZIP digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('control artifact too large')
        state=json.loads(z.read('control/control.json'));require_control(state)
        if state['dumps']['reference-order']['sha256']!=state['dumps']['f32-0']['sha256'] or state['comparisons'][2]['mean_kl']<=.0005:
            raise ValueError('this causality disposition requires bitwise reference-order restoration and failed membership-only quality')
        if z.read('source-sha.txt').decode().strip()!=state['source_sha']:raise ValueError('source mismatch')
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in z.read(tool+'.log').decode():raise ValueError('sanitizer incomplete')
        cases=[json.loads(line) for line in z.read('parity.jsonl').decode().splitlines() if line.startswith('{')]
        expected={(p,m,b) for p in (2040,2052,8192) for m in (0,1) for b in (False,True)}
        if cases[-1]!={'suite':'cublas_selection_control','cases':12,'pass':True}:raise ValueError('intervention suite incomplete')
        cases=cases[:-1]
        if len(cases)!=12 or {(r['p0'],r['mode'],r['membership_only']) for r in cases}!=expected or any(not r['pass'] for r in cases):
            raise ValueError('intervention component cases incomplete')
        totals={}
        for label,arm in state['arms'].items():
            env=json.loads(z.read('control/'+label+'/invocation.json'))['environment']
            if env['STRATA_GLM_CPU_LANE']!='0' or env['ABYSS_V100_IDX']!=('f32' if label.startswith('f32') else 'cublas'):
                raise ValueError('wrong model arm configuration')
            records=arm['selection_controls']
            if records:
                totals[label]={k:sum(int(r[k]) for r in records) for k in ('rows','ordered_equal_rows','membership_equal_rows','overlap')}
                if len(records)!=231:raise ValueError('incomplete selection intervention')
        return dict(schema=1,run=run,source_sha=state['source_sha'],artifact_zip_sha256=digest,
            comparisons=state['comparisons'],dumps=state['dumps'],selection_totals=totals,
            component_cases=12,sanitizers='memcheck and synccheck pass',
            disposition='F32 membership and ordering restore bitwise logits/zero KL; membership alone fails the unchanged mean-KL limit. '
            'Ordering is material on this deterministic CPU0/full128K trajectory. Counters describe intervened trajectories, not the original standalone candidate. '
            'No timing claim or production admission; correction must preserve ordering as well as membership. Local manifest integrity is not publisher authenticity; counters unavailable/QPI unmeasured.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',required=True,type=int);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
