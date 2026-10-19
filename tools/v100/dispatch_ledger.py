#!/usr/bin/env python3
"""Parse bounded P0 host-call counters and join decode service to issue positions."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse(text):
    rows=[];end=[]
    for line in text.splitlines():
        if line.startswith('V100_DISPATCH '):rows.append(json.loads(line[len('V100_DISPATCH '):]))
        elif line.startswith('V100_DISPATCH_END '):end.append(json.loads(line[len('V100_DISPATCH_END '):]))
    if len(end)!=1 or end[0]['dropped_calls'] or end[0]['keys']!=len(rows):
        raise ValueError('missing/duplicate terminal summary, dropped calls or truncated records')
    issues={}
    for r in rows:
        if r['path']=='request_issue':
            key=(r['layer'],r['request_seq'])
            if key in issues or r['calls']!=1:raise ValueError('ambiguous request issue; single-GPU capture required')
            issues[key]=r['position']
    unknown=[];totals=defaultdict(lambda:dict(calls=0,rows=0,logical_weight_bytes=0))
    for r in rows:
        if r['phase']==2 and r['path'].startswith('cpu_'):
            key=(r['layer'],r['request_seq'])
            r['position']=issues.get(key,-1)
            if r['position']<0:unknown.append(key)
        nt=r['nt']; group='T1' if nt==1 else 'T2_8' if 2<=nt<=8 else 'T9_plus' if nt>=9 else 'not_applicable'
        key=(r['phase'],r['layer'],r['type'],group,r['path'])
        for metric in ('calls','rows','logical_weight_bytes'):
            if r[metric]<0:raise ValueError('negative dispatch counter')
            totals[key][metric]+=r[metric]
    return dict(schema=1,host_records=rows,summary=end[0],
                unresolved_decode_service=sorted(set(unknown)),
                by_layer_type_token_group=[dict(phase=k[0],layer=k[1],type=k[2],token_group=k[3],path=k[4],**v)
                                           for k,v in sorted(totals.items())],
                interpretation='Host dispatch calls only. Prefill position is the chunk anchor, not '
                'the individual gathered expert row token. CPU nt is actual expert token count. '
                'Decode service position is joined by layer and request sequence. '
                'Logical matrix bytes are not measured DRAM or PCIe traffic. '
                'Setup/calibration is phase 0; prefill 1; decode 2. Single GPU only. '
                'Timing from a diagnostic run is excluded from throughput comparisons.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('log',type=Path)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    result=parse(args.log.read_text(errors='replace'))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if result['unresolved_decode_service']:raise SystemExit('Unresolved service positions; inspect ledger')

if __name__=='__main__':main()
