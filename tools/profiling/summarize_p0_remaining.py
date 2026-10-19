#!/usr/bin/env python3
"""Reproduce remaining-matrix summaries from digest-verified checkpoint ZIPs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tempfile

from summarize_p0_campaign import summarize,span


def paired(arms):
    groups={}
    for arm in arms:
        row=arm['requests'][1]
        groups.setdefault(arm['index']//2,{})[arm['base']]=row
    if set(groups)!={0,1,2} or any(set(v)!={'historical','current'} for v in groups.values()):
        raise ValueError('three complete base pairs required')
    rows=[]
    for pair,group in sorted(groups.items()):
        a,b=group['historical'],group['current']
        if (a['case'],a['output_tokens'])!=(b['case'],b['output_tokens']):
            raise ValueError('paired shapes differ')
        rows.append(dict(pair=pair,case=a['case'],output_tokens=a['output_tokens'],
            **{k+'_change_percent':100*(b[k]/a[k]-1) for k in ('prefill_ms','decode_ms','ttft_ms')},
            generated_ids_equal=a['output_ids_sha256']==b['output_ids_sha256']))
    return rows


def collect(root,artifacts):
    groups={};traces=[]
    for artifact in artifacts:
        unit=artifact['unit'];folder=root/f'u{unit}';archive=root/f'u{unit}.zip'
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=artifact['zip_sha256']:
            raise ValueError('artifact ZIP digest mismatch')
        if unit in (21,22):
            traces.append(dict(**artifact,timeline=json.loads((folder/'timeline.json').read_text()),
                               trace_identity=json.loads((folder/'trace-identity.json').read_text())))
            continue
        report=json.loads((folder/'unit.json').read_text());task=report['task']
        if report['unit']!=unit:raise ValueError('checkpoint unit mismatch')
        key=(task['kind'],task['length'],task['prefill'])
        group=groups.setdefault(key,dict(arms=[],artifacts=[],directories=[]))
        for arm in report['arms']:
            rows=arm['requests']
            if len(rows)!=2 or [r['request'] for r in rows]!=[0,1] or any(
                r['prompt_tokens']!=task['length'] or r['output_tokens']!=task['output'] for r in rows):
                raise ValueError('request count or exact token shape mismatch')
            name=f"{task['kind']}-{task['length']}-{task['prefill']}-{arm['index']}-{arm['base']}"
            directory=folder/name
            completion=json.loads((directory/'completion.json').read_text())
            if completion!=dict(status='pass',requests=2,returncode=0):
                raise ValueError('incomplete or unsuccessful process')
            if rows!=json.loads((directory/'requests.json').read_text()):
                raise ValueError('checkpoint requests disagree with process record')
            group['arms'].append(arm);group['directories'].append(directory)
        group['artifacts'].append(artifact)
    results=[]
    for (kind,length,prefill),group in sorted(groups.items()):
        arms=group['arms']
        if sorted(a['index'] for a in arms)!=list(range(6)):
            raise ValueError('six distinct process arms required')
        chronology=sorted(arms,key=lambda a:a['requests'][0]['begin_monotonic_s'])
        for a,b in zip(chronology,chronology[1:]):
            if a['requests'][-1]['end_monotonic_s']>b['requests'][0]['begin_monotonic_s']:
                raise ValueError('overlapping or incomparable process windows')
        with tempfile.TemporaryDirectory(prefix='abyss-p0-summary-') as temp:
            merged=Path(temp)
            for directory in group['directories']:shutil.copytree(directory,merged/directory.name)
            summary=summarize(merged,kind)
        for arm in summary['complete_arms']:
            samples=arm['telemetry'].pop('numa_mapped_bytes')
            nodes=sorted({node for sample in samples if sample for node in sample})
            arm['telemetry']['numa_mapped_bytes_by_node']={node:span(
                sample.get(node) for sample in samples if sample) for node in nodes}
        comparisons=paired(arms)
        summary.update(length=length,prefill=prefill,artifacts=group['artifacts'],
            actual_process_order=[a['index'] for a in chronology],paired=comparisons,
            paired_median_percent={k:statistics.median(p[k] for p in comparisons)
                                   for k in comparisons[0] if k.endswith('_percent')})
        results.append(summary)
    return dict(schema=1,source_run=39194497860,matrix=results,traces=traces,
        disposition='Digest-verified completed checkpoints; request zero excluded from warm summaries. '
                    'Each context starts a fresh process; separate from interrupted carry-over campaign. '
                    'Generated trajectories and sampled clocks/power/NUMA can differ. '
                    'Timing changes are observations, not causal base or chunk speedups. '
                    'Timeline kernel residence is a main-stream wall-lane proxy, not a complete causal DAG.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path)
    ap.add_argument('--artifacts',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args()
    args.output.write_text(json.dumps(collect(args.root,json.loads(args.artifacts.read_text())),indent=2)+'\n')


if __name__=='__main__':main()
