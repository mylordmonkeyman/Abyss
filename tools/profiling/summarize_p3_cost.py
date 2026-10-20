#!/usr/bin/env python3
"""Digest-checked P3 observations, conditioned on work count and disk misses."""
import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


def distribution(values):
    values=sorted(values)
    if not values:return dict(samples=0)
    if any(not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid cost')
    return dict(samples=len(values),sum_ns=sum(values),median_ns=statistics.median(values),
        p95_ns=values[math.ceil(.95*len(values))-1],maximum_ns=max(values))


def population(rows):
    """Keep zero-work fetch overhead separate from actual expert fetches."""
    groups=collections.defaultdict(list)
    for row in rows:
        groups[(row['request_index'],row['cpu_experts'],row['fetch_experts'],bool(row['miss']))].append(row)
    result=[]
    for (request,cpu,fetch,miss),values in sorted(groups.items()):
        result.append(dict(request=request,cpu_experts=cpu,fetch_experts=fetch,disk_miss_mask_nonzero=miss,
            records=len(values),costs={name:distribution([r[name] for r in values if name in r and
                (name!='cpu_completed_wall_ns' or r['cpu_experts'])]) for name in
                ('cpu_completed_wall_ns','fetch_ns','cpu_wait_ns')}))
    return result


def layer_summary(rows):
    clean=[r for r in rows if not r['miss']]
    return dict(records=len(rows),miss_free_records=len(clean),
        cpu_expert_histogram=dict(sorted(collections.Counter(r['cpu_experts'] for r in clean).items())),
        cpu_completed_wall_ns=distribution([r['cpu_completed_wall_ns'] for r in clean if r['cpu_experts']]),
        active_fetch_ns=distribution([r['fetch_ns'] for r in clean if r['fetch_experts']]),
        cpu_wait_ns=distribution([r['cpu_wait_ns'] for r in clean if 'cpu_wait_ns' in r]))


def request_strata(rows,minimum_samples=8):
    """Descriptive matched layer/count strata, NOT same-expert paired trials."""
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in rows:
        if r['cpu_experts'] and not r['miss']:
            groups[(r['device'],r['layer'],r['cpu_experts'],r['fetch_experts'])][r['request_index']].append(r)
    result=[]
    for (device,layer,cpu,fetch),requests in sorted(groups.items()):
        if min(len(requests[0]),len(requests[1]))<minimum_samples:continue
        costs={}
        for name in ('cpu_completed_wall_ns','fetch_ns','cpu_wait_ns'):
            values=[[r[name] for r in requests[q] if name in r] for q in (0,1)]
            if min(map(len,values))<minimum_samples:continue
            medians=[statistics.median(v) for v in values]
            if medians[0]<=0:raise ValueError('nonpositive stratum baseline')
            costs[name]=dict(samples=list(map(len,values)),medians_ns=medians,
                request1_vs_request0_change_pct=100*(medians[1]/medians[0]-1))
        result.append(dict(device=device,layer=layer,cpu_experts=cpu,fetch_experts=fetch,costs=costs))
    return result


def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('cost artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('cost artifact cap exceeded')
        def read(name):return json.loads(z.read(name))
        cost=read('costs.json');raw=read('raw-identity.json');invocation=read('invocation.json')
        source=z.read('source-sha.txt').decode().strip()
        binary=z.read('binary-sha256.txt').decode().split()[0]
        if raw['source_sha']!=source or raw['binary_sha256']!=binary or invocation['binary_sha256']!=binary:
            raise ValueError('source/binary identities disagree')
        if invocation['fixture_sha256']!=raw['fixture_sha256']:raise ValueError('fixture identities disagree')
        end=gzip.decompress(z.read('engine.log.gz')).decode()
        import re
        counts=re.findall(r'V100_COST_END ranges=(\d+) dropped=(\d+)',end)
        if len(counts)!=1 or int(counts[0][1]):raise ValueError('observer dropped or incomplete')
        rows=cost['rows'];requests=read('requests.json')
        context=invocation.get('context_tokens',8192)
        protocol=invocation.get('cache_protocol','excluded 16-output warm-up then two warmed requests')
        if raw.get('cache_protocol',protocol)!=protocol:raise ValueError('cache protocol identities disagree')
        warmup=read('warmup.json')
        if protocol=='first post-load request versus subsequent request; OS/file caches uncontrolled':
            if 'done' not in warmup or warmup.get('disposition')!=protocol:
                raise ValueError('cache protocol missing explicit first-request disposition')
            if warmup.get('done') is not None:raise ValueError('cache protocol unexpectedly warmed first request')
        elif protocol=='excluded 16-output warm-up then two warmed requests':
            if (warmup.get('done') or '').split()[:3]!=['DONE','16',str(context)]:
                raise ValueError('cache protocol missing excluded warm-up')
        else:raise ValueError('unsupported cache protocol')
        if len(requests)!=2 or any(r['request']!=i or r['done'].split()[:3]!=['DONE','32',str(context)]
            for i,r in enumerate(requests)):raise ValueError('unexpected request population')
        expected={(layer,token) for layer in range(43) for token in range(64)}
        if len(rows)!=len(expected) or {(r['layer'],r['token_index']) for r in rows}!=expected:
            raise ValueError('incomplete model layer-token population')
        if any(r['device']!=0 or r['request_index']!=r['token_index']//32 or
            r['cpu_experts']!=r['cpu'].bit_count() or r['fetch_experts']!=r['fetch'].bit_count() or
            r['cpu']&r['fetch'] for r in rows):raise ValueError('invalid row identity/masks')
        cpu=sum(bool(r['cpu_experts']) for r in rows)
        if cost['tokens']!=64 or cost['completed_requests']!=len(rows) or cost['cpu_completion_records']!=cpu:
            raise ValueError('cost population totals disagree')
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=source,binary_sha256=binary,
            raw_identity=raw,fixture_sha256=raw['fixture_sha256'],context_tokens=context,requests=requests,
            cache_protocol=protocol,
            completed_layer_tokens=len(rows),cpu_completion_records=cpu,
            observer_ranges_including_excluded_warmup=int(counts[0][0]),observer_dropped=0,
            nonzero_miss_mask_records=sum(bool(r['miss']) for r in rows),
            conditioned_costs=population(rows),per_layer={str(layer):layer_summary([r for r in rows if r['layer']==layer])
                for layer in range(43)},request_strata_minimum_samples=8,
            same_layer_count_request_strata=request_strata(rows),memory=read('memory_summary.json'),thermal=read('thermal.json'),
            scope='Profiled observations under the declared cache protocol, not known-cold OS caches; first prompt-final forward included in each request. '
            'CPU completed-call wall, mapped-read fetch kernel and exposed CPU wait overlap: do not add them. '
            'Zero-fetch overhead separated; nonzero miss mask is not pure RAM transfer. '
            'Different work counts/routes are not paired allocation experiments. No fitted estimator, physical bandwidth, '
            'clean throughput or production admission. Counters unavailable/QPI unmeasured; '
            'local-manifest model verification does not establish publisher authenticity.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('archive',type=Path)
    ap.add_argument('--sha256',required=True);ap.add_argument('--run',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.write_text(json.dumps(summarize(args.archive,args.sha256,args.run),indent=2)+'\n')
