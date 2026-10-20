#!/usr/bin/env python3
"""Join bounded NVTX request identities to completed CUPTI fetch/wait kernels."""
import bisect
import collections
import json
import re
import sqlite3
import statistics

def fields(label):return dict(re.findall(r'(\w+)=([^ ]+)',label))
def key(row):return tuple(int(row[k]) for k in ('device','layer','seq'))

def analyze_rows(annotations,kernels,runtime,expected_tokens=64):
    routes={};cpu={};scopes=collections.defaultdict(list)
    for event in annotations:
        label=event.get('label') or ''
        if label.startswith('P3_ROUTE '):
            row=fields(label);identity=key(row)
            if identity in routes:raise ValueError('duplicate request metadata')
            routes[identity]=row
        elif label.startswith('P3_COST '):
            row=fields(label);identity=key(row)
            if event.get('end') is None or event['end']<=event['start']:raise ValueError('uncompleted cost range')
            if row['kind']=='cpu':
                if identity in cpu:raise ValueError('duplicate completed CPU call')
                cpu[identity]=dict(ns=event['end']-event['start'],experts=int(row['experts']))
            elif row['kind'] in ('fetch','cpu_wait'):
                scopes[(event['globalTid'],row['kind'])].append(dict(event,identity=identity))
            else:raise ValueError('unsupported/deferred cost scope')
    if not routes:raise ValueError('no route metadata')
    starts={}
    for ident,values in scopes.items():
        values.sort(key=lambda e:e['start']);starts[ident]=[e['start'] for e in values]
    maxima=sorted(k['end'] for k in kernels if '::argmax_kernel(' in k['name'])
    if len(maxima)!=expected_tokens:raise ValueError('unexpected completed token population')
    apis=collections.defaultdict(list)
    for r in runtime:apis[r['correlationId']].append(r)
    durations=collections.defaultdict(lambda:collections.defaultdict(list));tokens={}
    for k in kernels:
        kind='fetch' if any(s in k['name'] for s in ('moe_fetch_kernel','moe_fetch_stream_kernel')) else 'cpu_wait' if 'moe_cpu_wait_kernel' in k['name'] else None
        if kind is None:continue
        matches=[]
        for api in apis[k['correlationId']]:
            ident=(api['globalTid'],kind);index=bisect.bisect_right(starts.get(ident,[]),api['start'])-1
            if index>=0:
                scope=scopes[ident][index]
                if api['end']<=scope['end']:matches.append(scope['identity'])
        if len(matches)!=1:raise ValueError('missing/ambiguous host-to-GPU cost correlation')
        identity=matches[0]
        if identity not in routes:raise ValueError('GPU cost missing request metadata')
        token=bisect.bisect_left(maxima,k['end'])
        if token>=expected_tokens:raise ValueError('cost beyond completed token')
        if identity in tokens and tokens[identity]!=token:raise ValueError('request crosses token identity')
        tokens[identity]=token;durations[identity][kind].append((k['start'],k['end']))
    result=[]
    for identity,route in sorted(routes.items()):
        if identity not in durations or not durations[identity]['fetch']:raise ValueError('request lacks completed fetch kernel')
        mask={name:int(route[name]) for name in ('cpu','fetch','miss')}
        if any(v<0 or v>255 for v in mask.values()) or mask['cpu']&mask['fetch']:raise ValueError('invalid/overlapping route masks')
        ne=mask['cpu'].bit_count();done=cpu.get(identity)
        if ne and (done is None or done['experts']!=ne):raise ValueError('CPU completion population mismatch')
        if not ne and done:raise ValueError('CPU call without assigned expert')
        times={}
        for kind,intervals in durations[identity].items():
            intervals.sort()
            if any(b[0]<a[1] for a,b in zip(intervals,intervals[1:])):raise ValueError('overlapping same-request GPU costs')
            times[kind+'_ns']=sum(b-a for a,b in intervals)
        result.append(dict(device=identity[0],layer=identity[1],sequence=identity[2],token_index=tokens[identity],
            request_index=tokens[identity]//32,**mask,cpu_experts=ne,fetch_experts=mask['fetch'].bit_count(),
            cpu_completed_wall_ns=done['ns'] if done else 0,logical_fetch_bytes=mask['fetch'].bit_count()*int(route['blob']),**times))
    if set(cpu)-set(routes):raise ValueError('orphan CPU completion')
    for layer in {r['layer'] for r in result}:
        population=[r['token_index'] for r in result if r['layer']==layer]
        if len(population)!=expected_tokens or set(population)!=set(range(expected_tokens)):
            raise ValueError('incomplete/duplicate layer-token population')
    layers={}
    for layer in sorted({r['layer'] for r in result}):
        rows=[r for r in result if r['layer']==layer];measure={}
        for name in ('cpu_completed_wall_ns','fetch_ns','cpu_wait_ns'):
            values=[r[name] for r in rows if name in r]
            if values:measure[name]=dict(samples=len(values),minimum=min(values),median=statistics.median(values),maximum=max(values))
        layers[str(layer)]=measure
    return dict(schema=1,completed_requests=len(result),cpu_completion_records=len(cpu),tokens=expected_tokens,rows=result,layers=layers,
        scope='Completed CPU-call wall time and CUPTI fetch-kernel/wait-kernel residence joined by device/layer/request sequence and runtime correlation. '
        'Fetch is the complete mapped-read kernel path, not isolated PCIe DMA or physical bandwidth. CPU, fetch and wait overlap; never sum them into latency. '
        'No estimator, plan change, clean throughput claim, hardware-counter/QPI measurement or NUMA experiment.')

def analyze(database,expected_tokens=64):
    db=sqlite3.connect('file:'+str(database)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    columns={r[1] for r in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
    if not {'text','textId','globalTid','start','end'}<=columns:raise ValueError('unsupported NVTX schema; preserve raw export')
    annotations=[dict(r) for r in db.execute('SELECT n.*,COALESCE(n.text,s.value) AS label FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId')]
    kernels=[dict(r) for r in db.execute('SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName')]
    runtime=[dict(r) for r in db.execute('SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME')]
    try:return analyze_rows(annotations,kernels,runtime,expected_tokens)
    finally:db.close()
