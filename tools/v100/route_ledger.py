"""Validate bounded per-token prompt lanes and raw decode cache decisions."""
from collections import Counter
import json

LANES=('resident_mmq','resident_light','prestaged_ram','staged_ram_h2d',
       'staged_vram_d2d','staged_disk_h2d','cpu')


def parse(text,dispatch):
    raw=[line[len('V100_ROUTE '):] for line in text.splitlines() if line.startswith('V100_ROUTE ')]
    ends=[json.loads(line[len('V100_ROUTE_END '):]) for line in text.splitlines()
          if line.startswith('V100_ROUTE_END ')]
    if len(ends)!=1 or ends[0]['records']!=len(raw) or ends[0]['dropped'] or \
       ends[0]['payload_bytes']!=sum(len(line.encode()) for line in raw):
        raise ValueError('route capture dropped, truncated or missing terminal summary')
    issues={(r['layer'],r['request_seq']):r['position'] for r in dispatch['host_records']
            if r['path']=='request_issue'}
    prefill=[];decode=[];totals=Counter();seen=set()
    for invocation,line in enumerate(raw):
        r=json.loads(line);r['invocation']=invocation
        if r['kind']=='prefill':
            n,k=r['tokens'],r['k'];counts=r['counts']
            if n<=0 or not 1<=k<=8 or len(counts)!=n*len(LANES) or any(c not in '012345678' for c in counts):
                raise ValueError('invalid compact prompt route counts')
            for t in range(n):
                row=list(map(int,counts[t*7:(t+1)*7]))
                if sum(row)!=k:raise ValueError('per-token prompt routes do not conserve top-k')
                totals.update(dict(zip(LANES,row)))
            prefill.append(r)
        elif r['kind']=='decode':
            key=(r['layer'],r['request_seq']);k=len(r['ids']);bits=(1<<k)-1
            if key in seen or key not in issues or not 1<=k<=8 or any(e<0 for e in r['ids']):
                raise ValueError('decode route missing/ambiguous issue or invalid expert IDs')
            seen.add(key);r['position']=issues[key]
            masks=[r[name] for name in ('miss','fetch','cpu','skip')]
            if any(not isinstance(m,int) or m<0 or m&~bits for m in masks+[r['promo']]):
                raise ValueError('decode mask outside selected top-k')
            if any(a&b for i,a in enumerate(masks) for b in masks[i+1:]):
                raise ValueError('decode route lanes overlap')
            if r['promo']&~(r['miss']|r['fetch']):raise ValueError('promotion outside fetched/disk experts')
            r['resident_mask']=bits&~(r['miss']|r['fetch']|r['cpu']|r['skip'])
            decode.append(r)
        else:raise ValueError('unknown route kind')
    expected=Counter()
    for r in dispatch['host_records']:
        if r['phase']==1 and r['path'] in ('mmq_glu_selected','mmq_glu_not_selected'):
            expected[(r['layer'],r['position'],r['chunk_tokens'],r['type'])]+=r['calls']
    observed=Counter((r['layer'],r['position'],r['tokens'],r['gu_type']) for r in prefill)
    if observed!=expected or set(issues)!=seen or not prefill or not decode:
        raise ValueError('incomplete layer/token route coverage')
    return dict(schema=1,capture=ends[0],prefill_lanes=LANES,prefill=prefill,decode=decode,
        prefill_route_totals=dict(totals),
        decode_route_totals={name:sum(r[name].bit_count() for r in decode)
                             for name in ('resident_mask','miss','fetch','cpu','promo','skip')},
        scope='Prompt counts are seven decimal digits per token at position+offset, per layer invocation. '
              'Decode positions join actual request sequence; masks index ids, promotions overlap fetch/miss. '
              'Prefill prestage is RAM-tier H2D, tail-resident staging D2D. '
              'Counts are observed planning/dispatch, not measured physical traffic or CPU duration. '
              'Diagnostic copies/synchronization can perturb production routing populations.')
