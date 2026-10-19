#!/usr/bin/env python3
"""Offline, fixed-pack reconstruction of the two immutable P0 prefill layouts.

This is a source-derived suballocation worksheet, not a fresh allocation trace.
All returned regions are INSIDE the already-accounted expert pool. Different
users of the union overlap; their byte counts must never be added together.
"""
import argparse
import gzip
import json
from pathlib import Path
import re


class Carve:
    def __init__(self):
        self.rows=[]
        self.size=0

    def take(self,name,count,width=1):
        size=count*width
        aligned=(size+255)//256*256
        self.rows.append(dict(name=name,offset=self.size,requested_bytes=size,
                              aligned_bytes=aligned,padding_bytes=aligned-size))
        self.size+=aligned


def q8_bytes(rows,cols):
    # Pinned ggml mmq.cuh: 128 int8 values plus a 16-byte scales/sums union.
    # moe_mmq.cu q8_bytes also reserves 128 whole blocks of scratch padding.
    return rows*((cols+511)//512*512)*144//128+128*144


def ds4(tokens,max_pools):
    c=Carve();D=512;HD=64*D;OL=8*1024
    for name,n,w in [('qr_raw',tokens*1024,4),('qr',tokens*1024,4),
        ('qr16',tokens*1024,2),('kv_raw',tokens*D,4),
        ('ckv',tokens*2*D,4),('csc',tokens*2*D,4),
        ('ikv',tokens*2*128,4),('isc',tokens*2*128,4),('iw',tokens*64,4),
        ('q',tokens*HD,4),('iq',tokens*64*128,4),('o',tokens*HD,4),
        ('o16',tokens*HD,2),('oa',tokens*OL,4),('oa16',tokens*OL,2),
        ('keys',(tokens+128)*D,2),('score',tokens*max(1,max_pools),4),
        ('sel',tokens*512,4)]:c.take(name,n,w)
    return c


def moe(tokens,sub=256):
    c=Carve();routes=tokens*6
    window=min(routes,max(tokens,((8192-6144)//3)*32))
    c.take('logits',tokens*256,4)
    for name in ('ids','rw','rank','pos','row_tok'):c.take(name,routes,4)
    for name,n in [('counts',256),('base',256),('bounds',8192)]:c.take(name,n,4)
    c.take('Xq',q8_bytes(window,4096));c.take('Hq',q8_bytes(window,2048))
    for name in ('OUTP','OUTPc'):c.take(name,window*4096,4)
    for name,w in [('sh_g',4),('sh_u',4),('sh16',2)]:c.take(name,min(tokens,sub)*2048,w)
    return c


def layout(tokens,base,max_context=135168,stride=10888976,prestage_slots=160):
    if base not in ('historical','current') or tokens<=0 or tokens%64:
        raise ValueError('pinned base and positive 64-token carve grid required')
    # These arguments describe the observed stock CUDA/default-control path.
    max_pools=max(1,max_context//4)
    room=moe(tokens).size;sub=256
    for candidate in (512,1024,2048,4096):
        if candidate>tokens or ds4(candidate,max_pools).size>room:break
        sub=candidate
    pers=Carve()
    for name,n,w in [('R',tokens*4*4096,4),('x',tokens*4096,4),
        ('x16',tokens*4096,2),('mixer',tokens*4096,4),('ffn',tokens*4096,4),
        ('pre',tokens*4,4),('post',tokens*4,4)]:pers.take(name,n,w)
    if base=='current':
        pers.take('pre2',tokens*4,4);pers.take('onehot',tokens*4,4)
    for name,n in [('comb',tokens*16),('ss',tokens),('mix',tokens*24),('iota',tokens*6)]:
        pers.take(name,n,4)
    attention=ds4(min(tokens,sub),max_pools);experts=moe(tokens)
    union=max(attention.size,experts.size)
    outer=Carve();outer.take('persistent_plus_reused_union',pers.size+union)
    # from_gguf_dsv4 sets kda_head_dim=32 and qk_nope=v_head=512;
    # weight_scratch then reserves 16 MiB F16 and 64 MiB F32 on CUDA.
    outer.take('weight_f16',32*64*4096,2)
    outer.take('weight_f32',64*512*512,4)
    outer.take('cublas_workspace',16*1024**2)
    outer.take('staged_expert_groups',3*4*stride)
    slots=min(prestage_slots,max(16,tokens//128)) if tokens>=1024 else 0
    outer.take('prestaged_experts',slots*stride)
    return dict(tokens=tokens,base=base,mixer_sub_batch=sub,prestage_slots=slots,
        borrowed_bytes=outer.size,outer=outer.rows,persistent_bytes=pers.size,
        persistent=pers.rows,union_bytes=union,union_offset=pers.size,
        union_views=dict(attention=dict(bytes=attention.size,regions=attention.rows),
                         moe=dict(bytes=experts.size,regions=experts.rows)),
        scope='Source-derived layout for this fixed DeepSeek pack, CUDA, default controls, '
              'no KDA/dense/NextN fill. Union views overlap. All bytes belong to expert pool.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--capture-root',type=Path,help='Extracted immutable memory artifact, optional log corroboration')
    args=ap.parse_args()
    # gpin is exactly 1,045,341,696 requested bytes for 96 landing slots in
    # both allocation traces (seq 26), hence the maximum expert stride below.
    stride=1045341696//96
    result=dict(schema=1,source_run=39194149121,artifact_id=12161799551,
        artifact_zip_sha256='fa1995c21d2fba4cacb9fa877d5c447def118fe47575d617466d2fa8f8c6e8e2',
        stride_bytes=stride,stride_basis='cudaHostAlloc seq26 / observed 96 landing slots',
        context=135168,configured_chunk=32768,expert_pool_bytes=24909026560,
        source_pins=dict(historical='47001f31cbfba14bcbf4c083e3af07c50707ec0c',
                         current='7c893ac17e61ec45c934b3278356807fc4626c3c',
                         ggml='3cf03257f219afbe7334045ff7c6a06ac68c627d'),
        layouts=[layout(t,b,stride=stride) for t in (512,640,1024,4096,8192,32768)
                 for b in ('historical','current')],
        limits='Derived offsets/sizes, not instrumented suballocation addresses or physical pages. '
               'Configured maximum reservation differs from actual short-chunk loan. '
               'No production arithmetic changed; no additional GPU capacity is claimed.')
    if args.capture_root:
        checks=[]
        for base in ('historical','current'):
            with gzip.open(args.capture_root/(base+'-allocations.jsonl.gz'),'rt') as f:
                event=next(json.loads(line) for line in f if json.loads(line)['seq']==26)
            if event['api']!='cudaHostAlloc' or event['result'] or event['bytes']!=96*stride:
                raise ValueError('landing ring stride no longer matches pinned capture')
            with gzip.open(args.capture_root/base/'engine.log.gz','rt') as f:log=f.read()
            m=re.search(r'chunks of (\d+) tokens.*mixer sub-batches of (\d+).*borrowing '
                        r'(\d+) MB.*activations (\d+), weight scratch (\d+), expert '
                        r'staging (\d+), prestage (\d+): (\d+) experts',log)
            if not m:raise ValueError('missing observed startup layout')
            a=layout(32768,base,stride=stride);outer={r['name']:r['aligned_bytes'] for r in a['outer']}
            expected=[32768,a['mixer_sub_batch'],a['borrowed_bytes']/1048576,
                      outer['persistent_plus_reused_union']/1048576,
                      (outer['weight_f16']+outer['weight_f32'])/1048576,
                      outer['staged_expert_groups']/1048576,outer['prestaged_experts']/1048576,160]
            if any(abs(int(v)-e)>0.5 for v,e in zip(m.groups(),expected)):
                raise ValueError('source layout disagrees with observed rounded startup sizes')
            checks.append(dict(base=base,landing_api_seq=26,landing_requested_bytes=event['bytes'],
                               rounded_startup_fields=list(map(int,m.groups())),status='pass'))
        result['capture_corroboration']=checks
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
