"""Cost observations must join completed CPU/GPU work without inventing latency."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p3_cost_analyze import analyze_rows
from summarize_p3_cost import distribution,population,summarize

class CostJoin(unittest.TestCase):
    def test_artifact_summary_rejects_inconsistent_evidence(self):
        rows=[dict(device=0,layer=layer,token_index=token,request_index=token//32,
            cpu=0,fetch=0,miss=0,cpu_experts=0,fetch_experts=0,cpu_completed_wall_ns=0,fetch_ns=1)
            for layer in range(43) for token in range(64)]
        entries={'costs.json':dict(rows=rows,tokens=64,completed_requests=2752,cpu_completion_records=0),
            'raw-identity.json':dict(source_sha='source',binary_sha256='binary',fixture_sha256='fixture'),
            'invocation.json':dict(binary_sha256='binary',fixture_sha256='fixture',context_tokens=32768),
            'requests.json':[dict(request=i,done='DONE 32 32768') for i in range(2)],
            'memory_summary.json':{},'thermal.json':{}}
        def check(data,engine='V100_COST_END ranges=1 dropped=0',digest=None):
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'cost.zip'
                with zipfile.ZipFile(path,'w') as z:
                    for name,value in data.items():z.writestr(name,json.dumps(value))
                    z.writestr('source-sha.txt','source\n');z.writestr('binary-sha256.txt','binary  engine\n')
                    z.writestr('engine.log.gz',gzip.compress(engine.encode()))
                return summarize(path,digest or hashlib.sha256(path.read_bytes()).hexdigest(),1)
        self.assertEqual(check(entries)['completed_layer_tokens'],2752)
        bad=copy.deepcopy(entries);bad['costs.json']['rows'].pop()
        with self.assertRaisesRegex(ValueError,'layer-token population'):check(bad)
        bad=copy.deepcopy(entries);bad['raw-identity.json']['source_sha']='other'
        with self.assertRaisesRegex(ValueError,'identities disagree'):check(bad)
        bad=copy.deepcopy(entries);bad['requests.json'][1]['done']='DONE 32 8192'
        with self.assertRaisesRegex(ValueError,'request population'):check(bad)
        with self.assertRaisesRegex(ValueError,'dropped or incomplete'):check(entries,'V100_COST_END ranges=2 dropped=1')
        with self.assertRaisesRegex(ValueError,'digest mismatch'):check(entries,digest='wrong')

    def test_conditioned_summary_keeps_noop_and_misses_separate(self):
        rows=[dict(request_index=0,cpu_experts=0,fetch_experts=0,miss=0,
            cpu_completed_wall_ns=0,fetch_ns=3),
            dict(request_index=0,cpu_experts=1,fetch_experts=1,miss=0,
                cpu_completed_wall_ns=100,fetch_ns=50,cpu_wait_ns=20),
            dict(request_index=0,cpu_experts=1,fetch_experts=1,miss=1,
                cpu_completed_wall_ns=1000,fetch_ns=500,cpu_wait_ns=200)]
        groups=population(rows)
        self.assertEqual(len(groups),3)
        self.assertEqual(groups[0]['costs']['cpu_completed_wall_ns'],dict(samples=0))
        self.assertEqual(groups[1]['costs']['fetch_ns']['median_ns'],50)
        self.assertTrue(groups[2]['disk_miss_mask_nonzero'])
        self.assertNotIn('total_latency_ns',groups[1]['costs'])
        self.assertEqual(distribution([1,3,2])['p95_ns'],3)
        with self.assertRaisesRegex(ValueError,'invalid cost'):distribution([float('nan')])

    def fixture(self):
        events=[dict(label='P3_ROUTE device=0 layer=2 seq=1 cpu=1 fetch=2 miss=0 blob=8',start=300,end=None,globalTid=2),
            dict(label='P3_COST kind=cpu device=0 layer=2 seq=1 experts=1',start=500,end=600,globalTid=2),
            dict(label='P3_COST kind=fetch device=0 layer=2 seq=1 experts=-1',start=100,end=200,globalTid=1),
            dict(label='P3_COST kind=cpu_wait device=0 layer=2 seq=1 experts=-1',start=210,end=220,globalTid=1)]
        kernels=[dict(name='moe_fetch_kernel',start=1000,end=1100,correlationId=1),
            dict(name='moe_cpu_wait_kernel',start=1200,end=1300,correlationId=2),
            dict(name='strata::argmax_kernel(',start=1400,end=1500,correlationId=3)]
        api=[dict(start=110,end=120,globalTid=1,correlationId=1),dict(start=211,end=212,globalTid=1,correlationId=2)]
        return events,kernels,api

    def test_completed_costs_and_masks_remain_separate(self):
        result=analyze_rows(*self.fixture(),expected_tokens=1);row=result['rows'][0]
        self.assertEqual((row['cpu_completed_wall_ns'],row['fetch_ns'],row['cpu_wait_ns']),(100,100,100))
        self.assertEqual(row['logical_fetch_bytes'],8)
        self.assertEqual(row['cpu_experts'],1)
        self.assertNotIn('total_latency_ns',row)

    def test_ambiguous_or_missing_gpu_correlation_rejected(self):
        events,kernels,api=self.fixture()
        for bad in ([],api+[copy.deepcopy(api[0])]):
            with self.assertRaisesRegex(ValueError,'cost correlation'):analyze_rows(events,kernels,bad,1)

    def test_missing_cpu_completion_and_overlapping_masks_rejected(self):
        events,kernels,api=self.fixture()
        with self.assertRaisesRegex(ValueError,'CPU completion'):analyze_rows(events[:1]+events[2:],kernels,api,1)
        events[0]['label']=events[0]['label'].replace('fetch=2','fetch=1')
        with self.assertRaisesRegex(ValueError,'route masks'):analyze_rows(events,kernels,api,1)

if __name__=='__main__':unittest.main()
