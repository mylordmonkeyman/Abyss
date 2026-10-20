"""Require matched complete timing populations before interpreting P2 results."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from summarize_p2_tile import timing_summary

class AttentionTiming(unittest.TestCase):
    def population(self):
        return [dict(timing='attention_tile',ratio=r,p0=131071,T=t,key_tile=k,warps=w,pair=p,f32_ms=10.,candidate_ms=12.)
                for r in (0,4,128) for t in (256,2048) for k in (16,32) for w in (4,8) for p in range(6)]

    def test_paired_changes_and_missing_shape(self):
        rows=self.population();result=timing_summary(rows)
        self.assertEqual(len(result),24)
        self.assertAlmostEqual(result[0]['median_paired_elapsed_change_pct'],20.)
        self.assertEqual(result[0]['candidate_faster_pairs'],0)
        with self.assertRaisesRegex(ValueError,'shape population'):timing_summary(rows[6:])

    def test_duplicate_pair_rejected(self):
        rows=self.population();rows[1]['pair']=0
        with self.assertRaisesRegex(ValueError,'six unique'):timing_summary(rows)

    def test_nonfinite_or_nonpositive_elapsed_rejected(self):
        for value in (float('nan'),float('inf'),0.,-1.):
            rows=self.population();rows[0]['candidate_ms']=value
            with self.assertRaisesRegex(ValueError,'invalid elapsed'):timing_summary(rows)

if __name__=='__main__':unittest.main()
