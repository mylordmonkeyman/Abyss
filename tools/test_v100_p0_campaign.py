"""Keep pairing, warm/cold separation and output-length strata correct."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).parent/'bench'))
from v100_p0_campaign import paired


class Pairing(unittest.TestCase):
    def test_reverse_order_and_warm_output_strata(self):
        def request(repetition,count,latency,identity):
            return dict(request=repetition,case='mixed-8192',output_tokens=count,
                        prefill_ms=latency,decode_ms=latency*2,ttft_ms=latency*3,
                        output_ids_sha256=identity)
        # Second process pair is B/A. Cold values are intentionally enormous;
        # different output lengths must remain distinct despite the same fixture.
        arms=[dict(index=2,base='current',requests=[request(0,128,99999,'warmup'),
                    request(1,128,90,'new'),request(1,1024,240,'same')]),
              dict(index=3,base='historical',requests=[request(0,128,88888,'warmup'),
                    request(1,128,100,'old'),request(1,1024,200,'same')])]
        rows=paired(arms)
        self.assertEqual(len(rows),2)
        by_count={r['output_tokens']:r for r in rows}
        self.assertAlmostEqual(by_count[128]['decode_ms_change_percent'],-10)
        self.assertAlmostEqual(by_count[1024]['decode_ms_change_percent'],20)
        self.assertFalse(by_count[128]['generated_ids_equal'])
        self.assertTrue(by_count[1024]['generated_ids_equal'])
        self.assertTrue(all(r['pair']==1 for r in rows))

    def test_incomplete_pair_is_not_an_ab_claim(self):
        self.assertEqual(paired([dict(index=0,base='historical',requests=[dict(
            request=1,case='prose-600',output_tokens=128)])]),[])


if __name__=='__main__':unittest.main()
