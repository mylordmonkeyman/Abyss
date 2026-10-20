"""The full-context gate needs both repeats and both matched comparisons."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from summarize_p1_mma_model import check_comparisons,PAIRS


class ModelGate(unittest.TestCase):
    def rows(self):
        return [dict(reference=a,candidate=b,mean_kl=1e-7,max_kl=1e-7,rows=32,top1_matches=32,
            positions=[dict(predicted_token_index_zero_based=i,kl=1e-7) for i in range(131040,131072)]) for a,b in sorted(PAIRS)]

    def test_both_repeats_and_cross_pairs_are_required(self):
        rows=self.rows();self.assertTrue(check_comparisons(rows))
        with self.assertRaises(ValueError):check_comparisons(rows[:-1])
        bad=copy.deepcopy(rows);bad[0]['mean_kl']=0
        with self.assertRaisesRegex(ValueError,'aggregate'):check_comparisons(bad)
        bad=copy.deepcopy(rows);bad[0]['positions'][0]['predicted_token_index_zero_based']=0
        with self.assertRaisesRegex(ValueError,'positions'):check_comparisons(bad)
        bad=copy.deepcopy(rows);bad[0]['positions'][0]['kl']=float('nan')
        with self.assertRaisesRegex(ValueError,'invalid model KL'):check_comparisons(bad)
        bad=copy.deepcopy(rows);bad[0].update(mean_kl=.001,max_kl=.001)
        for p in bad[0]['positions']:p['kl']=.001
        self.assertFalse(check_comparisons(bad))


if __name__=='__main__':unittest.main()
