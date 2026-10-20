"""Check that baseline triage detects drift without masking invalid distributions."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_f32_triage import comparison

class TriageMetrics(unittest.TestCase):
    def test_shift_invariant_repeat(self):
        a=np.tile(np.array([[-1.,2.,3.]]),(32,1))
        result=comparison(a,a+17,131039)
        self.assertLess(abs(result['mean_kl']),1e-12)
        self.assertEqual(result['top1_matches'],32)
        self.assertEqual(result['positions'][-1]['predicted_token_index_zero_based'],131071)

    def test_same_top1_does_not_hide_kl_failure(self):
        a=np.tile(np.array([[0.,1.]]),(32,1))
        b=np.tile(np.array([[-5.,1.]]),(32,1))
        result=comparison(a,b,131039)
        self.assertEqual(result['top1_matches'],32)
        self.assertGreater(result['mean_kl'],.0005)

    def test_invalid_population_or_nonfinite_rejected(self):
        for a,b in [(np.zeros((31,2)),np.zeros((31,2))),
                    (np.zeros((32,2)),np.zeros((32,3))),
                    (np.full((32,2),np.nan),np.zeros((32,2)))]:
            with self.assertRaises(ValueError):comparison(a,b,131039)
if __name__=='__main__':unittest.main()
