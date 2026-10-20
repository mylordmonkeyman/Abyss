"""A cutoff replay must disclose both recovered decisions and omitted winners."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_mma_cutoff_replay import replay


class CutoffReplay(unittest.TestCase):
    def test_near_cutoff_swap_is_recovered(self):
        candidate=np.linspace(2,1,600);candidate[512]=candidate[511]-1e-7
        reference=candidate.copy();reference[512]=reference[511]+1e-7
        result=replay(reference,candidate)
        self.assertFalse(result['original_order_identical'])
        self.assertTrue(result['baseline_top512_covered'])
        self.assertTrue(result['refined_order_identical'])
        self.assertLess(result['refined_rows'],600)

    def test_band_cannot_certify_an_unobserved_large_error(self):
        candidate=np.linspace(2,1,600);reference=candidate.copy();reference[-1]=3
        result=replay(reference,candidate)
        self.assertFalse(result['baseline_top512_covered'])
        self.assertFalse(result['refined_order_identical'])


if __name__=='__main__':unittest.main()
