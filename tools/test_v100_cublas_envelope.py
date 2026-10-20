"""Empirical margin guards must disclose ordering and missed-envelope cases."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_cublas_envelope import probe

class Envelope(unittest.TestCase):
    def test_near_cutoff_membership_error_is_corrected(self):
        candidate=np.linspace(2,1,600);candidate[512]=candidate[511]-1e-7
        reference=candidate.copy();reference[512]=reference[511]+1e-7
        row=probe(reference,candidate,32*2**-23)
        self.assertTrue(row['flagged']);self.assertFalse(row['original_membership_equal'])
        self.assertTrue(all(r['ordered_equal'] for r in row['corrections'].values()))

    def test_membership_guard_does_not_claim_ordering(self):
        candidate=np.linspace(2,1,600);candidate[1]=candidate[0]-1e-7
        reference=candidate.copy();reference[1]=reference[0]+1e-7
        row=probe(reference,candidate,32*2**-23)
        self.assertFalse(row['flagged']);self.assertTrue(row['original_membership_equal'])
        self.assertFalse(row['corrections']['band-flagged']['ordered_equal'])
        self.assertTrue(row['corrections']['band-all']['ordered_equal'])

    def test_uncalibrated_error_can_escape_empirical_guard(self):
        candidate=np.linspace(2,1,600);reference=candidate.copy();reference[-1]=3
        row=probe(reference,candidate,32*2**-23)
        self.assertFalse(row['flagged'])
        self.assertFalse(row['corrections']['band-all']['membership_equal'])
        with self.assertRaises(ValueError):probe(reference[:512],candidate[:512],32*2**-23)

if __name__=='__main__':unittest.main()
