"""Numerical admission checks for the baseline measurement code; no GPU or model."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'v100'))
from model_baseline import normalize_logprobs


class LogProbComparison(unittest.TestCase):
    def test_normalization_offsets_and_kl_direction(self):
        # Known categorical distributions; arbitrary logit offsets must not change KL.
        ref = normalize_logprobs(np.log([[0.9, 0.1]]) + 1000)
        test = normalize_logprobs(np.log([[0.6, 0.4]]) - 2000)
        self.assertAlmostEqual(float(np.exp(ref).sum()), 1, places=12)
        kl = float((np.exp(ref) * (ref - test)).sum())
        self.assertAlmostEqual(kl, 0.22628916118535888, places=10)
        self.assertNotAlmostEqual(kl, float((np.exp(test) * (test - ref)).sum()))

    def test_nonfinite_and_wrong_shape_are_rejected(self):
        for raw in ([[float('nan'), 0]], [[float('inf'), 0]], [0, 1]):
            with self.assertRaises(ValueError):
                normalize_logprobs(raw)


if __name__ == '__main__':
    unittest.main()
