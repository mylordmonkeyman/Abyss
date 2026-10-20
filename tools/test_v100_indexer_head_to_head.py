"""Actual-model correction coverage and accounting cannot silently shrink."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_indexer_head_to_head import validate_corrections
from p1_integrated_quality import GEOMETRIES

class HeadToHead(unittest.TestCase):
    def records(self,geometry):
        return [dict(layer=str(layer),p0=str(p),T='2048',policy='band-all',flagged_tokens='10',
            corrected_tokens='2048',full_tokens='0',rescored_rows=str(512*2048))
            for layer in range(2,43,2) for p in geometry['candidate_positions']]

    def test_both_frozen_geometries(self):
        for geometry in GEOMETRIES.values():validate_corrections(self.records(geometry),geometry,'band-all')

    def test_missing_duplicate_policy_or_invalid_correction_rejected(self):
        geometry=GEOMETRIES['128k']
        for mutate in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
            lambda r:r[0].update(corrected_tokens='2047'),lambda r:r[0].update(policy='band-flagged'),
            lambda r:r[0].update(rescored_rows='1'),lambda r:r[0].update(full_tokens='2049')):
            with self.subTest(mutate=mutate):
                records=self.records(geometry);mutate(records)
                with self.assertRaises(ValueError):validate_corrections(records,geometry,'band-all')

if __name__=='__main__':unittest.main()
