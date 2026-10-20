"""Correction campaigns require a completed, deterministic causality control."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_cublas_corrected_bench import require_control
from p1_integrated_quality import GEOMETRIES

def fixture():
    labels=('f32-0','reference-order','reference-membership','f32-1')
    state=dict(source_sha='2e1de6f02cdaa689d537101e3b3b3ef2ce92d09e',
        status='complete diagnostic intervention; inspect causality and ordering',
        geometry=copy.deepcopy(GEOMETRIES['128k']),cpu_lane=0,
        dumps={k:dict(bytes=16547840,sha256='a'*64) for k in labels},arms={},comparisons=[])
    for kind in labels:
        records=[]
        if kind.startswith('reference-'):
            records=[dict(kind=kind,layer=str(layer),p0=str(p),T='2048',rows='2048',
                ordered_equal_rows='2000',membership_equal_rows='2047',overlap=str(2048*512-1))
                for layer in range(2,43,2) for p in state['geometry']['candidate_positions']]
        state['arms'][kind]=dict(selection_controls=records)
    for candidate in ('f32-1','reference-order','reference-membership'):
        kl=.01 if candidate=='reference-membership' else 0.
        state['comparisons'].append(dict(reference='f32-0',candidate=candidate,mean_kl=kl,max_kl=kl,
            rows=32,top1_matches=32,positions=[dict(predicted_token_index_zero_based=i,kl=kl) for i in range(131040,131072)]))
    return state

class Causality(unittest.TestCase):
    def test_membership_failure_is_not_waived_but_does_not_block_correction(self):
        require_control(fixture())

    def test_incomplete_wrong_geometry_source_or_repeat_is_rejected(self):
        for mutate in (
            lambda s:s['arms']['reference-order']['selection_controls'].pop(),
            lambda s:s.update(source_sha='b'*40),
            lambda s:s.update(cpu_lane=28),
            lambda s:s['geometry'].update(score_start=1),
            lambda s:s['dumps']['f32-1'].update(sha256='b'*64),
            lambda s:s['comparisons'][1].update(mean_kl=.01),
            lambda s:s['comparisons'][0]['positions'][0].update(kl=float('nan')),
        ):
            with self.subTest(mutate=mutate):
                state=fixture();mutate(state)
                with self.assertRaises(ValueError):require_control(state)

if __name__=='__main__':unittest.main()
