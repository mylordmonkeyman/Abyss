"""Correction evidence distinguishes literal corrected order from unflagged drift."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from summarize_p1_cublas_corrected import validate_population

def fixture(variant='band-all'):
    return [dict(case='cublas_indexer',variant=variant,T=3,p0=2052,mode=0,pass_=True,
        relative_l2_vs_f32=1e-7,max_abs_vs_f32=1e-6,sampled_max_abs_vs_fp64=1e-6,
        selection_count=1536,selection_overlap=1536,**{'pass':True}),
        dict(case='correction',variant=variant,T=3,p0=2052,mode=0,flagged_tokens=1,corrected_tokens=3,
            rescored_rows=1536,unflagged_order_errors=0,unflagged_membership_errors=0,
            exact_corrected_scores_and_order=True,**{'pass':True})]

class Corrected(unittest.TestCase):
    def test_all_token_literal_contract(self):
        validate_population(fixture(),{('band-all',3,2052,0)})

    def test_flagged_policy_discloses_unflagged_order_errors(self):
        records=fixture('band-flagged');records[1].update(corrected_tokens=1,rescored_rows=512,unflagged_order_errors=2)
        validate_population(records,{('band-flagged',3,2052,0)})

    def test_missing_duplicate_or_false_exact_evidence_rejected(self):
        for mutate in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
            lambda r:r[1].update(corrected_tokens=2),lambda r:r[1].update(exact_corrected_scores_and_order=False),
            lambda r:r[0].update(relative_l2_vs_f32=float('nan'))):
            with self.subTest(mutate=mutate):
                records=fixture();mutate(records)
                with self.assertRaises(ValueError):validate_population(records,{('band-all',3,2052,0)})

if __name__=='__main__':unittest.main()
