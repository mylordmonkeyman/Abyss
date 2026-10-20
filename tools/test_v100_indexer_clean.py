"""Clean timing requires qualified model evidence and exact balanced dispatch."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_indexer_clean import qualified,environment,validate_dispatch,ORDERS,QUALITY,source_check,libraries
from p1_integrated_quality import GEOMETRIES

class Clean(unittest.TestCase):
    def test_failed_identity_probe_cannot_bypass_clean_guard(self):
        with patch('p1_indexer_clean.q.p0.command',return_value=dict(returncode=None,error='timeout')):
            with self.assertRaises(ValueError):source_check(dict(source_sha='a'*40))
            with self.assertRaises(ValueError):libraries(Path('/unresolved/binary'))

    def test_verified_real_model_ledger_and_failed_original_comparator(self):
        evidence=json.loads(QUALITY.read_text());qualified(evidence)
        self.assertFalse(evidence['geometries']['128k']['original_cublas_passes_mean_kl_limit'])
        for mutate in (lambda e:e['geometries'].pop('100k'),
            lambda e:e['geometries']['128k']['quality'].update(cpu_lane=28),
            lambda e:e['geometries']['128k']['quality']['dumps']['corrected-1'].update(sha256='0'*64),
            lambda e:e['geometries']['128k']['direct']['corrected'].update(ordered_identical=43007)):
            with self.subTest(mutate=mutate):
                bad=copy.deepcopy(evidence);mutate(bad)
                with self.assertRaises(ValueError):qualified(bad)

    def test_balanced_positions_and_ordered_predecessors(self):
        modes={'f32','cublas','corrected','volta'}
        for position in range(4):self.assertEqual({r[position] for r in ORDERS},modes)
        predecessors=Counter((a,b) for row in ORDERS for a,b in zip(row,row[1:]))
        self.assertEqual(predecessors,Counter({(a,b):1 for a in modes for b in modes if a!=b}))
        for mode in modes:
            env=environment(mode)
            self.assertEqual(env['ABYSS_V100_PROFILE'],'0');self.assertEqual(env['STRATA_GLM_CPU_LANE'],'0')
            self.assertEqual(env.get('STRATA_DS4_IDX')=='1',mode=='f32')
            self.assertEqual(env['ABYSS_V100_CUBLAS_CORRECTION'],'band-all' if mode=='corrected' else 'none')

    def test_dispatch_counts_repeated_100k_position_and_no_candidate_in_f32(self):
        geometries=list(GEOMETRIES.values())
        records=[dict(layer=str(layer),p0=str(pos),T='2048',scorer='cublas',refinement='band-all')
            for geometry in geometries for layer in range(2,43,2) for pos in geometry['candidate_positions'] for _ in range(2)]
        self.assertEqual(validate_dispatch(records,'corrected',geometries,2),504)
        for mutate in (lambda r:r.pop(),lambda r:r.append(copy.deepcopy(r[0])),
            lambda r:r[0].update(refinement='none'),lambda r:r[0].update(T='1024')):
            with self.subTest(mutate=mutate):
                bad=copy.deepcopy(records);mutate(bad)
                with self.assertRaises(ValueError):validate_dispatch(bad,'corrected',geometries,2)
        with self.assertRaises(ValueError):validate_dispatch(records,'f32',geometries,2)
        self.assertEqual(validate_dispatch([dict(scorer='f32')],'f32',geometries,2),0)

if __name__=='__main__':unittest.main()
