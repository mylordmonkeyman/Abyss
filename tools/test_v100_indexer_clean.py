"""Clean timing requires qualified model evidence and exact balanced dispatch."""
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_indexer_clean import qualified,environment,validate_dispatch,ORDERS,QUALITY,source_check,libraries,record_gate,logit_comparison,bridge,measured_arm
from p1_integrated_quality import GEOMETRIES

class Clean(unittest.TestCase):

    def test_measurement_failure_is_deferred_but_safety_failure_is_not(self):
        errors=[]
        self.assertIsNone(record_gate(errors,'quality',lambda:logit_comparison(
            np.zeros((32,2)),np.full((32,2),np.nan),0)))
        self.assertEqual(len(errors),1)
        self.assertEqual(record_gate(errors,'next',lambda:7),7)
        with self.assertRaises(RuntimeError):
            record_gate(errors,'safety',lambda:(_ for _ in ()).throw(RuntimeError('memory guard')))
        metrics=logit_comparison(np.zeros((32,2)),np.ones((32,2)),0)
        self.assertEqual(metrics['mean_kl'],0)
        self.assertFalse(metrics['bitwise_equal'])
        self.assertTrue(metrics['passes_mean_kl_limit'])

    def test_bridge_collects_all_nine_arms_despite_failed_hash_gate(self):
        import p1_indexer_clean as clean
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache=root/'cache';cache.mkdir()
            original=root/'original.f32';np.zeros((32,2),dtype='<f4').tofile(original)
            tokens=root/'tokens';tokens.write_text('1,2')
            geometries={name:dict(token_count=2,score_start=0,max_context=102400,candidate_positions=[])
                        for name in ('100k','128k')}
            dumps={mode+'-0':dict(path=str(original),bytes=original.stat().st_size,sha256=clean.q.p0.sha256(original))
                   for mode in ('f32','cublas','corrected','volta')}
            evidence=dict(geometries={name:dict(quality=dict(fixture_sha256=clean.q.p0.sha256(tokens),dumps=dumps))
                                      for name in geometries})
            def arm(binary,directory,tokens,dump,mode,geometry,**kwargs):
                self.assertNotIn('defer_dispatch_validation',kwargs)
                directory.mkdir();values=np.zeros((32,2),dtype='<f4');values[:,0]=1;values.tofile(dump)
                return dict(validation_errors=[],dispatch_records=[])
            with patch.object(clean,'manifest',return_value=dict(binary_sha256='a'*64)),patch.object(clean.q,'GEOMETRIES',geometries), \
                 patch.object(clean.q,'fixture',return_value=[1,2]),patch.object(clean.q,'arm',side_effect=arm) as run, \
                 patch.object(clean,'validate_dispatch',return_value=0):
                bridge(cache,evidence)
            state=json.loads((cache/'bridge/bridge.json').read_text())
            self.assertEqual(run.call_count,9)
            self.assertEqual(len(state['arms']),9)
            self.assertEqual(len(state['comparisons']),6)
            self.assertEqual(len(state['gate_failures']),9)
            self.assertFalse(state['passes_admission_gates'])
            self.assertEqual(state['status'],'complete; all clean bridge measurements collected')


    def test_only_post_execution_dispatch_errors_are_deferred(self):
        import p1_indexer_clean as clean
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'engine.log').write_text('p1_indexer scorer=f32 layer=2 p0=0 T=2048\n')
            args=(Path('/binary'),root,Path('/tokens'),Path('/dump'),'f32',{}, {})
            with patch.object(clean.q,'arm',side_effect=ValueError('missing actual scorer dispatch records')):
                result=measured_arm(*args)
            self.assertEqual(result['dispatch_records'][0]['scorer'],'f32')
            self.assertEqual(len(result['validation_errors']),1)
            for error in (RuntimeError('memory guard'),ValueError('unknown identity failure')):
                with patch.object(clean.q,'arm',side_effect=error):
                    with self.assertRaises(type(error)):measured_arm(*args)

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
