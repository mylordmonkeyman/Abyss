import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('v100_bases',Path(__file__).parent/'bench/v100_base_compare.py')
bases=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bases)


class BaseEvidence(unittest.TestCase):
    def test_protocol_counts_and_reused_prefix_are_checked(self):
        wire=['T 11', 'T 29']
        row=bases.parse_done('DONE 2 600 10.0 20.0 length 0 0 0 0 0',wire,600,2)
        self.assertEqual(row['output_ids'],[11,29])
        self.assertEqual(row['decode_tokens_s'],100)
        for bad in ['DONE 2 599 10 20 length 0 0 0', 'DONE 2 600 10 20 length 0 0 1']:
            with self.assertRaises(ValueError): bases.parse_done(bad,wire,600,2)
        with self.assertRaises(ValueError):
            bases.parse_done('DONE 2 600 10 20 length 0 0 0',wire[:1],600,2)

    def test_node_memory_prefix_and_mapping_page_units(self):
        self.assertEqual(bases.kb_fields('Node 1 MemFree: 123 kB\nVmRSS: 4 kB\n'),
                         {'MemFree':123*1024,'VmRSS':4096})
        maps='123 anon=3 N0=2 N1=1 kernelpagesize_kB=4\n456 N1=3 kernelpagesize_kB=2048\n'
        self.assertEqual(bases.numa_bytes(maps),{'0':8192,'1':4096+3*2048*1024})

    def test_incremental_kl_has_fixed_direction_and_position_labels(self):
        ref=np.log([[.8,.2],[.5,.5]])
        candidate=np.log([[.7,.3],[.5,.5]])
        result=bases.compare_dumps(ref,candidate)
        expected=(.8*np.log(.8/.7)+.2*np.log(.2/.3))/2
        self.assertAlmostEqual(result['mean_kl'],expected)
        self.assertFalse(result['within_incremental_screen'])
        self.assertEqual([r['predicted_token_index_zero_based'] for r in result['positions']],[2041,2042])
        self.assertTrue(bases.compare_dumps(ref,ref+8)['within_incremental_screen'])
        with self.assertRaises(ValueError): bases.compare_dumps(ref,np.full_like(ref,np.nan))


if __name__=='__main__': unittest.main()
