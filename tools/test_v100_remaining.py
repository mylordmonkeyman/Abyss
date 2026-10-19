import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent/'bench'))
from v100_p0_remaining import plan

class RecoveryPlan(unittest.TestCase):
    def test_three_pairs_per_context_and_bounded_separate_units(self):
        for offset,length in [(0,32768),(6,65536),(12,131072)]:
            tasks=[plan(i) for i in range(offset,offset+6)]
            self.assertEqual([x['bases'][0] for x in tasks],['historical','current','current','historical','historical','current'])
            self.assertTrue(all(x['length']==length and x['output']==1024 for x in tasks))
            self.assertTrue(all(len(x['bases'])==1 for x in tasks))

    def test_chunks_and_traces_are_distinct_from_performance_long_units(self):
        self.assertEqual([plan(i)['prefill'] for i in range(18,21)],['512','1024','4096'])
        self.assertTrue(all(len(plan(i)['bases'])==6 for i in range(18,21)))
        self.assertEqual([plan(i)['length'] for i in (21,22)],[32768,131072])
        for i in (-1,23):
            with self.assertRaises(ValueError):plan(i)

if __name__=='__main__':unittest.main()
