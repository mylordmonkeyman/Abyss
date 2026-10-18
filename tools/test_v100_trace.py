import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('analyze_v100_trace',Path(__file__).parent/'profiling/analyze_trace.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)


class ActivityIntervals(unittest.TestCase):
    def test_overlap_clipping_and_empty_intervals(self):
        intervals=[(-10,5),(2,8),(7,12),(15,25),(16,19)]
        # [0,12] and [15,20]; overlap must not be counted twice.
        self.assertEqual(trace.union_ns(intervals,0,20),17)
        self.assertEqual(trace.union_ns(intervals,12,15),0)
        self.assertEqual(trace.union_ns([],0,10),0)
        self.assertEqual(trace.union_ns([(0,100),(10,20)],15,25),10)


if __name__=='__main__': unittest.main()
