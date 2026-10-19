import importlib.util
from pathlib import Path
import unittest
import sqlite3
import tempfile

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


class MainStreamBudget(unittest.TestCase):
    def test_overlapping_other_stream_is_not_added_to_main_wall_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.sqlite';db=sqlite3.connect(path)
            db.executescript("""
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,
                    streamId INTEGER,registersPerThread INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER);
            """)
            names=['strata::attend_raw_kernel(', 'strata::argmax_kernel(', 'strata::moe_cpu_wait(', 'mul_mat_q(']
            db.executemany('INSERT INTO StringIds VALUES (?,?)',enumerate(names))
            records=[(0,9,0,1),(9,10,1,1),(10,25,2,1),(25,30,1,1),(0,20,3,2)]
            db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,32,256,1,1)',
                           [(a*1000000,b*1000000,n,stream) for a,b,n,stream in records])
            db.commit();db.close()
            r=trace.analyze(path,'DONE 2 8192 10 20',8192)
            self.assertIn('8192-token',r['scope'])
            pre=r['phases']['prefill'];dec=r['phases']['decode']
            self.assertEqual(pre['kernel_sum_ms'],20)
            self.assertEqual(pre['critical_path_proxy']['measured_main_stream_kernel_union_ms'],10)
            self.assertEqual(dec['critical_path_proxy']['unattributed_main_stream_gap_ms'],0)
            self.assertEqual(dec['critical_path_proxy']['top_three'][0]['name'],'CPU expert dependency wait on GPU')
            self.assertEqual(dec['critical_path_proxy']['top_three'][0]['wall_lane_ms'],15)
            db=sqlite3.connect(path)
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0,5000000,0,1,32,256,1,1)')
            db.commit();db.close()
            with self.assertRaises(ValueError):trace.analyze(path,'DONE 2 8192 10 20',8192)

if __name__=='__main__': unittest.main()
