"""Scaling must meet nominal precision even where old controls underflow."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from summarize_p1_mma_batched import check_numerical


class ScalingGate(unittest.TestCase):
    def record(self,scaled):
        return dict(scaled=scaled,mode=2,relative_l2_vs_f32=.03,max_abs_vs_f32=1e-6,
            sampled_max_abs_vs_fp64=1e-6,sampled_relative_l2_vs_fp64=.03,
            nominal_relative_pass=False,**{'pass':True})

    def test_scaled_tiny_queries_cannot_use_old_absolute_exception(self):
        check_numerical(self.record(False))
        with self.assertRaisesRegex(ValueError,'nominal MMA relative'):check_numerical(self.record(True))
        row=self.record(True);row.update(relative_l2_vs_f32=1e-7,nominal_relative_pass=True)
        with self.assertRaisesRegex(ValueError,'FP64 relative'):check_numerical(row)
        row['sampled_relative_l2_vs_fp64']=1e-7;check_numerical(row)
        row['sampled_relative_l2_vs_fp64']=float('nan')
        with self.assertRaises(ValueError):check_numerical(row)


if __name__=='__main__':unittest.main()
