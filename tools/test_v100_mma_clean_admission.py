"""Clean timing cannot bypass model quality, repeatability or build controls."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_mma_clean_ab import qualified,clean_cache
from p1_integrated_quality import GEOMETRIES
import test_v100_mma_model_summary as model_tests


class CleanAdmission(unittest.TestCase):
    def evidence(self):
        return dict(passes_bounded_model_screen=True,f32_repeats_bitwise=True,mma_repeats_bitwise=True,
            quality=dict(status='complete; inspect model qualification',source_sha='1'*40,
                comparisons=model_tests.ModelGate().rows(),attention_investigation=False,cpu_lane=0,
                geometry=GEOMETRIES['128k'],dumps={label:dict(sha256='2'*64)
                    for label in ('f32-0','f32-1','volta-0','volta-1')}))

    def test_failed_model_or_repeat_cannot_time(self):
        evidence=self.evidence();qualified(evidence)
        bad=copy.deepcopy(evidence);bad['quality']['attention_investigation']=True
        with self.assertRaisesRegex(ValueError,'attention'):qualified(bad)
        bad=copy.deepcopy(evidence);bad['quality']['cpu_lane']=28
        with self.assertRaisesRegex(ValueError,'CPU-lane-0'):qualified(bad)
        bad=copy.deepcopy(evidence);bad['quality']['dumps']['volta-1']['sha256']='3'*64
        with self.assertRaisesRegex(ValueError,'repeat hashes'):qualified(bad)
        bad=copy.deepcopy(evidence);row=bad['quality']['comparisons'][0]
        row.update(mean_kl=.001,max_kl=.001)
        for position in row['positions']:position['kl']=.001
        with self.assertRaisesRegex(ValueError,'quality'):qualified(bad)

    def test_instrumented_or_wrong_arch_build_cannot_time(self):
        config='CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CUDA_ARCHITECTURES:STRING=70\n'+''.join(
            name+':BOOL=OFF\n' for name in ('STRATA_V100_CAPTURE_INDEXER','STRATA_V100_CAPTURE_STAGES','STRATA_V100_CAPTURE_COST'))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'CMakeCache.txt';path.write_text(config);clean_cache(path)
            path.write_text(config.replace('INDEXER:BOOL=OFF','INDEXER:BOOL=ON'))
            with self.assertRaisesRegex(ValueError,'INDEXER'):clean_cache(path)
            path.write_text(config.replace('STRING=70','STRING=80'))
            with self.assertRaisesRegex(ValueError,'SM70'):clean_cache(path)


if __name__=='__main__':unittest.main()
