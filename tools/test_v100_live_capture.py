"""Reject invalid same-input evidence independently of the live GPU capture."""
import pathlib
import sys
import tempfile
import unittest
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/profiling'))
from p1_live_capture import compare_capture,SAMPLES
class LiveCapture(unittest.TestCase):
    def test_complete_population_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=pathlib.Path(tmp)
            for layer in range(2,43,2):
                d=base/f'layer-{layer}';d.mkdir()
                (d/'meta.txt').write_text(f'ABYSS_P1_LIVE_1\n{layer} 98304 2048 32768 25088\n'+ ' '.join(map(str,SAMPLES))+'\n')
                (d/'complete').write_text('diagnostic\n')
                (d/'selection-restored').write_text('1048576 int32 values; bitwise verified\n')
                for t in SAMPLES:
                    for arm in ('candidate','reference'):
                        n=(98304+t+1)//4
                        np.arange(n,0,-1,dtype='<f4').tofile(d/f't-{t}-{arm}-score.f32')
                        np.arange(512,dtype='<i4').tofile(d/f't-{t}-{arm}-ids.i32')
                        np.zeros(32768,dtype='<f4').tofile(d/f't-{t}-{arm}-attention.f32')
            result=compare_capture(base)
            self.assertEqual(len(result['samples']),105)
            self.assertTrue(all(x['overlap']==512 and x['attention_max_abs']==0 for x in result['samples']))
            with self.assertRaisesRegex(ValueError,'capture geometry changed'):
                compare_capture(base,118784)
            for layer in range(2,43,2):
                (base/f'layer-{layer}/full-selection.tsv').write_text(
                    'token_offset\tordered_equal\tmembership_overlap\n'+''.join(f'{t}\t1\t512\n' for t in range(2048)))
            result=compare_capture(base)
            self.assertEqual(len(result['full_selection']),21)
            self.assertTrue(all(x['ordered_identical']==2048 for x in result['full_selection']))
            report=base/'layer-2/full-selection.tsv'
            original=report.read_text();report.write_text(original.replace('0\t1\t512','0\t0\t511',1))
            with self.assertRaisesRegex(ValueError,'disagrees with raw sampled IDs'):compare_capture(base)
            report.write_text(original)
            path=base/'layer-2/t-0-candidate-ids.i32'
            bad=np.arange(512,dtype='<i4');bad[0]=30000;bad.tofile(path)
            with self.assertRaisesRegex(ValueError,'range/duplicate/future'):compare_capture(base)
            np.arange(512,dtype='<i4').tofile(path)
            path=base/'layer-2/t-0-candidate-attention.f32'
            bad=np.zeros(32768,dtype='<f4');bad[1]=1e-7;bad.tofile(path)
            with self.assertRaisesRegex(ValueError,'same selection gave different attention'):compare_capture(base)
if __name__=='__main__':unittest.main()
