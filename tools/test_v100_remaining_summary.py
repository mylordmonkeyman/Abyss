import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parent/'profiling'))
from summarize_p0_remaining import paired,collect


class RemainingSummary(unittest.TestCase):
    def test_pairs_use_second_request_and_preserve_trajectory_difference(self):
        arms=[]
        for i,base in enumerate(('historical','current','current','historical','historical','current')):
            r=dict(case='fixture',output_tokens=1024,prefill_ms=100 if base=='historical' else 110,
                   decode_ms=100,ttft_ms=100,output_ids_sha256=base)
            arms.append(dict(index=i,base=base,requests=[{},r]))
        result=paired(arms)
        self.assertEqual(len(result),3)
        self.assertAlmostEqual(result[0]['prefill_ms_change_percent'],10)
        self.assertTrue(all(not r['generated_ids_equal'] for r in result))
        with self.assertRaises(ValueError):paired(arms[:-1])
        arms[1]['requests'][1]['output_tokens']=128
        with self.assertRaises(ValueError):paired(arms)

    def test_changed_zip_is_rejected_before_reading_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'u12.zip').write_bytes(b'changed archive')
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                collect(root,[dict(unit=12,artifact_id=1,zip_sha256='0'*64)])


if __name__=='__main__':unittest.main()
