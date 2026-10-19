import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('campaign_summary',Path(__file__).parent/'profiling/summarize_p0_campaign.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary)

class ActiveWindowTest(unittest.TestCase):
    def test_startup_and_between_request_samples_are_excluded(self):
        rows=[dict(begin_monotonic_s=10,end_monotonic_s=20),dict(begin_monotonic_s=30,end_monotonic_s=40)]
        samples=[dict(monotonic_s=x) for x in [0,10,15,20,25,30,40,45]]
        self.assertEqual([x['monotonic_s'] for x in summary.active_samples(samples,rows)],[10,15,20,30,40])
        self.assertEqual(summary.active_samples([dict(monotonic_s=25)],rows),[])

    def test_missing_values_stay_unknown_and_io_units_unchanged(self):
        self.assertIsNone(summary.span([None]))
        self.assertEqual(summary.numeric('N/A'),None)
        self.assertEqual(summary.numeric('1380 MHz'),1380)
        self.assertEqual(summary.numbers('read_bytes: 4096\npgmajfault 7\n'),{'read_bytes':4096,'pgmajfault':7})

    def test_decode_windows_exclude_prefill_and_unknown_ttft(self):
        rows=[dict(begin_monotonic_s=10,end_monotonic_s=50,ttft_ms=12000),
              dict(begin_monotonic_s=60,end_monotonic_s=70,ttft_ms=None)]
        self.assertEqual(summary.phase_windows(rows,'decode_after_first_token'),
                         [dict(begin_monotonic_s=22,end_monotonic_s=50)])
        self.assertEqual(summary.phase_windows(rows,'prefill_to_first_token'),
                         [dict(begin_monotonic_s=10,end_monotonic_s=22)])
        rows[0]['ttft_ms']=50000
        with self.assertRaises(ValueError):summary.phase_windows(rows,'decode_after_first_token')

if __name__=='__main__':unittest.main()
