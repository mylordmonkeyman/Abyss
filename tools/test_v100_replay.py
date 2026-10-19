import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('base_replay',Path(__file__).parent/'bench/v100_base_replay.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


class FixedReplay(unittest.TestCase):
    def test_expected_count_finite_probability_and_protocol(self):
        self.assertEqual(replay.parse_lp('LP -12.5 128 0',128)['sum_log_probability'],-12.5)
        for line in ['LP nan 128 0','LP -1 127 0','LP -1 128 2','DONE 128','LP inf 128 0']:
            with self.assertRaises(ValueError): replay.parse_lp(line,128)


if __name__=='__main__': unittest.main()
