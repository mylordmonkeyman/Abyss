"""Overhead controls retain their population and do not admit an optimization."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p3_observer_control import contrasts,ORDER


class ObserverControl(unittest.TestCase):
    def arms(self):
        return [dict(mode=mode,requests=[dict(repetition=i,warmup=i==0,scored_tokens=32,
            wire_ms=(999 if i==0 else {'plain':100,'observer':110,'trace':220}[mode]))
            for i in range(4)]) for mode in ORDER]

    def test_reversed_control_contrasts_exclude_warmup(self):
        result=contrasts(self.arms())
        self.assertEqual(len(result),2)
        for r in result:
            self.assertAlmostEqual(r['observer_vs_plain_pct'],10)
            self.assertAlmostEqual(r['trace_vs_observer_pct'],100)
            self.assertNotIn('performance_claim_admitted',r)

    def test_incomplete_or_reordered_controls_rejected(self):
        arms=self.arms()
        with self.assertRaisesRegex(ValueError,'arm population'):contrasts(arms[:-1])
        bad=copy.deepcopy(arms);bad[0]['mode']='observer'
        with self.assertRaisesRegex(ValueError,'arm population'):contrasts(bad)
        for field,value in [('scored_tokens',31),('warmup',True),('wire_ms',0),('wire_ms',float('nan'))]:
            bad=copy.deepcopy(arms);bad[0]['requests'][1][field]=value
            with self.assertRaisesRegex(ValueError,'fixed-token requests'):contrasts(bad)


if __name__=='__main__':unittest.main()
