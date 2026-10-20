"""A root-cause control must intervene at every eligible layer/chunk."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from p1_cublas_selection_control import validate_controls
from p1_integrated_quality import GEOMETRIES

class SelectionControls(unittest.TestCase):
    def test_population_kind_and_counter_integrity(self):
        geometry=GEOMETRIES['128k'];kind='reference-order'
        records=[dict(layer=str(layer),p0=str(pos),kind=kind,T='2048',rows='2048',ordered_equal_rows='2040',
            membership_equal_rows='2045',overlap=str(2048*512-3)) for layer in range(2,43,2) for pos in geometry['candidate_positions']]
        validate_controls(records,kind,geometry)
        with self.assertRaisesRegex(ValueError,'incomplete'):validate_controls(records[:-1],kind,geometry)
        with self.assertRaisesRegex(ValueError,'wrong'):validate_controls(records,'reference-membership',geometry)
        records[0]['ordered_equal_rows']='2046'
        with self.assertRaisesRegex(ValueError,'accounting'):validate_controls(records,kind,geometry)

if __name__=='__main__':unittest.main()
