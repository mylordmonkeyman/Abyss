import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent/'v100'))
from memory_worksheet import kb_fields,worksheet

class WorksheetTest(unittest.TestCase):
    def sample(self):
        return dict(MemTotal=1000,MemFree=100,AnonPages=400,Shmem=50,Cached=250,
                    Buffers=10,Slab=20,KernelStack=5,PageTables=5,MemAvailable=280,
                    AnonHugePages=300,Mlocked=200,SReclaimable=10)

    def test_overlapping_views_do_not_inflate_total(self):
        r=worksheet(self.sample(),{'Rss':650,'Anonymous':400})
        self.assertEqual(sum(r['host_disjoint_bytes'].values()),1000)
        self.assertEqual(r['host_disjoint_bytes']['file_cache_excluding_shmem'],200)
        self.assertEqual(r['host_disjoint_bytes']['unassigned_kernel_driver_reserved_and_other'],210)
        self.assertEqual(r['headroom_bytes'],280)
        self.assertEqual(r['overlapping_process_view_bytes']['Rss'],650)

    def test_inconsistent_sample_rejected(self):
        m=self.sample();m['MemTotal']=500
        with self.assertRaises(ValueError):worksheet(m)
        m=self.sample();m['Cached']=0
        with self.assertRaises(ValueError):worksheet(m)
        with self.assertRaises(ValueError):worksheet({})

    def test_units_and_non_memory_lines(self):
        self.assertEqual(kb_fields('Rss: 2 kB\nVmFlags: rd wr\nCount: 3\n'),{'Rss':2048})

if __name__=='__main__':unittest.main()
