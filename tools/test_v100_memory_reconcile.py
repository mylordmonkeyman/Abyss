import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('reconcile_memory', Path(__file__).parent/'profiling/reconcile_p0_memory.py')
reconcile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reconcile)


class MemoryReconciliation(unittest.TestCase):
    def test_shared_backing_aliases_count_once_and_private_stays_unknown(self):
        maps = ('1000-3000 rw-s 00000000 00:01 7 /dev/zero (deleted)\n'
                '5000-7000 rw-s 00000000 00:01 7 /dev/zero (deleted)\n'
                '9000-a000 rw-p 00000000 00:00 0\n')
        result = reconcile.backing_union([(0x1000,4096),(0x5000,4096),(0x9000,4096)],maps)
        self.assertEqual(result['shared_backing_page_bytes'],4096)
        self.assertEqual(result['virtual_page_union_bytes'],12288)
        self.assertEqual(result['unresolved_virtual_page_bytes'],4096)
        with self.assertRaises(ValueError):reconcile.backing_union([(0x8000,4096)],maps)

    def test_pss_partition_conserves_host_total_without_adding_pin_bytes(self):
        host = dict(anonymous=20,shared_tmpfs=30,file_cache_excluding_shmem=20,free=10,buffers=2,
                    slab=3,kernel_stacks=1,page_tables=1,unassigned_kernel_driver_reserved_and_other=13)
        w = dict(host_disjoint_bytes=host,host_total_bytes=100,
                 overlapping_process_view_bytes=dict(Pss_Anon=5,Pss_Shmem=25,Pss_File=7,Pss=37))
        r = reconcile.physical_partition(w)
        self.assertEqual(r['sum_bytes'],100)
        self.assertEqual(r['bytes']['other_shared_tmpfs'],5)
        w['overlapping_process_view_bytes']['Pss']=38
        with self.assertRaises(ValueError):reconcile.physical_partition(w)

    def test_failed_free_and_timestamp_preserve_live_range(self):
        events = [dict(api='cudaMalloc',ptr=4096,bytes=100,result=0,monotonic_ns=10),
                  dict(api='cudaFree',ptr=4096,bytes=0,result=1,monotonic_ns=20),
                  dict(api='cudaFree',ptr=4096,bytes=0,result=0,monotonic_ns=30)]
        self.assertEqual(reconcile.live_at(events,25)['device'][4096]['bytes'],100)
        self.assertEqual(reconcile.live_at(events,30)['device'],{})

    def test_application_components_conserve_without_readding_borrowed_workspace(self):
        mallocs=[dict(seq=1,bytes=2*1024**3,caller=42),dict(seq=2,bytes=4096,caller=43)]
        symbols=dict(allocation_callers=[dict(runtime_caller=42,addr2line=dict(stdout='strata::core::Glm5Model::fast_setup(err)\n??:?'))])
        r=reconcile.attribute_device(mallocs,symbols)
        self.assertEqual(r['sum_bytes'],2*1024**3+4096)
        self.assertEqual(r['bytes']['expert_pool_including_borrowable_prompt_tail'],2*1024**3)
        self.assertEqual(r['bytes']['other_or_unresolved_application_allocations'],4096)


if __name__=='__main__':unittest.main()
