"""Lifecycle and real ELF-interposition tests; no CUDA toolkit/GPU needed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent/'v100'))
from allocation_ledger import reconstruct, union_bytes
ROOT=Path(__file__).resolve().parents[1]


def events(spec):
    return [dict(seq=i+1,pid=7,monotonic_ns=i,api=api,ptr=p,bytes=n,
                 flags=0,result=r,alias=a) for i,(api,p,n,r,a) in enumerate(spec)]


class LedgerTest(unittest.TestCase):
    def test_overlaps_and_aliases_are_not_additive(self):
        e=events([('cudaHostAlloc',4096,4096,0,0),
                  ('cudaHostRegister',6144,4096,0,0),
                  ('cudaHostGetDevicePointer',4096,0,0,65536),
                  ('cudaMalloc',65536,100,0,0)])
        r=reconstruct(e)
        self.assertEqual(r['peak']['host_unique_virtual'],6144)
        self.assertEqual(r['peak']['host_virtual_page_footprint'],8192)
        self.assertEqual(r['peak']['device'],100)
        self.assertEqual(len(r['device_aliases']),1)

    def test_failed_free_preserves_range_and_success_removes_it(self):
        r=reconstruct(events([('cudaMalloc',32,100,0,0),('cudaFree',32,0,1,0),
                              ('cudaFree',32,0,0,0)]))
        self.assertEqual(r['snapshots'][0]['device'],100)
        self.assertEqual(r['snapshots'][-1]['device'],0)
        self.assertEqual(len(r['failures']),1)
        self.assertEqual(r['ledger_errors'],[])

    def test_bad_lifecycle_and_missing_events(self):
        e=events([('cudaMalloc',32,100,0,0),('cudaFree',64,0,0,0)])
        self.assertTrue(reconstruct(e)['ledger_errors'])
        e[0]['seq']=2
        with self.assertRaises(ValueError):reconstruct(e)
        with self.assertRaises(ValueError):reconstruct([])
        with self.assertRaises(ValueError):reconstruct(events([('unknown',0,0,0,0)]))

    def test_page_union(self):
        self.assertEqual(union_bytes([(1,1),(4095,2)],4096),8192)
        self.assertEqual(union_bytes([(100,50),(110,5),(145,10)]),55)
        with self.assertRaises(ValueError):reconstruct([],3)

    @unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
    def test_real_preload_nested_calls_failure_and_pointer_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            (d/'fake.cpp').write_text('''
#include <cstdlib>
extern "C" int cudaHostAlloc(void** p, size_t n, unsigned) { *p=malloc(n); return 0; }
extern "C" int cudaMallocHost(void** p, size_t n) { return cudaHostAlloc(p,n,0); }
extern "C" int cudaFreeHost(void* p) { free(p); return 0; }
extern "C" int cudaMalloc(void** p, size_t n) { if(n==999) return 2; *p=malloc(n); return 0; }
extern "C" int cudaFree(void* p) { free(p); return 0; }
extern "C" int cudaHostRegister(void*, size_t, unsigned) { return 0; }
extern "C" int cudaHostUnregister(void*) { return 0; }
extern "C" int cudaHostGetDevicePointer(void** d, void* h, unsigned) { *d=h; return 0; }
''')
            (d/'main.cpp').write_text('''
#include <cstddef>
extern "C" int cudaMallocHost(void**,size_t);
extern "C" int cudaFreeHost(void*);
extern "C" int cudaMalloc(void**,size_t);
extern "C" int cudaFree(void*);
extern "C" int cudaHostGetDevicePointer(void**,void*,unsigned);
extern "C" int cudaHostRegister(void*,size_t,unsigned);
extern "C" int cudaHostUnregister(void*);
int main() { void *h=nullptr,*d=nullptr,*a=nullptr;
cudaMallocHost(&h,4096); cudaHostRegister(h,4096,0);
cudaHostGetDevicePointer(&a,h,0); cudaMalloc(&d,999); cudaMalloc(&d,128);
cudaFree(d); cudaHostUnregister(h); cudaFreeHost(h); }
''')
            commands=[['c++','-shared','-fPIC',str(d/'fake.cpp'),'-o',str(d/'libfake.so')],
                      ['c++','-shared','-fPIC','-std=c++17','-pthread',
                       str(ROOT/'tools/v100/cuda_alloc_observer.cpp'),'-ldl','-o',str(d/'observer.so')],
                      ['c++',str(d/'main.cpp'),'-L'+temp,'-lfake','-Wl,-rpath,'+temp,'-o',str(d/'main')]]
            for cmd in commands:subprocess.run(cmd,check=True,capture_output=True)
            env=dict(os.environ,LD_PRELOAD=str(d/'observer.so'),ABYSS_CUDA_ALLOC_LOG=str(d/'events'))
            subprocess.run([str(d/'main')],check=True,env=env)
            raw=[json.loads(s) for s in (d/'events').read_text().splitlines()]
            r=reconstruct(raw)
            self.assertEqual(r['event_count'],9) # nested cudaHostAlloc suppressed
            self.assertEqual(r['peak']['host_unique_virtual'],4096)
            self.assertEqual(r['peak']['device'],128)
            self.assertTrue(r['capture_complete'])
            self.assertEqual(len(r['failures']),1)
            self.assertEqual(r['ledger_errors'],[])
            self.assertTrue(all(not v for v in r['live_ranges'].values()))
            self.assertEqual(r['device_aliases'],[])

if __name__=='__main__':unittest.main()
