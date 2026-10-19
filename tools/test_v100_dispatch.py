import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).parent/"v100"))
from dispatch_ledger import parse
ROOT=Path(__file__).resolve().parents[1]

@unittest.skipUnless(shutil.which('c++'),'C++ compiler required')
class DispatchTest(unittest.TestCase):
    def test_exact_gather_ids_within_and_across_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            (d/'main.cpp').write_text('''
#include "strata/core/v100_dispatch.hpp"
#include <cassert>
int main() {
    using namespace strata::v100dispatch;
    Scope scope({1,7,4,1024,0});
    InputRows a{{0,1,1,2,1,3},{0,0,0,0},0};
    assert(a.count(0,4,17)); assert(a.unique==3);
    assert(a.count(4,2,17)); assert(a.unique==4);
    assert(!a.count(-1,2,17)); assert(!a.count(5,2,17));
    a.tokens[0]=4; assert(!a.count(0,1,17)); assert(a.unique==4);
    InputRows b{{0,0},{0,0,0,0},0};
    assert(b.count(0,2,17)); assert(b.unique==1);
}
''')
            subprocess.run(['c++','-std=c++17','-pthread','-I'+str(ROOT/'include'),
                            str(d/'main.cpp'),'-o',str(d/'test')],check=True,capture_output=True)
            p=subprocess.run([str(d/'test')],env=dict(os.environ,ABYSS_V100_DISPATCH='1'),
                             capture_output=True,text=True,check=True)
            rows=parse(p.stderr)['host_records']
            self.assertEqual(sum(r['calls'] for r in rows),3)
            self.assertEqual(sum(r['rows'] for r in rows),6)
            self.assertEqual({(r['nt'],r['path']) for r in rows},
                             {(4,'mmq_input_distinct:3'),(2,'mmq_input_distinct:2'),
                              (2,'mmq_input_distinct:1')})

    def test_shared_ledger_thread_context_scope_and_default_off(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp)
            (d/'worker.cpp').write_text('''
#include "strata/core/v100_dispatch.hpp"
void worker(strata::v100dispatch::Context c) {
    strata::v100dispatch::Scope scope(c);
    for(int i=0;i<100;i++) strata::v100dispatch::record("cpu_gu:iq256",16,3,2,32);
}
''')
            (d/'main.cpp').write_text('''
#include "strata/core/v100_dispatch.hpp"
#include <thread>
void worker(strata::v100dispatch::Context);
int main() {
    using namespace strata::v100dispatch;
    Context c{1,7,512,1024,0};
    std::thread a(worker,c),b(worker,c); a.join(); b.join();
    { Scope s({2,8,1,2048,51});record("request_issue",16,1); }
    record("setup",-1,0);
}
''')
            subprocess.run(['c++','-std=c++17','-pthread','-I'+str(ROOT/'include'),
                            str(d/'main.cpp'),str(d/'worker.cpp'),'-o',str(d/'test')],
                           check=True,capture_output=True)
            env=dict(os.environ,ABYSS_V100_DISPATCH='0')
            p=subprocess.run([str(d/'test')],env=env,capture_output=True,text=True,check=True)
            self.assertEqual(p.stderr,'')
            env['ABYSS_V100_DISPATCH']='1'
            p=subprocess.run([str(d/'test')],env=env,capture_output=True,text=True,check=True)
            rows=[json.loads(s.removeprefix('V100_DISPATCH ')) for s in p.stderr.splitlines()
                  if s.startswith('V100_DISPATCH ')]
            self.assertEqual(len(rows),3) # one shared ledger across translation units
            cpu=next(r for r in rows if r['path']=='cpu_gu:iq256')
            self.assertEqual((cpu['calls'],cpu['rows'],cpu['logical_weight_bytes']),(200,400,6400))
            self.assertEqual((cpu['phase'],cpu['layer'],cpu['position'],cpu['nt']),(1,7,1024,3))
            setup=next(r for r in rows if r['path']=='setup')
            self.assertEqual((setup['phase'],setup['layer'],setup['position']),(0,-1,-1))
            end=[json.loads(s.removeprefix('V100_DISPATCH_END ')) for s in p.stderr.splitlines()
                 if s.startswith('V100_DISPATCH_END ')]
            self.assertEqual(len(end),1)
            self.assertEqual(end[0]['dropped_calls'],0)

class ParserTest(unittest.TestCase):
    def test_service_join_and_actual_nt_group(self):
        base=dict(phase=2,layer=3,chunk_tokens=1,position=-1,request_seq=17,type=16,nt=1,
                  calls=1,rows=8,logical_weight_bytes=256)
        rows=[dict(base,path='request_issue',position=2048),dict(base,path='cpu_gu:ggml_vec_dot'),
              dict(base,phase=1,position=0,request_seq=0,path='cpu_gu:iq256',nt=5)]
        text=''.join('V100_DISPATCH '+json.dumps(r)+'\n' for r in rows)
        text+='V100_DISPATCH_END '+json.dumps(dict(keys=3,dropped_calls=0))
        r=parse(text)
        self.assertEqual(r['host_records'][1]['position'],2048)
        self.assertEqual(r['unresolved_decode_service'],[])
        self.assertIn('T2_8',[x['token_group'] for x in r['by_layer_type_token_group']])
        with self.assertRaises(ValueError):parse(text.replace('"dropped_calls": 0','"dropped_calls": 1'))
        with self.assertRaises(ValueError):parse(text.split('V100_DISPATCH_END')[0])

if __name__=='__main__':unittest.main()
