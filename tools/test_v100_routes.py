import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parent/'v100'))
from dispatch_ledger import parse as parse_dispatch
from route_ledger import parse
ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('c++'),'host compiler required')
class Routes(unittest.TestCase):
    def test_actual_emitter_roundtrip_and_bounded_capture_rejection(self):
        program=r'''
#include "strata/core/v100_dispatch.hpp"
#include <cassert>
int main(int argc,char**) {
 using namespace strata::v100dispatch;
 { Scope scope({1,0,2,0,0});
   InputRows in{{0,1,0,1},{0,0},0};std::vector<int8_t> lanes{0,3,6,5};
   assert(prefill_routes(in,lanes,2,17,18));
   lanes[0]=-1;assert(!prefill_routes(in,lanes,2,17,18));
   record("mmq_glu_not_selected",17,2);
 }
 { Scope scope({2,0,1,10,42});record("request_issue",17,1); }
 { Scope scope({2,0,1,-1,42});int ids[]{7,9};decode_route(2,ids,0,1,2,1,0); }
 if(argc>1)emit_route(std::string(16*1024*1024+1,'x'));
}
'''
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'test.cpp').write_text(program)
            subprocess.run(['c++','-std=c++17','-pthread','-I'+str(ROOT/'include'),
                            str(d/'test.cpp'),'-o',str(d/'test')],check=True,capture_output=True)
            env=dict(os.environ,ABYSS_V100_DISPATCH='1')
            p=subprocess.run([str(d/'test')],env=env,check=True,capture_output=True,text=True)
            r=parse(p.stderr,parse_dispatch(p.stderr))
            self.assertEqual(r['prefill'][0]['counts'],'10000010001010')
            self.assertEqual(r['prefill_route_totals']['cpu'],1)
            self.assertEqual(r['decode'][0]['position'],10)
            self.assertEqual(r['decode_route_totals']['promo'],1)
            self.assertEqual(r['decode_route_totals']['resident_mask'],0)
            p=subprocess.run([str(d/'test'),'overflow'],env=env,check=True,capture_output=True,text=True)
            with self.assertRaisesRegex(ValueError,'dropped'):parse(p.stderr,parse_dispatch(p.stderr))

    def test_parser_rejects_bad_conservation_and_unjoined_decode(self):
        d=dict(host_records=[dict(path='mmq_glu_not_selected',phase=1,chunk_tokens=1,calls=1,layer=0,position=0,type=17),
                             dict(path='request_issue',phase=2,layer=0,request_seq=42,position=10)])
        rows=[dict(kind='prefill',layer=0,position=0,tokens=1,k=2,gu_type=17,counts='1000001'),
              dict(kind='decode',layer=0,position=-1,request_seq=42,ids=[7,9],miss=0,fetch=1,cpu=2,promo=1,skip=0)]
        def wire():
            raw=list(map(json.dumps,rows))
            return '\n'.join('V100_ROUTE '+line for line in raw)+'\nV100_ROUTE_END '+json.dumps(
                dict(records=len(raw),payload_bytes=sum(len(s.encode()) for s in raw),dropped=0))
        self.assertEqual(parse(wire(),d)['decode'][0]['position'],10)
        rows[0]['counts']='0000001'
        with self.assertRaisesRegex(ValueError,'conserve'):parse(wire(),d)
        rows[0]['counts']='1000001';rows[1]['request_seq']=99
        with self.assertRaisesRegex(ValueError,'issue'):parse(wire(),d)
        rows[1]['request_seq']=42;rows[1]['miss']=1
        with self.assertRaisesRegex(ValueError,'overlap'):parse(wire(),d)


if __name__=='__main__':unittest.main()
