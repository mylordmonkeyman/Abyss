"""Compile the actual host dispatch policy, including legacy and failed force modes."""
import pathlib
import shutil
import subprocess
import tempfile
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
class IndexerPolicy(unittest.TestCase):
    @unittest.skipUnless(shutil.which('g++'),'host compiler unavailable')
    def test_dispatch_contract(self):
        source=r'''
#include "strata/kernels/v100_indexer_policy.hpp"
#include <cassert>
using namespace strata::kernels::dsv4::p1;
int main(){
 assert(parse(nullptr,nullptr)==Mode::automatic);
 assert(parse("auto",nullptr)==Mode::automatic);
 assert(parse("f32",nullptr)==Mode::f32);
 assert(parse("cublas",nullptr)==Mode::cublas);
 assert(parse("volta","1")==Mode::f32);
 assert(parse("bad","1")==Mode::f32);
 for(auto v:{"volta","bad"}){bool threw=false;try{parse(v,nullptr);}catch(const std::runtime_error&){threw=true;}assert(threw);}
 assert(eligible(64,128,512,2048,98304));
 assert(eligible(64,128,512,2048,120000));
 assert(!eligible(64,128,512,2048,65536));
 assert(!eligible(64,128,512,2048,120001));
 assert(!eligible(64,128,512,2047,98304));
 assert(!eligible(32,128,512,2048,98304));
 assert(!eligible(64,64,512,2048,98304));
 assert(!eligible(64,128,256,2048,98304));
 for(auto mode:{Mode::automatic,Mode::f32}) {
  assert(!use_cublas(mode,4,64,128,512,2048,98304));
  assert(backend_error(mode,true,0,0,64,128,512)==nullptr);
  assert(backend_error(mode,false,8,0,64,128,512)==nullptr);
 }
 for(int ratio:{0,128})assert(!use_cublas(Mode::cublas,ratio,64,128,512,2048,98304));
 for(int t:{0,1,256,1024,2047,2049,4096})assert(!use_cublas(Mode::cublas,4,64,128,512,t,98304));
 for(int pos:{0,2047,2048,2049,65536,98303,120001,131072})assert(!use_cublas(Mode::cublas,4,64,128,512,2048,pos));
 assert(use_cublas(Mode::cublas,4,64,128,512,2048,98304));
 assert(use_cublas(Mode::cublas,4,64,128,512,2048,120000));
 assert(backend_error(Mode::cublas,false,7,0,64,128,512)==nullptr);
 assert(std::strstr(backend_error(Mode::cublas,true,0,0,64,128,512),"HIP"));
 for(int major:{6,8,9})assert(std::strstr(backend_error(Mode::cublas,false,major,0,64,128,512),"SM70"));
 assert(std::strstr(backend_error(Mode::cublas,false,7,5,64,128,512),"SM70"));
 assert(backend_error(Mode::cublas,false,7,0,32,128,512));
 assert(backend_error(Mode::cublas,false,7,0,64,64,512));
 assert(backend_error(Mode::cublas,false,7,0,64,128,256));

}'''
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp);(p/'test.cpp').write_text(source)
            subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)
if __name__=='__main__':unittest.main()
