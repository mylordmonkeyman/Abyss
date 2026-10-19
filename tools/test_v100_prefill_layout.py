import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prefill_layout',ROOT/'tools/profiling/p0_prefill_layout.py')
layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)


def extract(source,name):
    begin=source.index(name)
    begin=source.rfind('\n',0,begin)+1
    brace=source.index('{',begin);depth=1;end=brace+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[begin:end]


class PrefillLayout(unittest.TestCase):
    def test_union_views_overlap_and_current_extra_routes(self):
        old=layout.layout(32768,'historical');new=layout.layout(32768,'current')
        self.assertEqual(old['borrowed_bytes'],7350343424)
        self.assertEqual(new['borrowed_bytes']-old['borrowed_bytes'],1048576)
        self.assertEqual(old['union_bytes'],max(v['bytes'] for v in old['union_views'].values()))
        self.assertLess(old['union_bytes'],sum(v['bytes'] for v in old['union_views'].values()))
        self.assertEqual(old['borrowed_bytes'],sum(v['aligned_bytes'] for v in old['outer']))
        self.assertEqual(layout.layout(640,'historical')['prestage_slots'],0)
        self.assertEqual(layout.layout(1024,'historical')['prestage_slots'],16)
        self.assertEqual(layout.layout(8192,'historical')['prestage_slots'],64)
        c=layout.Carve();c.take('unaligned',257);c.take('next',1)
        self.assertEqual(c.rows[1]['offset'],512)
        with self.assertRaises(ValueError):layout.layout(600,'historical')

    @unittest.skipUnless(shutil.which('c++'),'host compiler required')
    def test_python_offsets_against_actual_host_compiled_source_sizing(self):
        version=(ROOT/'VERSION').read_text().strip()
        self.assertIn(version,('1.3.28','1.3.31'),'audit layout for a new upstream version')
        base={'1.3.28':'historical','1.3.31':'current'}[version]
        source=(ROOT/'src/core/glm_prefill.cu').read_text()
        geometry=extract((ROOT/'include/strata/core/glm_model.hpp').read_text(),'struct Glm5Geometry')+';'
        definitions='\n'.join(extract(source,'struct '+s)+';' for s in
                               ('Carve','KdaBufs','DsaBufs','Ds4Bufs','DenseBufs','MoeBufs'))
        functions='\n'.join(extract(source,n+'(') for n in ('carve_kda','carve_dsa',
            'carve_ds4','carve_dense','moe_window_rows','carve_moe','mixer_sub','chunk_bytes'))
        program='''
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
namespace strata { class GgufFile; }
constexpr int kSub=256,kLightOff=6144,kLightMax=(8192-kLightOff)/3,kLightRows=32;
namespace mmq {
struct block { float scales[4]; int8_t values[128]; };
static_assert(sizeof(block)==144);
size_t q8_bytes(int64_t rows,int64_t cols) {
 return rows*((cols+511)/512*512)*sizeof(block)/128+128*sizeof(block);
}
}
'''+geometry+'\n'+definitions+'\n'+functions+'''
int main() {
 Glm5Geometry g;g.arch=1;g.n_embd=4096;g.n_expert=256;g.n_exp_used=6;
 g.n_head=64;g.kda_head_dim=32;g.q_lora=1024;g.kv_lora=512;
 g.qk_nope=g.v_head=512;g.idx_heads=64;g.idx_top_k=512;
 for(int t : {512,640,1024,4096,8192,32768}) {
  auto p=chunk_bytes(g,t,false,true,false,true,135168/4,0,false);
  printf("%d %zu %zu %d\\n",t,p.first,p.second,mixer_sub(t,g,false,true,false,true,135168/4,0));
 }
}
'''
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);(d/'sizes.cpp').write_text(program)
            subprocess.run(['c++','-std=c++17',str(d/'sizes.cpp'),'-o',str(d/'sizes')],
                           check=True,capture_output=True)
            env={'PATH':str(Path(shutil.which('c++')).parent)}
            result=subprocess.run([str(d/'sizes')],check=True,capture_output=True,text=True,env=env)
            for line in result.stdout.splitlines():
                t,p,u,sub=map(int,line.split());a=layout.layout(t,base)
                self.assertEqual((p,u,sub),(a['persistent_bytes'],a['union_bytes'],a['mixer_sub_batch']))


if __name__=='__main__':unittest.main()
