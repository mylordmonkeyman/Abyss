"""Model admission must preserve failed comparators and reject incomplete evidence."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent/'profiling'))
from summarize_p1_head_to_head import check_quality,checked_zip,MODES,LABELS,PAIRS
from p1_indexer_head_to_head import ORDER,POLICY
from p1_integrated_quality import GEOMETRIES

def fixture(name='128k'):
    geometry=copy.deepcopy(GEOMETRIES[name]);start=geometry['score_start']+1
    state=dict(status='complete diagnostic model comparison',source_sha='a'*40,binary_sha256='b'*64,
        cpu_lane=0,geometry=geometry,correction_policy=POLICY,order=list(ORDER),arms={k:{} for k in LABELS},
        dumps={k:dict(bytes=16547840,sha256=('d' if k.startswith('cublas') else 'c')*64,
            path='/actions-runner/abyss-p1-v9/indexer-head-to-head/7-1/'+name+'/'+k+'.f32') for k in LABELS},
        comparisons=[],qualified_mean_kl_gate_pass=True,original_cublas_mean_kl_gate_pass=False)
    for a,b in sorted(PAIRS):
        kl=.01 if a.startswith('f32') and b.startswith('cublas') else 0.
        state['comparisons'].append(dict(reference=a,candidate=b,mean_kl=kl,max_kl=kl,rows=32,top1_matches=32,
            positions=[dict(predicted_token_index_zero_based=p,kl=kl) for p in range(start,start+32)]))
    return state

def check(state,name='128k'):return check_quality(state,name,'a'*40,'b'*64,7)

class Evidence(unittest.TestCase):
    def test_failed_original_is_explicit_in_both_geometries(self):
        for name in GEOMETRIES:
            passed,original,tails=check(fixture(name),name)
            self.assertTrue(passed);self.assertFalse(original)
            self.assertEqual(len(tails),10)
            self.assertEqual(sum(not r['passes_mean_kl_limit'] for r in tails),2)

    def test_corrected_failure_is_never_waived_or_subtracted(self):
        state=fixture()
        for i in (0,1):state['dumps']['corrected-'+str(i)]['sha256']='e'*64
        r=next(r for r in state['comparisons'] if r['reference']=='f32-0' and r['candidate']=='corrected-0')
        r.update(mean_kl=.001,max_kl=.001)
        for p in r['positions']:p['kl']=.001
        with self.assertRaisesRegex(ValueError,'disposition'):check(state)
        state['qualified_mean_kl_gate_pass']=False
        self.assertFalse(check(state)[0])

    def test_missing_duplicate_geometry_repeat_or_invalid_kl_rejected(self):
        for mutate in (lambda s:s['comparisons'].pop(),
            lambda s:s['comparisons'].append(copy.deepcopy(s['comparisons'][0])),
            lambda s:s['geometry'].update(score_start=1),
            lambda s:s['dumps']['corrected-1'].update(sha256='e'*64),
            lambda s:s['dumps']['f32-0'].update(bytes=1),
            lambda s:s['comparisons'][0]['positions'][0].update(kl=float('nan')),
            lambda s:s['comparisons'][0].update(max_kl=99),
            lambda s:s.update(cpu_lane=28),lambda s:s.update(status='running model head-to-head')):
            with self.subTest(mutate=mutate):
                state=fixture();mutate(state)
                with self.assertRaises(ValueError):check(state)

    def test_archive_digest_traversal_symlink_and_duplicate_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'evidence.zip'
            for kind in ('valid','traversal','symlink','duplicate'):
                with warnings.catch_warnings(),zipfile.ZipFile(path,'w') as z:
                    warnings.simplefilter('ignore',UserWarning)
                    if kind=='symlink':
                        info=zipfile.ZipInfo('link');info.create_system=3;info.external_attr=0o120777<<16;z.writestr(info,'outside')
                    else:
                        z.writestr('../outside' if kind=='traversal' else 'source.txt','data')
                        if kind=='duplicate':z.writestr('source.txt','other data')
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                if kind=='valid':
                    with checked_zip(path,digest) as z:self.assertEqual(z.read('source.txt'),b'data')
                    with self.assertRaisesRegex(ValueError,'digest'):checked_zip(path,'0'*64)
                else:
                    with self.assertRaises(ValueError):checked_zip(path,digest)

if __name__=='__main__':unittest.main()
