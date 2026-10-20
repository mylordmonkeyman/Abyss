#!/usr/bin/env python3
"""Recompute digest-checked live capture metrics and explain internal rank swaps."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import tempfile
import zipfile
import numpy as np
from p1_live_capture import compare_capture

def summarize(archive,digest,run,diagnostic_failure=False):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('live artifact digest mismatch')
    with zipfile.ZipFile(archive) as z,tempfile.TemporaryDirectory(prefix='p1-capture-recheck-') as tmp:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('live artifact cap exceeded')
        result=json.loads(z.read('capture-result.json'));source=z.read('source-sha.txt').decode().strip()
        if result['source_sha']!=source or str(result['run'])!=str(run):raise ValueError('source/run mismatch')
        root=Path(tmp)
        prefix='cublas/p1-indexer-capture/'
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or (info.external_attr>>16)&0o170000==0o120000:raise ValueError('unsafe artifact member')
            if info.filename.startswith(prefix) and not info.is_dir():
                target=root/info.filename[len(prefix):];target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(z.read(info))
        direct=compare_capture(root,result.get('capture_position',98304));recorded=result['direct']
        if direct['files']!=recorded['files'] or len(direct['samples'])!=len(recorded['samples']):raise ValueError('raw capture identity changed')
        for a,b in zip(direct['samples'],recorded['samples']):
            if set(a)!=set(b):raise ValueError('capture schema changed')
            for key,value in a.items():
                if isinstance(value,float):
                    if not np.isclose(value,b[key],rtol=1e-10,atol=1e-20):raise ValueError('recorded metric mismatch: '+key)
                elif value!=b[key]:raise ValueError('recorded metric mismatch: '+key)
        swaps=[]
        for row in direct['samples']:
            if row['selection_identical']:continue
            stem=root/f"layer-{row['layer']}"/f"t-{row['token_offset']}"
            arrays={name:np.fromfile(str(stem)+'-'+name,dtype='<i4' if name.endswith('i32') else '<f4')
                for name in ('reference-ids.i32','candidate-ids.i32','reference-score.f32','candidate-score.f32')}
            ri,ci,rs,cs=(arrays[k] for k in ('reference-ids.i32','candidate-ids.i32','reference-score.f32','candidate-score.f32'))
            changes=[]
            for i in np.flatnonzero(ri!=ci):
                gap=abs(float(rs[ri[i]])-float(rs[ci[i]]))
                if row['overlap']==512 and gap>2*row['score_max_abs']:raise ValueError('internal order change outside score-error envelope')
                changes.append(dict(rank_1based=int(i)+1,reference_id=int(ri[i]),candidate_id=int(ci[i]),
                    reference_gap=gap,candidate_gap=abs(float(cs[ri[i]])-float(cs[ci[i]]))))
            swaps.append(dict(layer=row['layer'],token_offset=row['token_offset'],membership_overlap=row['overlap'],
                score_max_error=row['score_max_abs'],changes=changes))
        q=result['quality']
        if len(q['positions'])!=32:raise ValueError('wrong scored population')
        quality_pass=q['mean_kl']<=.0005
        if not quality_pass and not diagnostic_failure:raise ValueError('bounded model KL gate failed')
        telemetry={mode:dict(thermal=json.loads(z.read(mode+'/thermal.json')),memory=json.loads(z.read(mode+'/memory_summary.json')))
                   for mode in ('f32','cublas')}
        return dict(schema=1,run=run,source_sha=source,artifact_zip_sha256=digest,binary_sha256=result['binary_sha256'],
            model_identity=result['model_identity'],fixture_sha256=result['fixture_sha256'],quality=q,telemetry=telemetry,
            direct=direct,internal_rank_changes=swaps,geometry=result.get('geometry'),cpu_lane=result.get('cpu_lane',28),
            capture_position=result.get('capture_position',98304),passes_written_mean_kl_limit=quality_pass,
            disposition='Failed diagnostic; no admission' if not quality_pass else 'Bounded capture pass only; no automatic promotion',
            scope='Independent recomputation from raw paired scores/IDs/attention. Membership and ordering '
            'differences are traced separately; a diagnostic failure is never admission. No throughput claim from '
            'instrumented runs. Local-manifest model integrity is not publisher authenticity; counters/QPI remain unmeasured.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',required=True,type=int);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--diagnostic-failure',action='store_true',help='preserve failed quality evidence without admitting it')
    a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run,a.diagnostic_failure),indent=2)+'\n')
