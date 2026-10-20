#!/usr/bin/env python3
"""Verify and compact the CPU-lane-off baseline diagnostic, never P1 admission."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('diagnostic artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        def read(name):return json.loads(z.read(name))
        state=read('f32-triage/triage.json');source=z.read('source-sha.txt').decode().strip()
        if state['source_sha']!=source or str(state['source_run'])!=str(run):raise ValueError('source/run mismatch')
        if state.get('candidate_control'):raise ValueError('this summary is the two-arm baseline control only')
        if state['changed_variable']!={'STRATA_GLM_CPU_LANE':'0'}:raise ValueError('wrong diagnostic intervention')
        if set(state['arms'])!={'f32-0','f32-1'} or set(state['dumps'])!=set(state['arms']):raise ValueError('incomplete baseline repeat')
        if state['geometry']['score_start']!=131039 or state['geometry']['token_count']!=131072:raise ValueError('wrong failed geometry')
        if len({(d['bytes'],d['sha256']) for d in state['dumps'].values()})!=1:raise ValueError('baseline is not bitwise identical')
        c=state['comparison']
        if c['mean_kl']!=0 or c['max_kl']!=0 or c['top1_matches']!=32:raise ValueError('zero-drift claim unsupported')
        if [p['predicted_token_index_zero_based'] for p in c['positions']]!=list(range(131040,131072)):raise ValueError('scored population changed')
        telemetry={}
        for arm,record in state['arms'].items():
            env=read('f32-triage/'+arm+'/invocation.json')['environment']
            if env['STRATA_GLM_CPU_LANE']!='0' or env['STRATA_DS4_IDX']!='1' or env['ABYSS_V100_IDX']!='f32':raise ValueError('wrong arm environment')
            rows=record.pop('dispatch_records')
            if not rows or any(r['scorer']!='f32' for r in rows):raise ValueError('non-F32/missing dispatch')
            record['dispatch_summary']=dict(records=len(rows),candidate_calls=0)
            memory=read('f32-triage/'+arm+'/memory_summary.json')
            if memory['samples']<=0:raise ValueError('missing sustained telemetry')
            telemetry[arm]=dict(memory=memory,thermal=read('f32-triage/'+arm+'/thermal.json'))
        platform=read('platform-probe.json')
        sm80=z.read('sm80-binary-sha256.txt').decode().split()[0]
        if len(sm80)!=64 or 'sm_80' not in z.read('sm80-object-images.txt').decode():raise ValueError('missing SM80 binary/image evidence')
        return dict(schema=1,run=run,artifact_zip_sha256=digest,source_sha=source,diagnostic=state,telemetry=telemetry,
            platform=dict(inventory=platform,sm80_binary_sha256=sm80,sm80_disposition='Compile/link and embedded SM80 image pass; no SM80 hardware runtime test',
                          hip_disposition=platform['hip_disposition']),
            disposition='CPU-lane-off F32 is bitwise repeatable in these two runs. CPU/GPU expert execution is implicated as an investigation axis, not proven root cause. '
            'Original-upstream CPU-lane-28 control remains separate. No default-setting acceptance or P1 promotion.',
            model_identity=read('f32-triage/model-identity.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path)
    p.add_argument('--sha256',required=True);p.add_argument('--run',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
