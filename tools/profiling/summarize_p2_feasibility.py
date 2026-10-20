#!/usr/bin/env python3
"""Digest-checked selected-base P2 prerequisites, without a production claim."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

def summarize(archive,digest,run):
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=digest:raise ValueError('P2 artifact digest mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist())>64*1024**2:raise ValueError('P2 artifact cap exceeded')
        frozen=json.loads(z.read('frozen-replay.json'));source=z.read('source-sha.txt').decode().strip()
        if source!=frozen['source_sha'] or str(run)!=str(frozen['source_run']):raise ValueError('source/run mismatch')
        if len([r for r in frozen['results'] if 'real_stage' in r])!=33 or not all(r.get('pass') for r in frozen['results']):
            raise ValueError('frozen replay population/failure')
        suites={}
        for name,count in [('qkpv-parity.jsonl',32),('latent-parity.jsonl',21)]:
            rows=[json.loads(line) for line in z.read(name).decode().splitlines() if line.startswith('{')]
            if not rows or not all(r.get('pass') for r in rows) or rows[-1].get('cases')!=count:raise ValueError('synthetic prerequisite failure')
            suites[name]=dict(summary=rows[-1],records=len(rows),results=rows[:-1])
        for tool in ('memcheck','synccheck'):
            if 'ERROR SUMMARY: 0 errors' not in z.read('frozen-'+tool+'.log').decode():raise ValueError('missing sanitizer success')
        return dict(schema=1,run=run,source_sha=source,artifact_zip_sha256=digest,frozen_replay=frozen,synthetic=suites,
            scope='Selected-base isolated P2 prerequisites pass on physical SM70. No new fused kernel, full P2 latent integration, '
            'learned output-weight replay, model KL or speed admission is established by this checkpoint. '
            'Preserve historical capture authenticity limitations; hardware counters/QPI remain unavailable/unmeasured.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--sha256',required=True)
    p.add_argument('--run',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(summarize(a.archive,a.sha256,a.run),indent=2)+'\n')
