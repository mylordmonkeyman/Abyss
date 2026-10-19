#!/usr/bin/env python3
"""Reproduce ported P0b trace evidence from bounded digest-verified artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def collect(artifacts):
    results=[]
    for a in artifacts:
        archive=Path(a['archive'])
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=a['zip_sha256']:
            raise ValueError('trace artifact digest mismatch')
        with zipfile.ZipFile(archive) as z:
            def read(name):return json.loads(z.read(name))
            invocation=read('invocation.json');window=read('capture-window.json')
            if invocation['base']!='working' or invocation['fixture_tokens']!=a['tokens']:
                raise ValueError('trace base/shape mismatch')
            shutdown=read('shutdown.json')
            stopped=shutdown['returncode']==0 or (shutdown['returncode']==1 and
                shutdown.get('stderr','').strip()=='There are no active sessions.')
            if window['request']['output_tokens']!=16 or not stopped or any(
                r['returncode'] for r in read('profiler-commands.json')):
                raise ValueError('trace request/profiler incomplete')
            result={k:a[k] for k in ('run','artifact_id','zip_sha256','tokens')}
            result.update(invocation=invocation,trace_identity=read('trace-identity.json'),
                timeline=read('timeline.json'),memory=read('memory_summary.json'),thermal=read('thermal.json'),
                profiler_shutdown=shutdown)
            results.append(result)
    if len({a['tokens'] for a in results})!=len(results):raise ValueError('duplicate context')
    return dict(schema=1,traces=results,scope='Ported selected-base P0b revalidation. '
        'Rootless CUDA activity after an excluded warmup; raw reports and SQLite remain on runner. '
        'Main-stream kernel residence and gaps are a wall-lane proxy, not a complete causal DAG. '
        'No P1 candidate executes, no paired performance or hardware-counter claim.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    a=ap.parse_args();r=collect(json.loads(a.artifacts.read_text()))
    a.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
