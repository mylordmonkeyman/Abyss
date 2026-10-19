#!/usr/bin/env python3
"""Rootless long-context CUDA timeline, bounded summaries; raw reports stay local."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools'),str(ROOT)]
import nsys_trace
from analyze_trace import analyze
from p0_accounting import fixture
import v100_p0_campaign as campaign
from model_baseline import fixed_env
from v100_base_compare import PACK,parse_done
import p0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tokens',type=int,choices=[32768,131072],required=True)
    ap.add_argument('--base',choices=['historical','current'],default='historical')
    ap.add_argument('--stream-fetch',type=int,choices=[0,1])
    args=ap.parse_args();out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=False)
    if campaign.memory_available()<16*1024**3:raise RuntimeError('initial memory safety margin below 16 GiB')
    if args.stream_fetch is not None and args.base!='current':
        raise ValueError('stream-fetch control is absent in the historical pin')
    binary=campaign.BINARIES[args.base]
    if p0.sha256(binary)!=campaign.EXPECTED[args.base]:raise ValueError('immutable baseline binary changed')
    identity=campaign.model_identity();p0.write_json(out/'model-identity.json',identity)
    ids=fixture('mixed-'+str(args.tokens));tokens=','.join(map(str,ids))
    nsys=nsys_trace.install(out)
    suffix=args.base+'-fetch'+str(args.stream_fetch)
    cache=Path('/actions-runner/abyss-p0-v9/traces')/(os.environ['GITHUB_RUN_ID']+'-'+str(args.tokens)+'-'+suffix)
    cache.mkdir(parents=True,exist_ok=False)
    session='abyssp0'+os.environ['GITHUB_RUN_ID']+'c'+str(args.tokens)+suffix
    commands=[]
    def cli(args,timeout=180):
        result=p0.command([str(nsys),*args],timeout=timeout);commands.append(result)
        p0.write_json(out/'profiler-commands.json',commands)
        if result['returncode']:raise RuntimeError('profiler command failed; inspect preserved command record')
        return result
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1')
    if args.stream_fetch is not None:env['STRATA_GLM_STREAM_FETCH']=str(args.stream_fetch)
    argv=[str(binary),'--glm-pack',str(PACK),'--serve','--max-context','135168',
          '--max-new','16','--prefill','auto','--temperature','0','--eos-ids','2147483647']
    p0.write_json(out/'invocation.json',dict(argv=argv,base=args.base,binary_sha256=campaign.EXPECTED[args.base],
        fixture_tokens=args.tokens,fixture_sha256=p0.sha256(campaign.CAMPAIGN/'fixtures'/('mixed-'+str(args.tokens)+'.tokens')),
        environment={k:v for k,v in env.items() if k.startswith('STRATA_')},
        cache_scope='first request excluded warmup, no prefix reuse; OS file cache uncontrolled',
        timing_scope='profile diagnostic only, excluded from ordinary throughput comparison',
        security='rootless CUDA activity; CPU sampling/context switches and hardware counters disabled'))
    with tempfile.TemporaryDirectory(prefix='abyss-p0-trace-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tmp=Path(tmp);fifo=tmp/'input.pipe';wire=tmp/'wire.log';log=tmp/'engine.log';status=tmp/'exit.txt';pidfile=tmp/'pid.txt'
        os.mkfifo(fifo);env.update(ABYSS_TRACE_INPUT=str(fifo),ABYSS_TRACE_STDOUT=str(wire),
            ABYSS_TRACE_STDERR=str(log),ABYSS_TRACE_STATUS=str(status),ABYSS_TRACE_PID=str(pidfile))
        shell='"$@" < "$ABYSS_TRACE_INPUT" > "$ABYSS_TRACE_STDOUT" 2> "$ABYSS_TRACE_STDERR" & child=$!; printf "%s\\n" "$child" > "$ABYSS_TRACE_PID"; wait "$child"; rc=$?; printf "%s\\n" "$rc" > "$ABYSS_TRACE_STATUS"; exit "$rc"'
        launch=[str(nsys),'launch','--session-new='+session,'--trace=cuda,nvtx,osrt',
            '--cuda-graph-trace=node','/bin/bash','-c',shell,'abyss-p0-trace',*argv]
        fd=os.open(fifo,os.O_RDWR);writer=os.fdopen(os.dup(fd),'w',buffering=1)
        pos=0;launched=None
        def until(prefix,monitor,timeout=1800):
            nonlocal pos
            deadline=time.monotonic()+timeout;lines=[]
            while time.monotonic()<deadline:
                if pidfile.exists() and monitor.pid is None:
                    value=pidfile.read_text().strip()
                    if value:monitor.pid=int(value)
                if campaign.memory_available()<8*1024**3:
                    if monitor.pid and Path(f'/proc/{monitor.pid}/exe').resolve()==binary:
                        os.kill(monitor.pid,9)
                    raise RuntimeError('host memory safety stop below 8 GiB')
                if wire.exists():
                    with wire.open() as f:
                        f.seek(pos)
                        for line in f:
                            pos+=len(line.encode());lines.append(line.rstrip())
                            if line.startswith('ERR '):raise RuntimeError(line)
                            if line.startswith(prefix):return line.rstrip(),lines
                if status.exists():raise RuntimeError('engine exited: '+status.read_text())
                if launched.poll() is not None:raise RuntimeError('profiler launcher exited before expected engine reply')
                time.sleep(.1)
            raise TimeoutError(prefix)
        try:
            with campaign.Monitor(out) as monitor:
                with (out/'launch.log').open('w') as f:launched=subprocess.Popen(launch,env=env,stdout=f,stderr=subprocess.STDOUT)
                until('READY ',monitor)
                writer.write('GEN 16 temperature=0 '+tokens+'\n');writer.flush()
                warm,_=until('DONE ',monitor)
                p0.write_json(out/'warmup.json',dict(done=warm))
                cli(['start','--session='+session,'--output='+str(cache/'request'),
                     '--force-overwrite=true','--stop-on-exit=true','--sample=none','--cpuctxsw=none'])
                begin=time.monotonic();writer.write('GEN 16 temperature=0 '+tokens+'\n');writer.flush()
                done,lines=until('DONE ',monitor)
                parsed=parse_done(done,lines,args.tokens,16)
                p0.write_json(out/'capture-window.json',dict(begin_monotonic_s=begin,end_monotonic_s=time.monotonic(),done=done,request=parsed))
                cli(['stop','--session='+session])
                writer.write('QUIT\n');writer.flush()
                deadline=time.monotonic()+120
                while not status.exists() and time.monotonic()<deadline:time.sleep(.1)
                if not status.exists() or status.read_text().strip()!='0':raise RuntimeError('engine did not exit cleanly')
                launched.wait(timeout=120)
                if launched.returncode:raise RuntimeError('profiler launcher failed after engine shutdown')
        finally:
            writer.close();os.close(fd)
            p0.write_json(out/'shutdown.json',p0.command([str(nsys),'shutdown','--session='+session,'--kill=sigkill'],timeout=30))
            if launched is not None and launched.poll() is None:launched.kill();launched.wait()
            for path,name in [(wire,'wire.log'),(log,'engine.log')]:
                if path.exists():shutil.copyfile(path,out/name)
    reports=list(cache.glob('*.nsys-rep'))
    if len(reports)!=1:raise ValueError('missing or ambiguous CUDA trace')
    report=reports[0];database=cache/'request.sqlite'
    cli(['export','--type=sqlite','--output='+str(database),'--force-overwrite=true',str(report)],timeout=300)
    result=analyze(database,done,args.tokens)
    p0.write_json(out/'timeline.json',result)
    p0.write_json(out/'trace-identity.json',dict(raw_report_runner_path=str(report),bytes=report.stat().st_size,
        sha256=p0.sha256(report),sqlite_runner_path=str(database),sqlite_sha256=p0.sha256(database),
        retention='raw trace and SQLite retained only on runner; summary/identities uploaded'))
    if args.stream_fetch is not None:
        launches=result['fetch_kernel_launches']
        streamed=sum(v for k,v in launches.items() if 'moe_fetch_stream_kernel' in k)
        ordinary=sum(v for k,v in launches.items() if 'moe_fetch_kernel' in k)
        p0.write_json(out/'fetch-dispatch.json',dict(base=args.base,control=args.stream_fetch,
            streamed_launches=streamed,ordinary_launches=ordinary,
            scope='Actual CUPTI kernel activity, not symbol presence or a speedup claim'))
        if (args.stream_fetch==1 and (not streamed or ordinary)) or \
           (args.stream_fetch==0 and (not ordinary or streamed)):
            raise ValueError('observed fetch kernels disagree with explicit current-base control')
    print(json.dumps(dict(tokens=args.tokens,top_three={phase:v['critical_path_proxy']['top_three'] for phase,v in result['phases'].items()})),flush=True)


def cap():
    out=Path(os.environ['EVIDENCE']).absolute()
    if not out.exists():return
    total=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if total>64*1024**2:raise ValueError('trace summary exceeds 64 MiB; upload refused')
    p0.write_json(out/'upload-cap.json',dict(bytes_before_manifest=total,raw_reports_uploaded=False))
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'],'a') as f:f.write('P0_REMAINING_CAP=true\n')

if __name__=='__main__':
    try:main()
    finally:cap()
