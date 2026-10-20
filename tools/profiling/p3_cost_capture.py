#!/usr/bin/env python3
"""Rootless P3 completed-cost observation; no new synchronization or estimator."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/profiling'),str(ROOT/'tools/bench'),str(ROOT/'tools/v100'),str(ROOT/'tools')]
import p0
import nsys_trace
import v100_p0_campaign as campaign
from model_baseline import fixed_env,PACK
from p0_accounting import fixture,compress
from p3_cost_analyze import analyze

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replay-run',help='analyze a completed numeric run-attempt cache without loading a model')
    ap.add_argument('--context',type=int,choices=(8192,32768,131072),default=8192,
        help='genuine frozen history length; no synthetic position offsets')
    ap.add_argument('--first-versus-warmed',action='store_true',
        help='observe first post-load request and subsequent request, without asserting cold OS caches')
    args=ap.parse_args()
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=True)
    if args.replay_run:
        if not re.fullmatch(r'\d+-\d+',args.replay_run):raise ValueError('invalid replay run-attempt')
        cache=Path('/actions-runner/abyss-p3-v9/cost')/args.replay_run
        identity=json.loads((cache/'identity.json').read_text());database=cache/'request.sqlite'
        if p0.sha256(database)!=identity['sqlite_sha256']:raise ValueError('completed raw SQLite changed')
        p0.write_json(out/'replayed-identity.json',identity)
        p0.write_json(out/'costs.json',analyze(database));return
    binary=ROOT/'build-v100-p3-cost/strata'
    model=campaign.model_identity();identity=json.loads((campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=identity['model_sha256'] or identity['pack']!=json.loads((PACK/'v100_pack_complete.json').read_text()):
        raise ValueError('frozen model/pack identity changed')
    fixture_name='mixed-'+str(args.context)
    history=fixture(fixture_name)
    if len(history)!=args.context:raise ValueError('frozen history length mismatch')
    tokens=','.join(map(str,history))
    nsys=nsys_trace.install(out);run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    cache=Path('/actions-runner/abyss-p3-v9/cost')/run;cache.mkdir(parents=True,exist_ok=False)
    session='abyssp3cost'+run;commands=[]
    def cli(args,timeout=180):
        result=p0.command([str(nsys),*args],timeout=timeout);commands.append(result)
        p0.write_json(out/'profiler-commands.json',commands)
        if result['returncode']:raise RuntimeError('profiler command failed; preserve checkpoint')
        return result
    env=fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1',STRATA_GLM_KDA_GRAPH='0',STRATA_GLM_DEFER='0')
    argv=[str(binary),'--glm-pack',str(PACK),'--serve','--max-context','135168','--max-new','32',
          '--prefill','auto','--temperature','0','--eos-ids','2147483647']
    cache_protocol=('first post-load request versus subsequent request; OS/file caches uncontrolled'
        if args.first_versus_warmed else 'excluded 16-output warm-up then two warmed requests')
    p0.write_json(out/'invocation.json',dict(argv=argv,binary_sha256=p0.sha256(binary),model_identity=model,
        context_tokens=args.context,fixture_sha256=p0.sha256(campaign.CAMPAIGN/('fixtures/'+fixture_name+'.tokens')),
        cache_protocol=cache_protocol,
        environment={k:v for k,v in env.items() if k.startswith('STRATA_')},
        scope='Two 32-output uncached genuine-history requests under the declared cache protocol in a rootless CUDA/NVTX trace. '
        'No cold-cache, competing-load, alternate-NUMA, estimator or clean throughput claim.'))
    with tempfile.TemporaryDirectory(prefix='abyss-p3-cost-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tmp=Path(tmp);fifo=tmp/'input';wire=tmp/'wire';log=tmp/'engine';status=tmp/'status';pidfile=tmp/'pid'
        os.mkfifo(fifo);env.update(ABYSS_TRACE_INPUT=str(fifo),ABYSS_TRACE_STDOUT=str(wire),ABYSS_TRACE_STDERR=str(log),
            ABYSS_TRACE_STATUS=str(status),ABYSS_TRACE_PID=str(pidfile))
        shell='"$@" < "$ABYSS_TRACE_INPUT" > "$ABYSS_TRACE_STDOUT" 2> "$ABYSS_TRACE_STDERR" & child=$!; printf "%s\\n" "$child" > "$ABYSS_TRACE_PID"; wait "$child"; rc=$?; printf "%s\\n" "$rc" > "$ABYSS_TRACE_STATUS"; exit "$rc"'
        launch=[str(nsys),'launch','--session-new='+session,'--trace=cuda,nvtx','--cuda-graph-trace=node',
                '/bin/bash','-c',shell,'abyss-p3-cost',*argv]
        fd=os.open(fifo,os.O_RDWR);writer=os.fdopen(os.dup(fd),'w',buffering=1);pos=0;child=None
        def until(prefix,monitor,timeout=1800):
            nonlocal pos
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if pidfile.exists() and monitor.pid is None:
                    value=pidfile.read_text().strip()
                    if value:monitor.pid=int(value)
                if campaign.memory_available()<8*1024**3:
                    if monitor.pid and Path(f'/proc/{monitor.pid}/exe').resolve()==binary:os.kill(monitor.pid,9)
                    raise RuntimeError('own-child cost capture memory stop below 8 GiB')
                if wire.exists():
                    with wire.open() as f:
                        f.seek(pos)
                        for line in f:
                            pos+=len(line.encode())
                            if line.startswith('ERR '):raise RuntimeError(line)
                            if line.startswith(prefix):return line.rstrip()
                if status.exists():raise RuntimeError('cost engine exited: '+status.read_text())
                if child.poll() is not None:raise RuntimeError('cost profiler launcher exited')
                time.sleep(.1)
            raise TimeoutError(prefix)
        try:
            if campaign.memory_available()<16*1024**3:raise RuntimeError('initial host margin below 16 GiB')
            with campaign.Monitor(out) as monitor,(out/'launch.log').open('w') as launchlog:
                child=subprocess.Popen(launch,env=env,stdout=launchlog,stderr=subprocess.STDOUT)
                until('READY ',monitor)
                if args.first_versus_warmed:
                    p0.write_json(out/'warmup.json',dict(done=None,disposition=cache_protocol))
                else:
                    writer.write('GEN 16 temperature=0 '+tokens+'\n');writer.flush()
                    p0.write_json(out/'warmup.json',dict(done=until('DONE ',monitor)))
                cli(['start','--session='+session,'--output='+str(cache/'request'),'--stop-on-exit=true','--sample=none','--cpuctxsw=none'])
                requests=[]
                for request in range(2):
                    writer.write('GEN 32 temperature=0 '+tokens+'\n');writer.flush();done=until('DONE ',monitor)
                    if int(done.split()[1])!=32:raise ValueError('short completed request')
                    requests.append(dict(request=request,done=done));p0.write_json(out/'requests.json',requests)
                cli(['stop','--session='+session]);writer.write('QUIT\n');writer.flush()
                deadline=time.monotonic()+120
                while not status.exists() and time.monotonic()<deadline:time.sleep(.1)
                if not status.exists() or status.read_text().strip()!='0':raise RuntimeError('cost engine did not exit cleanly')
                child.wait(timeout=120)
                if child.returncode:raise RuntimeError('cost launcher failed after shutdown')
        finally:
            writer.close();os.close(fd)
            p0.write_json(out/'shutdown.json',p0.command([str(nsys),'shutdown','--session='+session,'--kill=sigkill'],timeout=30))
            if child is not None and child.poll() is None:child.kill();child.wait()
            for path,name in [(wire,'wire.log'),(log,'engine.log')]:
                if path.exists():shutil.copyfile(path,out/name)
    text=(out/'engine.log').read_text()
    end=re.findall(r'V100_COST_END ranges=(\d+) dropped=(\d+)',text)
    if len(end)!=1 or int(end[0][1])!=0:raise ValueError('missing/truncated cost observation ledger')
    compress(out/'engine.log')
    reports=list(cache.glob('*.nsys-rep'))
    if len(reports)!=1 or reports[0].stat().st_size>2*1024**3:raise ValueError('missing/oversized raw cost report')
    report=reports[0];database=cache/'request.sqlite'
    cli(['export','--type=sqlite','--output='+str(database),str(report)],timeout=300)
    if database.stat().st_size>4*1024**3:raise ValueError('oversized raw SQLite export')
    raw_identity=dict(source_sha=p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        binary_sha256=p0.sha256(binary),context_tokens=args.context,
        cache_protocol=cache_protocol,
        fixture_sha256=p0.sha256(campaign.CAMPAIGN/('fixtures/'+fixture_name+'.tokens')),
        report_path=str(report),report_bytes=report.stat().st_size,report_sha256=p0.sha256(report),
        sqlite_path=str(database),sqlite_sha256=p0.sha256(database),raw_uploaded=False)
    p0.write_json(out/'raw-identity.json',raw_identity);p0.write_json(cache/'identity.json',raw_identity)
    p0.write_json(out/'costs.json',analyze(database))
if __name__=='__main__':main()
