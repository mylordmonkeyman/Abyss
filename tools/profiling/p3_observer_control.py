#!/usr/bin/env python3
"""Bounded fixed-token observer/profiler controls; no allocation or math change."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import p3_cost_capture as cost
from v100_base_replay import parse_lp

ORDER=('plain','observer','trace','trace','observer','plain')


def contrasts(arms):
    if [a['mode'] for a in arms]!=list(ORDER):raise ValueError('incomplete control arm population')
    medians=[]
    for arm in arms:
        rows=arm['requests']
        if len(rows)!=4 or [r['repetition'] for r in rows]!=list(range(4)) or any(
            r['scored_tokens']!=32 or r['warmup']!=(i==0) or not math.isfinite(r['wire_ms']) or r['wire_ms']<=0
            for i,r in enumerate(rows)):raise ValueError('incomplete fixed-token requests')
        medians.append(statistics.median(r['wire_ms'] for r in rows[1:]))
    return [dict(order_half=half,plain_median_wire_ms=medians[plain],
        observer_median_wire_ms=medians[observer],trace_median_wire_ms=medians[trace],
        observer_vs_plain_pct=100*(medians[observer]/medians[plain]-1),
        trace_vs_observer_pct=100*(medians[trace]/medians[observer]-1))
        for half,(plain,observer,trace) in enumerate(((0,1,2),(5,4,3)))]


def arm(binary,mode,index,out,cache,nsys,ids):
    out.mkdir();cache.mkdir(parents=True,exist_ok=False)
    env=cost.fixed_env();env.update(STRATA_DS4_IDX='1',STRATA_DS4_ATTN='1',
        STRATA_GLM_KDA_GRAPH='0',STRATA_GLM_DEFER='0')
    argv=[str(binary),'--glm-pack',str(cost.PACK),'--serve','--max-context','12288',
        '--max-new','32','--prefill','auto','--temperature','0','--eos-ids','2147483647']
    session='abyssp3control'+os.environ['GITHUB_RUN_ID']+'-'+str(index)+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    commands=[];requests=[];child=None;position=0
    def cli(args,timeout=180,required=True):
        r=cost.p0.command([str(nsys),*args],timeout=timeout);commands.append(r)
        cost.p0.write_json(out/'profiler-commands.json',commands)
        if required and r['returncode']:raise RuntimeError('control profiler command failed')
    cost.p0.write_json(out/'invocation.json',dict(argv=argv,mode=mode,binary_sha256=cost.p0.sha256(binary),
        environment={k:v for k,v in env.items() if k.startswith('STRATA_')},
        cpu_affinity=sorted(os.sched_getaffinity(0)),controller_poll_bound_ms=10))
    with tempfile.TemporaryDirectory(prefix='abyss-p3-control-',dir=os.environ.get('RUNNER_TEMP')) as temp:
        temp=Path(temp);fifo=temp/'input';wire=temp/'wire';log=temp/'engine';status=temp/'status';pidfile=temp/'pid'
        os.mkfifo(fifo);env.update(ABYSS_TRACE_INPUT=str(fifo),ABYSS_TRACE_STDOUT=str(wire),
            ABYSS_TRACE_STDERR=str(log),ABYSS_TRACE_STATUS=str(status),ABYSS_TRACE_PID=str(pidfile))
        shell='"$@" < "$ABYSS_TRACE_INPUT" > "$ABYSS_TRACE_STDOUT" 2> "$ABYSS_TRACE_STDERR" & child=$!; printf "%s\\n" "$child" > "$ABYSS_TRACE_PID"; wait "$child"; rc=$?; printf "%s\\n" "$rc" > "$ABYSS_TRACE_STATUS"; exit "$rc"'
        launch=['/bin/bash','-c',shell,'abyss-p3-control',*argv]
        if mode=='trace':launch=[str(nsys),'launch','--session-new='+session,'--trace=cuda,nvtx',
            '--cuda-graph-trace=node',*launch]
        fd=os.open(fifo,os.O_RDWR);writer=os.fdopen(os.dup(fd),'w',buffering=1)
        def until(prefix,monitor,timeout=1800):
            nonlocal position
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if pidfile.exists() and monitor.pid is None:
                    value=pidfile.read_text().strip()
                    if value:monitor.pid=int(value)
                if cost.campaign.memory_available()<8*1024**3:
                    if monitor.pid and Path(f'/proc/{monitor.pid}/exe').resolve()==binary:os.kill(monitor.pid,9)
                    raise RuntimeError('own-child observer control memory stop below 8 GiB')
                if wire.exists():
                    with wire.open() as f:
                        f.seek(position)
                        for line in f:
                            position+=len(line.encode())
                            if line.startswith('ERR '):raise RuntimeError(line)
                            if line.startswith(prefix):return line.strip()
                if status.exists():raise RuntimeError('control engine exited: '+status.read_text())
                if child.poll() is not None:raise RuntimeError('control launcher exited')
                time.sleep(.01)
            raise TimeoutError(prefix)
        try:
            if cost.campaign.memory_available()<16*1024**3:raise RuntimeError('initial host margin below 16 GiB')
            with cost.campaign.Monitor(out) as monitor,(out/'launch.log').open('w') as launchlog:
                child=subprocess.Popen(launch,env=env,stdout=launchlog,stderr=subprocess.STDOUT)
                until('READY ',monitor)
                for repetition in range(4):
                    # LOGP snapshots equal-context requests independently of GEN's
                    # NO_REUSE control. An explicit different context forces reset.
                    writer.write('LOGP 2 '+','.join(map(str,ids[:3]))+'\n');writer.flush()
                    parse_lp(until('LP ',monitor),1)
                    if mode=='trace' and repetition==1:
                        cli(['start','--session='+session,'--output='+str(cache/'requests'),
                            '--stop-on-exit=true','--sample=none','--cpuctxsw=none'])
                    before=dict(loadavg=Path('/proc/loadavg').read_text(),cpu_stat=Path('/proc/stat').read_text())
                    begin=time.monotonic();writer.write('LOGP 8192 '+','.join(map(str,ids))+'\n');writer.flush()
                    response=until('LP ',monitor);elapsed=1000*(time.monotonic()-begin)
                    row=dict(parse_lp(response,32),repetition=repetition,warmup=repetition==0,
                        wire_ms=elapsed,host_before=before,host_after=dict(loadavg=Path('/proc/loadavg').read_text(),
                        cpu_stat=Path('/proc/stat').read_text()))
                    requests.append(row);cost.p0.write_json(out/'requests.json',requests)
                if mode=='trace':cli(['stop','--session='+session])
                writer.write('QUIT\n');writer.flush();deadline=time.monotonic()+120
                while not status.exists() and time.monotonic()<deadline:time.sleep(.1)
                if not status.exists() or status.read_text().strip()!='0':raise RuntimeError('control did not exit cleanly')
                child.wait(timeout=120)
                if child.returncode:raise RuntimeError('control launcher failed')
        finally:
            writer.close();os.close(fd)
            if mode=='trace':cli(['shutdown','--session='+session,'--kill=sigkill'],timeout=30,required=False)
            if pidfile.exists() and not status.exists():
                value=pidfile.read_text().strip()
                if value:
                    try:
                        pid=int(value)
                        if Path(f'/proc/{pid}/exe').resolve()==binary:os.kill(pid,9)
                    except (ProcessLookupError,FileNotFoundError):pass
            if child is not None and child.poll() is None:child.kill();child.wait()
            for src,name in ((wire,'wire.log'),(log,'engine.log')):
                if src.exists():shutil.copyfile(src,out/name)
            if (out/'engine.log').exists():cost.compress(out/'engine.log')
    import gzip
    text=gzip.decompress((out/'engine.log.gz').read_bytes()).decode()
    ledgers=re.findall(r'V100_COST_END ranges=(\d+) dropped=(\d+)',text)
    if mode=='plain':
        if ledgers:raise ValueError('compiled-out control unexpectedly has observer')
    elif len(ledgers)!=1 or int(ledgers[0][0])==0 or int(ledgers[0][1]):
        raise ValueError('missing/truncated observer control ledger')
    result=dict(mode=mode,requests=requests,observer_ledger=ledgers)
    if mode=='trace':
        reports=list(cache.glob('*.nsys-rep'))
        if len(reports)!=1 or reports[0].stat().st_size>2*1024**3:raise ValueError('missing/oversized control trace')
        result['raw_report']=dict(path=str(reports[0]),bytes=reports[0].stat().st_size,
            sha256=cost.p0.sha256(reports[0]),uploaded=False)
    cost.p0.write_json(out/'result.json',result)
    return result


def main():
    out=Path(os.environ['EVIDENCE']).absolute();out.mkdir(parents=True,exist_ok=True)
    model=cost.campaign.model_identity();frozen=json.loads((cost.campaign.CAMPAIGN/'identity.json').read_text())
    if model['sha256']!=frozen['model_sha256'] or frozen['pack']!=json.loads((cost.PACK/'v100_pack_complete.json').read_text()):
        raise ValueError('frozen model/pack identity changed')
    ids=cost.fixture('mixed-32768')[:8192+32]
    if len(ids)!=8224:raise ValueError('wrong fixed-token control length')
    data=','.join(map(str,ids));nsys=cost.nsys_trace.install(out)
    run=os.environ['GITHUB_RUN_ID']+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
    if not re.fullmatch(r'\d+-\d+',run):raise ValueError('invalid control run identity')
    cache=Path('/actions-runner/abyss-p3-v9/observer-control')/run
    cache.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(cache).free<8*1024**3:raise RuntimeError('control raw trace disk margin below 8 GiB')
    binaries=dict(plain=cost.ROOT/'build-v100-p3-control/strata',observer=cost.ROOT/'build-v100-p3-cost/strata')
    report=dict(schema=1,source_sha=cost.p0.command(['git','rev-parse','HEAD'])['stdout'].strip(),
        model_identity=model,pack=frozen['pack'],binary_sha256={k:cost.p0.sha256(v) for k,v in binaries.items()},
        source_fixture='mixed-32768.tokens',source_fixture_sha256=frozen['fixtures']['mixed-32768.tokens']['sha256'],
        fixed_token_sha256=hashlib.sha256(data.encode()).hexdigest(),context_tokens=8192,scored_tokens=32,
        order=list(ORDER),arms=[],status='running',
        scope='Fixed token bytes, F32 and fixed startup split; fresh process per arm, one excluded and three warmed LOGP replays. '
        'LOGP wire time includes prefill, host logits/log-softmax and <=10 ms controller polling: not normal decode throughput. '
        'OS cache, clocks and CPU-lane-28 router variability remain uncontrolled; identical input tokens do not prove identical expert trajectories. '
        'Reverse order is descriptive overhead screening, not a causal speed claim or adaptive-plan admission. '
        'No cache drops, placement changes, privileges, arithmetic or allocation changes. Local-manifest model verification is not '
        'independent publisher authenticity; hardware counters/QPI remain unavailable/unmeasured.')
    def checkpoint():
        cost.p0.write_json(out/'control.json',report);cost.p0.write_json(cache/'checkpoint.json',report)
    checkpoint()
    try:
        for index,mode in enumerate(ORDER):
            label=str(index)+'-'+mode;print('Starting observer control',label,flush=True)
            binary=binaries['plain' if mode=='plain' else 'observer']
            report['arms'].append(arm(binary,mode,index,out/label,cache/label,nsys,ids));checkpoint()
        report.update(status='complete; overhead screening only',contrasts=contrasts(report['arms']));checkpoint()
    except BaseException as error:
        report.update(status='incomplete; inspect failure before retry',error=str(error));checkpoint();raise


if __name__=='__main__':main()
