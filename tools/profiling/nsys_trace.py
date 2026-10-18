#!/usr/bin/env python3
"""Bounded CUDA timeline via a checksum-pinned vendor CLI, without privileged runner changes."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v100'))
from model_baseline import ROOT, PACK, ENGINE, fixed_env, Monitor
import p0

URL = ('https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/'
       'nsight-systems-2024.6.2_2024.6.2.225-1_amd64.deb')
SHA256 = 'ebb52666c5e1d68d07f2e293774765f6c8639101f70fd970a5fd7c006a8ec442'
SIZE = 374017610


def install(out):
    root = Path('/actions-runner/abyss-tools/nsys-2024.6.2')
    root.mkdir(parents=True, exist_ok=True)
    deb = root / 'vendor.deb'
    if not deb.exists() or deb.stat().st_size != SIZE or p0.sha256(deb) != SHA256:
        part = root / 'vendor.deb.partial'
        with urllib.request.urlopen(URL, timeout=60) as response, part.open('wb') as f:
            shutil.copyfileobj(response, f, 8*1024*1024)
        if part.stat().st_size != SIZE or p0.sha256(part) != SHA256:
            raise ValueError('vendor package checksum/size mismatch')
        part.rename(deb)
    extracted = root / 'extracted'
    if not (root / 'complete.json').exists():
        subprocess.run(['dpkg-deb', '-x', str(deb), str(extracted)], check=True, timeout=120)
        p0.write_json(root / 'complete.json', {'package_sha256':SHA256})
    candidates = sorted(p for p in extracted.rglob('nsys') if p.is_file() and os.access(p, os.X_OK))
    # The target CLI uses its bundled relative support files; no system install/symlink.
    preferred = [p for p in candidates if p.parent.name == 'target-linux-x64']
    if not candidates:
        raise ValueError('vendor package has no executable nsys')
    nsys = (preferred or candidates)[0]
    version = p0.command([str(nsys),'--version'])
    p0.write_json(out/'nsys_provenance.json', {'vendor_package_url':URL,'vendor_package_sha256':SHA256,
                  'vendor_package_size':SIZE,'cli_path':str(nsys),'cli_sha256':p0.sha256(nsys),
                  'version':version,'source_index':'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/Packages.gz',
                  'system_installation':False, 'binary_redistribution':False,
                  'license_files':[{'path':str(p),'sha256':p0.sha256(p)} for p in extracted.rglob('*')
                                   if p.is_file() and p.name.lower() in ('eula.txt','license.txt','eula','license')]})
    if version['returncode']:
        raise RuntimeError(version)
    return nsys


def main():
    out = Path(os.environ['EVIDENCE']).absolute()
    out.mkdir(parents=True, exist_ok=True)
    nsys = install(out)
    for command in ['launch','start','stop','shutdown','stats']:
        helptext = p0.command([str(nsys),command,'--help'])
        (out/f'nsys-{command}-help.txt').write_text(helptext.get('stdout','')+helptext.get('stderr',''))
    tokens = (ROOT / 'results/ci-v100-39139393175-1/bench-2048/input.txt').read_text().strip()
    if len(tokens.split(',')) != 2048:
        raise ValueError('expected the completed baseline\'s 2048-token prompt')
    (out/'input.txt').write_text(tokens+'\n')
    session = 'abyss' + os.environ['GITHUB_RUN_ID']
    env = fixed_env()
    commands, windows = [], []
    def cli(args, timeout=120):
        r = p0.command([str(nsys),*args], timeout=timeout)
        commands.append(r)
        p0.write_json(out/'profiler_commands.json', commands)
        if r['returncode']:
            raise RuntimeError(r)
        return r
    with tempfile.TemporaryDirectory(prefix='abyss-trace-',dir=os.environ.get('RUNNER_TEMP')) as tmp:
        tmp = Path(tmp)
        fifo, stdout, stderr, status = (tmp/n for n in ['input.pipe','wire.log','engine.log','exit.txt'])
        os.mkfifo(fifo)
        env.update(ABYSS_TRACE_INPUT=str(fifo), ABYSS_TRACE_STDOUT=str(stdout),
                   ABYSS_TRACE_STDERR=str(stderr), ABYSS_TRACE_STATUS=str(status))
        argv = [str(ENGINE),'--glm-pack',str(PACK),'--serve','--max-context','36864',
                '--max-new','64','--prefill','auto','--temperature','0','--eos-ids','2147483647']
        shell = '"$@" < "$ABYSS_TRACE_INPUT" > "$ABYSS_TRACE_STDOUT" 2> "$ABYSS_TRACE_STDERR"; printf "%s\\n" "$?" > "$ABYSS_TRACE_STATUS"'
        # CLI session is isolated by run id. CPU sampling/context switches are disabled.
        launch_argv = [str(nsys),'launch','--session-new='+session,'--trace=cuda,nvtx,osrt',
                       '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','/bin/bash','-c',shell,'abyss-trace',*argv]
        p0.write_json(out/'trace_request.json',{'argv':argv,'input_sha256':p0.sha256(out/'input.txt'),
                      'trace_scope':'one warmed 2048-token prefill and 32 generated tokens',
                      'no_hardware_counter_claim':True,'no_cpu_sampling_claim':True})
        # Open both ends in this controller so the shell's redirection cannot deadlock launch.
        fd = os.open(fifo,os.O_RDWR)
        writer = os.fdopen(os.dup(fd),'w',buffering=1)
        read_pos = 0
        def until(prefix, timeout=600):
            nonlocal read_pos
            deadline = time.monotonic()+timeout
            while time.monotonic() < deadline:
                if stdout.exists():
                    with stdout.open() as f:
                        f.seek(read_pos)
                        for line in f:
                            read_pos += len(line.encode())
                            if line.startswith('ERR '):
                                raise RuntimeError(line)
                            if line.startswith(prefix):
                                return line.strip()
                if status.exists():
                    raise RuntimeError('engine exited: '+status.read_text()+'; '+(stderr.read_text() if stderr.exists() else ''))
                time.sleep(.1)
            raise TimeoutError(prefix)
        try:
            with Monitor(out):
                result = subprocess.run(launch_argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                        text=True,timeout=60)
                (out/'launch.log').write_text(result.stdout)
                if result.returncode:
                    raise RuntimeError(result.stdout)
                until('READY ')
                writer.write('GEN 64 temperature=0 '+tokens+'\n');writer.flush()
                until('DONE ')
                cli(['start','--session='+session,'--output='+str(out/'warmed-request'),'--force-overwrite=true','--stop-on-exit=false'])
                begin = {'utc':p0.now(),'monotonic_s':time.monotonic()}
                writer.write('GEN 32 temperature=0 '+tokens+'\n');writer.flush()
                done = until('DONE ',timeout=120)
                windows.append({'begin':begin,'end':{'utc':p0.now(),'monotonic_s':time.monotonic()},'done':done})
                p0.write_json(out/'capture_windows.json',windows)
                cli(['stop','--session='+session],timeout=180)
                writer.write('QUIT\n');writer.flush()
                deadline=time.monotonic()+60
                while not status.exists() and time.monotonic()<deadline: time.sleep(.1)
                if not status.exists() or status.read_text().strip()!='0':
                    raise RuntimeError('engine did not exit cleanly')
        finally:
            writer.close();os.close(fd)
            p0.write_json(out/'shutdown.json',p0.command([str(nsys),'shutdown','--session='+session,'--kill=sigkill'],timeout=30))
            for src,name in [(stdout,'wire.log'),(stderr,'engine.log')]:
                if src.exists(): shutil.copyfile(src,out/name)
    reports = list(out.glob('*.nsys-rep'))
    if len(reports)!=1:
        raise ValueError('missing or ambiguous Nsight report')
    report = reports[0]
    summary = cli(['stats','--report=cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum',
                   '--format=csv',str(report)],timeout=180)
    (out/'kernel-api-summary.csv').write_text(summary.get('stdout',''))
    p0.write_json(out/'report_identity.json',{'file':report.name,'bytes':report.stat().st_size,'sha256':p0.sha256(report)})
    # Reports only, never profiler binaries. Keep upload data bounded even on a broken profiler.
    for path in out.glob('*.sqlite'): path.unlink()
    for path in out.glob('*.qdstrm'): path.unlink()
    if report.stat().st_size > 128*1024*1024:
        destination=Path('/actions-runner/abyss-tools')/('trace-'+os.environ['GITHUB_RUN_ID']+'.nsys-rep')
        report.rename(destination)
        raise ValueError('report exceeded 128-MiB upload cap; retained only in runner tools directory')
    print(summary.get('stdout','')[-12000:],flush=True)


def cap_upload():
    out = Path(os.environ['EVIDENCE']).absolute()
    if not out.exists():
        return
    retained = []
    cache = Path('/actions-runner/abyss-tools') / ('trace-' + os.environ['GITHUB_RUN_ID'])
    # Also enforced on exceptions: never accidentally upload a huge partial trace.
    files = sorted((p for p in out.rglob('*') if p.is_file()), key=lambda p:p.stat().st_size, reverse=True)
    total = sum(p.stat().st_size for p in files)
    for p in files:
        if p.suffix in ('.sqlite','.qdstrm') or total > 128*1024*1024:
            cache.mkdir(parents=True,exist_ok=True)
            size = p.stat().st_size
            retained.append({'file':p.name,'bytes':size,'sha256':p0.sha256(p),'runner_path':str(cache/p.name)})
            p.rename(cache/p.name)
            total -= size
    p0.write_json(out/'upload_cap.json',{'retained_only_on_runner':retained,'uploaded_bytes_before_manifest':total})


if __name__ == '__main__':
    try:
        main()
    finally:
        cap_upload()
