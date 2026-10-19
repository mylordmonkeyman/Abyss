#!/usr/bin/env python3
"""Bounded P0 logit comparison and warm, uncached prompt benchmarks on the mounted model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import p0
from p0 import ROOT

PACK = Path('/actions-runner/abyss-models/deepseek-v4-flash-0731-ud-q2-k-xl/pack-dsv4')
ENGINE = ROOT / 'build-v100-diagnostics/strata'


def fixed_env():
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(('STRATA_', 'ABYSS_V100_')):
            del env[k]
    env.update(STRATA_DS4_SPEC='0', STRATA_GLM_RAM_GB='72', STRATA_GLM_CPU_LANE='28',
               STRATA_GLM_PCIE_SHARE='0.21', STRATA_GLM_USAGE='0', STRATA_GLM_NO_REUSE='1',
               STRATA_GLM_SLOTS='0')
    return env


def corpus():
    return ''.join(f'Observation {i}: A server combines processors, memory, storage and communication links. '
                   'Matrix multiplication and data transfers can overlap. Explain assumptions, compare alternatives, '
                   'and distinguish measurements from predictions.\n' for i in range(3000))


class Monitor:
    def __init__(self, out):
        self.out, self.stop, self.samples = out, threading.Event(), []
        self.pid = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            sample = {'utc': p0.now(), 'monotonic_s': time.monotonic(),
                      'gpu': p0.command(['nvidia-smi', '-i', os.environ.get('ABYSS_GPU', '0'), '-q', '-x']),
                      'meminfo': Path('/proc/meminfo').read_text()}
            for label, path in [('process_status', f'/proc/{self.pid}/status'),
                                ('process_numa_maps', f'/proc/{self.pid}/numa_maps'),
                                ('cgroup_memory_current', '/sys/fs/cgroup/memory.current'),
                                ('cgroup_memory_stat', '/sys/fs/cgroup/memory.stat'),
                                ('cgroup_memory_events', '/sys/fs/cgroup/memory.events')]:
                try:
                    sample[label] = Path(path).read_text()
                except OSError:
                    pass
            self.samples.append(sample)
            self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        p0.write_json(self.out / 'telemetry.json', self.samples)
        parsed = []
        for s in self.samples:
            try:
                parsed.append(p0.xml_sample(s['gpu']['stdout']))
            except (KeyError, ValueError) as e:
                parsed.append({'error': str(e)})
        p0.write_json(self.out / 'thermal.json', p0.thermal_assessment(parsed))


def launch(argv, env, out, timeout=1800):
    started = time.monotonic()
    with Monitor(out) as monitor, (out / 'engine.log').open('w') as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
        monitor.pid = proc.pid
        try:
            rc = proc.wait(timeout=timeout)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
    result = {'argv': list(map(str, argv)), 'environment': {k: v for k, v in env.items() if k.startswith('STRATA_')},
              'returncode': rc, 'wall_s_including_load': time.monotonic() - started}
    p0.write_json(out / 'process.json', result)
    if rc:
        raise RuntimeError((out / 'engine.log').read_text()[-12000:])
    return result


def normalize_logprobs(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('expected finite rows of log probabilities')
    mx = raw.max(axis=1, keepdims=True)
    return raw - (mx + np.log(np.exp(raw - mx).sum(axis=1, keepdims=True)))


def score(out, tok, vocab):
    context, rows = 2000, 64
    # Output distribution j predicts token index context+1+j (zero based): includes 2048.
    ids = [0] + tok.encode(corpus())[:context + rows]
    assert len(ids) == context + rows + 1
    tokens = out / 'teacher-forced-input.txt'
    tokens.write_text(','.join(map(str, ids)) + '\n')
    stats = {}
    with tempfile.TemporaryDirectory(prefix='abyss-score-', dir=os.environ.get('RUNNER_TEMP')) as tmp:
        dumps = []
        for mode in ['token', 'batch']:
            case = out / mode
            case.mkdir()
            dump = Path(tmp) / (mode + '.f32')
            env = fixed_env()
            env.update(STRATA_GLM_SCORE=str(context), STRATA_GLM_SCORE_DUMP=str(dump))
            if mode == 'batch':
                env['STRATA_GLM_SCORE_PREFILL'] = '1'
            argv = [str(ENGINE), '--glm-pack', str(PACK), '--tokens-file', str(tokens), '--max-new', '1',
                    '--max-context', '4096', '--prefill', 'auto', '--temperature', '0']
            launch(argv, env, case)
            if dump.stat().st_size != rows * vocab * 4:
                raise ValueError('wrong teacher-forced distribution count')
            raw = np.fromfile(dump, dtype='<f4').reshape(rows, vocab).astype(np.float64)
            if not np.isfinite(raw).all():
                raise ValueError(f'nonfinite log probabilities: {mode}')
            stats[mode] = {'dump_bytes': dump.stat().st_size, 'dump_sha256': p0.sha256(dump),
                           'finite_values': int(raw.size)}
            # Normalize stored FP32 log probabilities in FP64 before computing KL.
            dumps.append(normalize_logprobs(raw))
        ref, test = dumps
        kl = (np.exp(ref) * (ref - test)).sum(axis=1)
        same = ref.argmax(axis=1) == test.argmax(axis=1)
        per_position = [{'predicted_token_index_zero_based': context + 1 + i, 'kl': float(k),
                         'same_top1': bool(same[i]), 'max_abs_logprob_difference': float(np.abs(ref[i]-test[i]).max())}
                        for i, k in enumerate(kl)]
        result = {'scope': 'same pack, batch prefill vs one-token path; not an external oracle',
                  'context_parameter': context, 'scored_positions': rows, 'vocabulary': vocab,
                  'mean_kl': float(kl.mean()), 'max_kl': float(kl.max()), 'top1_agreement': float(same.mean()),
                  'raw_dump_retention': 'temporary; summary and hashes retained, no raw tensors uploaded',
                  'dumps': stats, 'positions': per_position,
                  'review_threshold_mean_kl': 0.002,
                  'passed_initial_screen': bool(kl.mean() <= 0.002)}
        p0.write_json(out / 'logit_comparison.json', result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('positions', 'dumps')}), flush=True)
        if not result['passed_initial_screen']:
            raise ValueError('batch/token KL exceeds provisional review threshold; investigate before promotion')


def bench(out, tok, template, length):
    text = corpus() + '\nWrite a detailed technical report with at least thirty sections and worked examples.'
    ids = tok.encode(template.render([{'role': 'user', 'content': text}], enable_thinking=False), parse_special=True)
    # Keep the template ending and request while making the token count exact.
    ids = ids[:length - 128] + ids[-128:]
    assert len(ids) == length
    (out / 'input.txt').write_text(','.join(map(str, ids)) + '\n')
    argv = [str(ENGINE), '--glm-pack', str(PACK), '--serve', '--max-context', '36864',
            '--max-new', '256', '--prefill', 'auto', '--temperature', '0', '--eos-ids', '2147483647']
    env = fixed_env()
    results, lines = [], queue.Queue()
    with Monitor(out) as monitor, (out / 'engine.log').open('w') as stderr:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=stderr, text=True, bufsize=1)
        monitor.pid = proc.pid
        def reader():
            with (out / 'wire.log').open('w') as log:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    lines.put(line.rstrip())
            lines.put(None)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        def read_until(prefix, timeout=900):
            deadline, collected = time.monotonic() + timeout, []
            while True:
                line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
                if line is None or line.startswith('ERR '):
                    raise RuntimeError(f'engine stopped: {line}; see engine.log')
                collected.append(line)
                if line.startswith(prefix):
                    return line, collected
        try:
            read_until('READY ')
            # First request warms expert residency; subsequent five are measured.
            for repetition in range(6):
                started = time.monotonic()
                proc.stdin.write('GEN 256 temperature=0 ' + ','.join(map(str, ids)) + '\n')
                proc.stdin.flush()
                done, wire = read_until('DONE ')
                f = done.split()
                generated = [int(x.split()[1]) for x in wire if x.startswith('T ')]
                if len(generated) != 256 or int(f[1]) != 256 or int(f[2]) != length or int(f[8]) != 0:
                    raise ValueError('wrong output/prompt length or unexpected prefix reuse')
                row = {'prompt_tokens': length, 'output_tokens': 256, 'repetition': repetition,
                       'warmup': repetition == 0, 'prefill_ms': float(f[3]), 'decode_ms': float(f[4]),
                       'prefill_tokens_s': length / (float(f[3])/1000),
                       'decode_tokens_s': 256 / (float(f[4])/1000),
                       'wire_wall_s': time.monotonic()-started, 'done': done,
                       'output_sha256': hashlib.sha256(','.join(map(str, generated)).encode()).hexdigest()}
                results.append(row)
                p0.write_json(out / 'requests.json', results)
                (out / f'output-{repetition}.txt').write_text(tok.decode(generated))
                print(json.dumps(row), flush=True)
            proc.stdin.write('QUIT\n')
            proc.stdin.flush()
            proc.wait(timeout=120)
            if proc.returncode:
                raise RuntimeError(f'engine exit {proc.returncode}')
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            thread.join(timeout=5)
    measured = results[1:]
    summary = {'scope': 'synthetic exact-length prompts, warmed expert tiers, prefix reuse disabled',
               'profiling': False, 'prompt_tokens': length, 'warm_repeats': len(measured),
               'median_prefill_tokens_s': float(np.median([x['prefill_tokens_s'] for x in measured])),
               'median_decode_tokens_s': float(np.median([x['decode_tokens_s'] for x in measured])),
               'prefill_range': [min(x['prefill_tokens_s'] for x in measured), max(x['prefill_tokens_s'] for x in measured)],
               'decode_range': [min(x['decode_tokens_s'] for x in measured), max(x['decode_tokens_s'] for x in measured)],
               'identical_generated_tokens_across_repeats': len({x['output_sha256'] for x in results}) == 1,
               'argv': argv, 'environment': {k:v for k,v in env.items() if k.startswith('STRATA_')}}
    p0.write_json(out / 'benchmark_summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    from model_bringup import NAMES, SIZES
    from gguf_reader import GGUFFile
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['score', 'bench'])
    parser.add_argument('--length', type=int, choices=[2048,8192,32768], default=2048)
    args = parser.parse_args()
    out = Path(os.environ['EVIDENCE']).absolute() / (args.mode if args.mode == 'score' else f'bench-{args.length}')
    out.mkdir(parents=True, exist_ok=False)
    for name, n in zip(NAMES,SIZES):
        if (Path('/models') / name).stat().st_size != n:
            raise ValueError('wrong model shard size')
    signature = json.loads((PACK / 'v100_pack_complete.json').read_text())
    if signature['packer_sha256'] != p0.sha256(ROOT / 'tools/iq_pack.py'):
        raise ValueError('pack provenance mismatch')
    p0.write_json(out / 'provenance.json', {'commit': p0.command(['git','rev-parse','HEAD']),
                  'binary_sha256': p0.sha256(ENGINE), 'pack': signature,
                  'model_verification': 'size/structure only, not model SHA256',
                  'cpu_affinity': sorted(os.sched_getaffinity(0)),
                  'cgroup_memory_max': Path('/sys/fs/cgroup/memory.max').read_text()})
    gg = GGUFFile(Path('/models') / NAMES[0])
    tok = Tokenizer.from_dir(PACK / 'tokenizer')
    if args.mode == 'score':
        score(out, tok, len(gg.metadata['tokenizer.ggml.tokens']))
    else:
        template = ChatTemplate(PACK / 'tokenizer/chat_template.jinja',
                                bos_token=gg.metadata['tokenizer.ggml.tokens'][gg.metadata['tokenizer.ggml.bos_token_id']])
        bench(out, tok, template, args.length)


if __name__ == '__main__':
    main()
