#!/usr/bin/env python3
"""Size-checked, read-only GGUF bring-up; not a full P0 performance qualification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import p0

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from gguf_reader import GGUFFile
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate

SIZES = (5257664, 49437013568, 47390237120)
NAMES = tuple(f'DeepSeek-V4-Flash-0731-UD-Q2_K_XL-{i:05d}-of-00003.gguf' for i in range(1, 4))


def main():
    out = Path(os.environ['EVIDENCE']).absolute()
    out.mkdir(parents=True, exist_ok=True)
    root = Path('/actions-runner/abyss-models/deepseek-v4-flash-0731-ud-q2-k-xl')
    root.mkdir(parents=True, exist_ok=True)
    model = [Path('/models') / name for name in NAMES]
    identity = {'integrity': 'exact byte sizes and GGUF structural checks; SHA256 not verified',
                'huggingface_revision': 'fbbb5b93fb787c21338159b0af3318bb3f4d9768',
                'files': [{'name': p.name, 'bytes': p.stat().st_size, 'expected_bytes': n}
                          for p, n in zip(model, SIZES)]}
    p0.write_json(out / 'model_identity.json', identity)
    for src, n in zip(model, SIZES):
        if src.stat().st_size != n:
            raise ValueError(f'wrong model size: {src}')
        dest = root / src.name
        if dest.is_symlink() and dest.readlink() == src:
            continue
        if dest.exists() or dest.is_symlink():
            raise ValueError(f'refusing to replace existing path: {dest}')
        dest.symlink_to(src)
    gg = GGUFFile(model[0])
    if gg.metadata.get('general.architecture') != 'deepseek4':
        raise ValueError('expected deepseek4 GGUF')
    p0.inventory(model, out / 'p0_tensor_inventory.csv')
    p0.write_json(out / 'toolchain_manifest.json', p0.manifest(os.environ.get('ABYSS_GPU', '0')))
    # Header census is static metadata, not a runtime kernel/residency census.
    p0.write_json(out / 'model_geometry.json', {k: v for k, v in gg.metadata.items()
                  if k.startswith('deepseek4.') or k.startswith('general.')})
    pack = root / 'pack-dsv4'
    signature = {'source_sizes': list(SIZES), 'packer_sha256': hashlib.sha256(
        (ROOT / 'tools/iq_pack.py').read_bytes()).hexdigest()}
    marker = pack / 'v100_pack_complete.json'
    if not marker.exists() or json.loads(marker.read_text()) != signature:
        if pack.exists():
            raise ValueError('existing pack lacks matching completion marker; refusing to overwrite')
        stage = root / 'pack-dsv4.partial'
        if stage.exists():
            shutil.rmtree(stage)  # Only our unfinished staging directory.
        with (out / 'pack.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'tools/iq_pack.py'), '--gguf',
                            str(root / NAMES[0]), '--out', str(stage), '--compat-bf16'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        for name in ('index.txt', 'native_experts.txt', 'dense.bin', 'tokenizer/vocab.json'):
            if not (stage / name).is_file():
                raise ValueError(f'incomplete pack: {name}')
        p0.write_json(stage / marker.name, signature)
        stage.rename(pack)
    p0.write_json(out / 'pack_provenance.json', signature)
    tok = Tokenizer.from_dir(pack / 'tokenizer')
    bos_id = gg.metadata.get('tokenizer.ggml.bos_token_id')
    bos = gg.metadata['tokenizer.ggml.tokens'][bos_id] if isinstance(bos_id, int) else ''
    template = ChatTemplate(pack / 'tokenizer/chat_template.jinja', bos_token=bos)
    cases = [('short', 'Explain why the sky is blue in two sentences.', 16),
             ('prefill', ('Context: Volta GPUs have tensor cores and high bandwidth memory.\n' * 32)
              + 'Summarize this context and explain how memory bandwidth affects inference.', 256)]
    results = []
    for name, text, count in cases:
        rendered = template.render([{'role': 'user', 'content': text}], enable_thinking=False)
        ids = tok.encode(rendered, parse_special=True)
        token_file = out / f'{name}-input.txt'
        token_file.write_text(','.join(map(str, ids)) + '\n')
        (out / f'{name}-prompt.txt').write_text(rendered)
        argv = [str(ROOT / 'build-v100-diagnostics/strata'), '--glm-pack', str(pack),
                '--tokens-file', str(token_file), '--max-new', str(count), '--max-context', '4096',
                '--prefill', 'auto', '--temperature', '0']
        env = dict(os.environ, STRATA_DS4_SPEC='0', ABYSS_V100_PROFILE='1', STRATA_GLM_RAM_GB='72')
        samples, stop = [], threading.Event()
        def sample():
            while not stop.is_set():
                samples.append({'monotonic_s': time.monotonic(), 'gpu': p0.command([
                    'nvidia-smi', '-i', os.environ.get('ABYSS_GPU', '0'), '-q', '-x']),
                    'meminfo': Path('/proc/meminfo').read_text()})
                stop.wait(5)
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        started = time.monotonic()
        rc, status = None, 'timeout'
        try:
            with (out / f'{name}.log').open('w') as log:
                proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=2400)
                rc, status = proc.returncode, 'completed'
        except subprocess.TimeoutExpired:
            pass
        finally:
            stop.set()
            thread.join()
            p0.write_json(out / f'{name}-telemetry.json', samples)
        log = (out / f'{name}.log').read_text(errors='replace')
        generated = [int(line.split()[1]) for line in log.splitlines() if line.startswith('T ')]
        done = [line for line in log.splitlines() if line.startswith('DONE ')]
        result = {'case': name, 'argv': argv, 'status': status, 'returncode': rc,
                  'wall_s_including_load': time.monotonic() - started, 'prompt_tokens': len(ids),
                  'generated_tokens': len(generated), 'done': done,
                  'passed': rc == 0 and len(generated) == count and len(done) == 1}
        (out / f'{name}-output.txt').write_text(tok.decode(generated))
        results.append(result)
        p0.write_json(out / 'bringup_results.json', results)
        print(json.dumps(result), flush=True)
        if not result['passed']:
            print(log[-12000:], flush=True)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
