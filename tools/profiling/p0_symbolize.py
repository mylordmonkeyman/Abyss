#!/usr/bin/env python3
"""Read-only ELF attribution for the completed immutable-base allocation capture."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools/bench'), str(ROOT / 'tools/v100')]
from v100_base_replay import BINARIES, EXPECTED
import p0


def elf_address(caller, maps, loads, executable):
    for line in maps.splitlines():
        fields = line.split()
        lo, hi = (int(x, 16) for x in fields[0].split('-'))
        if lo <= caller < hi:
            if len(fields) < 6 or fields[-1] != str(executable):
                return None
            offset = caller - lo + int(fields[2], 16)
            for file_offset, virtual, file_bytes in loads:
                if file_offset <= offset < file_offset + file_bytes:
                    return virtual + offset - file_offset
            raise ValueError('caller file offset has no ELF LOAD segment')
    return None


def main():
    out = Path(os.environ['EVIDENCE']).absolute()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / 'results/ci-v100-p0-ledgers-39194149121-1/memory'
    report = dict(source_run=39194149121, scope='read-only ELF/source attribution; no CUDA invocation', bases={})
    for base, binary in BINARIES.items():
        if p0.sha256(binary) != EXPECTED[base]:
            raise ValueError('immutable binary identity changed')
        with gzip.open(source / (base + '-allocations.jsonl.gz'), 'rt') as f:
            events = [json.loads(line) for line in f]
        with gzip.open(source / base / 'telemetry.jsonl.gz', 'rt') as f:
            maps = next(s['process_maps'] for s in map(json.loads, f) if s.get('process_maps'))
        headers = p0.command(['readelf', '-lW', str(binary)])
        if headers['returncode']:
            raise ValueError('ELF program headers unavailable')
        loads = []
        for line in headers['stdout'].splitlines():
            fields = line.split()
            if fields and fields[0] == 'LOAD':
                loads.append((int(fields[1], 16), int(fields[2], 16), int(fields[4], 16)))
        callers = sorted({e['caller'] for e in events if e.get('caller')})
        resolved = []
        for caller in callers:
            address = elf_address(caller, maps, loads, binary)
            symbol = p0.command(['addr2line', '-Cf', '-e', str(binary), hex(address - 1)]) if address else None
            # Return addresses point after the call; subtract one for source attribution.
            resolved.append(dict(runtime_caller=caller, elf_return_address=address, addr2line=symbol))
        getters = {name: p0.command(['objdump', '-d', '--disassemble=' + name, str(binary)])
                   for name in ('ggml_cpu_has_avx2', 'ggml_cpu_has_avx512')}
        report['bases'][base] = dict(binary_sha256=EXPECTED[base], headers=headers,
                                    allocation_callers=resolved, compiled_cpu_getters=getters)
    p0.write_json(out / 'symbols.json', report)
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > 4 * 1024**2:
        raise ValueError('symbol report exceeds 4 MiB')
    if os.environ.get('GITHUB_ENV'):
        with open(os.environ['GITHUB_ENV'], 'a') as f:
            f.write('P0_SYMBOLS_CAP=true\n')


if __name__ == '__main__':
    main()
