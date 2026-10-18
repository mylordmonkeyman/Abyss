#!/usr/bin/env python3
"""v8 P0 ancestry/feature evidence; read-only source/header census, not a quality pass."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

import p0

ROOT = p0.ROOT
STRATA = '477ac13cbe8e582c20484871ce459d20d82df09d'
GGML = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


def inspect():
    paths = ['src/prefill/moe_mmq.cu', 'src/core/glm_prefill.cu',
             'src/core/glm_fast_path.cu', 'src/kernels/cpu/native_expert.cpp',
             'src/kernels/cpu/iq_avx2.cpp', 'src/kernels/cuda/dsv4_fast.cu']
    source = {p: (ROOT / p).read_text() for p in paths}
    # Explicit source observations, scoped to these symbols. Neither literal
    # absence nor presence proves a runtime bottleneck or a feature enabled.
    candidates = [
        dict(id='P3.12', status='absent', file=paths[0], symbol='quantize',
             evidence='Indexed quantize_mmq_q8_1_cuda; no engine scatter call. glm_prefill run_set uses '
                      'row_tok repeatedly over window-sized expert groups; down GU rows are expert-specific.',
             decision='Isolated vendor API test first; full-route scatter is not a drop-in for window-sized Xq.'),
        dict(id='P3.2f', status='different', file=paths[3], symbol='pick_rows1 / iq256_rows1_supported',
             evidence='Existing exact singleton IQ3_XXS/IQ3_S; iq_avx2.cpp supports singleton types 18/21.',
             decision='Named IQ2_XXS case is inapplicable if absent from inventory. Upstream exact-one also '
                      'explicitly supports type 17 (IQ2_XS): retain that separately validated CPU candidate.'),
        dict(id='P4i', status='different', file=paths[2], symbol='STRATA_GLM_CPU_PLAN / STRATA_GLM_PCIE_SHARE',
             evidence='Startup CPU/DMA/mapped-access calibration and fixed route plan already exist.',
             decision='Observe per-layer prediction error before introducing online allocation.'),
        dict(id='P3.13', status='absent', file=paths[0], symbol='Context::run / dispatch',
             evidence='Contiguous Product::w; no STRATA_MMQ_DIRECT pointer-table path.',
             decision='Optional after real gather cost; MXFP4 cannot inherit another format whitelist.'),
        dict(id='P3.14', status='absent', file=paths[2], symbol='STRATA_HC_Q8',
             evidence='No upstream HC Q8 control in audited engine source.',
             decision='Optional lossy storage only after separate resident HC census.'),
        dict(id='P4j', status='different', file=paths[2], symbol='numa_pinned / bg_pace / promote',
             evidence='Existing pinned mapped pool, service and paced background promotion; no upstream '
                      'STRATA_EXCHANGE_ROTATE/STRATA_PIN_BY_NEED control.',
             decision='No replacement cache; require trace of actual missing overlap/host commit copies.'),
        dict(id='P4k', status='absent', file=paths[1], symbol='prefill-head-alias',
             evidence='No upstream head-alias option in audited prefill source.',
             decision='Optional only after measured head/workspace pressure.'),
        dict(id='P2g', status='inapplicable', file=paths[5], symbol='attend_batch_kernel',
             evidence='Per-query n_raw and raw0 addressing; existing V100 trace launches attend_batch.',
             decision='Do not port Xolotl masked-union workaround; retain direct per-query gathers.'),
        dict(id='Abyss RAM_THP', status='different', file=paths[2], symbol='numa_pinned',
             evidence='buddyinfo compaction guard already exists; newer STRATA_GLM_RAM_THP control absent.',
             decision='No base update; later one-node/partial hugepage policy not incorporated.'),
        dict(id='Abyss STREAM_FETCH', status='absent', file=paths[2], symbol='STRATA_GLM_STREAM_FETCH',
             evidence='Later Abyss streaming-fetch control absent.',
             decision='No base update; existing staged fetch preserved.'),
    ]
    vendor = ROOT / 'third_party/llama.cpp'
    header = vendor / 'ggml/src/ggml-cuda/quantize.cuh'
    api = header.read_text() if header.is_file() else None
    return dict(schema_version=1, specification='8.0',
                specification_sha256=p0.sha256(ROOT / 'docs/v100/IMPLEMENTATION_SPEC.md'),
                working_commit=p0.command(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
                git_describe=p0.command(['git', 'describe', '--always', '--dirty', '--tags'])['stdout'].strip(),
                pinned_abyss_base=p0.BASE, strata_reference=STRATA,
                abyss_differential_reference='7c893ac17e61ec45c934b3278356807fc4626c3c',
                base_changed=False, ggml_expected=GGML,
                ggml_actual=p0.command(['git', '-C', str(vendor), 'rev-parse', 'HEAD']).get('stdout', '').strip(),
                vendor_scatter_api_declared=None if api is None else 'void quantize_scatter_mmq_q8_1_cuda(' in api,
                source_sha256={p: hashlib.sha256(s.encode()).hexdigest() for p,s in source.items()},
                candidates=candidates,
                model_integrity='Sizes and GGUF structure only; shard SHA256 unverified (not fabricated).',
                proposed_controls_implemented=[], production_change=False,
                unresolved_gates=['Hardware counters denied: ERR_NVGPUCTRPERM',
                                  'QPI counters unmeasured', 'Greedy repeat variability unisolated',
                                  'Model-exact extended parity and dedicated MXFP4 row parity incomplete'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='First mounted shard; header/size census only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = inspect()
    if args.model:
        files = p0.shards([args.model])
        rows = p0.inventory(files, args.output / 'p0_tensor_inventory.csv')
        routed = [r for r in rows if r['role'].startswith('routed_expert_')]
        report['shards'] = [dict(name=p.name, bytes=p.stat().st_size, sha256=None) for p in files]
        report['routed_inventory'] = [dict(role=role, type=typ, tensors=n)
                                     for (role,typ),n in sorted(collections.Counter(
                                         (r['role'],r['type']) for r in routed).items())]
        report['iq2_xxs_routed_tensors'] = sum(r['type'] == 'IQ2_XXS' for r in routed)
        report['iq2_xxs_candidate_applicable_to_this_pack'] = report['iq2_xxs_routed_tensors'] > 0
        report['mxfp4_routed_layers'] = sorted({int(r['layer']) for r in routed if r['type'] == 'MXFP4'})
    p0.write_json(args.output / 'p0_upstream_delta.json', report)
    p0.write_json(args.output / 'toolchain_manifest.json', p0.manifest(os.environ.get('ABYSS_GPU', '0')))
    with (args.output / 'p0_strata_delta_candidates.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['id','status','file','symbol','evidence','decision'])
        writer.writeheader(); writer.writerows(report['candidates'])
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','candidates')}), flush=True)


if __name__ == '__main__':
    main()
