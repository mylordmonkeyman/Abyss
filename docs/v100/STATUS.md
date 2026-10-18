# V100 implementation status — 18 October 2026

Branch base: upstream `47001f31cbfba14bcbf4c083e3af07c50707ec0c` (v1.3.28).
Production arithmetic and dispatch remain unchanged. No performance improvement is claimed.

## Confirmed on the actual V100

- [Initial build/device run 39100453015](https://github.com/mylordmonkeyman/Abyss/actions/runs/39100453015),
  commit `41affbb`: CUDA 12.8.93, Tesla V100-PCIE-32GB, cc 7.0; SM70 engine and device
  executable built; device allocation/kernel smoke passed; memcheck and synccheck each
  reported **0 errors**. The existing build cache was reused on this run.
- [DeepSeek component run 39100790344](https://github.com/mylordmonkeyman/Abyss/actions/runs/39100790344),
  commit `1a3dc6d`: **26 synthetic cases passed** under default dispatch and again with
  `STRATA_DS4_IDX=1 STRATA_DS4_ATTN=1`. The full component suite also passed memcheck
  and synccheck, both with **0 errors**.
- Indexer cases check FP64 scores, GPU-score ordering, tie-breaking, meaningful-margin
  FP64 membership, causal visibility, 512/513 cutoff, 8,192 visible rows, 32-token tile
  tails and untouched output guards. Raw-ring cases check wrap/chunk load/store.
- Attention cases use the real 64-head × 512 latent K=V shape, raw/CSA/HCA paths,
  sink regimes, 128-token raw-window and compression boundaries, and plain/YaRN inverse
  RoPE. Maximum observed attention relative L2 versus the scalar FP64 reference was
  about **6.10e-7**, maximum absolute error about **7.40e-8**, on these synthetic inputs.
  These are component results, not full-model quality numbers.

The first parity-harness build exposed an NVCC initializer-list deduction issue;
explicit shape arrays fixed it in `1a3dc6d` before the passing run above.

## Other changes and checks

- P0 capture/inventory tooling: nine host-only tests pass; synthetic GGUF census and
  host-only CMake build passed previously and in hosted CI.
- `8324f9e` repairs the inherited HIP `cudaMemcpyKind` mapping. The mapping checker and
  its two tests pass locally; hosted CI passes all 130 server and 202 installer/tool
  tests. This does not claim a new ROCm runtime qualification.
- Isolated P1a MMA884 primitive/test added for four quadpairs, all accumulator coordinates,
  typed PTX operands, masks/K tails, signed/zero/subnormal stress, and CUDA graph replays.
  It is **not wired into inference**. [Run 39101028811](https://github.com/mylordmonkeyman/Abyss/actions/runs/39101028811),
  commit `ed41547`, passed all **15 MMA contract cases**, device-scalar/FP64 comparison,
  memcheck/synccheck (0 errors), and selected SM70 image/PTX/SASS checks. All contract
  cases reported zero absolute error on these exactly representable synthetic inputs,
  including the subnormal case. This does not establish error bounds for arbitrary
  full-model FP32-to-FP16 splits. The emitted PTX contains the expected MMA884 opcode
  and the selected SM70 SASS contains HMMA. Full-engine opcode auditing is not claimed.
- [Hosted CI run 39101028820](https://github.com/mylordmonkeyman/Abyss/actions/runs/39101028820)
  also passed on `ed41547`. The GPU container and existing build cache were reused;
  no runner/container replacement was needed for these tests.

## P0 remains partial

The component test bypasses q-projection/compressor construction and grouped output
projections; it does not establish complete model-exact MLA stage parity, quantized
MXFP4 expert correctness, model logits, or task behavior. The indexer tests intentionally
invoke the scorer directly, including below top-512 visibility, without changing the
production caller's skip policy.

The read-only `/models` mount is now available. [Full-model run 39137399940](https://github.com/mylordmonkeyman/Abyss/actions/runs/39137399940),
commit `852ef95`, passed both initial runtime cases with coherent generated text:

| Prompt tokens | Generated tokens | Prompt time | Decode time |
| --- | --- | --- | --- |
| 14 | 16 | 844.7 ms | 758.9 ms |
| 464 | 256 | 1,892.4 ms | 9,449.7 ms |

The second case used batched prefill for 463 tokens and the token path for the final
prompt token. These runs enable stage/gap profiling: the approximately 245 prompt
tokens/s and 27.09 generated tokens/s in the second case are **instrumented bring-up
observations**, not an unprofiled baseline or an improvement claim. Loading and
shutdown are excluded from those phase rates; total process wall time was about
101 seconds. Speculation was off, capacity 4,096, RAM budget 72 GiB, CPU lane 28 threads.
The second process reused the first process's expert-usage profile, so these two
cases are not a controlled A/B pair.

Static inventory confirms MXFP4 routed down projections at layers **26 and 42**.
The runner reports 6.17 GiB dense GPU weights, 24.18 GiB GPU expert pool and
68.28 GiB pinned expert tier; these are not a complete process/page-cache ledger.
Five-second GPU samples observed 36–44 C and no active thermal slowdown reason;
they do not establish sustained benchmark conditions. Driver 580 uses
`instant_power_draw`/`current_power_limit` XML tags; the capture parser now supports
these tags, with a regression test. Original XML was preserved, allowing recovery
of power data from this run without rerunning inference.

The pack is persistent under `/actions-runner/abyss-models/`; GGUF shards are
symlinked to the read-only mount, with no 97-GB duplicate or model edits. Model
verification here is exact byte sizes and GGUF structure, **not a verified SHA256**.
Artifact `12140800382` is 48,483 bytes; its SHA256 is
`506165216495d657c35bf9578e6f018ad8a0516981b8d19de3ea86f9b2629a8f`.
The initial harness incorrectly supplied a `generate` subcommand; `852ef95` fixed
that invocation before the passing run. Hosted CI and GPU components also passed
again on that commit (130 server / 202 tool tests, with existing skips).
Hosted CI now uses the same pinned allowlisted checkout action and the image's
Python, removing the extra setup-python action dependency.

Still required: 2K/8K/32K prompts and warm-repeat benchmarks, dedicated MXFP4-down parity,
teacher-forced logits/boundary and reused-cache tests, runtime type/shape/kernel census,
VRAM/pinned/page-cache/workspace ledger, sustained thermal/QPI/PCIe evidence, supported
profiler counters and an annotated critical-path trace. No production P1/P2/P3 routing
may be promoted on these synthetic tests alone.
