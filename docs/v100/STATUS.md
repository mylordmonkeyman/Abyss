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

The target runner reports **no `/models` directory and zero visible GGUFs** in
`model_availability.json`. Full DeepSeek runtime bring-up therefore requires making
the shipped model/pack available through the explicit model-only mount. See
[runner security/model access](../../deploy/v100-runner/SECURITY.md#optional-model-access).
Do not expose unrelated host files to solve that prerequisite.

Still required: shipped UD-Q2_K_XL full inference/prefill and both MXFP4-down layers,
teacher-forced logits/boundary and reused-cache tests, runtime type/shape/kernel census,
VRAM/pinned/page-cache/workspace ledger, sustained thermal/QPI/PCIe evidence, supported
profiler counters and an annotated critical-path trace. No production P1/P2/P3 routing
may be promoted on these synthetic tests alone.
