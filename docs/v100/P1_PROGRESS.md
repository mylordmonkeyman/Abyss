# P1 — Volta indexer implementation

Working branch: `feature/deepseek-v100-p1`, based on selected upstream v1.3.31
`7c893ac17e61ec45c934b3278356807fc4626c3c`, with the P0 evidence/tooling history
merged without rewriting `feature/deepseek-v100-perf` or its frozen references.
The automatic merge was clean; this is not evidence that runtime parity passes.

## Initial checkpoint — 19 October 2026

- P0 measured indexer residence at 22.49% of 32K and 49.38% of 128K historical
  prefill phase wall, warranting the P1 experiment. See `P0_LONG_RESULTS.md`.
- The previously tested MMA884 primitive is reused with its four independent
  quad-pair layout and typed `r`/`+f` register constraints.
- `dsv4_volta_idx.cuh` is an isolated, one-token correctness-first scorer. Four
  or eight warps/CTA score 32 key rows/warp, cache FP16 query hi/lo and FP32 head
  weights, keep independent FP32 accumulation chains, then apply ReLU per head.
  Masked key tails still participate with all lanes in each MMA instruction.
- The isolated executable compares every score to FP64 and same-branch F32,
  invokes the **unchanged upstream selector**, checks membership/error envelope,
  tie ordering, guards, repeat graph launches, and records top-512 overlap and
  rank-512/513 margin. Input populations are identical for four/eight warps.
- V100 workflow collects memcheck/synccheck, PTX/SASS, ptxas resources, runtime
  occupancy, six interleaved timing pairs and sampled clock/power/thermal state.
  These are synthetic scorer timings, not model throughput evidence.
- The source-compiled workspace test now selects the current immutable layout
  for VERSION 1.3.31, including upstream's `pre2`/`onehot` buffers. Both frozen
  layout formulas and P0 evidence remain unchanged. All 48 host tooling tests pass.

## Gates before integration or promotion

Physical scorer/sanitizer/timing results are pending. The workflow also requests
a bounded selected-base dispatch/allocation/route audit. This is **part of** the
required repeated P0b audit, not a replacement for long-context critical-path
and model quality revalidation on the ported branch.

Production code does not call this scorer. No new runtime control/default is
introduced. Finite FP32 queries outside the unscaled FP16 range are unsupported
by this candidate; microtests explicitly reject that input domain rather than
clamp it. Tiny-query residual underflow is reported against FP64. Synthetic
error screening does not qualify model arithmetic.

Remaining P1 work: inspect physical results and fix failures; finish selected-
base P0b revalidation; implement/test score-only runtime dispatch with legacy
F32 override; batched correctness and key-reuse candidates; bounded FP32-scratch
cuBLAS candidate; score/selection boundaries, same-pack teacher-forced logits,
real-model paired performance and sm80/HIP regression checks. Keep F32 for
unsupported or losing shapes. P2/P3 work is outside this checkpoint.
