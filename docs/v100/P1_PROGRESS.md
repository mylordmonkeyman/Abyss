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

The initial physical scorer/sanitizer run and selected-base dispatch audit pass
in [run 39256759600](https://github.com/mylordmonkeyman/Abyss/actions/runs/39256759600),
source `a14cbe935c04c98c30da16a626f0f5fb440e3cec`. This is **part of** the
required repeated P0b audit, not a replacement for long-context critical-path
and model quality revalidation on the ported branch.

## First physical result: keep F32, correct layout before further measurement

`p1-indexer-2026-10-19.json` is reproduced by
`tools/profiling/summarize_p1_indexer.py` from artifact `12187579922`, ZIP SHA256
`fc08c550c3e3a2ceb35b9bb168e61288ef65c5ebe2823d3bd4be813dae1dd2af`.
All 26 synthetic cases, graph replay and unchanged selector checks pass;
memcheck/synccheck report zero errors for fragment/scorer/F32 component tests.
Emitted PTX contains MMA884, SM70 SASS contains HMMA. Normal-query FP64 relative
L2 is about 2.7–3.5e-7 at 513/31,250 rows, with 512/512 selected membership.
The 4/8-warp candidates use 54/56 registers, 33,024 shared bytes, zero reported
local bytes and two active CTAs/SM. This is reported resource occupancy, not a
hardware-counter measurement. Three timing-window telemetry samples show no
thermal reason; this short window does not establish sustained model behavior.

The candidate **loses every timed shape**: four-warp median score elapsed changes
are +61.57% at 8,192 rows and +79.10% at 31,250, with worse short shapes. The
eight-warp candidate is +214.38%/+80.57%. No optimized runtime dispatch is
admitted. Tiny queries scaled by 2^-20 exhibit **4.308% relative L2**, despite
only 1.50e-6 maximum absolute error. That stress case passes its explicit
absolute-error diagnostic bound, **not** the nominal component relative target.
No model quality or tiny-query numerical qualification is claimed.

A second isolated experiment pads query shared-memory rows from 128 to 130
halves. The original stride maps simultaneous head reads to the same banks;
padding changes that static access pattern. Run `39257756365` passes all 26
cases and both sanitizers, but still **loses every timed shape**. Four-warp
median changes are +19.43% at 8,192 and +30.75% at 31,250 rows; eight-warp
changes are +134.03%/+35.33%. Separate campaigns are not an interleaved
original-versus-padded comparison. `p1-indexer-padded-2026-10-19.json` preserves
the six F32-versus-padded pairs; artifact `12190076611`, ZIP SHA256
`d471f100267428f6ddc1e471e488d22f5195ced073a2bf7551786a070d610113`.
Shared storage is 33,536 bytes, still two active CTAs/SM, zero local bytes.
The tiny-query limitation remains. Keep production F32 and upstream selectors.

The hosted HIP-name check found two CUDA-only diagnostic resource-query types
outside an explicit CUDA guard. The test now has `#ifndef STRATA_USE_HIP` and
the local compatibility check passes, without adding HIP aliases or changing
HIP execution. Both hosted jobs pass on the corrected branch.

## Selected-base bounded model audit

`p1-base-audit-2026-10-19.json` summarizes artifact `12187754461`, ZIP SHA256
`fa1dab43ab8d6453461437c90a182c5659af4d67f409e9f8494be00a1c61cc8c`.
Four full-model requests complete: 600 and 8,192 tokens, two each, 16 outputs.
There are 26,338 dispatch keys, zero dropped calls/unresolved decode joins,
2,920 conserved route records and zero route drops. The 177-event allocation
capture is complete, with no failures or ledger errors. Active-window minimum
MemAvailable is 49,648,242,688 bytes; no sampled thermal reason, power limits
observed. Context and reservations differ from immutable long-context memory
captures; these peaks must not be used as a regression comparison.

Rootless working-branch 32K/128K critical-path revalidation passes in run
`39257756365`, using the same engine binary SHA256 as the dispatch audit:
`8e8f8278fd3e3729ae02d16d4af18625ccf1f31d211e15c5554b08f051e0dee3`.
`p1-base-traces-2026-10-19.json` and `summarize_p1_traces.py` preserve both
digest-verified artifacts and raw report identities. At 32K, prefill MMQ,
attention and indexer residence are 28.23%, 26.15%, 22.46%; at 128K, indexer,
attention and MMQ are 49.45%, 21.19%, 16.26%. This is a main-stream wall-lane
proxy, not a complete causal DAG. Decode populations/gaps vary; do not pool
them or assign gaps to CPU/disk. Minimum observed MemAvailable is 47.06 GB
(decimal) across both captures, no sampled thermal reason; power caps occur.
Raw profiler reports remain on the runner. Full branch quality comparison,
batched scorer candidates and P1 acceptance remain open.

`v100-p1-quality.yml` adds a separately serialized ported-F32 check. It reuses
the frozen teacher fixture and digest-verified immutable v1.3.31 distributions,
then runs the working branch twice. Missing cached reference distributions are
regenerated from the immutable binary; changed cached bytes fail explicitly.
Its 16/8 GiB preflight/own-child safety stops, process telemetry, 64 MiB artifact
cap and temporary raw distributions preserve the existing runner constraints.
This is a port/repeatability screen, not optimized scorer model qualification.

## Batched cuBLAS prototype (P1e), awaiting physical validation

`test_dsv4_cublas_indexer` implements a separate, test-only scorer using **FP32
SGEMM**, not an asserted tensor-core path. Queries retain their original FP32
values; each 512-row FP16 key tile is converted exactly to FP32 once and reused
across 32-token query tiles. The per-head dot matrix stays FP32. A warp epilogue
applies ReLU independently to every head, then its weight, with token-specific
visibility. No full `[T, keys, 64]` global intermediate is allocated.

Explicit scratch is 20.25 MiB: 16 MiB cuBLAS workspace, 4 MiB dot tile and
256 KiB converted keys, statically capped below 32 MiB. Common input/output
arrays and opaque cuBLAS handle/module costs are separate; this is not a total
device-memory accounting claim. Production integration requires reconciliation.

The prototype uses unchanged upstream selection in **T single-token launches**,
while the upstream batch baseline uses one batched selection launch. Reported
event times include this prototype penalty, conversion, GEMM and epilogue; they
are full isolated indexer stages, never score-only or model throughput. A loss
does not rule out an improved batched-selector integration. Both candidate and
baseline compute small visible populations for component testing; production's
existing skip when top-512 covers every row remains intact.

Tests compare every visible score against same-branch F32 (relative L2, p99/max
absolute error), sample FP64 scores, reject future/duplicate IDs, verify selected
membership against a measured error envelope, and disclose minimum 512/513
margins and overlap. Causal tails, T=1/3/8/33, exact ties and tiny queries are
included. The timed matrix covers T=1/2/3/8/16/32/256 against approximately
512/2K/8K/31K compressed rows, plus T=1024/4096 at 2K/8K/32K/120K starting
contexts. Six interleaved pairs per shape include telemetry and own-child
16/8 GiB memory guards. The separate key-reuse **MMA** batched candidate,
real-model candidate quality and dispatch acceptance are still pending.

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
