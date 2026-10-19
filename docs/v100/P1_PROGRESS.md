# P1 — Volta indexer implementation

Working branch: `feature/deepseek-v100-p1`, based on selected upstream v1.3.31
`7c893ac17e61ec45c934b3278356807fc4626c3c`, with the P0 evidence/tooling history
merged without rewriting `feature/deepseek-v100-perf` or its frozen references.
The automatic merge was clean; this is not evidence that runtime parity passes.

## Current checkpoint

| Gate | Measured state |
| --- | --- |
| Selected-base port | Full-model dispatch/route audit, 32K/128K timelines, bounded F32 repeatability screen pass; frozen P0 evidence preserved |
| One-token MMA884 scorer | Synthetic contracts and sanitizers pass within disclosed bounds; both layouts lose versus F32, tiny-query relative error remains unqualified. Keep F32 |
| Batched FP32 cuBLAS scorer | Three bounded variants measured, then an expanded 48-shape matrix. At the model's actual T=2048, indexer stage elapsed is -9.20% at 96K and -11.82% at 120K starting context; 64K is noise, shorter contexts lose |
| Production integration / P1 exit | **Open**: explicit integration, graph/sanitizer and bounded 100K-context model-KL screen pass; direct attention/boundary and whole-model performance acceptance pending. Default arithmetic unchanged |

Next: run the gated same-binary whole-model A/B at 8K and 128K, three
process pairs with two requests per fixture. Actual long-context dispatch and
32-position model-KL/repeatability now pass. The batched MMA correctness/key-reuse
experiment and full boundary, direct downstream-attention and sm80/HIP acceptance
remain P1 work. No automatic dispatch or default arithmetic is changed.

## Experimental integration checkpoint — bounded model validation passed

The shared cuBLAS scorer now has an **explicit experimental** runtime path.
`ABYSS_V100_IDX=cublas` on SM70 uses only C4, 64 heads, dimension 128,
512 selection, T=2048 and starting positions 98,304 through 120,000;
other chunks retain F32 and log the fallback reason. This interval is an
experimental investigation regime, not a qualified automatic threshold.
`auto` remains upstream; `f32` forces the old scorer and the legacy
`STRATA_DS4_IDX=1` wins over every experimental setting. Forced `volta`
fails clearly because the measured MMA candidate remains unqualified.
Forced cuBLAS on HIP/non-SM70 or incompatible model geometry fails clearly.

The runtime and microtest share one implementation. It borrows 32 MiB from
`B.o` (dead until `attend_batch`) and 256 KiB from `B.oa` (dead until output
LoRA writes with beta=0), and uses the existing prefill handle/16 MiB workspace
on the same stream. No new persistent allocation or handle mutation is added.
The caller checks capacities; existing score/selection buffers remain distinct.
Nondefault-stream graph capture/replay is added to each component parity case.
The shared scorer passes nondefault-stream graph capture/replay and both
CUDA sanitizers; the actual model path additionally passes the bounded logit
screen below. This does not replace direct attention/selection diagnostics.

The integration workflow builds the engine and shared scorer, runs parity,
graph replay and both sanitizers, then the 48-shape matrix before four
same-binary long-context quality arms (F32/candidate/candidate/F32).
A verified frozen mixed-fixture prefix places a full 2048-token chunk at
98,304, followed by 32 scored positions. Each candidate arm must contain actual
cuBLAS dispatch records at that shape. Logit dumps stay temporary; identities,
KL, dispatch records and sustained telemetry are bounded artifacts. These
profiled quality arms are excluded from ordinary throughput comparisons.
[Integration run 39328349428](https://github.com/mylordmonkeyman/Abyss/actions/runs/39328349428),
source `c157de24442a7c4ea71c2af8464b6d214c7548b4`, has passed the SM70 engine
build, shared scorer parity/nondefault-stream graph replay, both sanitizers
and the 48-shape matrix. Hosted CI also passed. The four real-model quality
arms completed successfully. No P1 exit or model speedup is claimed.
`summarize_p1_integrated.py` will digest-check the bounded artifact and require
actual target-shape dispatch and all four comparisons. `p1_model_ab.py` is
prepared for three same-binary process pairs with 8K/128K fixtures and verified
checkpoint reuse; it refuses a binary differing from the committed quality
evidence. Whole-model dispatch is gated on that quality evidence.

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
Raw profiler reports remain on the runner. Batched candidates and P1 acceptance
remain open; the bounded port-quality result follows.

`v100-p1-quality.yml` adds a separately serialized ported-F32 check. It reuses
the frozen teacher fixture and digest-verified immutable v1.3.31 distributions,
then runs the working branch twice. Missing cached reference distributions are
regenerated from the immutable binary; changed cached bytes fail explicitly.
Its 16/8 GiB preflight/own-child safety stops, process telemetry, 64 MiB artifact
cap and temporary raw distributions preserve the existing runner constraints.
This is a port/repeatability screen, not optimized scorer model qualification.

Run `39258176612` passes the bounded F32 port/repeatability screen:
`p1-base-quality-2026-10-19.json`, artifact `12190232042`, ZIP SHA256
`b38ffef13845d1446708156985fae36b9353b90e6d623563ac4481d6c67424a4`.
Working repeat mean KL is 1.0291e-7; immutable-v1.3.31 versus working comparisons
are 1.5664e-7 and 1.9662e-7, max position 2.5819e-6, all top-1 32/32. These
32 teacher-forced positions are a port screen, not the P1 candidate corpus gate.

## Batched cuBLAS prototype (P1e)

`test_dsv4_cublas_indexer` implements a separate, test-only scorer using **FP32
SGEMM**, not an asserted tensor-core path. Queries retain their original FP32
values; each 512-row FP16 key tile is converted exactly to FP32 once and reused
across 32-token query tiles. The per-head dot matrix stays FP32. A warp epilogue
applies ReLU independently to every head, then its weight, with token-specific
visibility. No full `[T, keys, 64]` global intermediate is allocated.

The initial 32-token prototype's explicit scratch is 20.25 MiB: 16 MiB cuBLAS workspace, 4 MiB dot tile and
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

### First physical batched result

Run `39260845000` passes 11 parity cases, memcheck/synccheck, and all 36 timed
shapes (216 timing pairs). `p1-cublas-2026-10-19.json` is reproduced by
`summarize_p1_batched.py`; artifact `12191830882`, ZIP SHA256
`04c37c10d3ffd3452deb3fafaa01ebd3c8c815fb18c995f1f6148f39bf1b1028`.
Maximum relative L2 versus F32 is 3.1553e-7 and sampled FP64 max absolute
error 1.0859e-5. Every timed selected membership matches F32, including tiny
query stress in the parity population. No sampled thermal reason in 17 samples.

Several **small batched** shapes win against upstream's **batched** F32 scorer.
These T=1 observations must not be compared to the faster upstream **decode**
warp scorer. Large prefill does not win: T=256/~31K visible rows is +4.00%;
T=4096 at 120K starting context is +3.91%; shorter contexts lose more. A losing
full stage cannot be admitted based on presumed GEMM speed.

The next bounded experiment assigns eight scores per epilogue warp instead of
one, reducing its CTA count eightfold while retaining each score's head reduction
order. Key/GEMM tiles, workspace and selector launches remain fixed. The extra
tie/tiny cases now use 513 visible rows to exercise the 512/513 cutoff directly.
This scheduling change must pass the complete parity/sanitizer/timing matrix
before its result is used. F32 production remains unchanged.

The eight-score variant passes all checks in run `39261596608`, including the
513-row cutoff fixtures, but still loses bulk prefill: +7.54% at T=256/~31K
rows and +7.41% at T=4096/120K. Retain its separate evidence in
`p1-cublas-loop8-2026-10-19.json`; artifact `12191851575`, ZIP SHA256
`40cbed15126faa27567a32d3f6b2522a0008efeef402ae9bd5ff47d2b06eab31`.
All memberships match F32, maximum relative L2 3.1553e-7, zero sanitizer errors.
These separate campaigns do not establish a causal loop-1/loop-8 speed ratio.

The next test restores one score/warp and increases the query tile from 32 to
256 tokens, reducing repeated GEMM/epilogue calls for bulk shapes. Its explicit
scratch is **48.25 MiB**, below a hard 64 MiB caller-owned cap; common buffers
and opaque library costs remain separate. This is a bounded experiment, not a
production memory reservation or a performance claim. Re-run the full matrix
before deciding whether any bulk shape merits model integration.

### Wider query tile: long-context integration shortlist

Run `39262110290` passes parity, both sanitizers and all 36 timed shapes.
`p1-cublas-wide-2026-10-19.json` is reproduced by `summarize_p1_batched.py`;
artifact `12191212424`, ZIP SHA256
`3f06d8591d6257a37f860bab22bab2b538ddc133be74c305bc6e135f01543847`.
Maximum relative L2 remains 3.1553e-7; every parity/timed selected membership
matches F32. Sixteen telemetry samples contain no thermal reason; they do not
establish sustained whole-model behavior. Hosted CI also passes.

| Tokens T | Starting context | Median isolated stage elapsed change vs batched F32 | Repeats |
| --- | --- | --- | --- |
| 256 | 125,000 | **-13.02%** | All six pairs faster |
| 1,024 | 120,000 | **-12.37%** | All six pairs faster |
| 4,096 | 120,000 | **-12.12%** | All six pairs faster |
| 256 | 32,768 | +24.03% | Retain F32 |
| 1,024 | 32,768 | +27.04% | Retain F32 |
| 4,096 | 32,768 | +25.89% | Retain F32 |

Negative elapsed changes favor the candidate. These are synthetic, fixed-input
**full isolated indexer stages**, including the prototype's T selector launches.
They are not model throughput, an auto-dispatch threshold or evidence that all
intermediate geometries win. Only the explicitly measured long-context regimes
merit the next model-level investigation. Keep the losing shorter shapes on
F32. SGEMM uses FP32 math; no tensor-core dispatch is claimed for this candidate.

The actual 128K trace startup uses **2,048-token mixer sub-batches**, so the
next component matrix explicitly adds T=2048 and 64K/96K starting contexts.
This probes the measured engine geometry and the untested gap between losing
32K and winning 120K contexts; no threshold is inferred by interpolation.

### Actual mixer geometry and crossover checkpoint

Run `39262911919` passes all 11 parity fixtures, both sanitizers and the expanded
48-shape matrix (288 timing pairs), plus hosted CI. Evidence:
`p1-cublas-real-shapes-2026-10-19.json`, artifact `12190314513`, ZIP SHA256
`b47990cfbbd90dcf3d43ca04d576033d5a66d7d1fb33a55e6d25bfc308204b3e`.
All selected memberships match F32; maximum relative L2 3.1553e-7 and sampled
FP64 maximum absolute error 1.2722e-5. Thirty-nine telemetry samples contain no
thermal reason; power limiting is observed. Host peak RSS is about 2.52 GB in
this model-free process, not a model memory budget.

| Actual T=2048, starting context | Median indexer stage elapsed change | Decision |
| --- | --- | --- |
| 32,768 | +26.74% | Retain F32 |
| 65,536 | +0.10% | Retain F32; no reliable win |
| 98,304 | **-9.20%** | All six pairs faster; shortlist model investigation |
| 120,000 | **-11.82%** | All six pairs faster; shortlist model investigation |

The other tested bulk sizes (256/1024/4096) also win around 9–10% at 96K and
12–13% at ~120K. Results at 64K are mixed/noise for most bulk sizes. This is a
**component integration shortlist**, not an automatic threshold or model
performance admission. Keep default dispatch F32 while implementing and testing
the guarded model path. No further unchanged microbenchmark campaign is needed
before that next implementation step.

### Integration feasibility to verify

`prefill_ds4_mixer` prepares queries, scores/selects, then writes attention into
`B.o`, followed by grouped output projection into `B.oa`, all on stream `s`.
For the fixed model, `B.o` has `T*64*512` FP32 elements: enough for the bounded
dot tile `min(T,256)*64*512`. `B.oa` has `T*8192` FP32 elements: enough for
65,536 converted key values when T>=8. Investigate borrowing those still-dead
outputs and reusing `S->blas`/`S->ws` instead of adding persistent per-stream
allocations. Prove liveness, stream ordering, capacities and graph behavior in
the integrated path; this source observation is **not** a measured zero-memory
overhead claim. Keep score/selection buffers distinct. Attribute vendor SGEMM
to indexer phases using event/range evidence, not by relabeling every model GEMM.

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

## Integrated long-context logit and repeatability result — 19 October 2026

`p1-integrated-quality-2026-10-19.json` is reproduced from artifact `12225080552`,
ZIP SHA256 `77cd2653b10e360161196c2cf572adda5eadf0e50eebbbb5e9a2925b7f92fa13`,
run `39328349428`, source `c157de24442a7c4ea71c2af8464b6d214c7548b4`.
Engine SHA256: `69080a4fd395c987d24048386a8eb9b1f0a4741201ee9a8c03441c47ae4cc584`.
Each candidate arm has 21 actual cuBLAS calls, one in every C4 layer (2,4,…,42),
at p0=98,304/T=2,048. The 32 predicted positions are 100,354 through 100,385.
F32 arms have zero candidate calls; all arms retain 1,029 actual dispatch rows
in the artifact. The committed summary retains checked coverage counts.

| Comparison | Mean KL | Max position KL | Top-1 matches |
| --- | --- | --- | --- |
| F32 repeat | 1.81510e-7 | 2.77680e-6 | 32/32 |
| cuBLAS repeat | 1.73335e-7 | 4.47440e-6 | 32/32 |
| F32-0 vs cuBLAS-0 | 2.92070e-7 | 6.34302e-6 | 32/32 |
| F32-1 vs cuBLAS-1 | 1.33124e-7 | 3.20676e-6 | 32/32 |

This passes the bounded same-pack incremental mean-KL gate (0.0005), where the
candidate actually executed. It is not a full task-quality, all-context or P1
exit claim. The exact same binary runs every arm, with original F32 queries,
FP16 keys, FP32 SGEMM dots and unchanged selection. All 48 component shapes,
graph replays, memcheck and synccheck pass. Each model arm has 31 telemetry
samples; no sampled thermal reason, power limiting observed in all four.
Minimum host MemAvailable is 48,259,117,056 bytes across the arms. RSS/PSS,
anonymous memory, registered/pinned pages and cgroup charge overlap; never add
them. Profile/dispatch-enabled arm timings are excluded from ordinary model
throughput. Model SHA verifies the local manifest, not independent publisher
authenticity; hardware counters and QPI traffic remain unavailable/unmeasured.

The next workflow reuses the exact verified engine binary and refuses a runtime
source change, failed quality evidence or overlapping GPU process. It records
three A/B, B/A, A/B process pairs, short 8K/128-output and long 128K/16-output
requests, with the second uncached request per case as the primary comparison.
Completed arm checkpoints live on the existing runner and are reusable only
with matching binary/model/fixture/quality-evidence identities.

## Batched MMA experiment — implemented, physical validation queued

The specification's separate batched MMA experiment is now isolated in
`dsv4_volta_idx_batch.cuh` and `test_dsv4_mma_batched`. The one-query/grid-Y
variant establishes the batched visibility/output contract; the eight-query
variant loads each packed K fragment once per head-band/K step and reuses it
across eight independent query hi/lo accumulator sets. Both use four warps,
128 compressed rows/CTA, padded shared storage for one eight-head band, and
per-head ReLU/weights before summation. No global token/key/head intermediate
or production dispatch is added. Query tails and key tails keep all lanes
participating in every aligned MMA.

The planned physical test has 19 diagnostic cases per variant (38 total),
including T=1/3/8/9/33, 510–515-row visibility boundaries, query-tile tails,
zero/tiny/duplicate/large-query cases, unchanged selection and two graph
replays. Tiny FP16-residual underflow uses an explicit absolute diagnostic
bound, records nominal relative acceptance separately, and remains unqualified
for runtime arithmetic. The matrix has 15 shapes per variant, six interleaved
pairs each: T=256/1024/4096 at 2K/8K/32K/120K, plus actual mixer T=2048 at
64K/96K/120K. These full isolated stage times include the T unchanged single-token
selection launches, compared with the baseline's one batched selector launch.
Sanitizers, PTX/SASS and actual registers/shared/local bytes/occupancy are gates;
source register reuse is not a measured DRAM-traffic claim. This test is queued
in the same `abyss-v100` concurrency group behind whole-model A/B, with no
container or privilege changes. Physical results remain pending.
