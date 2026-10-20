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
| Production integration / P1 exit | **Evaluation concluded: keep F32**, as revision 9 permits. No candidate promoted. Full-128K deterministic candidate KL fails; late capture finds near-tie membership/attention effects. See `P1_COMPLETION.md`. SM80 compilation passes; HIP SDK unavailable. |

Current priority: **P1 batched MMA884 reopened on 20 October at user direction**.
The optional query-scaling experiment was not exhausted by the earlier fallback
decision. Implement and physically test it before further P3 work. Existing
cuBLAS rejection and production F32 defaults remain in force.

The new isolated matrix keeps unscaled one/eight-query controls and adds
per-query/head power-of-two normalization before the FP16 hi/lo split. Each
head's combined FP32 dot is rescaled before ReLU and its learned weight.
Exponents are integers, so tiny FP32 input needs no overflowing scale value.
No clamping, changed selection or relaxed precision threshold is introduced.
All scaled cases, including the former 2^-20 query failure, must meet relative
L2 <1e-5 versus F32 and the sampled FP64 oracle. Additional 2^-100, 2^80 and
mixed per-head scales test the extended domain. Normal-case FP64 absolute
bounds are retained. Graph replays, padding/visibility/selection guards,
memcheck/synccheck, resources, PTX/SASS and a 60-shape / 360-pair stage matrix
include normalization cost. The first physical result is recorded below; a passing competitive
variant must next receive actual-model input-range/selection and quality tests.

### Scaled MMA first physical result and next normalization schedule

[Run 39432795421](https://github.com/mylordmonkeyman/Abyss/actions/runs/39432795421)
passes all 82 diagnostic cases, both sanitizers, graphs, emitted SM70 MMA checks
and 60 numerical timing shapes / 360 pairs. Artifact `12269201557`, ZIP SHA256
`9005382d02c98fa10eb91eaa8b01958087e1c28127d8490943f58f9b1d2c6662`,
recomputes into `p1-mma-scaled-2026-10-20.json`.

Scaling repairs the former tiny-query precision failure: relative L2 is
3.75204915e-7 instead of 3.0464%, with selection membership 4608/4608.
Scaled 2^-100, 2^80 and mixed-head stresses also pass nominal F32 and sampled
FP64 relative gates and full membership. The eight-query scaled kernel uses
186 registers, 33792 shared bytes, zero local bytes, two active CTAs/SM.
The one-query scaled control reports eight local bytes and is not shortlisted.

The first scaled schedule loses every timed shape: actual T=2048 changes
are +15.34% at 96K and +12.51% at 120K. These include normalization and literal
selection launches. This is an accuracy repair, not a speed qualification.
Before model dispatch, the next schedule shares exact normal power-of-two
factors per head, using FP32 multiplication for query normalization and dot
restoration. Extreme exponents retain the full integer-exponent `scalbnf`
path. FP32-subnormal queries (2^-130) and large queries with small keys explicitly
exercise both extreme-exponent fallbacks with finite outputs. No thresholds or
selector semantics change. Recheck all 86 diagnostic
cases/graphs/sanitizers, and time only the scaled eight-query tile (15 shapes,
90 pairs); unscaled controls and the one-query loser already have their timing
dispositions. This tests the revised normalization overhead directly.

The shared-factor schedule passes all 86 cases, graphs, sanitizers and 90
timing pairs in [run 39434241151](https://github.com/mylordmonkeyman/Abyss/actions/runs/39434241151).
Artifact `12269701720`, ZIP SHA256
`6c6656de84ce2dc1ed0ac655790f255385257faba3363ce8328b2b7a323d0ca1`,
recomputes `p1-mma-scaled-fast-2026-10-20.json`. Both extreme-exponent paths
pass; the 2^-130 case has relative L2 1.8032e-6 and full membership. The
eight-query kernel has zero local bytes, 182 registers and 34304 shared bytes.
It still loses all timed shapes: +12.13% at actual T=2048/96K and +9.20% at
120K. A model-speed claim or automatic dispatch remains unqualified.

Next prepare scaled hi/lo fragments once per token, rather than recomputing
normalization in every key tile. This distinct variant uses one positive
power-of-two scale common to the token's heads, preserves per-head ReLU and
weights, and restores the final row score. Commutation holds in real arithmetic;
floating-point order/underflow effects must pass the same F32/FP64 and selection
checks. The existing per-head and unscaled candidates remain diagnostic controls.
The new 110-case matrix includes zero/tiny/large/mixed/exponent extremes and
graph/sanitizer checks, with 15 shapes/90 pairs including the preparation launch.
Caller-owned packed-query scratch is `(T*64*128+T)*4` bytes (64 MiB + 8 KiB at
T=2048), bounded explicitly and included in resource accounting. No model
runtime is enabled by this isolated experiment.

P3 checkpoint: the P0-admitted completed-cost observation in `P3_PROGRESS.md`.
P2 also concludes with F32 fallback; see `P2_COMPLETION.md`.
The fallback disposition and evidence inventory are in `P1_COMPLETION.md`.
The following investigation checkpoint is preserved as history, superseded by
the completed late capture and fallback decision.

The next compile-only capture uses the original 128K fixture/partial-tail score
geometry, CPU lane 0 and p0=118784, after all eleven candidate chunks have run.
It compares F32 against cuBLAS on the same live inputs within that chunk,
records paired scores/IDs/attention at five sampled tokens across all 21 C4
layers, and reports selection overlap/order for every one of its 2048 tokens.
Raw paired tensors remain capped at 56 MiB and the complete artifact at 64 MiB.
It bitwise restores the candidate IDs before model continuation. This isolates
local rounding/selection effects after accumulation; it does not locate the first
whole-model divergence or establish throughput. Production arithmetic/defaults
and the <=0.0005 mean-KL threshold remain unchanged.

## Original-upstream and deterministic candidate controls — 20 October 2026

[Original-upstream control 39360007121](https://github.com/mylordmonkeyman/Abyss/actions/runs/39360007121)
uses the immutable selected v1.3.31 binary and working F32 in upstream/working/
working/upstream order, with the same CPU-lane-28 settings and exact failed
128K fixture/scoring geometry. Collection completed; all comparisons exceed the
written KL limit. A successful collection job is not a quality pass.

| Comparison | Mean KL | Maximum KL | Top-1 matches |
| --- | ---: | ---: | ---: |
| Upstream repeat | 0.01145396 | 0.11544902 | 32/32 |
| Working F32 repeat | 0.00853500 | 0.06895841 | 31/32 |
| Upstream-0 / working-0 | 0.00894022 | 0.07876772 | 32/32 |
| Upstream-1 / working-1 | 0.01329741 | 0.12122638 | 31/32 |

This closes the missing upstream baseline characterization and shows the
original variability is not unique to our branch. It does not prove port parity
under a deterministic configuration. `p1-upstream-repeat-2026-10-20.json`
preserves artifact 12238690139, ZIP SHA256
`b9546e611e241b74487dad2eabe5e9464bf9e1765db6f3a7e414ae294230224f`.

[CPU-lane-off candidate control 39361291597](https://github.com/mylordmonkeyman/Abyss/actions/runs/39361291597)
completes all four arms and fails quality for a separate reproducible reason:

| Comparison | Mean KL | Maximum KL | Top-1 matches |
| --- | ---: | ---: | ---: |
| F32 repeat | 0 | 0 | 32/32 |
| cuBLAS repeat | 0 | 0 | 32/32 |
| F32 / cuBLAS, both matched pairs | 0.01040476 | 0.12950428 | 31/32 |

Each scorer's two raw distributions have identical hashes. Both candidate arms
execute all 231 intended calls. The F32 dump also matches the previous lane-off
baseline hash. Thus candidate drift persists independently of CPU-lane-28
repeatability. Do not subtract the upstream repeat KL or relax thresholds.
`p1-lane-off-candidate-2026-10-20.json` preserves artifact 12238708864, ZIP SHA256
`16073a26d3b44a121d9c66ffda9a9e418484a8e66016999b0d86ce51a3809a0c`.
Both controls are reproduced by `summarize_p1_controls.py`; raw distributions
remain on the existing runner, with hashes and per-position KL in the ledgers.

## CPU-lane-off control and platform result — 20 October 2026

[Run 39359164238](https://github.com/mylordmonkeyman/Abyss/actions/runs/39359164238)
passes both F32 repeats at the exact failed geometry after changing only
`STRATA_GLM_CPU_LANE` from 28 to 0. Both 16,547,840-byte dumps have identical
SHA256 `ea857dda662787ae630d699d94d34f940eeab0087606997dc37964fb13417c81`:
mean/max KL **0**, top-1 **32/32**. All indexer dispatches remain F32.
This makes CPU/GPU expert execution a useful investigation axis, without proving
which part causes the original variability or establishing a candidate regression.
Original-upstream CPU-lane-28 repeatability is now characterized above.

The independent SM80 compile/link and embedded-image check pass; binary SHA256
`4524ccf401db1e4b60cc925012ead29c8481c0a21a3f9127437d0feec8deeb85`.
The visible device inventory contains only SM70, so this is not SM80 inference.
HIP compilation/runtime are unavailable because no `hipcc` exists on PATH or
at `/opt/rocm/bin/hipcc`; host policy and HIP-name checks pass, without claiming
an actual HIP build. Neither privileges nor toolchains were changed.

`p1-f32-triage-2026-10-20.json` is reproduced by `summarize_p1_triage.py` from
artifact 12237460320, ZIP SHA256
`c97f37929540495c10fce414956cc549d0360d2e9d626928b8637f56c88e0301`.
Each arm has 42 telemetry samples, no sampled thermal reason and observed power
limiting. Minimum host MemAvailable is 48,571,547,648 bytes; overlapping memory
ledgers are not summed. Raw logits and checkpoints stay on the existing runner.

The subsequent F32/cuBLAS/cuBLAS/F32 control at this same CPU-lane-off
configuration fails the matched comparisons as recorded above. No production
math, threshold or default dispatch is changed.

## Full-128K admission blocker — 20 October 2026

[Run 39353531268](https://github.com/mylordmonkeyman/Abyss/actions/runs/39353531268)
completed all four arms, then failed the unchanged mean-KL <= 0.0005 gate:

| Comparison | Mean KL | Maximum KL | Top-1 matches |
| --- | ---: | ---: | ---: |
| F32 repeat | 0.0070883041 | 0.0668990567 | 32/32 |
| cuBLAS repeat | 0.0189530553 | 0.2957759258 | 30/32 |
| F32-0 / cuBLAS-0 | 0.0142452077 | 0.1263606029 | 32/32 |
| F32-1 / cuBLAS-1 | 0.0078669786 | 0.0589207369 | 30/32 |

`p1-128k-failure-2026-10-20.json` preserves failed evidence from artifact
12236690630, ZIP SHA256
`b25fddd238b96505b5dfab1a7f340ebe4ba1b103a38fada1c85b3fbc99a281fb`.
All 231 intended candidate calls occurred in each candidate arm. The binary is
`fc7a26eca2da31c1a63819b3b4a6440616cbedd59d775c4f70dc8cc51c8c2392`.
The frozen 131072-token fixture scores positions 131040–131071 after prefilling
131038 tokens, leaving a 2014-token mixer tail. The passing 100K fixture had an
aligned 2048-token tail. Tail geometry and CPU/GPU expert assignment are
investigation axes, not established causes. Do not subtract baseline drift,
relax thresholds, repeat unchanged, or infer candidate correctness from top-1.
Original raw logit dumps were temporary and no longer exist; their hashes,
per-position KL, dispatch records, compressed logs and telemetry remain.
SM80/HIP steps were skipped after the quality failure; no SDK-unavailable or
compile-pass disposition can be inferred from that skip.

The same binary passes the default-off guard regression in
[run 39353474002](https://github.com/mylordmonkeyman/Abyss/actions/runs/39353474002):
48 timed shapes, graph replay, both component sanitizers and four 100K quality
arms. `p1-guard-regression-2026-10-20.json` is regenerated by
`summarize_p1_integrated.py` from artifact 12235461004, ZIP SHA256
`40fa4d71faf8bf06bee803aad14d37c56b5921010b9bd669f8b8f25213a4fbe5`.
F32-repeat mean KL is 5.36e-8; all four comparisons are <= 2.61e-7 and top-1
32/32. This bounded pass does not override the full-context failure.

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

## Whole-model A/B result — 19 October 2026

[Run 39333942954](https://github.com/mylordmonkeyman/Abyss/actions/runs/39333942954)
completed six process arms / 24 requests on the same verified binary
`69080a4fd395c987d24048386a8eb9b1f0a4741201ee9a8c03441c47ae4cc584`.
`p1-model-ab-2026-10-19.json` is independently recomputed by
`summarize_p1_model_ab.py` from artifact `12229354268`, ZIP SHA256
`399fab74971d32f45579aeaecac61b26e3c8b0d09edb86b1be887d9f8024efd9`.

| Prompt | Three paired prefill latency changes | Median paired change |
| --- | --- | --- |
| 8,192 / 128 outputs | +0.470%, -0.186%, -0.544% | -0.186% (near noise) |
| 131,072 / 16 outputs | -1.535%, -1.496%, -1.585% | **-1.535%** |

Median raw 128K prefill rates are 357.14 F32 and 362.74 candidate tokens/s;
these ratios of population medians are not the paired statistic. The explicit
candidate executes 462 cuBLAS calls per candidate process (two long requests,
11 long mixer chunks and 21 C4 layers), with zero short-context candidate
calls. F32 arms have zero candidate calls. No profiler is enabled. Every arm
has 77–78 telemetry samples, no sampled thermal reason, and observed power
limiting. Minimum host MemAvailable is 47,791,554,560 bytes. OS cache and expert
residency carry over within processes; clocks/cache are not perfectly matched.
This is consistent observed improvement across three pairs, not a universal
or clock-normalized performance guarantee.

Do **not** claim decode improvement from these numbers. Decode is unchanged;
primary decode variations are +8.31/-14.90/-6.38% at 128K, with only 16 output
tokens. Five of six primary case-pairs have different greedy output IDs,
including all three 8K pairs where the candidate never runs. Teacher-forced
KL/repeatability, not greedy string identity, is the numerical screen. These
trajectory/tier differences confound decode attribution. Keep `auto` unchanged
until the remaining P1 gates are reviewed.

## Batched MMA physical result — 19 October 2026

[Run 39334367431](https://github.com/mylordmonkeyman/Abyss/actions/runs/39334367431)
passes all 38 diagnostic cases, two graph replays each, memcheck/synccheck,
30 timed numerical shapes / 180 pairs and emitted SM70 MMA884/HMMA checks.
Artifact `12230530011`, ZIP SHA256
`59ad7e5d66b8da0a3590961a2b19a38bf63eab782046c6fa15ae770e2070341c`,
is summarized by `summarize_p1_mma_batched.py` into
`p1-mma-batched-2026-10-19.json`. Normal-case relative L2 <=4.20e-7.

| Variant | Registers | Static shared | Local bytes | Active CTAs/SM |
| --- | --- | --- | --- | --- |
| One query | 71 | 4,192 B | 0 | 7 |
| Eight queries, K reused | 196 | 33,536 B | 0 | 2 |

The one-query batched grid loses every timed shape. At actual T=2048, the
eight-query variant's full isolated indexer latency changes are -1.41% at
64K, -10.40% at 96K and -13.85% at 120K, all six pairs faster at each.
At 120K, T=256/1024/4096 change -13.54/-13.68/-14.01%. Shorter shapes lose.
Separate cuBLAS/MMA campaigns are **not** a paired head-to-head comparison.
The matrix includes the extra single-token selector-launch penalty and is
not score-only timing or model throughput. Sixty-five telemetry samples show
no sampled thermal reason; this is not hardware-counter evidence.

Tiny-query stress still gives 3.0464% relative L2, maximum absolute error
1.3414e-6, membership 4607/4608 across nine tokens in **both** variants.
These pass their stated absolute diagnostic bound, **not** nominal relative
qualification. There is no scaling/clamping change or MMA runtime integration.
The key-reuse experiment is complete as a measured isolated candidate; normal
component wins do not qualify this arithmetic for model use.

## Next diagnostic implementation — physical validation pending

`STRATA_V100_CAPTURE_INDEXER` is an internal compile-only option, default OFF.
At p0=98,304/T=2,048 in every C4 layer, it samples token offsets
0/255/256/1023/2047, retains candidate IDs, executes original F32 score/select
and attention on the same live inputs, then bitwise restores **all** candidate
IDs before normal attention/model continuation. It adds no GPU allocation.
Raw paired scores, selected IDs and attention rows total under a 56 MiB hard
capture cap; the complete artifact is refused above 64 MiB. No input/weight
repacking, global cache operations or privilege changes are added.

`p1_live_capture.py` checks exact layer/row populations, selector ties/order,
range/duplicate/future IDs, score-error-conditioned disagreements, rank-512/513
margins, attention differences and a two-arm 32-position model-KL screen.
Identical selection must reproduce identical attention. Attention above the
1e-4 investigation screen stops admission for review alongside margins/KL.
The diagnostic uses its own build directory and is excluded from timing claims.
Default-off integration is revalidated separately; working binaries are frozen
by digest on the existing runner before build-tree reuse. Counter/QPI and model
publisher-authenticity limitations remain unchanged.

## Direct diagnostic and default-off regression — 20 October 2026 UTC

[Default-off run 39344305603](https://github.com/mylordmonkeyman/Abyss/actions/runs/39344305603)
passes engine/component build, 48 shapes, graph/sanitizers and all four 100K-context
quality/repeatability arms after the compile-only instrumentation was added.
Artifact `12231382117`, ZIP SHA256
`56ebb3fa82c03b8342b7dbc589bf0bbd10f55203a7bbebbb8dbede75c72169e2`,
is reproduced into `p1-off-regression-2026-10-20.json`. The preceding throughput
binary is preserved by digest on the existing runner before build-tree reuse.

[Live capture run 39344362285](https://github.com/mylordmonkeyman/Abyss/actions/runs/39344362285)
passes all 105 sampled rows, five per C4 layer, with no membership mismatch,
meaningful-margin violation, range/duplicate/future index, restoration or attention
screen failure. Artifact `12231837565`, ZIP SHA256
`55bfb48f8936f2dca4782a28092a417f7db991b328f5d013a9ec8ebdd52c21ce`,
is independently recomputed from raw tensors by `summarize_p1_capture.py` into
`p1-live-capture-2026-10-20.json`. Raw artifact size is 50,769,524 bytes before
ZIP compression; the 64 MiB upload cap passes. Every layer bitwise restores
all 1,048,576 candidate IDs before model continuation.

- Score relative L2 <=1.7174e-7, maximum absolute difference 1.43052e-6.
- Top-512 **membership is 512/512 in all 105 rows**; 103 rows also have identical
  order. Two internal adjacent rank swaps are traced, not discarded as noise.
- Layer 2 / t=1023 swaps ranks 185/186: F32 gap 4.76837e-7 becomes a candidate
  exact tie, correctly resolved by index. Per-row maximum score error is
  7.15256e-7. Layer 26 / t=256 swaps ranks 497/498: F32 gap 1.63913e-7,
  candidate gap 2.98023e-8, within per-row maximum score error 9.53674e-7.
  Neither affects the 512/513 cutoff or membership.
- Maximum direct attention relative L2 4.24527e-9 and absolute difference
  1.19210e-7; identical ordered selections reproduce identical attention.
- Capture-build mean model KL 9.30197e-7, maximum position KL 1.70362e-5,
  top-1 32/32. This passes the bounded mean-KL gate, not arbitrary task quality.

The extra reference kernels, copies and synchronization are compile-only and
excluded from performance conclusions. Model local-manifest/publisher-authenticity,
power-limit, unavailable counters and unmeasured QPI limitations are unchanged.

## Final-boundary/platform work — implemented, physical validation pending

The actual runtime backend guard is now a shared host policy, tested for legacy
precedence, auto/F32 preservation, HIP/SM75/SM80/SM90 rejection of forced cuBLAS,
invalid geometry and both context/tile boundaries. These are host policy tests,
**not** inference on absent devices. Arithmetic and dispatch eligibility are
unchanged; the default-off integration workflow revalidates the refactored guard.

The next quality screen extends the frozen fixture to 131,072 tokens, scoring
positions 131,040–131,071. Each candidate arm must execute all 231 expected
layer/chunk pairs (21 layers × 11 chunks), rather than validating just the first
eligible chunk. The four F32/candidate/candidate/F32 arms keep existing safety
margins, same-pack identities and bounded temporary dumps. Quality-only evidence
can reuse component evidence only when the actual engine SHA256 matches.
SM80 gets a separate full compile/link and object-image check. A read-only HIP
SDK/device probe records unavailability honestly; if the SDK exists, a HIP
compile is required. No other-device runtime is claimed without actual hardware.
No automatic admission or P1 completion is asserted by this checkpoint.
