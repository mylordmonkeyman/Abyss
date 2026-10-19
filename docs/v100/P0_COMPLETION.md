# P0 completion checkpoint

User instruction: continue autonomously through completion of P0, with commits and
runner use authorized. Follow revision 9, sections 1–2. P1/P2 implementation is
paused; their remaining optimization gates do not redefine P0.

Status: **in progress**, not P0-complete. Hourly continuation is enabled; disable
it only after the completion ledger is committed and the user is notified.

## Fixed decisions and preserved evidence

- Working base: historical v1.3.28 `47001f31`; v1.3.31 `7c893ac` remains the
  comparison. v9 expressly permits retaining the historical base for isolation.
  Both unmodified upstream binaries already build/run on the physical V100.
- Same-pack base logit screen passes; initial three-pair 600/4K/8K comparison and
  fixed-history replay are preserved in `v9-bases-2026-10-18.json` and
  `v9-replay-2026-10-18.json`. Small mixed timing differences do not prove causal
  improvements. The bounded repeatability screen is now quantified below.
- Actual formats: IQ2_XS/IQ3_XXS gate/up, IQ3_XXS down except MXFP4 layers 26/42.
  Named IQ2_XXS/Q2_K experiments are N/A for this pack. Ordinary SM70 MMQ executes;
  fused GLU's Turing guard is not a global MMQ eligibility test.
- Rootless Nsight Systems timeline works. Privileged hardware counters return
  ERR_NVGPUCTRPERM; keep the v9 event/timeline alternative. QPI traffic remains
  unmeasured, not inferred from page placement. Never weaken runner security.
- Full model SHA was previously unverified. The completion campaign now has full
  shard SHA256 values, all matching the local SHA256SUMS file. This verifies byte
  identity against that host manifest, not an independent publisher signature.
  Earlier runs retain their original limited-integrity description.

## Remaining gates

| Gate | Status / next evidence |
| --- | --- |
| Exact source/build/model identity | Full shard hashes now measured and match local SHA256SUMS; preserve upstream binaries and selected SM70 image evidence |
| Baseline repeatability | Completed 32-row screen: both same-base pairs and cross-base top-1 32/32; GPU-only historical pair byte-identical; bounded corpus scope retained |
| Extended short matrix | Complete: 108 requests, three interleaved pairs and excluded first repetition; active-window telemetry reviewed, no causal speedup claimed |
| Filled context | 32K and 64K three pairs complete with 1024 outputs; 128K units progressing, original historical 128K also preserved |
| Chunk settings | 512/1024/4096 plus auto; 128 unsupported on the upstream 256-token chunk grid, not silently relabeled |
| Runtime dispatch | Captured actual CPU/AVX2, MMQ and GLU/fallback decisions; per-token route/cache and exact per-call input-duplication detail remain open |
| Unique memory worksheet | Conserved per-SHA host/GPU budget and alias-deduped backing pages; symbol-backed allocation components and source-derived borrowed-arena layout reconciled |
| Long-context critical path | Rootless CUDA timeline/events at filled context; report top three wall-path contributors and overlap limits |
| Sustained telemetry | Active-window clocks/power/thermal, faults/I/O, NUMA placement, pressure and preserved safety floor |
| Final decisions | Base, indexer/attention hotspots, repeated quantization, CPU type path, static balance and ≤3 P3 admissions with evidence |

## Accounting instrumentation checkpoint

The physical collection passed in run `39194149121`. See
`P0_ACCOUNTING_RESULTS.md`, `p0-memory-accounting-2026-10-19.json` and the
collector instructions in `P0_MEMORY_ACCOUNTING.md`. Each immutable base has
165 contiguous allocation events, no failures, clean exit markers and complete
release of observed allocations. Live host-pin ranges resolve to 41 shared
backing objects with no unresolved private ranges. Their page footprint is
about 71.85 GiB, inside PSS/Shmem rather than additive to RSS. Each disjoint
host budget conserves 135,068,450,816 bytes. The source-derived 948,249,856-byte
raw/CSA/HCA/indexer arena matches the actual allocation in both bases.
`p0-prefill-layout-2026-10-19.json` now resolves the borrowed arena into disjoint
outer regions and explicitly overlapping MoE/attention union views. Its maximum
loan is 7,350,343,424 / 7,351,392,000 bytes, historical/current, entirely within
the existing expert-pool budget. Both startup logs and exact landing-ring API
bytes corroborate the reconstruction; host-compiled actual sizing functions
match at six chunk sizes. These derived offsets are not instrumented device
addresses or physical pages. 44 local tooling tests pass; no PFN/QPI claim is made.

## Dispatch instrumentation checkpoint

`ABYSS_V100_DISPATCH=1` adds bounded host counters at the actual CPU row and
MMQ branches, with worker scopes and decode request/position joins. See
`P0_DISPATCH_LEDGER.md`. The CUDA build and device checks passed in run `39185866653`; physical
capture now passed in run `39194149121`: 23,285 keys, no dropped calls, no
unresolved decode service positions. See `p0-dispatch-2026-10-19.json`.
Ordinary MMQ ran 6,064 products; fused GLU ran zero. IQ2_XS T=1 uses GGML dots,
IQ3_XXS T=1 uses IQ256 rows1, multi-row IQ2_XS/IQ3_XXS uses IQ256, and MXFP4
down uses GGML dots. No IQ512 calls were observed. At least 3,615,488 repeated
input quantizations are proven by a distinct-row upper bound; exact per-call
counts and detailed per-token prefill route/cache attribution are still open.
Read-only run `39199082932` passed: both exact binaries' compiled
`ggml_cpu_has_avx2` returns 1 and `ggml_cpu_has_avx512` returns 0, verified by
getter disassembly. Actual allocation callers are symbolicated and application
component bytes reconcile to the API total. 40 local tooling tests pass;
profile timings are excluded.

The next diagnostic copies actual sorted gather IDs after expert scatter and
counts the distinct source tokens in each MMQ input call, plus the union within
each layer/chunk invocation. Its histogram separates duplication within calls
from duplication across calls, including repeated prompts. Host tests cover
invalid IDs/ranges, independent invocation unions and incomplete captures.
42 local tooling tests pass. CUDA compilation and physical collection of this
extension are pending; the existing 3,615,488 lower bound remains the measured
result. The added copy/synchronization can perturb planning and overlap, so
these diagnostic timings and route populations are not production measurements.
CUDA build/device run `39203344685` passed this extension. All remaining matrix
and timeline units in run `39194497860` passed. A dispatch-only ledger collection
is next: the workflow now defaults to that phase and requires the exact-count
summary to reconcile before accepting the capture. Immutable memory arms are
repeated only by an explicit manual `phase=all` selection. The long/chunk/trace
artifacts still need review; workflow success alone does not close P0.

## Interrupted long-context campaign

The campaign ended unsuccessfully during the second long arm. Read-only
inspection run `39193892390`, artifact `12161480287`, preserved the partial arm.
Historical v1.3.28 completed all genuine 32K/64K/128K histories with 1024 outputs,
two requests each. Current v1.3.31 completed both 32K requests before interruption.
No engine survived inspection. Pre-interruption telemetry had >44 GiB minimum
MemAvailable, no cgroup OOM events, and the last sample at 03:01:24Z; GitHub
marked failure at 03:10:58Z without finalized job logs. Subsequent `/proc/stat`
confirms a new host boot at 03:02:59Z (previous boot: Oct18 04:59:52Z). The runner
disappeared during a host reboot; what triggered that reboot remains unknown.
This is not an engine OOM diagnosis. Completed/partial data remains preserved.

Extended results: `p0-extended-2026-10-19.json`. Active-request samples showed
SM clock medians 1380 MHz on all six arms, no sampled thermal reason, occasional
power cap activity, minimum MemAvailable >45.27 GiB. Median paired prefill
changes were within ±0.96%; decode changes ranged -3.96% to +3.05% by stratum
(negative favors current). All compared generated trajectories differed. These
observations do not establish a causal base performance improvement/regression.

## Recovery execution checkpoint

`P0_RECOVERY.md` defines the separate accounting and replacement long-context
collection. Allocation/dispatch run `39194149121` completed successfully after
a passing CUDA rebuild. Remaining run `39194497860` is active under the same GPU lock.
New long units use consistent fresh processes per context and immediately upload
individual checkpoints. Do not merge their timing strata with the original
carry-over-residency long arm. All six 32K units passed in the intended
A/B, B/A, A/B order, verified by actual request start times. Results are in
`p0-32k-2026-10-19.json`; artifact ZIP digests were verified. Paired median
prefill time change is +0.00019%, decode time -5.8683% (negative favors current).
All three generated trajectories differ, so this is not a causal speedup claim.
Warm decode sample clocks were 1380/877 MHz in all arms, without sampled
thermal/power-cap reasons; prefill clock/power behavior varied.

All six 64K units (6–11) now passed, with artifact ZIP digests verified and actual
process order 0,1,2,3,4,5. `p0-64k-2026-10-19.json` preserves their summary.
Paired median prefill time change is +0.07117%, decode +2.07808%, TTFT +0.06468%
(positive means current took longer). Warm prefill medians are 497.23 / 497.15
tokens/s, decode 23.50 / 23.02, historical/current respectively. Generated
trajectories differ in all three pairs; no causal base regression follows.
Warm decode samples show 1380/877 MHz on both bases and no sampled thermal or
power-cap reason. Prefill samples include power caps; minimum active-window
MemAvailable is 45.36 GiB. The 128K units are next; inspect fresh job state
before reporting later completion counts.

## Original campaign (preserved, interrupted)

Active run: https://github.com/mylordmonkeyman/Abyss/actions/runs/39184856034
(commit `61e5abb`). Its quality phase passed; the extended short/output matrix
completed all 108 requests. Quality artifact `12158440606` is preserved in
`p0-quality-2026-10-19.json`. Same-base mean KL: historical 1.39448e-7, current
8.59050e-8; all six comparisons have top-1 agreement 32/32. The GPU-only pair is
byte-identical (KL 0). Normal-lane variability is consistent with host lane /
reduction behavior; this is not proof of a base regression or a sole cause.
No sampled thermal reason was observed; minimum MemAvailable stayed above
45.5 GiB. These short score windows do not establish sustained performance.

`v100-p0.yml` invokes `tools/bench/v100_p0_campaign.py` on the existing cached
immutable binaries; there is no engine rebuild, model repack or container change.
Persistent checkpoints live at `/actions-runner/abyss-p0-v9/campaign-1`.
Separate workflow steps upload quality, extended, long and chunk checkpoints, so
results can be reviewed while later phases run. Raw full-vocabulary distributions
stay local and are excluded from artifacts. Completed arms are reused; incomplete
arms are preserved and require investigation rather than blind retries.

The main GPU job may take several hours. Do not dispatch another GPU workload
while it is active. Independent source inspection, accounting instrumentation,
parsers and documentation can proceed. Future iterations must inspect current
branch and job state before editing or dispatching to avoid overlapping work.

Every arm has the same fixed controls, input IDs and binary checksums. "First"
means new-process expert residency; OS file cache is uncontrolled, with no global
cache drop. Later fixtures inherit expert residency. Per-case first and second
requests remain distinct. No performance claim follows from unmatched clocks,
power, NUMA placement or CPU contention.

Safety: require 16 GiB MemAvailable before each arm and terminate an arm if it
falls below 8 GiB; preserve the event and stop larger work. Do not force all model
pages onto node0, reset GPUs, change power clocks or add container privileges.
P0 must not be marked complete merely because a campaign ended successfully:
the dispatch, memory, critical-path and decision rows above still require review.
