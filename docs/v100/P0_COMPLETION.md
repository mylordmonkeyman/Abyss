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
  improvements. Greedy/likelihood repeatability still needs quantification.
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
| Filled context | Genuine 32K/64K/128K histories, fixed 1024 decode, three pairs where safe |
| Chunk settings | 512/1024/4096 plus auto; 128 unsupported on the upstream 256-token chunk grid, not silently relabeled |
| Runtime dispatch | Exact per-layer/type CPU row/ISA, MMQ/quantize/GLU/fallback calls, T=1 versus 2–8 and prefill |
| Unique memory worksheet | Exact pinned/registered allocations and aliases; GPU component bytes; reconcile physical host views without adding overlaps |
| Long-context critical path | Rootless CUDA timeline/events at filled context; report top three wall-path contributors and overlap limits |
| Sustained telemetry | Active-window clocks/power/thermal, faults/I/O, NUMA placement, pressure and preserved safety floor |
| Final decisions | Base, indexer/attention hotspots, repeated quantization, CPU type path, static balance and ≤3 P3 admissions with evidence |

## Accounting instrumentation checkpoint

The opt-in CUDA allocation observer and disjoint host worksheet are implemented;
27 local V100 tooling tests pass. See `P0_MEMORY_ACCOUNTING.md` for coverage,
collection instructions and remaining physical-accounting gaps. Real cudart
interposition must be verified on the runner after the unprofiled campaign.
Do not mark the unique-memory gate complete from host tests or virtual ranges.

## Dispatch instrumentation checkpoint

`ABYSS_V100_DISPATCH=1` adds bounded host counters at the actual CPU row and
MMQ branches, with worker scopes and decode request/position joins. See
`P0_DISPATCH_LEDGER.md`. The CUDA build and device checks passed in run `39185866653`; physical
capture remains pending. 32 local tooling tests now pass (including accounting,
active-window summary and main-stream budget tests). Prefill counters identify chunks and expert
batch sizes, not individual gathered row tokens. No dispatch gate is closed
from source presence or host-only tests.

## Interrupted long-context campaign

The campaign ended unsuccessfully during the second long arm. Read-only
inspection run `39193892390`, artifact `12161480287`, preserved the partial arm.
Historical v1.3.28 completed all genuine 32K/64K/128K histories with 1024 outputs,
two requests each. Current v1.3.31 completed both 32K requests before interruption.
No engine survived inspection. Pre-interruption telemetry had >44 GiB minimum
MemAvailable, no cgroup OOM events, and the last sample at 03:01:24Z; GitHub
marked failure at 03:10:58Z without finalized job logs. Cause is not yet proven;
this is not an engine OOM diagnosis. Preserve completed data, inspect host uptime
and use smaller checkpointed units before resuming missing long/chunk work.

Extended results: `p0-extended-2026-10-19.json`. Active-request samples showed
SM clock medians 1380 MHz on all six arms, no sampled thermal reason, occasional
power cap activity, minimum MemAvailable >45.27 GiB. Median paired prefill
changes were within ±0.96%; decode changes ranged -3.96% to +3.05% by stratum
(negative favors current). All compared generated trajectories differed. These
observations do not establish a causal base performance improvement/regression.

## Running campaign

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
