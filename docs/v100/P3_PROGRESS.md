# P3 — bounded P0-admitted investigations

20 October 2026. P1 and P2 conclude with documented F32 fallback decisions;
neither promotes approximate arithmetic. Work remains on the current branch
to preserve the selected v1.3.31 base and evidence history.

`P0_DECISIONS.md` admits only two investigations: (1) per-layer CPU/PCIe split
cost evidence, then (2) quantize-once/scatter if its small measured wall-time
share still warrants it. No estimator or third optimization is admitted yet.

## Completed-cost observation, before an estimator

`STRATA_V100_CAPTURE_COST` is a default-OFF, CUDA-only internal diagnostic
build option. It adds bounded NVTX markers containing the actual service
request's CPU/fetch/miss masks and logical blob size, plus ranges around the
completed CPU worker call and host enqueue of fetch/CPU-wait kernels. It adds
no CUDA event/synchronization, allocation decision, expert-selection or math
change. The observer refuses records beyond its 32768-annotation cap and the
campaign requires zero drops. Default builds contain no observer calls.

The dedicated build reuses the immutable model/pack and pinned profiler. One
excluded 16-output warm-up precedes a capture of two 32-output uncached 8K
requests. The existing CPU lane stays at 28 threads, startup plan unchanged;
speculation, deferred expert combines and KDA graphs are explicitly off in
this diagnostic. No caches are dropped or memory policy changed. These are
warmed observations, not a cold-cache or production performance comparison.

The analyzer joins NVTX labels to CUDA runtime correlation IDs and completed
CUPTI kernels, then to request/device/layer and argmax-delimited token IDs.
It rejects ambiguous/missing joins, missing CPU completions, overlapping masks
and incomplete per-layer token populations. CPU completed-call wall time,
mapped-read fetch-kernel elapsed and exposed CPU-wait-kernel residence are
reported separately. They overlap and must not be summed into token latency;
fetch elapsed/logical bytes are not physical PCIe/QPI bandwidth counters.

Raw report/SQLite stay on the existing runner; the summary upload is capped at
64 MiB. No extra model pages, NUMA pinning, GPU reset, privilege change or
container restart is involved. Historical SHA-unverified captures retain their
original labels; current local-manifest verification is not publisher authenticity.

Before fitting any estimator, inspect per-layer cost/count variation, miss-free
versus disk-involved routes and the diagnostic's overhead. Cold-to-warm, longer
context, competing-load and alternate-placement conditions are still outstanding;
this first dataset cannot replace them. Any future bounded policy needs exact
expert/result preservation, no thrashing, minimum samples/update bounds, safe
memory/NUMA behavior and repeated clean 128/512/1024-output A/B improvement.

## First completed dataset: genuine 8K history

[Run 39398479434](https://github.com/mylordmonkeyman/Abyss/actions/runs/39398479434)
passes on source `69e0fec911eb6a521a788196ab8a24e5cb8e95b3`.
Artifact `12254960180`, ZIP SHA256
`302c5efbb3e5e974d660c8caf7d478542e2c08d7f1786b3750eff4b21093c44d`,
is independently population-checked and conditioned by request, CPU/fetch expert
count and miss status by `summarize_p3_cost.py`. Reproduce with that script's
`--sha256`, `--run 39398479434` and `--output` arguments; the committed output is
`p3-cost-8k-2026-10-20.json`.

All 43 layers have all 64 completed token-path forwards (two 32-output requests,
including each prompt-final forward): 2752 joined records, 2489 completed CPU
calls. The observer reports 13493 annotations including excluded warm-up, zero
drops. Five records have a nonzero disk-miss mask and are separated from the
miss-free populations. Zero-fetch no-op kernels are also separated.

For example, miss-free medians in the first/second request are:

| CPU experts / fetched experts | Records | CPU call ms | Fetch kernel ms | Exposed CPU wait ms |
| --- | --- | --- | --- | --- |
| 2 / 0 | 428 / 408 | 0.6583 / 0.6730 | 0.0033 / 0.0033 | 0.5500 / 0.5680 |
| 2 / 1 | 323 / 207 | 0.6808 / 0.6863 | 0.7974 / 0.7964 | 0.0158 / 0.0149 |
| 4 / 1 | 58 / 28 | 1.2763 / 1.2893 | 0.8659 / 0.8484 | 0.3287 / 0.3122 |

These observations show work-count-dependent costs and exposed waits, not a
counterfactual proof that reallocating the same experts is faster. Request route
populations differ; the CPU-lane-28 F32 repeatability limitation is unchanged.
No estimator is fitted or enabled and no timing columns are added together.

Minimum sampled host MemAvailable was 46,975,135,744 bytes; peak RSS was
86,158,761,984 bytes and peak PSS 86,150,887,424 bytes (overlapping, not additive).
Fourteen telemetry samples report no thermal flag, but observed power limiting;
this is neither sustained clock qualification nor a clean performance claim.
Raw profiler report and SQLite identities are recorded and retained on the
runner for replay without loading the model.

Next capture extends the same observational protocol to frozen genuine 32K
history. It keeps the startup plan, 28-thread lane, F32 dispatch and all security,
memory, profiler and artifact limits. No synthetic position offset substitutes
for long-context routes. Longer context alone does not close observer-overhead,
cold-to-warm, competing-load or alternate-placement evidence gaps.
