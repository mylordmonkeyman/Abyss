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
