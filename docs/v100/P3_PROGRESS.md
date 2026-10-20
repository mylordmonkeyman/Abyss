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

The 32K capture extends the same observational protocol to frozen genuine 32K
history. It keeps the startup plan, 28-thread lane, F32 dispatch and all security,
memory, profiler and artifact limits. No synthetic position offset substitutes
for long-context routes. Longer context alone does not close observer-overhead,
cold-to-warm, competing-load or alternate-placement evidence gaps.

The 32K extension is dispatched as
[run 39405805983](https://github.com/mylordmonkeyman/Abyss/actions/runs/39405805983)
from `dc409c2d532270b549d06bc8070714b758ef257e`. Its successful result is
reviewed below. Artifact-summary regression tests additionally reject
incomplete populations, source/binary identity disagreement, wrong context,
observer drops and ZIP digest mismatch. All 62 local V100 tooling tests and
the HIP-name mapping check pass; this does not substitute for physical HIP or
other-device inference.

## Genuine 32K result and within-layer/count variation

Run 39405805983 passes. Artifact `12257890824`, ZIP SHA256
`5b1e44a6fb03803cc4e4d255e8c260667249a13852049c66a41f15cac1cd3ef2`,
recomputes into `p3-cost-32k-2026-10-20.json`. The actual engine binary SHA256
is identical to the 8K capture (`be1e0c02b7087b49b94232c32794a5ac736eea1f05fab1e1281b2f4483f4f602`);
the fixture SHA256 differs as expected for genuine longer history.

All 2752 layer/token records join; 2542 have completed CPU calls. There is one
nonzero miss-mask record and zero observer drops. The raw report is 23,200,145
bytes, retained with its SHA256 and SQLite digest on the runner. Minimum sampled
MemAvailable is 46,361,104,384 bytes; RSS/PSS peak at 86,613,348,352 /
86,604,806,144 bytes (overlapping). All 24 telemetry samples are valid; no
thermal flag is sampled, but power limiting is observed. No clean speed claim.

The summarizer now compares request medians **only within the same device,
layer, CPU expert count and fetch expert count**, excludes disk-miss records,
and requires at least eight samples in each request. The 32K capture provides
43 such CPU-active strata. Examples of completed CPU-call wall medians:

| Layer / CPU experts / fetch experts | Request samples | First ms | Second ms | Relative change |
| --- | --- | --- | --- | --- |
| 15 / 2 / 0 | 15 / 14 | 0.6650 | 0.4332 | -34.85% |
| 28 / 2 / 0 | 8 / 8 | 0.6530 | 0.4444 | -31.95% |
| 19 / 2 / 0 | 9 / 8 | 0.6472 | 0.7379 | +14.01% |

This is descriptive variation, not paired execution of identical experts or a
significance test. Expert identities, cache locality and host scheduling remain
uncontrolled; pooled count-only medians also mix layers/formats. Neither the
cause of a slow call nor a better allocation follows from these medians. The
minimum sample screen is not an estimator admission threshold. The 8K ledger
is regenerated with the same descriptive strata for reproducibility.

The next serialized capture uses frozen genuine 128K history with unchanged
32-output requests and safety limits. Only then proceed to observer-overhead
and cache/load/placement controls; no bounded estimator or production change
is admitted by these warmed context observations alone. All 63 local tooling
tests and the HIP-name check pass before dispatch.

## Genuine 128K result

[Run 39412803044](https://github.com/mylordmonkeyman/Abyss/actions/runs/39412803044)
passes on `babc202cdcfced6aa288f92a29281bb38529b8ff`. Artifact `12261181483`,
ZIP SHA256 `e6adb31e81123a9fe4c15a743ac953eed3eb38c69e7576ad9a45882d22a495eb`,
reproduces `p3-cost-128k-2026-10-20.json` with the same summarizer/digest checks.
The engine binary SHA256 is identical across all three context captures; each
history is genuine, frozen and checked by its distinct fixture digest.

All 2752 layer/token records join, with 2342 CPU completions, no nonzero miss
masks and zero observer drops. There are 58 same-layer/count request strata
with at least eight samples each. The largest absolute CPU-call median change
among those strata is 8.11%; this remains descriptive, not a controlled change
in allocation, expert identity or CPU load. Absence of disk-miss bits in this
warmed trace is not proof of a cold-cache working set or absence of all I/O.

Minimum sampled MemAvailable is 45,803,896,832 bytes. RSS/PSS peaks are
86,875,045,888 / 86,867,211,264 bytes, overlapping. All 109 telemetry samples
are valid, with no sampled thermal flag and observed power limiting. Profiler
report size is 54,619,734 bytes, retained with its digest and SQLite identity
on the runner. Profiled request timings and differing cache/CPU populations
are not clean throughput evidence.

## Observer/profiler controls — next bounded experiment

`p3_observer_control.py` runs plain / observer / traced / traced / observer /
plain in fresh processes. The two isolated build trees have the same source
and settings except `STRATA_V100_CAPTURE_COST=OFF/ON`; neither replaces the
working or immutable upstream binaries. No runtime observer switch or
production-source change is introduced.

Each arm uses the identical 8224-token prefix derived from frozen
`mixed-32768.tokens`: 8192 context tokens and 32 teacher-forced continuations.
One excluded LOGP replay precedes three measured replays. A different explicit
two-token context before every replay prevents LOGP's equal-context snapshot
reuse (GEN's NO_REUSE setting alone does not control that path). The traced
arms use the existing pinned, rootless CUDA/NVTX session API, with warm-up
excluded from collection. Plain/observer arms do not launch Nsight. ON arms
must have a nonempty, zero-drop annotation ledger; OFF arms must have none.

These wire times include prefill, host logits/log-softmax and up to 10 ms of
controller polling, not normal decode throughput. Fixed input bytes reduce
generated-trajectory differences but do not eliminate CPU-lane-28 router
variability, cache, scheduling, clock or order effects. Host load/stat,
placement, memory and GPU telemetry remain recorded. Contrasts are descriptive
overhead screens, not statistically established causal overhead or optimization
admission. No thresholds are relaxed and no estimator is enabled.

All GPU work remains serialized by `abyss-v100` and owner/trusted-branch guards.
The 64 MiB upload cap, 16/8 GiB host margins and bounded runner-side reports
are retained; cleanup targets only the controller's own process/session.
No global cache drops, all-pages NUMA binding, competing load, container churn
or privilege changes are used. Cold-to-warm, load and placement controls remain
separate outstanding measurements. All 65 local tooling tests and the HIP-name
mapping check pass before this control dispatch.

## Observer controls reviewed; natural cache evolution next

[Run 39419705642](https://github.com/mylordmonkeyman/Abyss/actions/runs/39419705642)
passes on `72c2474c3600c9a0d28c8d4af66aa06ec80fc4d4`. Artifact `12263990850`,
ZIP SHA256 `8da7dc1547e941fae08f3a85a8a002134024174fb7424c240450b4237edc1de1`,
is recomputed by `summarize_p3_control.py` into
`p3-observer-control-2026-10-20.json`. It checks wire replies, arm order and
population, binary/source identities, equal environments, observer ledgers,
profiler start/stop success and checkpoint consistency before recomputing
contrasts. All four observer-enabled arms have nonempty, zero-drop ledgers;
both compiled-out arms have no observer ledger. The observer binary is the
same digest used in the preceding context captures.

| Order half | Plain median wire ms | Observer median ms | Traced median ms | Observer vs plain | Traced vs observer |
| --- | --- | --- | --- | --- | --- |
| Forward | 12657.87 | 12740.76 | 13319.98 | +0.65% | +4.55% |
| Reverse | 12901.06 | 12832.57 | 13287.07 | -0.53% | +3.54% |

No consistent observer-only slowdown is established by these two contrasts;
they are not a confidence bound or proof of zero overhead. Tracing is slower
in both order halves. Different likelihoods and annotation counts preserve
the known CPU-lane-28 trajectory limitation even with fixed token bytes.
Therefore profiled cost captures remain diagnostics, never clean throughput
or a causal predictor of a better split. No KL threshold is waived or baseline
variance subtracted. Minimum sampled host MemAvailable is 48,256,970,752 bytes;
no thermal flag is sampled and power limiting is observed in four of six arms.
Raw reports of 25,300,658 and 23,725,990 bytes stay on the runner with digests.

Next the existing cost harness omits the excluded warm-up and observes the
**first post-load request and its subsequent request**, both with genuine 8K
history and 32 outputs. The protocol is recorded in invocation/raw identities
and the summary rejects a mislabeled warm-up. This is natural process/device
cache evolution, not a known-cold OS/file cache: startup itself primes experts
and all model pages retain their existing policy. No cache drop, eviction
workload, NUMA bind, artificial CPU load or new privileges are introduced.
It does not substitute for separate load and alternate-placement conditions.
All 66 local tooling tests and the HIP-name mapping check pass before dispatch.
