# P0 current-base feature dispositions

The pins remain historical `47001f31` and current `7c893ac1`; no mutable
upstream binary or model pack is substituted. Both complete the benchmark
matrix and bounded same-pack F32 screen. Final selection is recorded in
`P0_DECISIONS.md`; there is no causal base speedup/regression claim.

## RAM THP

Active registered-THP expert tier: **no** in the measured configuration.
THP 0/1 performance experiment: **N/A for the observed active tier**. Both
immutable allocation traces have zero cudaHostRegister calls; their live host
pins use cudaHostAlloc and shared backing, with observed AnonHugePages zero.
The pinned v1.3.31 source applies its anonymous mmap/MADV_HUGEPAGE policy in
`numa_pinned`, followed by mbind/registration, with cudaHostAlloc fallback when
that path cannot be used. The active shared pinned ranges are not anonymous
registered THP ranges. There is no demonstrated THP performance benefit to
toggle on those ranges. This does not assert that no startup madvise was
attempted, identify which fallback guard failed, or generalize to other hosts.
No attempt was made to force registration, privileged access, compaction,
cache dropping or node-local placement. Existing evidence and safety limits
take precedence over manufacturing a THP comparison.

Historical `STRATA_GLM_RAM_THP`: **absent/N/A**, not a silently ignored 0/1
experiment. Current default-on source presence is recorded separately from
actual resident-page backing. See `P0_ACCOUNTING_RESULTS.md` and its digests.

## Streaming fetch

Historical control: **absent/N/A**. Current control: default on in the pinned
host source. It supplies gate/up bytes, down bytes and read-chunk count to
`gf::moe_fetch`; the pinned CUDA dispatcher selects `moe_fetch_stream_kernel`
when those arguments are positive, otherwise `moe_fetch_kernel`. The current
host disk service also uses the control for per-chunk landing/availability.
These source facts are not proof of executed kernels or a measured speedup.

The targeted feature workflow now collects two serialized rootless 32K traces
of the immutable current binary with explicit 0/1 settings. Each has one
excluded warmup, no prefix reuse, frozen tokens, 16 output tokens and the
existing 16 GiB initial / 8 GiB runtime memory floors. The collector requires
ordinary-only fetch kernels at 0 and streamed-only fetch kernels at 1, using
actual CUPTI activity names. Source/build/model hashes and raw report paths
are retained; uploads are bounded. Inspect failures before retrying. No speed
claim or setting promotion follows merely from a passing dispatch assertion.
Run `39236730013` passed both arms. Artifact 12179412869 (control 0) has ZIP
SHA256 `86482f668b47acb454b8da960f22a16ec9ca3e35d1dd2b630c49219463689d03`;
artifact 12179602638 (control 1) has ZIP SHA256
`0a6c91fdcfbe089583969b938ac28ef8e6bcd417dc8300a62cbb3665ce2ab4de`.
Both downloaded archives match. Actual CUPTI activity reports **688 ordinary,
zero streamed** launches at 0 and **688 streamed, zero ordinary** at 1. The
immutable binary SHA256 matches the existing current-base manifest in both.
`p0-fetch-2026-10-19.json` preserves dispatch, source/model/fixture controls,
native request records, CPU calibration, route summaries, timelines and raw
trace identities. Profiled output IDs are identical, but cache/CPU/fetch
populations differ: final native VRAM-hit summaries are 69.82% / 56.95% at
0 / 1. Consequently 761.3 / 878.5 ms decode times are not a matched-residency
speed comparison. This closes execution eligibility; no upstream default is
overridden. Current 32K prefill again has MMQ/attention/indexer as its top three
main-stream contributors, consistent with the historical 32K profile.

GPU hardware counters and QPI traffic remain unavailable/unmeasured. The
timeline alternative is retained, without any security or privilege change.
