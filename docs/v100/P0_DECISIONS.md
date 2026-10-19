# P0 decisions — revision 9, 19 October 2026

These decisions close baseline selection and experiment eligibility. They do
not promote optimized arithmetic or claim a production performance improvement.

## Base decision

**Yes: select v1.3.31 `7c893ac17e61ec45c934b3278356807fc4626c3c` for the next
implementation branch**, as v9 §1.1 directs after both baselines pass.
Its immutable binary SHA256 is
`5f11a611b284af9bb0df85d5e049e41bbbe7e02b81eb8176abce28fc6426ca89`.
Both bases build/run on V100, pass the bounded same-pack F32 comparison, and
complete the short/output and filled-context matrix with safe observed memory.
Extended short prefill paired differences are within ±0.96%; long prefill
differences are near zero. Decode differences vary with generated trajectories
and residency. There is no reproducible, matched-work causal regression here
that justifies overriding v9's future-base default.

The initial ~5% short-prompt observation is preserved, not erased: fixed-history
replay reduced the overall paired difference to +0.523%, and the extended
matrix did not establish the same broad slowdown. Normal-lane logit variation
exists within each base; all six bounded quality comparisons retain top-1
32/32. Neither observation proves a sole cause or full release qualification.

The P0 evidence branch remains based on historical `47001f31` for reproducible
isolation; **no rebase/merge is claimed**. Freeze both upstream binaries, the
pack, fixtures and this evidence. Starting the next implementation branch from
v1.3.31 and porting diagnostics must repeat P0b before using newly ported
results, as v9 requires. This migration is not an arithmetic/default promotion
and is not silently performed by a baseline-selection document.

Current STREAM_FETCH 0/1 dispatch is now measured, not inferred: 688 ordinary
launches at 0 and 688 streamed launches at 1, with no opposite-path launches.
The single profiled pair emits identical token IDs but has different cache/CPU/fetch populations, so its
761.3 / 878.5 ms decode intervals are not a causal control comparison and do
not support overriding the upstream default. THP has no active registered
anonymous tier in the measured configuration; its experiment is N/A for that
path. See `P0_FEATURE_DISPOSITIONS.md` and `p0-fetch-2026-10-19.json`.

## Hotspots and path decisions

| Decision | Yes / no / N/A | Evidence and scope |
| --- | --- | --- |
| Filled-context indexer hotspot | Yes | 22.49% of 32K and 49.38% of 128K prefill main-stream wall lane |
| DeepSeek attention hotspot | Yes | 26.17% / 21.14% of 32K / 128K prefill; also material in decode |
| Ordinary SM70 MMQ executes | Yes | Actual products, quantizers and CUPTI kernels; no inference from GLU eligibility |
| Fused GLU executes | No | Actual selection/run counts zero on SM70; ordinary MMQ still runs |
| Repeated input quantization | Yes | Exact 3,615,700 repeated rows, 83.04%; within/across-call counts and histogram digest retained |
| IQ2_XXS / Q2_K CPU hotspot | N/A | Neither routed type exists in this pack; no port of a nonexistent path |
| Actual CPU ISA/type paths | Yes | AVX2=1, AVX512=0 in both exact binaries; IQ2_XS/IQ3_XXS IQ256/GGML and MXFP4 GGML calls recorded |
| Static-plan exposed CPU/PCIe dependency waits | Yes | CPU wait kernels lead 32K/128K historical decode (27.27% / 22.81%); current 32K traces also expose CPU/fetch spans |
| Proven better online split | No | No controlled estimator A/B has run; waits alone do not establish misallocation or a winning update policy |
| Speculative verifier baseline | N/A | Speculation deliberately disabled in these baseline controls; T=2–8 CPU observations are expert prefill batches |

Top-three measurements use an ordered main-stream wall-lane budget plus
explicit unassigned gaps, not a complete causal CPU/GPU DAG. Do not add copy
stream or CPU sums to these wall-time shares. All per-token route masks
conserve selected experts; promotions overlap fetch/miss. The captured decode
has zero skip bits. Actual diagnostic populations are not clean-run residency
statistics because observation adds synchronization.

## P3 admission rank (two bounded experiments, after P1/P2)

| Rank | Experiment | Admission rationale | Mandatory stop / next-stage gate |
| --- | --- | --- | --- |
| 1 | P3.3 bounded online CPU/PCIe allocation | Exposed dependency waits dominate decode more than repeated quantization; actual CPU/fetch routes and NUMA placement are available | First obtain per-layer completed CPU/transfer cost and end-of-stage variation, with no extra hot-path synchronization. Retain startup plan unless repeated clean end-to-end A/B improves; stop on overhead, thrashing, NUMA or memory regression |
| 2 | P3.1 quantize-once/scatter | Ordinary q8_1 MMQ and exact duplication proven; isolated SM70 gather/scatter byte parity and sanitizers passed | Quantization is a small wall-time share (named layout-0 kernel 0.740% / 0.418% at 32K / 128K). Require repeatable full-prompt/TTFT benefit beyond noise; close N/A if prep is immaterial or scatter writes erase benefit |
| — | P3.2 Broadwell IQ2_XXS singleton | N/A for the actual pack | Reopen only with an actually executed matching format, not a scalar/AVX512 strawman |

These are admissions to bounded investigations, **not passed P3 acceptance
gates or authorizations to replace the startup plan now**. Rank 1 must collect
the listed per-layer costs before implementing an estimator. Re-profile after
P1/P2; their changes can alter the ranking. No third P3 item is admitted. All
other appendix candidates remain uncommitted. Arithmetic, expert selection,
quantized weights and defaults remain unchanged in this P0 closure.

## Limits retained

The current CPU calibration is 28 threads and 0.346 ms per expert in both fetch
arms. The default first supported calibration layer is layer 0 (IQ2_XS gate/up,
IQ3_XXS down). With E=4096 and n_ff=2048, its quantized weight payload is
`2*2048*4096/256*74 + 4096*2048/256*98 = 8,060,928` bytes. The 74/98-byte
256-value blocks are specified by the pinned ggml `block_iq2_xs` /
`block_iq3_xxs` definitions and static assertions. The resulting effective
weight-payload rate is approximately **23.30 decimal GB/s**. This application
rate includes compute/scheduling and rounded calibration time; it is not an
independent DRAM-bandwidth measurement and excludes activations/metadata/cache
effects. Do not substitute the maximum expert stride or nominal GGUF size.

- GPU occupancy/HBM/L2 hardware counters: unavailable (`ERR_NVGPUCTRPERM`);
  use the v9 rootless timeline/event alternative. Static launch resources are
  not achieved occupancy. CPU physical DRAM and QPI traffic are unmeasured;
  logical bytes and calibrated CPU/PCIe expert times are not traffic counters.
- Model SHA256 matches the host manifest, not an independent publisher
  signature. Earlier SHA-unverified experiments retain that original scope.
- Memory is a disjoint host/GPU budget with overlapping PSS/pin views and a
  source-derived borrowed-arena worksheet. It is not PFN/pagemap measurement.
  File-cache footprint does not establish its minimum required working set.
- No thermal reason was sampled; power-cap samples and clock variation remain
  disclosed. Sampling cannot exclude unsampled contention or throttle events;
  flagged/confounded comparisons are preserved and excluded from speed claims.
- The 32-row quality/repeatability screen is a P0 baseline screen, not P1/P2
  arithmetic acceptance or P4 release quality. P1/P2/P3/P4 remain separate.
