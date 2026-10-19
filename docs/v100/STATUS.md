# V100 implementation status — 18 October 2026

Branch base: upstream `47001f31cbfba14bcbf4c083e3af07c50707ec0c` (v1.3.28).
Production arithmetic and dispatch remain unchanged. No performance improvement is claimed.

## Active specification

The implementation now follows **v9** (`IMPLEMENTATION_SPEC.md`).
See [v9 adoption and paired-base campaign](V9_ADOPTION.md). It supersedes v8's
task ordering; the [v8 applicability audit](V8_ADOPTION.md) retains actual-format corrections,
newer-Abyss differences and the isolated quantize-once/scatter contract. The pinned
base and measured baseline are preserved. P0 base-selection, dispatch, memory and
quality gates remain open; unavailable counters are disclosed under v9's event/
timeline alternative. Adopting v9 does not make proposed flags functional.

### v9 initial paired-base result

[Run 39150769368](https://github.com/mylordmonkeyman/Abyss/actions/runs/39150769368)
completed on `c2d490e`: both **unmodified upstream SHAs** built for SM70 and
completed 36 total requests, 128 output tokens each, across 600/4K/8K prompts.
Three process pairs used A/B, B/A, A/B ordering, identical pack and fixture IDs,
explicit FP32 overrides, speculation/prefix reuse/persistent usage disabled.

| v1.3.31 versus v1.3.28 | Median paired prefill latency change | Median paired decode latency change |
| --- | --- | --- |
| 600 prose | +5.21% | +4.91% |
| 4,096 code | -0.14% | -2.09% |
| 8,192 mixed language/numeric | -0.17% | +1.97% |

Negative values favor v1.3.31. These are three-pair initial observations, not
clean causal speedups: per-pair variation is substantial, power-limit activity
is not matched in every arm, and greedy strings differ. Teacher forcing across
the 2,048 boundary gives **mean KL 1.5014e-7**, max **1.0741e-6**, top-1 **32/32**.
This passes the small v9 incremental screen, not the full quality/reuse gate.

All six process arms report the same rounded **24.18 GiB GPU expert pool** and
**68.28 GiB pinned RAM tier**. Peak sampled RSS is about **78.60–78.62 GiB**;
minimum host MemAvailable remains above **45 GiB**. No sampled thermal slowdown;
power-cap activity occurs in five arms. AnonHugePages is zero in these samples,
so the newer THP policy has no demonstrated realized benefit here. RSS, CUDA
registered pages, cgroup memory and file cache are overlapping views.

**Decision: retain v1.3.28 for isolation**, investigate short-prompt variability
and actual placement/dispatch before adopting v1.3.31. The merge dry run is clean;
no merge/rebase or production optimization was performed. Extended 512/1024-output,
filled 32K+, full unique memory accounting and per-layer dispatch remain open.
See [paired evidence](v9-bases-2026-10-18.json) and
[memory worksheet](v9-memory-2026-10-18.csv). Artifact `12145906706` is
2,869,760 bytes, SHA256 `e7570b41bbeaf6502b678b4e2e12d65eb9ed7508768183d49c85af33a58963e7`.

### Fixed-token follow-up and P2a next step

[Replay run 39171655697](https://github.com/mylordmonkeyman/Abyss/actions/runs/39171655697)
passed all 36 fixed-history requests (one warmup plus five measured requests per
process). The paired median LOGP wall-time changes are **+1.260%, +0.380%, +0.523%**
for v1.3.31; median across pairs **+0.523%**. LOGP includes host-logit transfer and
CPU normalization, so these are not decode throughput changes. The smaller gap
does not reproduce the initial ~5% observation, but all three paired signs remain
positive. Within-version sum log-probabilities vary across identical requests;
fixed tokens do not establish identical internal routes or arithmetic repeatability.

Retain v1.3.28 for isolation under v9. The next narrowly scoped base investigation
is repeated fixed-history logits/router/cache-state capture, rather than treating
this small replay difference as a proven source regression. No thermal slowdown
was sampled; placement remains a confounder. The complete replay artifact is
`12153042649`, 1,931,188 bytes, SHA256
`e8874df89dab98a1d15c1b33909fc790a58586a7cbd7e6692b744ca5216fcd0d`.
See [compact replay evidence](v9-replay-2026-10-18.json).

The isolated [P2a QK/sink-softmax/PV diagnostic](P2_ARITHMETIC.md) is implemented
with 32 synthetic cases and graph replay, FP64/device-FP32 references, guard
checks, SM70 resource/SASS capture and sanitizer steps. Its [first physical V100 run](https://github.com/mylordmonkeyman/Abyss/actions/runs/39172763525)
passed all 32 cases and both sanitizers (zero errors). It does not enable production arithmetic or satisfy the full
latent/gather/RoPE/output-projection/model-quality gate.

### P2 cached-row gather and inverse-RoPE result

[Run 39173519298](https://github.com/mylordmonkeyman/Abyss/actions/runs/39173519298),
commit `8c74a3f`, passes **21 cached-row cases / 54 checked token positions**,
with three graph replays per position and zero memcheck/synccheck errors. It
compares shared staged SM70 QK/sink-softmax/PV plus inverse RoPE with FP64 and
actual FP32 `attend_batch`, using real ring load/store and query preparation.
Raw wrap/chunk tails, exact full-versus-split state/output, CSA visibility around
top-512, HCA chronology and malformed-selection rejection pass. Maximum relative
L2 versus existing FP32 attention is **1.0040e-6**; versus FP64 after inverse RoPE
**1.4413e-6**. See [all case/extrema evidence](p2-latent-gathers-2026-10-18.json).
Hosted Python/CPU-build CI passes.

The staged gather path is diagnostic only. At that stage, full P2 still required learned input
projections and actual compressor/norm construction, grouped output projection,
real stage dumps and same-branch logits, then a fused profitable geometry. No
production dispatch, public switch or performance improvement is enabled.

### P2 compressor construction and grouped output result

[Run 39178758945](https://github.com/mylordmonkeyman/Abyss/actions/runs/39178758945),
commit `a7a0727`, passes **10 actual-kernel compressor cases / 54 paired grouped
output-projection checks**, plus the 21 cached-row attention cases. Both
sanitizers report zero errors; hosted Python and CPU-build CI pass. Compressor
cache/ring state is byte-identical for full versus split input; incomplete blocks
leave the cache untouched. Rounded-FP64 compressor relative L2 reaches 2.0676e-5;
the broader pre-storage FP64 comparison reaches 2.0848e-4, including F16 rounding.

The 8-group × 1024-rank output check uses the actual prefill cuBLAS shapes,
leading dimensions and two F16 conversion points with synthetic sparse weights.
Maximum candidate-versus-fallback post-output relative L2 is **5.8131e-5**.
See [all case results](p2-components-2026-10-18.json). Learned input GEMMs, raw-KV
and independent query preparation, real quantized output weights, bounded model
stage capture/logits and fused resource/performance gates remain open. No
production dispatch or performance claim is enabled.

### P2 real model activation replay

[Run 39180592084](https://github.com/mylordmonkeyman/Abyss/actions/runs/39180592084),
commit `d155daf`, passes **3 actual-model layers / 33 stage checks**, graph replay
and both sanitizers (zero errors). Frozen capture `ce1091a` covers position 2,048
in raw layer 0, CSA layer 2 and HCA layer 3. Independent preparation references
consume actual learned projection outputs; replay uses actual caches, selections
and sinks. All repeated arithmetic investigations used identical hashed tensors,
without another model load. The normal engine cache was restored afterward.

Real CSA data exposed a 1.17e-5 relative attention difference that synthetic inputs
did not. The same-probability scalar comparison isolated long PV accumulation.
64-element short independent hi/lo chains plus compensated FP32 combination
reduce it to **9.91e-7**, without widening the target. Raw/HCA differences are
5.69e-7 / 5.33e-7. New QK/PV use 64/63 registers with zero local bytes.
[Normal GPU CI 39180592052](https://github.com/mylordmonkeyman/Abyss/actions/runs/39180592052)
and hosted Python/CPU-build checks also pass. See [all real-stage evidence](p2-real-stages-2026-10-18.json).

This is isolated real-input numerical validation, not a runtime improvement or
full-model KL pass. Actual output-weight replay, downstream model-logit/quality
gates, wider layer/position coverage and profitable fused geometry remain.
Production arithmetic/dispatch stay unchanged.

[v8 run 39149507205](https://github.com/mylordmonkeyman/Abyss/actions/runs/39149507205),
commit `f7138de`, passed **89 isolated Q8_1 scatter byte/guard cases**, with zero
memcheck/synccheck errors. Existing DeepSeek/MMA checks also passed; the engine
binary is unchanged. The new source-feature audit confirms the actual mixed IQ
and MXFP4 inventory and absence of routed IQ2_XXS/Q2_K. Upstream exact-one also
supports the relevant IQ2_XS type, retained as a separately validated CPU candidate.
Scatter integration is deferred: all MMQ quantization represents only 0.638% of
the existing 2K trace's prefill wall time, and full-route scatter needs a resource
design for Abyss's smaller W-row buffers. The ~1.5x isolated quantizer result is
not a whole-model speedup. v9 hosted CI passes 130 server / 209 tool tests.

## Confirmed on the actual V100

- [Initial build/device run 39100453015](https://github.com/mylordmonkeyman/Abyss/actions/runs/39100453015),
  commit `41affbb`: CUDA 12.8.93, Tesla V100-PCIE-32GB, cc 7.0; SM70 engine and device
  executable built; device allocation/kernel smoke passed; memcheck and synccheck each
  reported **0 errors**. The existing build cache was reused on this run.
- [DeepSeek component run 39100790344](https://github.com/mylordmonkeyman/Abyss/actions/runs/39100790344),
  commit `1a3dc6d`: **26 synthetic cases passed** under default dispatch and again with
  `STRATA_DS4_IDX=1 STRATA_DS4_ATTN=1`. The full component suite also passed memcheck
  and synccheck, both with **0 errors**.
- Indexer cases check FP64 scores, GPU-score ordering, tie-breaking, meaningful-margin
  FP64 membership, causal visibility, 512/513 cutoff, 8,192 visible rows, 32-token tile
  tails and untouched output guards. Raw-ring cases check wrap/chunk load/store.
- Attention cases use the real 64-head × 512 latent K=V shape, raw/CSA/HCA paths,
  sink regimes, 128-token raw-window and compression boundaries, and plain/YaRN inverse
  RoPE. Maximum observed attention relative L2 versus the scalar FP64 reference was
  about **6.10e-7**, maximum absolute error about **7.40e-8**, on these synthetic inputs.
  These are component results, not full-model quality numbers.

The first parity-harness build exposed an NVCC initializer-list deduction issue;
explicit shape arrays fixed it in `1a3dc6d` before the passing run above.

## Other changes and checks

- P0 capture/inventory tooling: nine host-only tests pass; synthetic GGUF census and
  host-only CMake build passed previously and in hosted CI.
- `8324f9e` repairs the inherited HIP `cudaMemcpyKind` mapping. The mapping checker and
  its two tests pass locally; hosted CI passes all 130 server and 202 installer/tool
  tests. This does not claim a new ROCm runtime qualification.
- Isolated P1a MMA884 primitive/test added for four quadpairs, all accumulator coordinates,
  typed PTX operands, masks/K tails, signed/zero/subnormal stress, and CUDA graph replays.
  It is **not wired into inference**. [Run 39101028811](https://github.com/mylordmonkeyman/Abyss/actions/runs/39101028811),
  commit `ed41547`, passed all **15 MMA contract cases**, device-scalar/FP64 comparison,
  memcheck/synccheck (0 errors), and selected SM70 image/PTX/SASS checks. All contract
  cases reported zero absolute error on these exactly representable synthetic inputs,
  including the subnormal case. This does not establish error bounds for arbitrary
  full-model FP32-to-FP16 splits. The emitted PTX contains the expected MMA884 opcode
  and the selected SM70 SASS contains HMMA. Full-engine opcode auditing is not claimed.
- [Hosted CI run 39101028820](https://github.com/mylordmonkeyman/Abyss/actions/runs/39101028820)
  also passed on `ed41547`. The GPU container and existing build cache were reused;
  no runner/container replacement was needed for these tests.

## P0 remains partial

The component test bypasses q-projection/compressor construction and grouped output
projections; it does not establish complete model-exact MLA stage parity, quantized
MXFP4 expert correctness, model logits, or task behavior. The indexer tests intentionally
invoke the scorer directly, including below top-512 visibility, without changing the
production caller's skip policy.

The read-only `/models` mount is now available. [Full-model run 39137399940](https://github.com/mylordmonkeyman/Abyss/actions/runs/39137399940),
commit `852ef95`, passed both initial runtime cases with coherent generated text:

| Prompt tokens | Generated tokens | Prompt time | Decode time |
| --- | --- | --- | --- |
| 14 | 16 | 844.7 ms | 758.9 ms |
| 464 | 256 | 1,892.4 ms | 9,449.7 ms |

The second case used batched prefill for 463 tokens and the token path for the final
prompt token. These runs enable stage/gap profiling: the approximately 245 prompt
tokens/s and 27.09 generated tokens/s in the second case are **instrumented bring-up
observations**, not an unprofiled baseline or an improvement claim. Loading and
shutdown are excluded from those phase rates; total process wall time was about
101 seconds. Speculation was off, capacity 4,096, RAM budget 72 GiB, CPU lane 28 threads.
The second process reused the first process's expert-usage profile, so these two
cases are not a controlled A/B pair.

Static inventory confirms MXFP4 routed down projections at layers **26 and 42**.
The runner reports 6.17 GiB dense GPU weights, 24.18 GiB GPU expert pool and
68.28 GiB pinned expert tier; these are not a complete process/page-cache ledger.
Five-second GPU samples observed 36–44 C and no active thermal slowdown reason;
they do not establish sustained benchmark conditions. Driver 580 uses
`instant_power_draw`/`current_power_limit` XML tags; the capture parser now supports
these tags, with a regression test. Original XML was preserved, allowing recovery
of power data from this run without rerunning inference.

The pack is persistent under `/actions-runner/abyss-models/`; GGUF shards are
symlinked to the read-only mount, with no 97-GB duplicate or model edits. Model
verification here is exact byte sizes and GGUF structure, **not a verified SHA256**.
Artifact `12140800382` is 48,483 bytes; its SHA256 is
`506165216495d657c35bf9578e6f018ad8a0516981b8d19de3ea86f9b2629a8f`.
The initial harness incorrectly supplied a `generate` subcommand; `852ef95` fixed
that invocation before the passing run. Hosted CI and GPU components also passed
again on that commit (130 server / 202 tool tests, with existing skips).
Hosted CI now uses the same pinned allowlisted checkout action and the image's
Python, removing the extra setup-python action dependency.

## Warm baseline, teacher forcing, and actual CUDA timeline

[Baseline run 39139393175](https://github.com/mylordmonkeyman/Abyss/actions/runs/39139393175),
commit `c2bbbad`, completed an excluded warm-up plus **five measured warm repetitions**
per length, 256 output tokens each, profiling/speculation/prefix reuse off. Context
capacity was 36,864, RAM budget 72 GiB, CPU lane 28 threads. Requested PCIe share
0.21 becomes about 0.17 in the engine's integer per-route plan. Persistent usage
history was disabled. Same-process expert residency still warms across requests.

| Prompt | Prefill median (range), tokens/s | Decode median (range), tokens/s |
| --- | --- | --- |
| 2,048 | 549.8 (540.7–563.7) | 34.17 (33.51–34.91) |
| 8,192 | 709.1 (707.5–712.8) | 32.15 (31.31–33.29) |
| 32,768 | 623.7 (621.3–624.7) | 30.55 (29.86–31.54) |

These are **synthetic baseline observations, not improvements**. All lengths used
the same executable and packing. Greedy output differed across repetitions; at 2K,
the first mismatch versus warm repetition 1 was at generated index 22 or 34. No
cause is established. Treat this as a baseline-variability investigation before
using generated-string equality as an optimization acceptance gate.

Peak sampled RSS was 78.68 / 78.75 / 79.07 GiB; peak container charge 73.16 /
73.24 / 73.59 GiB. These measurements are not additive. No cgroup OOM events were
recorded. Maximum sampled temperatures were 58 / 66 / 71 C, with no sampled
thermal slowdown. Power-cap activity occurred in the 8K and 32K runs. Full-process
samples include loading/idle intervals, so their clock ranges do not prove steady
active clocks. NUMA mapping was about 54/24 GiB on nodes 0/1 at 2K and 32K but
24/55 GiB at 8K (at peak RSS); policy labels were default. That is mapped residency,
not measured QPI traffic. Cross-length changes are not a controlled causal A/B.

[Same-engine teacher-forced run 39138840797](https://github.com/mylordmonkeyman/Abyss/actions/runs/39138840797)
compared **64 full-vocabulary distributions**, crossing the ~2K selection and
compression boundaries: mean KL **5.5087e-7**, max **7.2320e-6**, top-1 **64/64**.
Both dumps contained 8,273,920 finite values. This fixture was highly predictable;
it is an internal path screen, not a representative task-quality corpus.

[External CPU reference run 39140902521](https://github.com/mylordmonkeyman/Abyss/actions/runs/39140902521)
compared Abyss against pinned llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`
on **129 varied teacher-forced positions**, including the raw-window wrap. Mean
KL was **0.0135402**, max **0.0529301**, top-1 **124/129**. Mean NLL was 3.63659
(reference) versus 3.65062 (Abyss). This calibrates one small reference fixture;
it is not an unconditional quality pass or a like-for-like comparison with the
upstream README's different document. Raw distributions were temporary; summaries,
hashes and per-position results were uploaded (artifact 933,184 bytes).

[Rootless timeline run 39140801624](https://github.com/mylordmonkeyman/Abyss/actions/runs/39140801624)
succeeded with Nsight Systems **2024.6.2**, a checksum-pinned package acquired from
NVIDIA's CUDA repository into the runner's tool cache. No system installation,
container recreation or privilege change was needed. The **9,206,422-byte** artifact
contains the report and SQLite export for one warmed 2K prefill + 32-token request.
CPU sampling/context-switch tracing was disabled. The first argmax marks the
prefill/decode split; the SQLite interval matches reported decode time within
0.022 ms. Analysis is reproducible with `tools/profiling/analyze_trace.py`.

| Captured phase | Largest source-tagged shares of summed kernel time |
| --- | --- |
| Prefill | Routed-expert MMQ 50.46%; DeepSeek attention 18.04%; dense/shared/projection GEMM 15.99% |
| Decode | Dense/shared/projection GEMV 31.50%; CPU-expert dependency wait 25.93%; routed-expert arithmetic 15.11%; RAM fetch kernel 8.53% |

Indexer score/select accounts for about 2.38% / 2.04% of prefill/decode kernel time
in this capture. These are kernel-time shares, **not wall-time savings or proof of
compute/HBM limits**. The timeline records about 867 ms of prefill intervals with
an explicit copy active but no compute kernel; CPU activity and zero-copy PCIe
reads need separate interpretation. The report contains the FP32 `idx_score*` /
`attend*` symbols, **zero `qsa_*` launches**, **64 MXFP4 down-kernel launches** over
32 token-path forwards (including the final prompt forward), and 17 MXFP4 MMQ
launches. Presence/finite end-to-end output is not standalone MXFP4 arithmetic parity.

The shipped expert inventory is important for the next profiler campaign:
42 layers' gate/up weights are **IQ2_XS**, one **IQ3_XXS**; down weights are
**IQ3_XXS** on 41 layers and **MXFP4** on layers 26/42. Dense cuBLAS/CUTLASS Volta
Tensor Core kernels are already present; our isolated MMA884 primitive remains
unconnected to inference. No new arithmetic or production dispatch is enabled.

[Counter-access run 39139110081](https://github.com/mylordmonkeyman/Abyss/actions/runs/39139110081)
verified installed Nsight Compute **2025.1.1**, but its SM70 smoke returned
**ERR_NVGPUCTRPERM**. The profiler-gate report remains blocked. No capabilities,
seccomp policy or host driver settings were changed. Kernel activity timestamps
and static launch attributes do not replace HBM/L2/SM/HMMA/occupancy/spill counters.

Next: isolate CPU-lane/cache-dependent greedy variability, broaden controlled
teacher forcing, verify MXFP4 rows, and obtain permitted hardware-counter/QPI
measurements before choosing a production optimization. The current 2K timeline
makes MoE staging/actual IQ formats, CPU dependency and dense GEMV the first
**profiling candidates**; a long-context trace is needed before extrapolating that
ranking to 32K+. Extended 1,024-token and 64K/125K runs and reused-cache checks remain.
See [machine-readable baseline](baseline-2026-10-18.json) and
[timeline analysis](timeline-2026-10-18.json) for commands, hashes and full values.
