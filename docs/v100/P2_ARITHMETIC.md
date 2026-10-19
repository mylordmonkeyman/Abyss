# P2a staged latent arithmetic diagnostic

This is an isolated mathematical check, not an inference dispatch or a complete
MLA parity gate. It uses the already V100-validated `volta_mma884.cuh` primitive
with all four independent quadpairs and separate FP32 hi/lo accumulation chains.

`tests/v100/dsv4_qkpv_parity.cu` covers 64 FP32 query heads × 512 dimensions,
contiguous shared FP16 K=V, QK, per-head sink-inclusive softmax and PV. The sink
contributes to the denominator and never to the value numerator. It compares
stages against host FP64 math and separate device FP32 scalar dot/product paths.
Key counts 0/1/7/8/15/16/31/32/33/127/128/129/512/640 cover sink-only, tile tails,
raw-window-sized and CSA-sized inputs. Additional cases cover dominant/negligible
sinks, zero queries and larger finite queries. All outputs have sentinel guards;
the staged graph is replayed four times per case.

Two PV encodings are measured independently: ordinary FP16 hi/lo probabilities
and probabilities multiplied by the exact power of two 1,024 before splitting,
then divided by 1,024 after accumulation. The latter can preserve small residuals
that otherwise underflow in FP16. Neither variant changes probability normalization;
both introduce rounding and must be evaluated against the reference. The staged
component screens are relative L2 1e-5 for QK/softmax and scaled PV, 1e-4 for
unscaled PV, with absolute-error reporting for sink-dominated tiny outputs.
These are diagnostic screens, not replacements for v9's model ΔKL/semantic gate.

QK uses a one-warp 8-head × 32-key tile. PV uses a one-warp 8-head × 32-output-
dimension tile. This is deliberately simple arithmetic geometry, not a tuned
attention CTA. Global score/probability scratch and a serial head-softmax kernel
make stage comparison straightforward. At 640 keys each score/probability array
is 160 KiB; production fusion must avoid these diagnostic buffers and prove its
own register/shared-memory budget and all-in benefit.

The test reports CUDA function registers/local bytes and theoretical CTA residency,
and the workflow retains ptxas logs and selected SM70 SASS. These do not replace
achieved occupancy, spill traffic or bandwidth counters. Memcheck/synccheck and
the physical V100 launch remain required before reporting a pass.

Still required for the full P2 latent gate: projected/normalized real query and
compressor inputs, raw/CSA/HCA addressing and masks, selected-key identity,
RoPE/inverse RoPE and grouped `wo_a/wo_b`, malformed-selection defenses, complete
model logits and a measured profitable production geometry. No new public switch
or automatic path is enabled by this diagnostic.

## Physical V100 result

Run [39172763525](https://github.com/mylordmonkeyman/Abyss/actions/runs/39172763525)
on `da9364a` passed all 32 cases, four graph replays each, memcheck and synccheck
with zero errors. Selected SM70 SASS contains HMMA. QK/PV report 40/39 registers,
zero local bytes and 32 theoretical one-warp CTAs/SM; achieved occupancy is unmeasured.
At 640 keys, unscaled PV relative L2 is 1.0601e-5 and scaled PV is 7.8376e-7.
The dominant-sink case reports relative L2 1.0 from FP16 underflow, with maximum
absolute error 5.84e-35; it passes only the explicitly stated absolute screen.
This is not proof that probability scaling is safe for arbitrary model inputs.
See [all case results](p2-qkpv-2026-10-18.json).

## Cached-row latent follow-up

`test_dsv4_mla_latent_parity` reuses the same staged arithmetic from
`tests/v100/qkpv_math.cuh`. Twenty-one synthetic cases exercise actual
`win_load`, `win_store`, `q_prep_batch` and FP32 `attend_batch`, with raw-window
lengths and chunk tails, wrap, ratios 0/4/128, CSA top-512 visibility around
2,048, and YaRN positions around 4K/8K. Exact host row bytes/absolute IDs are
checked independently of the device gather. Split chunks must reproduce the
same historical window and existing FP32 output exactly. Selected tokens include
chunk ends and compressor completion boundaries. The candidate compares QK,
probability, sink denominator, PV and inverse-RoPE stages against FP64 equations
and its final output against existing FP32 attention. CUDA graphs repeat the
staged path three times per checked token. Negative/future/duplicate CSA IDs are
rejected by the diagnostic before memory addressing; malformed IDs are never
passed into the upstream attention API.

Cached K=V rows here are synthetic, rotated at raw position or compressed block
start. Actual learned projections, KV normalization and compressor creation are
not covered. Queries use the existing preparation kernel, so this is not an
independent q-preparation oracle. Grouped output projection, real stage dumps,
full-model logits and fused-kernel resource/performance gates remain open. This
staged gather path is diagnostic only and adds no public control.

### Physical result

[Run 39173519298](https://github.com/mylordmonkeyman/Abyss/actions/runs/39173519298)
on `8c74a3f` passed all 21 cases, 54 checked token positions and 324 stage
comparisons; memcheck/synccheck each report zero errors. The maximum relative
L2 is 1.0189e-6 for QK, 4.9045e-7 for probabilities, 1.0313e-6 for the sink
denominator, 9.4576e-7 for PV, 1.4413e-6 after inverse RoPE versus FP64, and
1.0040e-6 versus existing FP32 attention. Exact row identity/order, split-window
state/output, future-row exclusion and malformed-selection rejection passed.
This is an addressing/arithmetic diagnostic, not the complete P2 latent gate.
Hosted Python tests and CPU build also pass on this commit.

The first compile exposed a mixed-type `auto` declaration, fixed in `ef55687`.
The next run passed the shared arithmetic checks but stopped on an accidental
extra cuobjdump filename, fixed in `8c74a3f` before the passing complete run.
See [case results and stage extrema](p2-latent-gathers-2026-10-18.json); full
per-token max/p99/relative stage observations remain in artifact `12153153780`
(117,185 bytes, SHA256
`569d02af98f02baf2b5f43c738220e84d4662552932e0956d061d88a4a61ca19`).

## Compressor construction and output projection follow-up

The cached-row test now builds ratio-4 overlapping CSA and ratio-128 HCA caches
with the actual `compress_batch` kernel. Independent FP64 softmax pooling,
per-channel biases, RMS normalization and block-start RoPE are compared before
and after FP16 storage rounding. Full versus split batches must produce identical
cache bytes and final rings; separate incomplete-block cases must leave the cache
untouched. The resulting device-created rows feed the existing staged attention
checks, instead of independently synthesized compressed rows. Inputs are synthetic
projection outputs; the learned input GEMMs and raw-KV prep are still outside scope.

Every checked latent output also passes through the prefill output sequence:
FP16 conversion, 8 groups of 4096-to-1024 `wo_a`, FP16 conversion, then
8192-to-4096 `wo_b`, using the same cuBLAS T/N operations and tensor-op FP32
accumulation API. Candidate and fallback form two batch columns to exercise actual
leading dimensions and group offsets. Synthetic FP16 matrices have eight nonzero
weights per row, making independent FP64 stage oracles affordable while cuBLAS
runs full dense shapes. This tests layout and conversion placement, not learned
quantized weight dequantization or full-model quality.

Compressor FP64-plus-F16-storage relative screen is 5e-4; rounded-FP64 comparison
is 2e-4, with 0.004 maximum absolute limits. Projection GEMM stage screens are
1e-5; the post-projection candidate/fallback screen is 1e-3 because two explicit
FP16 conversion boundaries are included. All measured errors are retained. These
component screens do not replace or widen the model delta-KL or semantic gates.

### Physical result

[Run 39178758945](https://github.com/mylordmonkeyman/Abyss/actions/runs/39178758945),
commit `a7a0727`, passes 10 compressor cases, the 21 attention cases and 54
paired output-projection checks. Memcheck and synccheck each report zero errors;
hosted Python tests and CPU build also pass. All full/split cache and ring bytes
match, and both incomplete-block cases leave the cache untouched.

Maximum compressor relative L2 versus FP64 before F16 storage rounding is
2.0848e-4 (max absolute 0.0019342); versus the rounded FP64 cache it is 2.0676e-5
(max absolute 0.001953125). The difference between the two comparisons exposes
the storage-rounding floor. Maximum grouped GEMM backend relative L2 is
7.6659e-9 for `wo_a` and zero for `wo_b` on these sparse matrices. After both
conversion boundaries, candidate versus fallback maximum relative L2 is
5.8131e-5, maximum absolute 3.0518e-5 and maximum per-check p99 absolute
4.2915e-6. These component observations do not qualify arbitrary learned weights
or model distributions.

See [all compressor, projection and latent cases](p2-components-2026-10-18.json).
Artifact `12155207022`: 123,710 bytes, SHA256
`d8ac23db76563fb0437b550c5cd13c0944c965a483a9d43150a5fc19ef3a3e8d`.
The earlier cached-row section's compressor/projection gaps are now covered
for synthetic inputs as described here. Remaining P2 work requires real learned
projection activations and quantized weights, raw-KV/query preparation oracles,
bounded real stage dumps and same-branch full-model logits, then a fused
profitable kernel with validated resources. No production path is enabled.

## Bounded real-model stage replay

`STRATA_V100_CAPTURE_STAGES` is an internal compile-time diagnostic, default OFF
and requiring CUDA plus V100 tests. Only `glm_prefill.cu` receives the definition.
It captures one token at absolute position 2,048 from the first raw, CSA and HCA
layers encountered, into a fresh per-run directory. Device tensor writes have
a combined hard 32-MiB cap; the workflow caps all uploaded evidence at 64 MiB.
There is no new public environment control or candidate inference dispatch.
Normal build configuration explicitly disables capture, and the capture workflow
restores/rebuilds the normal engine cache even after failure.

A deterministic 2,400-token repeated frozen fixture reaches position 2,048. This
is a diagnostic activation workload, not a semantic-quality or throughput corpus.
Captures retain actual pre/post-preparation queries, KV projection output and norm
weights, raw/constructed compressed cache rows, real CSA selections, sinks and
FP32 latent output. Existing `wo_a`/mixer outputs are retained as provenance, but
actual output-weight replay and same-branch full-model delta-KL remain open.

The latent executable accepts the capture root as its sole optional argument.
It independently checks query RMS/RoPE and raw-KV RMS/norm/RoPE/F16 storage from
real learned projection outputs, exact gather identities/causality, QK, sink
softmax, PV and inverse RoPE versus FP64 and captured FP32 attention. Three graph
replays run per captured layer. Every tensor has an exact size and a manifest
SHA256; the original model itself remains only size/structure checked.

### Physical real-model result and reduction investigation

Capture source `ce1091a` produced 3,127,475 bytes of tensors at position 2,048:
raw layer 0 (128 keys), CSA layer 2 (128 raw + 512 selected compressed), and
HCA layer 3 (128 raw + 16 compressed). The diagnostic explicitly sets existing
`STRATA_GLM_PREFILL_SUB=256`; the engine otherwise may enlarge mixer sub-batches.
The 2,400-token repeated fixture is not a semantic-quality corpus. Capture bytes
are frozen under `/actions-runner/abyss-v100-stage-captures/frozen-2048-v1`, with
source commit, binary, fixture, pack and every tensor hash. Replay validates all
hashes, reuses the exact original bytes and does not reload the model.

The first real CSA check exposed relative L2 1.1740e-5 against captured FP32
attention, above the 1e-5 component target. Captured FP32 itself agrees closely
with the FP64 oracle. Shortening QK chains improved scores but left the output
error at 1.1652e-5. A separate scalar PV using identical device probabilities
isolated the dominant error to the long PV accumulation (1.1679e-5).

New isolated `qk_mma_chunked` / `pv_mma_chunked` retain independent FP16 hi/lo
chains within 64-dimension / 64-key groups, then use compensated FP32 combination.
The original variants remain available. On the same captures, the combined
variant reduces CSA output error to **9.9051e-7**; same-probability PV error is
**9.4330e-7**. No screen was widened. This is numerical error reduction, not a
runtime speedup or model quality claim. QK/PV report **64/63 registers and zero
local bytes**; achieved occupancy and profitable fused geometry are unmeasured.

[Real-stage run 39180592084](https://github.com/mylordmonkeyman/Abyss/actions/runs/39180592084)
on `d155daf` passes all three ratios / 33 stage checks, graph replays, memcheck
and synccheck (zero errors). Relative L2 against captured FP32 attention is
5.6936e-7 raw, 9.9051e-7 CSA and 5.3347e-7 HCA. Independent query-prep error is
at most 1.4644e-7; all three raw-KV rows match the rounded FP64 preparation
reference exactly in this capture. The normal-cache restoration step succeeded.
[Component run 39180592052](https://github.com/mylordmonkeyman/Abyss/actions/runs/39180592052)
also passes the original/new contiguous arithmetic, cached-row/compressor/output
checks and sanitizers; hosted Python and CPU-build CI pass.

See [complete real-input evidence](p2-real-stages-2026-10-18.json). Real-stage
artifact `12156555253`: 2,646,327 bytes, SHA256
`c0dc2efc09b8206f2de86d88917ad172764eddb07839fd951707cdc1ea0c7e02`.
The actual learned input projections, prepared raw/compressed caches and CSA
selection are now represented by captured production stages. This single
position in three layers does not complete the corpus/long-context gate. Actual
quantized output-weight replay, candidate downstream/model logits and quality
gates, broader layer/position coverage, and fused resource/performance validation
remain open. There is still no optimized inference dispatch or new public flag.
