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
