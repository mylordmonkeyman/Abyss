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
