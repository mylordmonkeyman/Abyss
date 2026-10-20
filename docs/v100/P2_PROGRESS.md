# P2 — Volta DeepSeek prefill attention

20 October 2026. Active work continues on `feature/deepseek-v100-p1`, preserving
its selected v1.3.31 base and P0/P1 history. P1 concludes with the documented
F32 fallback in `P1_COMPLETION.md`; no candidate indexer is promoted.

The first P2 checkpoint revalidates the existing isolated QK/softmax/PV and
latent-gather/compressor/inverse-RoPE/grouped-projection diagnostics against this
selected base. `v100-p2-feasibility.yml` uses a dedicated build directory and
does not overwrite the working engine or immutable upstream binaries. It runs
synthetic arithmetic/latent parity, then verifies every cached P0 tensor hash
and replays all three ratio-0/4/128 captures with memcheck and synccheck.
No model load, pack conversion, new capture or GPU allocation for the model is
needed. A missing/changed immutable capture fails explicitly.

## Selected-base prerequisite result

[Run 39378667079](https://github.com/mylordmonkeyman/Abyss/actions/runs/39378667079)
passes 32 arithmetic cases (four graph replays each), 21 synthetic latent cases
and 33 real-stage comparisons on 3,127,475 preserved activation bytes, with zero
memcheck/synccheck errors. `p2-feasibility-2026-10-20.json` is reproduced by
`summarize_p2_feasibility.py` from artifact 12244905362, ZIP SHA256
`7ebcd1b44b579d917c945749a1c9c8a1c32af3aa2dfab4077facfd3e9f80879c`.
The synthetic suite still reports `projection_compressor_gate_complete=false`;
this result must not be relabeled as the full P2 integration/model gate.

## Isolated online-softmax candidate

`tests/v100/attention_tile.cuh` adds a correctness-first CTA for one query and
eight heads. It uses independent Q-hi/Q-lo MMA884 chains for 512-D QK, FP32
sink-aware online max/denominator, and independent P-hi/P-lo MMA884 chains for
PV against the identical FP16 K=V rows. Unnormalized probabilities are split
after fixed x1024 scaling, then PV is divided by 1024. FP32 running numerators
combine chains at each key-tile boundary and rescale when the maximum changes.
Only one warp does each tile's QK; PV dimension subtiles are spread over all
warps. This simple schedule may lose to F32; it is a feasibility comparator.

| Key tile | Query hi/lo | Key rows | Probability hi/lo | Scores | Online state | Total dynamic shared |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16384 B | 16384 B | 512 B | 512 B | 96 B | 33888 B |
| 32 | 16384 B | 32768 B | 1024 B | 1024 B | 96 B | 51296 B |

Both use four/eight warps, explicitly opt in to their checked dynamic shared
size, and record actual registers/local memory and resource occupancy. The
32-key layout alone limits V100 shared-memory occupancy to one CTA per SM;
the actual compiled occupancy must still be inspected. No full persistent
score/probability matrix or new inference allocation is introduced.

The new executable compares all output elements against current `attend_batch`
and a first-query/head FP64 oracle, checks guards and nondefault-stream graph
replay, then runs both sanitizers before timing six alternating pairs per
raw/CSA/HCA shape at T=256/2048. Synthetic timing uses a valid 64-D identity RoPE to isolate latent
attention; it does not test inverse RoPE/output projection or establish model
speed. The earlier stage gates remain separate and production dispatch is unchanged.

## First tile run: invalid comparator corrected

[Run 39384906696](https://github.com/mylordmonkeyman/Abyss/actions/runs/39384906696)
builds successfully, then fails the second numerical case before sanitizer or
timing steps. The first 16-key/four-warp layout reports 127 registers, zero
local bytes, 33888 shared bytes and two resource-limited CTAs/SM. This is not a
performance or complete parity pass. Artifact 12247255442 has ZIP SHA256
`759865770eb23c03a3f61df3ad6a3f4d525a1ba1ad6926b3781f6a2959081f30`.

Source review found the harness passed n_rot=0 to `attend_batch_kernel`, whose
inverse-RoPE epilogue divides/modulos by n_rot/2. That unsupported setting makes
the comparison invalid. The harness now uses n_rot=64 and freq_scale=0,
ext_factor=0, attn_factor=1, yielding a mathematically identity rotation through
the valid upstream path. Candidate arithmetic and numerical thresholds are
unchanged. Metrics are emitted before assertion, including the independent
FP64 oracle, so another failure retains its numerical evidence. This corrects
the test; it does not establish that the candidate will pass the corrected run.

This is a prerequisite check, not a new fused kernel or P2 performance result.
After it passes: establish bounded candidate schedules and byte/register/shared
memory ledgers, then compare isolated MMA QK/FP32 sink-softmax/MMA PV with the
current `attend_batch` at matched raw/CSA/HCA shapes. New integration requires
the mandatory physical latent-parity gate and subsequent same-pack model
quality, short/long context and whole-model performance evidence.

K=V is the existing 512-D latent cache; sink is a softmax logit with no value.
Preserve raw-window indexing, selected CSA IDs, chronological HCA rows and
inverse RoPE before output projection. No generic MLA reconstruction or Qwen
attention semantics are substituted. Production attention remains unchanged.
