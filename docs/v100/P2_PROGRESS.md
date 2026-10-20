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
