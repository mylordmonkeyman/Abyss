# cuBLAS correction and common MMA comparison

User-authorized priority, 20 October 2026. Preserve the operational F32 reference,
production defaults, model/pack/fixtures and mean KL limit **0.0005**. No baseline
KL subtraction. Refined MMA's passing 128K control is recorded separately.

1. **Causality intervention (passed, run 39453135796).** Keep the
   original cuBLAS score calculation but substitute F32 selection at every
   eligible call. Compare F32 membership+order against F32 membership ordered
   by cuBLAS scores. Bracket with deterministic F32 repeats at the failed 128K
   geometry. Record membership/ordering counters and full-model KL. This is
   diagnostic work, excluded from speed claims. If reference-order substitution
   fails, inspect non-selection effects before claiming the cause is proved.
   Result: reference-order substitution restores bitwise F32 logits / zero KL;
   membership-only substitution fails mean KL 0.01026781025. Ordering is material,
   so flagged-only membership repair is diagnostic, not a sufficient admission.
2. **Adaptive full-token F32 fallback.** Keep SGEMM. On GPU, measure the 512/513
   margin against a calibrated conservative error envelope. Recompute flagged
   tokens using operational batch-F32 dimension/head accumulation and original
   selection. Record correction fraction and all costs; no CPU decision/sync.
   A measured empirical envelope must be labeled as such, not called a proof.
   Passed in isolated run 39471901764; flagged-only correction leaves internal
   ordering differences and therefore is not selected for model admission.
3. **Cutoff/top-k+delta rescore.** Reuse validated reference-matched arithmetic
   and GPU compaction/selection work from the MMA investigation where practical.
   Recompute candidates within the conservative cutoff envelope. Overflow must
   trigger full reference scoring, never truncate candidates. Validate exact
   selected scores, ordering/ties, ratio-4 visibility, graph capture, guards and
   both sanitizers before model tests. Count corrected tokens and scored rows.
   Implemented flagged-only and all-token policies with an empirical 32eps guard;
   all-token rescoring also repairs internal top-512 ordering. Counterfactual late
   capture calibration is not a global bound or actual-model correction result.
   The all-token variant passed all component selected-score/order checks and
   wins actual T=2048 by 45.584% /46.491% at96K/120K. Advance that exact entry
   point, with its empirical guard disclosed, to common model qualification.
4. **Reference-order epilogue.** Separately test sequential 64-head weighted
   F32 accumulation against the current warp reduction. Preserve per-head ReLU,
   F32 SGEMM outputs and original FP16 keys. Measure whether selection differences
   and correction rate decrease; include epilogue cost. Keep the original variant
   as a named comparator. Do not combine changes without isolated evidence.
   Sequential heads passed components but are slower than the warp-sum plus
   all-token cutoff winner; sequential alone still has diagnostic order errors.
5. **Common quality and performance.** Use deterministic same-pack 100K/128K
   fixtures and CPU-lane-0 repeats first. Compare F32, original cuBLAS, corrected
   cuBLAS and refined MMA in the same source/binary/configuration. Report mean,
   p99/worst KL, logit hashes, top-512 membership/order, correction fractions,
   stage times and end-to-end prefill/TTFT/decode with overhead included. Verify
   clean-build arithmetic before uninstrumented sustained interleaved timing.
   Use balanced process order and retain individual deltas, telemetry and unique
   memory/NUMA accounting. Different generated trajectories confound causal
   decode comparisons and must be disclosed. CPU-lane-28/default/task/platform
   gates remain separate; a bounded CPU-lane-0 pass is not general admission.
   Quality completed in39473440895 and all raw captures independently verified:
   corrected cuBLAS and refined MMA are bitwise F32 with zero KL at both fixtures.
   Original cuBLAS still fails128K atmeanKL0.01040475762. The clean campaign next
   requires eight matching clean logit hashes before four balanced, serialized
   end-to-end timing blocks at8K/100K/128K, including direct corrected/MMA deltas.

Do not initially change quantization, softmax, GEMM output precision, TF32 mode
or KL estimator, and do not use FP64 scoring everywhere. Every GPU campaign uses
`abyss-v100` serialization, the existing trusted runner and immutable upstream
binaries. No privilege/container/cache/GPU-reset/NUMA-page-placement changes.
Publisher authenticity, unavailable hardware counters and unmeasured QPI remain
limitations. Resume P3/P4 after these P1 decisions, not in parallel with GPU work.
