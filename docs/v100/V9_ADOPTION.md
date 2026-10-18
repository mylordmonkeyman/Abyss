# v9 adoption and P0a base comparison — 18 October 2026

`IMPLEMENTATION_SPEC.md` now contains the supplied v9 document. It is the sole
forward plan. The v8 design remains available in git history at `f7138de`, with
its measured results retained as evidence rather than a competing task list.

The document's introductory "never run" / unproven-MMQ statements describe the
upstream source audit. This fork has already run complete V100 inference and
captured actual ordinary MMQ kernels. Preserve those results in `STATUS.md`;
do not reset them to untested merely because the new design was prepared separately.
Fused GLU's Turing gate does not disable the ordinary V100 MMQ path.

## Revised order

1. Compare **unmodified** upstream v1.3.28 `47001f31` and v1.3.31 `7c893ac`.
   Separate persistent source/build/output directories preserve the historical
   implementation build and canonical model pack. No rebase or automatic new-base
   selection occurs in the comparison harness.
2. Reconcile dispatch, memory, thermal and quality evidence. The actual expert
   types are IQ2_XS/IQ3_XXS plus two MXFP4 down layers, rather than the draft's
   example IQ2_XXS/Q2_K combinations. Named absent-format experiments close N/A;
   any IQ2_XS work requires its own applicable-source and measured-cost case.
3. Continue P1/P2 model-specific arithmetic and measured shape decisions, then
   admit at most three worthwhile P3 experiments. Missing privileged counters
   remain disclosed; v9 permits CUDA events/Nsight Systems without weakening
   runner permissions.

## Initial paired campaign

`v100-bases.yml` builds the exact two upstream SHAs with CUDA 12.8, SM70 and the
same portable ggml AVX2/FMA backend. The same original pack/tokenizer and exact
fixture IDs are supplied to both executables. No pack conversion is run.
`tools/bench/v100_base_compare.py` performs A/B, B/A, A/B process ordering,
600-token prose, 4K code and 8K mixed-language/numeric prompts, 128 generated
tokens, and two requests per fixture per process (one excluded warmup and one
warm measurement). Prefix reuse, speculation, persistent usage and profiling
are disabled; legacy FP32 scorer/attention overrides are explicit.

Only the first request in a newly loaded process has cold expert residency;
later fixtures inherit earlier residency. OS page-cache state is uncontrolled,
and no host-wide cache drop is authorized or performed. Thus this is an initial
process-start/warm comparison, not a claim of storage-cold performance.
Warm results are paired by process round and fixture; preserve all first-request
results separately rather than pooling cache states.

One matched 32-position, full-vocabulary teacher-forced comparison spans the
2,048 selection boundary, relative to historical Abyss FP32, with mean/p99/max KL,
top-1 and dump hashes. Dumps are temporary, with no raw tensors uploaded.
This uses v9's 0.0005 exact-math review screen; the earlier external llama.cpp
KL is not the allowed incremental error. One fixture does not meet the entire
task/reuse/rollback gate or measure each base's internal repeat variability.

Telemetry records RSS/PSS, THP, host/cgroup pressure, I/O, NUMA mappings, thread
affinity, clocks/power/throttle reasons and actual GPU memory/link XML. These
overlapping memory views are not added together. Full component pin accounting,
QPI, sustained matched conditions, 512/1024 outputs and filled 32K+ remain open.
Both existing public-runner guards and a 128-MiB uncompressed evidence cap apply.

The base remains v1.3.28 until the evidence supports selecting v1.3.31 under v9.
No new production switch, arithmetic, cache policy or runner privilege is added.
