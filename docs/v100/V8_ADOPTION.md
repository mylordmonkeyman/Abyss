# v8 implementation adoption — 18 October 2026

The supplied consolidated v8 specification now replaces `IMPLEMENTATION_SPEC.md`.
Its source-audit statements and proposed flags remain design requirements, not
claims of completed implementation. Actual measurements remain in `STATUS.md`.
The build base stays `47001f31cbfba14bcbf4c083e3af07c50707ec0c`;
the preserved measured baseline is `c2bbbad9d3c3d3f777411c3a99b8610d9ab51240`.
No Strata or newer Abyss commits have been imported and no engine arithmetic,
cache policy, CPU lane or dispatch has changed in this adoption batch.

## P0.2c source and format reconciliation

Source inspection uses Strata `477ac13cbe8e582c20484871ce459d20d82df09d`,
ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d` and the separate newer
Abyss reference `7c893ac17e61ec45c934b3278356807fc4626c3c`.
`tools/v100/upstream_delta.py` exports the actual build SHA, source hashes,
feature map, toolchain, header census and candidate CSV into the GPU artifact.
Its feature classifications are scoped source observations, not runtime proof.

| Candidate | Finding on this implementation lineage | Action |
| --- | --- | --- |
| P3.12 scatter | Vendor already exports indexed and scatter Q8_1 APIs; engine only calls indexed | Isolated byte/layout test; keep production integration deferred |
| P3.2f CPU IQ2 | Actual gate/up: 42 IQ2_XS layers, one IQ3_XXS; down: 41 IQ3_XXS and two MXFP4 | No IQ2_XXS/Q2_K expert work in this pack. Upstream exact-one **also supports IQ2_XS (type 17)**, a relevant independently guarded follow-up |
| Existing CPU singleton | Exact IQ3_XXS/IQ3_S already implemented in Abyss | Preserve; compare actual-format IQ2_XS separately |
| P4i CPU/PCIe | Startup calibration and fixed plan already exist | Measure observed per-layer completion versus plan before adaptive policy |
| P3.13 direct MMQ | No pointer-table dispatch in current adapter | Conditional on gather critical-path cost and slot safety |
| RAM_THP delta | Existing buddyinfo compaction guard; newer partial/one-node control absent | Do not duplicate old guard or implicitly change base |
| STREAM_FETCH delta | Newer explicit control absent | Keep staged baseline; any import needs separate P0 |
| P2g masked union | Existing attention uses per-query raw0/n_raw | Xolotl workaround inapplicable to this route |

The specification's example IQ2_XXS/Q2_K expert distribution is superseded by
the measured inventory for experiment selection; quant formats are not interchangeable.
The original design document is retained as supplied. Model shard verification
remains sizes/structure, not SHA256; missing shard hashes are explicitly null.

## P3.12 preliminary measurement and isolated contract

Re-analysis of the existing bounded trace (run `39140801624`, artifact
`12141194059`) finds **1,416 indexed MMQ quantizer launches, 24.813115 ms**
inside a **3,889.8 ms** 2K prefill: **0.637902% of wall time**. These calls
include gate/up **and expert-specific down activations**. The latter cannot be
shared across experts. This is a source/API and CUPTI activity observation,
not hardware-counter evidence. Removing quantizer GPU work alone has very
limited headroom in this capture; saved host launch gaps need separate evidence.
No long-context or end-to-end gain follows from an isolated kernel speedup.

The current `glm_prefill.cu::run_set` quantizes `S->x` via each group's `row_tok`;
the same token activation can recur across routed experts. However, `M.Xq` is
allocated for **W window rows**, not the complete T×top-k route matrix. The vendor
scatter API requires an inverse token-slot→destination map over its destination
layout. Simply replacing the function name is incorrect. For illustration,
T=2,048, K=4,096 and top-k=6 would require 54 MiB of Q8_1 payload versus
9 MiB for a W=T window, excluding existing MMQ tails. An intermediate quantized
token buffer plus packed-row gather is another design, with additional storage
and launches to measure. Neither variant is integrated here.

`test_v100_mmq_scatter` compares the existing vendor implementations bit for bit:

- T=1/2/3/8/31/64/512/2048, top-k=6, randomized expert-sorted route maps;
- actual IQ2_XS/IQ3_XXS/MXFP4 activation layouts plus Q2_K/Q4_K layout controls;
- D4/DS4/D2S6, padded columns, nontrivial input stride, single-slot control;
- repeated source IDs, non-tile group counts, 43 synthetic routing seeds;
- zeros, signed zeros, scale extremes, subnormals and nonfinite probes;
- complete packed payload and padding, 128-block MMQ tail and outer sentinel guards.

Destination rows are unique. Repeated expert IDs are permitted as synthetic
map inputs, but duplicate destination writes would race and are excluded.
NaN/Inf byte parity is only a compatibility probe: it does not make such
activations valid model inputs. The test is not real captured 43-layer routing,
model-exact expert output parity, or an independent scalar quantizer oracle.
`--bench` saves five alternating-order paired CUDA-event samples per shape;
it excludes map construction, allocation and copying and is not a TTFT test.
The workflow retains V100 telemetry and runs memcheck/synccheck.

## Next admission decisions

P0 remains partial: counters denied, QPI unmeasured, greedy-repeat variation
unisolated, extended model and MXFP4 parity pending. Complete isolated validation
without promoting a production optimization. Defer scatter integration if its
small timing share persists. The actual-format IQ2_XS exact AVX2 comparison and
CPU/PCIe completion ledger are stronger follow-ups than implementing absent
IQ2_XXS or Q2_K expert cases just to match examples in the draft.
