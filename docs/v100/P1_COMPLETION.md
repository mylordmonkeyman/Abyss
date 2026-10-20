# P1 evaluation disposition — keep F32

**Reopened 20 October 2026 at user direction:** this document records the
earlier fallback evaluation, not exhaustion of all optional MMA refinements.
The per-head query-scaling experiment is active in `P1_PROGRESS.md`; physical
component and any subsequent model qualification remain pending. The cuBLAS
quality failure is unchanged, and no MMA candidate is admitted for production.

20 October 2026. Revision 9 permits a P1/P2 evaluation to conclude “keep F32”
(critical-path introduction and P1.3.4/P1 exit). P1 experimentation concludes
with that disposition. **No new indexer arithmetic is admitted for production.**
This is an evaluation/fallback closure, not a claim that a candidate passed
quality or that the shipping engine gained the experimental measured speedup.

| Candidate / gate | Evidence and disposition |
| --- | --- |
| MMA884 primitive | Fragment contracts, emitted PTX/SASS, physical SM70 tests and sanitizers recorded in P0/P1 ledgers. Arithmetic instruction contract passes. |
| One-token MMA | Both measured layouts lose to F32; tiny-query relative error remains unqualified. No runtime integration; forced `volta` fails clearly. |
| Batched MMA, correctness-first and key reuse | 38 parity cases, graph replay, sanitizers and 30-shape/180-pair study pass within disclosed diagnostic bounds. Eight-query reuse wins some long shapes; tiny-query relative error 3.0464% is unqualified. No runtime admission. |
| FP32 cuBLAS | Shared component parity, graphs, sanitizers and 48-shape matrix pass. At actual T=2048, isolated indexer elapsed improves 9–12% in the eligible long regime. Experimental whole-model median prefill latency improves 1.53%, but full-128K quality fails. Reject production admission. |
| Same-pack quality / upstream baseline | CPU-lane-28 original-upstream repeat mean KL 0.01145396; working F32 repeat 0.00853500. This variability is not unique to the branch and is not an allowance to subtract from candidate error. |
| Deterministic quality control | With CPU lane 0, F32 and candidate each repeat bitwise. Matched F32/candidate mean KL 0.01040476 exceeds 0.0005; top-1 31/32. A separate reproducible candidate difference exists. |
| Same-input selection/attention | First eligible chunk passes 105 sampled rows. Later chunk p0=118784 changes top-512 membership in 6/43008 token/layer rows. One sampled near-tie swap has attention relative L2 0.00085353. Explains a local sensitivity mechanism, not the entire model divergence. |
| Boundaries / fallback | Actual host dispatch policy tests preserve legacy F32 override, C4/head/dimension/T/context restrictions and clear unsupported forced-mode errors. `auto` remains upstream; no experimental arithmetic selected automatically. Short-context fallback and component tails remain recorded. |
| Other platforms | SM80 compile/link and embedded image pass. No SM80 inference hardware available. HIP SDK absent on runner; no HIP compile/runtime claim. Host HIP-name compatibility passes. |

The late diagnostic is run
[39372924674](https://github.com/mylordmonkeyman/Abyss/actions/runs/39372924674),
artifact 12242394864, ZIP SHA256
`609c432afb9c3108f1ee10a69ea2705b8e371feaca8a69a1bc3a3946f74146bb`.
`p1-late-capture-2026-10-20.json` independently recomputes its raw paired
scores, IDs and attention via `summarize_p1_capture.py --diagnostic-failure`.
The diagnostic binary yields exactly the same F32/candidate distribution hashes
as the preceding deterministic quality control; capture did not change that result.
Its 54,694,479-byte uncompressed artifact satisfies the 64 MiB limit.

At layer 2 / absolute token 120831, F32's rank-512/513 margin is
7.1525574e-7, versus maximum measured score error 9.5367432e-7; the candidate
creates a tie and chooses a different row under the unchanged selector.
Sampled score relative L2 is at most 1.70018e-7. No meaningful-margin,
invalid/future/duplicate selection was found in this capture. Tiny score errors
therefore cannot by themselves establish model parity.

All earlier measured data and limitations remain in `P1_PROGRESS.md` and its
linked JSON ledgers. Experimental `ABYSS_V100_IDX=cublas` remains an explicit
research switch for reproduction, **unqualified for general 128K use**; use
the existing automatic upstream path or `ABYSS_V100_IDX=f32` /
`STRATA_DS4_IDX=1`. No threshold waiver, baseline subtraction or default change.
Reopening admission requires corrected arithmetic or a demonstrated decision
preservation scheme, then fresh matched quality and performance evidence.

The critical path now moves to P2 prefill attention feasibility. Existing P0
latent-math primitives and frozen activation captures are reused and revalidated
against the selected base before any fused/raw-only attention integration.
Immutable model/pack provenance remains unchanged; local manifest hashes are
not independent publisher authentication. Counters/QPI remain unavailable or
unmeasured, and overlapping memory accounting is never summed.
