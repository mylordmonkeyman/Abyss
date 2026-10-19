# P0 remaining matrix and long-context timelines

Run [39194497860](https://github.com/mylordmonkeyman/Abyss/actions/runs/39194497860)
completed all 23 units successfully on 19 October 2026. The last eleven ZIP
artifacts were downloaded and their SHA256 digests checked. Their exact IDs,
digests, phase telemetry and timing summaries are in
`p0-remaining-2026-10-19.json`. Reproduce with:

```sh
python tools/profiling/summarize_p0_remaining.py EXTRACTED_ROOT \
  --artifacts EXTRACTED_ROOT/artifacts.json --output summary.json
```

The root contains `u12.zip` through `u22.zip` and their `uN/` extracted trees;
`artifacts.json` lists unit, artifact_id and zip_sha256 (also embedded in the
committed summary). The parser requires successful process exits, exact
prompt/output counts, both requests per arm, all three base pairs, agreement
between checkpoint/process records, and nonoverlapping request windows.

## Completed benchmark observations

All four groups ran in actual index order 0,1,2,3,4,5 (A/B, B/A, A/B).
The first request per fresh process is excluded. The genuine 128K case emits
1,024 tokens; the 8K chunk cases emit 128. All use the frozen pack/fixtures and
immutable binaries. These fresh-process strata remain separate from the
interrupted campaign's carry-over expert residency.

| Prompt / requested chunk | Historical prefill tok/s | Current prefill tok/s | Paired median prefill elapsed change | Paired median decode elapsed change |
| --- | ---: | ---: | ---: | ---: |
| 128K / auto | 356.96 | 357.01 | -0.01239% | -0.47698% |
| 8K / 512 | 253.25 | 255.71 | -0.96266% | +2.88384% |
| 8K / 1024 | 381.54 | 386.62 | -0.94174% | -1.37020% |
| 8K / 4096 | 629.81 | 636.47 | -0.58471% | +3.61755% |

Negative elapsed changes favor current. Per-base rates are medians, while
changes are medians of paired elapsed ratios; these are different statistics.
128K warm decode medians are 20.69 / 20.27 tok/s, historical/current. Generated
trajectories differ in every 128K pair and all chunk pairs except one 4096 pair.
Do not infer causal speedups or a base regression from these observations.
Chunk groups were collected sequentially, not interleaved across chunk sizes;
their large prefill differences are configuration observations with different
routing/cache histories, not isolated arithmetic improvements.

No active-window thermal reason was sampled. Minimum MemAvailable was 44.94 GiB
at 128K and 50.77 GiB or more in the chunk groups. Phase clock/power samples,
global fault/swap/I/O changes and per-process I/O deltas remain in the JSON.
Sampling every ten seconds does not establish unsampled thermal/clock state or
exclude CPU contention. NUMA summaries are mapped-page observations, not QPI
traffic. The original SHA-unverified experiments retain that limitation.

## Long-context wall-lane evidence

Units 21/22 use the immutable historical binary, one warmup followed by a
profiled genuine prompt and 16 generated tokens. This is diagnostic timing,
separate from the 1024-output sustained benchmark. Raw reports and SQLite stay
at their recorded runner paths; uploaded identities contain their SHA256 values.
Only summary/identity evidence is committed.

| Filled context / phase | First contributor | Second contributor | Third contributor |
| --- | --- | --- | --- |
| 32K prefill | MMQ 28.22% | Attention 26.17% | Indexer 22.49% |
| 128K prefill | Indexer 49.38% | Attention 21.14% | MMQ 16.24% |
| 32K decode | CPU-expert dependency wait 27.27% | Combined GEMV 21.25% | Attention 14.24% |
| 128K decode | CPU-expert dependency wait 22.81% | Combined GEMV 17.94% | Attention 17.46% |

These are disjoint kernel-residence shares on the main compute stream divided
by phase wall time, with argmax boundaries checked against native decode timing.
The unassigned main-stream gaps are 1.31% / 0.84% of 32K / 128K prefill and
2.54% / 2.18% of decode. They are not silently charged to CPU or disk. Copy
streams and CPU work overlap; a wait-kernel span is an exposed dependency,
not standalone CPU compute duration or proof that a different split wins.
Combined GEMV roles cannot be separated by a shared kernel name.
This wall-lane proxy is not a full causal dependency graph.

The captured `quantize_mmq_q8_1` layout-0 symbol totals 389.47 ms at 32K and
1,541.99 ms at 128K prefill (0.740% / 0.418% of phase wall time). These are
that named kernel's totals, not an assertion that every activation-quantization
variant is covered. Quantize-once is eligible by actual dispatch, but this cost
does not justify ranking it above the measured indexer/attention work.

## Decisions supported now; remaining closure

- Indexer hotspot: **yes** at filled context, especially 128K. P1 has a measured
  target; its arithmetic qualification remains a separate stage.
- Attention hotspot: **yes** at both contexts. P2 remains separate from P0.
- Ordinary SM70 MMQ: **yes**, fused GLU **no**. Repeated input quantization is
  proven; dispatch-only run `39221345537` passed and measured 3,615,700 exact
  repeated rows (83.04% of routed inputs). See `p0-exact-input-reuse-2026-10-19.json`.
  Its timings and perturbed planning populations do not enter this matrix.
- Named IQ2_XXS/Q2_K CPU experiment: **N/A** for this pack; the actual
  IQ2_XS/IQ3_XXS/MXFP4 paths and AVX2 getters are already recorded.
- Static CPU/PCIe imbalance: exposed CPU dependencies are **yes**; a beneficial
  alternative split is **not established** by wait spans alone. Final P3
  admission must distinguish these statements rather than infer a scheduling
  win or cross-socket bottleneck.

P0 is still open for detailed per-token route/cache evidence,
current-base THP/fetch dispositions, and the final base/P3
decision ledger. Hardware counters remain unavailable under the existing
permissions; QPI traffic remains unmeasured. No production arithmetic changed.
