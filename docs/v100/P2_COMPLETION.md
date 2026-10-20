# P2 evaluation disposition — keep F32

20 October 2026. Revision 9 P2.2.4 explicitly permits retaining the correct
component/microbenchmark work and shipping FP32 when a candidate cannot be made
correct and profitable. P2 evaluation concludes with that fallback disposition.
No new production attention, arithmetic selector, cache format or default is
admitted. Full model quality/projection/release qualification is not claimed.

The selected-base prerequisites pass 32 staged arithmetic cases, 21 synthetic
latent/addressing cases, and 33 frozen real-stage comparisons with memcheck and
synccheck. Exact metadata and the historical activation provenance remain in
`p2-feasibility-2026-10-20.json`. Its incomplete projection/compressor integration
flag is preserved; it is not promoted to a full engine acceptance claim.

Two online-softmax schedules were physically evaluated, each at 16/32 keys and
four/eight warps. Both preserve shared 512-D K=V, denominator-only sink, causal
raw/CSA/HCA addressing, and independent hi/lo QK/PV chains. Both pass 32 parity/
graph cases, 24 timed numerical cases and both sanitizers, with zero local bytes.
These feasibility tests use a valid 64-D identity RoPE; they do not integrate
model RoPE/output projection into a new inference kernel.

| Schedule | Median paired elapsed changes across 24 shapes | Faster pairs |
| --- | ---: | ---: |
| Serial QK warp | +234.72% to +749.16% | 0/144 |
| QK dimension partitions across all warps | +257.81% to +609.44% | 0/144 |

Each matrix includes raw/CSA/HCA at T=256/2048 and six alternating pairs per
shape against actual F32 `attend_batch` on identical inputs. Separate campaigns
are not a direct paired serial-versus-partitioned comparison. Maximum full
output relative L2 versus F32 is 4.02383e-6 / 1.32341e-6 respectively. Correctness
alone does not justify replacing a substantially faster baseline.

The partitioned result is
[run 39390712443](https://github.com/mylordmonkeyman/Abyss/actions/runs/39390712443),
artifact 12249168224, ZIP SHA256
`abe706e0147742b76e7c5ca17aac3f8b26b0bf87c9762d90ebd15f22216e139f`.
`p2-tile-parallel-2026-10-20.json` is reproduced by `summarize_p2_tile.py`.
Four/eight-warp register counts are 127/96 at 16 keys and 128/125 at 32 keys;
resource-limited CTAs/SM are two and one respectively. Six telemetry samples
show no thermal reason, without establishing sustained model behavior. Resource
occupancy is not a hardware-counter measurement. No inference speedup is claimed.

The first run's unsupported n_rot=0 comparator and its failed observation are
preserved in `P2_PROGRESS.md`; the corrected identity-RoPE matrices supersede it.
There is no reason to run costly model qualification for these losing schedules.
Reopening P2 requires a materially different, measured profitable design, then
the complete latent/model/quality/platform gates before dispatch integration.

The critical path moves to the two P0-admitted P3 investigations: per-layer
completed CPU/fetch/wait costs first, then bounded quantize-once/scatter if
cost remains material. The calibrated startup allocation and all experts remain
unchanged. Model authenticity and unavailable counters/QPI limitations remain.
