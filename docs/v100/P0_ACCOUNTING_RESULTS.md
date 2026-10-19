# P0 accounting and actual dispatch results

Run [39194149121](https://github.com/mylordmonkeyman/Abyss/actions/runs/39194149121)
completed successfully. Memory used both checksum-verified immutable upstream
binaries at genuine 32K and 128K histories. Actual host dispatch used the working
fork, with opt-in profiling, 600/8192 prompts and 16 output tokens, two requests
per fixture. None of these diagnostic timings is performance acceptance data.

## Memory budget

`p0-memory-accounting-2026-10-19.json` contains a conserved host partition per
SHA, the source-derived attention suballocations, GPU capacity budget, CUDA calls,
mapped backing identities, node memory and remaining attribution limits. All
three model shard hashes match the host manifest. Source artifact ZIP hashes
were independently checked after download; older experiments retain their
original limited-integrity descriptions.

| Observation | Historical v1.3.28 | Current v1.3.31 |
| --- | ---: | ---: |
| Live CUDA host-pin API requested GiB | 71.84855 | 71.84883 |
| Unique shared-backing page footprint, bytes | 77,146,800,128 | 77,147,095,040 |
| Application PSS at selected checkpoint, GiB | 79.68891 | 79.69122 |
| Live device API requested GiB | 30.35677 | 30.35677 |
| MemAvailable at selected checkpoint, GiB | 45.57745 | 45.50484 |
| Successful/failed `cudaHostRegister` calls | 0 / 0 | 0 / 0 |
| Realized application `AnonHugePages` | 0 | 0 |

These are overlapping views. CUDA pin bytes are **inside** application shared
backing/PSS and host Shmem, not an extra 71.85 GiB to add to RSS or Cached. All
captured live host-pin ranges resolve to 41 shared backing objects. Dedupe uses
`(device,inode,file offset)`, covering every pinned page without unresolved private
ranges. CUDA device-pointer aliases add no host pages. This is shared-backing
identity plus the successful pin API contract, corroborated by PSS; it is not a
PFN/pagemap measurement. Both processes released every observed allocation, with
165 contiguous events, clean exit markers and no failed allocation calls.

The physical host budget partitions anonymous, shmem, file-cache excluding
shmem, kernel/accounting categories, a labeled residual and free pages. PSS
attributes the application's share *within* the first three categories. Each
partition sums to **135,068,450,816 bytes**. MemAvailable stays a separate
reclaim-aware headroom measure. Observed file-cache footprint is included; its
minimum sufficient working-set size and exact per-file resident pages remain
unmeasured. Thread stacks/runtime are included in application anonymous PSS,
without pretending they were separately attributed.

The GPU budget sums to the nominal 32 GiB using live application API bytes,
unattributed used memory, driver-reserved memory and free capacity, with an
explicit -1 MiB NVML rounding adjustment. At the historical checkpoint the
unattributed used residual is **709,541,292 bytes**; it combines context/modules,
vendor pools, alignment and any other client allocations. No fabricated split
assigns those bytes individually. NVIDIA reports host PIDs while `/proc` exposes
container PIDs here; per-process NVML attribution remains unverified. Whole-GPU
capacity accounting does not require inventing that PID mapping.

The attention allocation is **948,249,856 bytes**, identically observed in both
binaries and independently reconstructed from the source and model geometry:

| Attention arena component | Bytes |
| --- | ---: |
| Raw window rings | 5,812,224 |
| CSA compressed KV | 726,684,672 |
| CSA indexer keys | 181,671,168 |
| HCA compressed KV | 21,647,360 |
| Compressor recurrence rings | 12,206,080 |
| Attention activations/selection | 228,352 |

Read-only symbol run 39199082932 passed (artifact 12162959473; ZIP digest verified).
It corroborates dense/expert/attention allocations against their exact immutable
ELF callers, and identifies cached MMQ workspace and the attention snapshot.
Source line numbers are unavailable in the release binaries. The
prefill workspace/staging borrows the expert-pool tail; adding its rounded log
totals again would double-count GPU capacity. The fine layout is now reconstructed
in `p0-prefill-layout-2026-10-19.json` using `tools/profiling/p0_prefill_layout.py`.
At the observed 32,768-token maximum chunk, its disjoint outer carve is:

| Borrowed expert-pool region | Historical bytes | Current bytes |
| --- | ---: | ---: |
| Persistent activations plus reused union | 5,376,776,192 | 5,377,824,768 |
| F16 projection-weight scratch | 16,777,216 | 16,777,216 |
| F32 projection-weight scratch | 67,108,864 | 67,108,864 |
| cuBLAS workspace | 16,777,216 | 16,777,216 |
| Three staged groups of four experts | 130,667,776 | 130,667,776 |
| 160 prestaged experts | 1,742,236,160 | 1,742,236,160 |
| Total inside expert pool | 7,350,343,424 | 7,351,392,000 |

The persistent activations use 4,033,740,800 / 4,034,789,376 bytes. Their
1,048,576-byte difference is current-base `pre2` and `onehot` routing storage.
The shared union reserves **1,343,035,392** bytes: its MoE view occupies that
amount and its attention view 1,168,769,024 bytes at the 2,048-token mixer
sub-batch. These two views overlap; do not sum them. The JSON lists individual
buffer offsets, payload and alignment padding, including MMQ quantized inputs,
output windows, attention score/selection and projections. Short chunks borrow
less: their prestage carve is zero below 1,024 tokens, then capped by the
configured slots. The maximum reservation is not every request's actual loan.

The maximum expert stride is 10,888,976 bytes, corroborated by the successful
1,045,341,696-byte landing-ring allocation (96 slots, sequence 26) in each
immutable trace. Both rounded startup layouts agree with the reconstructed
bytes. Host compilation of the actual source sizing functions independently
matches the Python persistent/union/sub-batch values at six chunk sizes.
Relevant layout functions are identical between the two pinned sources except
for the two current-base persistent routing buffers. This is source-derived
suballocation accounting, not newly observed device addresses, peak residency
or physical-page measurement. No additional bytes enter the GPU budget.

| Application device component | Historical bytes | Current bytes |
| --- | ---: | ---: |
| Dense arena | 6,620,498,688 | 6,620,498,688 |
| Expert pool including borrowable prompt tail | 24,909,026,560 | 24,909,026,560 |
| Attention raw/CSA/HCA/indexer arena | 948,249,856 | 948,249,856 |
| Fast activations/routing/expert scratch | 88,061,548 | 88,065,644 |
| Cached MMQ workspace | 9,830,400 | 9,830,400 |
| Attention state snapshot | 18,018,304 | 18,018,304 |
| Model-load auxiliary buffers | 1,468,072 | 1,468,072 |
| Prefill auxiliary device buffers | 131,072 | 131,072 |
| Other/unresolved application requests | 45,120 | 45,120 |

These measured API component bytes sum exactly to each application's live
device total. The host expert RAM tier is 74,304,782,336 requested bytes in each
base. Remaining host pin groups cover embedding/routing/reader, prompt landing
and CPU exchange buffers; the complete disjoint requested-range partition is in
the JSON. They remain subsets of the host physical budget, not extra terms.

NUMA placement is materially asymmetric. The selected historical checkpoint
reports 23,586,148,352 mapped resident bytes on node0 and 61,985,579,008 on node1;
current reports 23,705,186,304 and 61,869,039,616. These process/node views overlap
the host budget. Node-local meminfo is retained separately; node1 has very little
free memory while node0 holds most free capacity. This supports investigating
placement, but **does not measure QPI bytes, remote latency or causal slowdown**.
No model pages were forced onto node0.

v1.3.31 source adds RAM_THP (default on) and STREAM_FETCH (default on). The
captured allocation path used `cudaHostAlloc`, not registered anonymous memory,
and realized zero anonymous huge pages in either base. A THP benefit is therefore
not demonstrated. Current's stream-fetch branch is present/default-on in source;
its exact kernel execution still needs current-base timeline corroboration. The
historical pin lacks these two named switches. Do not label the default base
comparison a separate 0/1 switch experiment.

## Actual host dispatch

`p0-dispatch-2026-10-19.json` preserves per-layer/type/token-group totals plus
compact global totals. The original bounded artifact retains the complete
23,285-key ledger. No calls were dropped and every decode service request joins
to its issued layer/position. The CPU is Broadwell with AVX2 and no AVX512 flags.
Read-only getter disassembly confirms that **both exact upstream binaries**
compile `ggml_cpu_has_avx2` to return 1 and `ggml_cpu_has_avx512` to return 0.
This is independent of the working fork's actual IQ256/no-IQ512 row counters.

| Path | Observed branch |
| --- | --- |
| MMQ built | yes |
| All 43 gate/up and down formats supported | yes |
| Ordinary SM70 MMQ products | 6,064 calls |
| Fused GLU selected / `run_glu` executed | 0 / 0 |
| Input Q8_1 gather quantization | 3,032 calls; 4,353,848 routed rows |
| Hidden/down Q8_1 quantization | 3,032 calls; 4,353,848 rows |
| IQ2_XS gate/up, T=1 | GGML `vec_dot` fallback |
| IQ2_XS / IQ3_XXS, T=2–8 and larger | IQ256 multi-row path |
| IQ3_XXS down, T=1 | IQ256 rows1 |
| MXFP4 down, T=1 and larger batches | GGML `vec_dot` fallback |
| IQ512 / direct IQ512 row calls | none observed |
| Named routed IQ2_XXS / Q2_K experiments | N/A: absent from this pack |

Repeated MMQ input quantization is proven: the measured 4,353,848 routed input
rows exceed **738,360** possible distinct layer/chunk input rows, allowing for
every repeated prompt invocation. At least **3,615,488** input-row quantizations
repeat an input within a layer/chunk invocation. Exact distinct counts per call
and per token were not captured; the report states a bound, not an invented
exact count. This proves eligibility for investigating quantize-once/scatter,
not priority: the earlier 2K timing measured only 0.638% in these quantizers.

T=2–8 observations here are actual expert prefill batches. Speculation was off;
they are not speculative-verifier throughput evidence. The host ledger does not
measure physical DRAM/PCIe traffic or per-format exposed CPU time. Aggregate
cache/fetch/wait outcomes remain in engine logs; detailed per-token route/cache
attribution and the long-context main-stream budget remain open P0 evidence.

## Interruption and continuing work

The original campaign's old host boot time was 2026-10-18 04:59:52Z. The subsequent
build's `/proc/stat` records **2026-10-19 03:02:59Z**, immediately after the last
surviving 03:01:24Z sample. A host reboot, rather than a diagnosed engine OOM,
therefore explains the runner's disappearance. What triggered the reboot is
unknown. No earlier data or partial arm was deleted.

Remaining matrix/timeline run 39194497860 is progressing with per-unit uploads.
Its full 32K three-pair comparison is preserved in `p0-32k-2026-10-19.json`.
Actual chronological order matches the planned A/B, B/A, A/B sequence. Median
paired prefill change is +0.00019%, decode time -5.8683%; all output trajectories
differ, and no causal speedup is claimed. Warm decode sample clocks are
1380/877 MHz in every arm, with no sampled thermal or power-cap reason. Prefill
clock/power behavior varies; ten-second samples cannot guarantee unsampled state.
Final base selection, critical-path top three, exact P3 ranking and unresolved
runtime/arena details still prevent declaring P0 complete. Production arithmetic
is unchanged. Validation: 40 local V100 tooling tests pass.
