# Abyss on Tesla V100 (sm_70): DeepSeek V4 Flash Performance Port

**Implementation specification — revision 9.0 (critical-path and acceptance-gate consolidation)**  
**Date:** 18 October 2026 · **Replaces:** v8 (18 October 2026) · **Status:** implementation plan; no V100 DeepSeek-V4-Flash run proven yet

**Target:** one Tesla V100-PCIe-32GB (sm_70) on Ubuntu 24.04, CUDA 12.8, two Xeon E5-2697A v4 CPUs (32 physical cores/64 threads, AVX2/FMA, **no AVX-512**) and 128 GB dual-socket DDR4. GPU attaches to NUMA node 0; remote DRAM uses **QPI**, not UPI.  
**Model:** Unsloth DeepSeek-V4-Flash-0731 `UD-Q2_K_XL` (approximately 96.8 GB GGUF; do not infer expert or resident bytes from nominal file size).  
**Repository:** [zephel01/Abyss](https://github.com/zephel01/Abyss). Two controlled P0 baselines: **v1.3.28 `47001f31cbfba14bcbf4c083e3af07c50707ec0c`** (historical implementation pin) and **v1.3.31 `7c893ac17e61ec45c934b3278356807fc4626c3c`** (current comparison head as inspected 18 October). Upstream source changes must be checked against the selected working branch and its vendored dependencies. Do not rebase blindly.

## 1. Critical path — read this first

**The committed work is a sequence, not a queue of 25 parallel projects.** P0 establishes which expensive operations *actually execute on the V100*, P1/P2 implement/evaluate the Volta math, **at most two or three** P3 optimizations are admitted based on the measurements, and P4 hardens and releases the result. Everything beyond this sequence is in **Appendix A: Candidates, not committed**.

| Stage | Next action | Advance only when | Output |
|---|---|---|---|
| **P0a — bring-up and base selection** | Build and **run** the complete model at both `47001f31` and `7c893ac`; compare resource use, logits, prefill and decode | A full V100 response, model parity and no unexplained regression at selected base | Reproducible selected base SHA, complete source/build/model manifest |
| **P0b — bottleneck and dispatch audit** | Profile filled contexts, record real CPU/MMQ/GLU/MXFP4 dispatch, PCIe/NUMA/thermal state, and memory worksheet | Three leading critical-path contributors and path-level yes/no eligibility | Machine-readable baseline and decision ledger |
| **P1 — Volta lightning indexer** | Prove `mma.sync.m8n8k4` fragment contract, benchmark/indexer-integrate only where needed | Correct scores, top-512 and model behavior; measurable winning shapes or documented F32 fallback | Indexer PR(s) |
| **P2 — Volta DeepSeek attention** | Prove model-specific K=V latent math; implement raw→CSA/HCA batched attention on SM70 | Correct raw/CSA/HCA/sink/inverse-RoPE; winning shapes or documented F32 fallback | Attention PR(s) |
| **P3 — choose ≤3 experiments** | Prefer the largest measured P0 bottlenecks. Candidate shortlist: activation quantize-once/scatter **if SM70 MMQ really runs**; exact Broadwell IQ2_XXS **if it beats existing path**; online CPU/PCIe balance **if static planning leaves stalls** | Each has independent stop rule and end-to-end V100 A/B | Only admitted P3 PRs |
| **P4 — release** | Restore-safe CI, sustained performance, documentation and reversibility | All §6 gates and real V100 runtime CI pass | Release candidate and benchmark ledger |

**Do not invert the sequence because upstream reports a large speedup on another card.** A P1/P2 PR can conclude "keep F32" with evidence. P3 work can close N/A. Unsupported formats, absent runtime paths, insignificant measured costs, and worse whole-token latency are valid outcomes. No default lossy router pruning, skipped experts, or cache quantization in the first release.

### 1.1 P0 decisions that change the first batch

1. **P0/MMQ yes/no:** `glu_supported()` in [`src/prefill/moe_mmq.cu`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/prefill/moe_mmq.cu) uses `turing_mma_available()` but guards the **fused GLU**, not all nonfused MMQ. The same source has an ordinary `mmq::supported()` switch including IQ2_XXS, Q2_K and MXFP4. Record the **actual DeepSeek prefill calls** to `mmq::quantize`, `Context::run`, `glu_supported`, `run_glu` and the dequant/F16 fallback, with format, SM70 device, token count, and launch counters. In particular, whether an MMQ kernel is *built* is not proof that it is *selected*. Mark **P3.1 N/A** if the duplicated q8_1 activation MMQ path never runs or is below the profiling noise floor. Do **not** make `glu_supported` alone the eligibility test.
2. **P0/Broadwell dispatch:** Record the exact call site, `ggml` CPU instruction-set selection and type-specific implementation for **IQ2_XXS gate/up, Q2_K down and MXFP4** in one-token decode and 2–8-token verify/prefill. [`docs/deepseek4/PORT.md`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/docs/deepseek4/PORT.md) describes AVX-512 CPU-row optimizations on some hosts; those do **not** run on E5-2697A v4. But Abyss **does** have AVX2 multi-token and one-token IQ3-specialized paths. Compare upstream's newer exact AVX2 IQ2_XXS path against the **actual** Abyss fallback, not an assumed scalar implementation. Close **P3.2 N/A** if the incumbent path is already equal/faster or rarely used.
3. **P0/base:** Default future development to **v1.3.31 after both baselines pass**, because its `STRATA_GLM_RAM_THP` and `STRATA_GLM_STREAM_FETCH` changes can matter on this 96.8-GB/128-GB host. If it regresses performance, accuracy, stability, memory pressure or V100 build, retain v1.3.28 for isolation and open a narrowly scoped fix/port investigation. A development rebase must repeat P0b and preserve pinned reference results. Any new Abyss release after these dates requires a fresh SHA and audit rather than automatic adoption.

### 1.2 Release defaults and controls

The initial user-facing control surface comprises **only three new `ABYSS_V100_*` variables**, introduced by the PR that implements each:

| Control | Owner PR | Meaning |
|---|---|---|
| `ABYSS_V100_PROFILE` (`0`, `1`) | P0b | Enable the baseline profiling/diagnostic ledger; off by default |
| `ABYSS_V100_IDX` (`auto`, `volta`, `cublas`, `f32`) | P1b | Tested indexer dispatch; `auto` may choose F32 on unsuitable shapes |
| `ABYSS_V100_ATTN` (`auto`, `volta`, `f32`) | P2c | Tested attention dispatch; F32 always available |

No such variable is to be documented as functional until its implementing PR is merged. Keep existing Abyss `STRATA_DS4_IDX`, `STRATA_DS4_ATTN`, `STRATA_DS4_IDX_ALL`, `STRATA_DS4_SPEC`, `STRATA_GLM_NUMA` etc., with documented precedence in the actual code. Each later optional experiment adds **at most one justified temporary switch in its own PR**, deletes it if the experiment closes N/A, and does not publish the approximately 30 earlier *proposed* controls as an interface.

## 2. P0 — actual V100 bring-up and platform evidence

**Starting point: “builds, never run.”** Abyss [`docs/deepseek4/PORT.md`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/docs/deepseek4/PORT.md) expressly lists sm_70 build support from v1.3.7–1.3.11 (CUDA 12, BF16 projections using FP32 GEMMs) and says a V100 **“has not been run.”** Neither successful `nvcc` nor a passing CUDA unit test constitutes full-model inference. P0a is specifically tasked to close that gap on **one V100-PCIe-32GB** and the actual DeepSeek GGUF.

### 2.1 P0a: two SHA baselines and executable proof

Use isolated checkout/build/output locations for the two SHAs; never overwrite the model pack, benchmark fixtures or historical binary. Record actual dependencies and compiler flags. Baseline build pattern (adapt paths to the pinned repo's CMake; determine target at configure time):

```bash
# Run once per checkout: historical pin and v1.3.31
cmake -S . -B build-v100 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
cmake --build build-v100 --target strata -j 32
```

First classify build-only, startup, model weights, single-token forward, prefill, full generated response, long-context and server health as **separate** statuses. Report first failing stage, exit code, shape/type, GPU trace and memory at failure. For both SHAs use the identical GGUF shards/model hash, prompt IDs, maximum context, sampling/seed, CPU pinning and power state. Verify the CUDA 12.8 target image is actually sm_70 (do not infer from CMake alone); never build sm_70 with CUDA 13. No GPU/runner churn merely to reproduce an unchanged failure.

**Paired base-selection requirement:** publish prompt 600/4K/8K (and filled 32K only after stable) with fixed decode length, clean cold and warmed runs, teacher-forced logits/KL, peak VRAM, RSS, pinned pages, major faults, NVMe read bytes, TTFT, prefill tokens/s, decode tokens/s, correct responses, crash/timeout count. Do not compare unreconciled binaries, branch-specific tokenizer/packer differences or dissimilar clocks as a speed claim.

### 2.2 P0a: mandatory 128-GB memory-budget worksheet

Store a CSV or JSON ledger **per SHA**, distinguishing decimal GB from binary GiB and total virtual mappings from committed/pinned physical memory. Complete **each row**, not a single model-size estimate:

| Entry | Required measurement and units | Placement / accounting |
|---|---|---|
| GGUF shards/model | On-disk byte sizes, unique tensor/quant roles, packed routed experts and dense weights | Read-only mappings; not automatically 96.8 GB resident |
| GPU dense and context | Weights, CUDA context/modules, per-layer staging, KV raw/CSA/HCA/indexer, workspace, expert cache, reserve | Exact GiB out of V100 32 GiB; memory peak during prefill and speculation |
| Host pinned / registered | Bytes `cudaHostAlloc`, `cudaHostRegister`, mapped aliases, expert arena, staging, exchange buffers | Distinguish allocated/pinned, avoid double counting mapped same physical pages |
| Host unpinned | Thread stacks, tokenizer/runtime, buffers, heap, snapshots, unrepacked host source | RSS/PSS; startup and steady state |
| Page cache and file mapping | File cache used by model shards and disk-landed experts, page faults, cache churn and direct-I/O bytes | Do not count file-backed virtual size as physical RAM automatically |
| OS / Docker / safety floor | Kernel/userspace, container/runner, file-backed overhead, other resident services, MemAvailable and swap/oom | Preserve measured safety margin, no forced >RAM pin |
| Socket 0 / socket 1 | `numastat -p`, `/proc/<pid>/numa_maps`, per-node `free`/page placement, local vs QPI expert reads and H2D | GPU node0; **interleave** may favor aggregate CPU bandwidth, but hurt node1→GPU DMA |
| THP and SSD fetch | THP actual `AnonHugePages`, compaction stalls, `STRATA_GLM_RAM_THP=0/1`; `STRATA_GLM_STREAM_FETCH=0/1` | Distinguish v1.3.31 additions from absent historical-pin flags |

**Compute** `(OS+other workloads) + committed model RAM + pinned buffers + app heap + required page-cache working set + free safety floor` and show node-local breakdown. Require no host OOM, sustained reclaim storms, severe remote-DRAM stalls or pinning failures. Do not force all ~100 GB onto NUMA node0: its available memory is approximately half of the total. QPI rather than UPI is the intersocket link on Broadwell-EP.

### 2.3 P0b: runtime dispatch, critical-path profiler and telemetry

Produce a per-layer, per-token/full-prompt ledger of **GPU compute, expert-cache hit/miss, CPU expert rows, PCIe DMA vs mapped-read, nonresident NVMe reads, host waits, CUDA graph/workspace and synchronizations**. Separate cold/warm, decoder 1 token vs verifier 2–8 tokens, and prompt chunks. Trace exact kernels, group counts, fallbacks and bytes for (a) lightning indexer, (b) raw/CSA/HCA attention, (c) expert MMQ quantize/gather/GU/activation/down, (d) CPU row implementation and ISA, (e) cached/staged/repacked expert memory.

**MMQ explicit yes/no ledger (model-specific):** for each normal layer format combination IQ2_XXS/Q2_K and anomalous MXFP4 layers, record `mmq::built`, `mmq::supported`, fused `glu_supported`, SM70 `Context::run` invocations, F16/cuBLAS fallback invocations, quantizer calls and how many distinct activations each call repeats. The fact that `glu_supported` refuses Volta is **not** itself an MMQ=no result; capture actual branch decisions in `src/prefill/prefill.cpp`.

**CPU explicit yes/no ledger:** record `/proc/cpuinfo`/`lscpu` flags, `ggml_cpu_has_avx2/avx512` and compiled ggml backend, actual `native_gu_rows`/`native_down_rows`, `exact_rows_nt` branch chosen at T=1 and T>1, parallel thread count, socket placement, RAM throughput and share of decode critical path. Record format-specific fallback for IQ2_XXS/Q2_K/MXFP4. Do not assume upstream `PORT.md` AVX-512 optimizations can execute on Broadwell.

**Sustained hardware telemetry:** one warm-up and representative sustained decode interval per arm; sampled PCIe link state/throughput, NVML/DCGM clocks, SM/memory clock, power cap/usage, temperature, utilization and **power/thermal throttle reasons** where available. Separate thermal throttling, power limit and CPU/DRAM/PCIe limitation. If `ERR_NVGPUCTRPERM` denies GPU counters, retain Nsight Systems/CUDA-events timing and mark unavailable counters; do not blindly run a privileged container. Pin a locally available Volta-capable Nsight Compute version and digest **without publishing proprietary binaries**. Reject comparisons with materially different clocks, power states, CPU contention or memory pressure; preserve rather than erase flagged runs.

**Architecture CI:** examine the correct SM70 PTX/SASS/fatbin image (not regexes over all architectures), capture `ptxas -v` registers/spills and resource limits, and use an actual V100 kernel execution test. Compile-only and runtime gates are independent. The first PR must prove that Abyss **runs**, not simply that `sm_70` code compiles.

### 2.4 P0b: benchmark matrix and decisions

- Short prompts: 600, 4K, 8K; prefill chunks 128/512/1K/4K if memory permits; steady decoded outputs 128/512/1024 with fixed settings.
- Filled-context: 32K, 64K, 128K (larger only if memory-safe), with genuine history. Synthetic position offsets may isolate **attention-only** kernel cost but never substitute for genuine routing/cache hit rates.
- Cases: prose, code, multilingual and numeric, fixed exact prompt IDs; at least three interleaved A/B repeats where feasible. Keep raw timing, step counters, provenance and thermal samples.
- Ground model logits to a fixed same-pack Abyss F32 baseline, and retain llama.cpp comparison as *external reference*, not as the implementation's ΔKL budget. Verify head, sinks, RoPE, top-512 ties and boundaries.
- Decide and record: base selected? indexer hotspot? attention hotspot? real MMQ repeated q8_1 quantization path? CPU IQ2_XXS hotspot? static CPU/PCIe imbalance? exact P3 admission rank? Each item **yes/no/N/A with evidence**.


## 3. P1 — Volta Tensor Core lightning-indexer scoring (detailed)
### P1.0 Scope and exact semantics

P1 replaces only the **indexer score calculation** for DeepSeek's CSA (`ratio=4`) layers. Existing `q_prep`, selection, compressor and top-k behavior stay unchanged.

For token `t`, compressed row `j`, head `h`:

```text
dot[t,j,h]   = Σ(k=0..127) iq[t,h,k] * idx_K[j,k]
score[t,j]   = Σ(h=0..63) max(dot[t,j,h], 0) * idx_weight[t,h]
n_visible(t) = floor((absolute_position(t)+1)/4)  [existing integer convention]
top512(t)    = existing scorer selection applied only to j < n_visible(t)
```

The **ReLU occurs independently before each head's weighted contribution**. The query `iq` and weights are FP32; the indexed keys are FP16. Preserve `q_prep`'s normalization, Hadamard and compressed RoPE. Visibility and tie behavior must agree with the existing kernel/selection; never read stale scores beyond `n_visible`.

**Do not optimize away** Abyss's existing skip of score/selection where the top-512 already covers all visible rows. Below ~2K source tokens, P1 often has no useful decode work.

### P1.1 Primitive and reference implementation

Propose an included CUDA implementation `src/kernels/cuda/dsv4_volta_idx.cuh` (or a `.cu` added to CMake) with:

```text
device-code guard:  defined(__CUDA_ARCH__) && 700 <= __CUDA_ARCH__ < 750
runtime guard:      cuda device major == 7 && minor == 0
instruction:        mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32
source fragment map: Strata-V100 src/kernels/cuda/qsa_prompt_attn.cu
```

Strata-V100’s source describes **four independent 8×8×4 MMA computations per warp**, indexed by its quad-pair layout. **Cross-check the entire map** against Superfast’s [`ggml/src/ggml-cuda/mma.cuh`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mma.cuh), which documents a mirrored Volta input layout; register order and head/row assignment must be independently tested rather than assumed identical:

```text
QP q: lanes [4q..4q+3] and [4q+16..4q+19]
C register i:
  row = (lane & 1) + 4*(lane >> 4) + 2*((i >> 1) & 1)
  col = (lane & 2) + 4*(i >> 2) + (i & 1)
```

Re-use this layout **only after a standalone synthetic test** validates every accumulator coordinate, row/column ordering, and masked edge on an actual V100. The four quad pairs are **independent**; do not mistake one warp for a single 8×8 result tile. Validate compiler output and every warp lane's register packing.

**P1.1.1 — Inline PTX register/operand contract (hard arithmetic gate).** For `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, the packed pairs of F16 A/B operands are 32-bit bit-pattern carriers and should use an inline-assembly **`"r"` 32-bit register constraint** (two carriers per operand fragment), while each of the eight FP32 accumulator/destination registers uses **`"+f"`**. PTX `.f16x2` fragment storage may be represented in appropriately sized 32-bit containers; never substitute a float value for the packed half-bit-pattern or reinterpret a PTX `.f32` accumulator as `uint32_t` without a defined bitcast. Follow the [PTX `mma` fragment specification](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html) and [inline assembly constraint documentation](https://docs.nvidia.com/cuda/archive/12.2.2/inline-ptx-assembly/index.html), not an assumed register map from a newer MMA shape.

- Compile with `-arch=sm_70` and warnings/error checks. Inspect generated PTX **and** SM70 SASS, ensuring the instruction is the expected Volta FP16 tensor operation and neither compiler nor source silently widened/narrowed differently. Verify register operand count/order, `+` read/write constraints, and that *all 32 active lanes execute an identical aligned MMA instruction*; divergent participation violates `.sync.aligned` semantics.
- A lane-map oracle constructs A/B with distinct, exactly representable values per row/column, executes the primitive once, then multiple dependent K steps, and compares **all 4 independent 8×8 products × every accumulator element** against a straightforward FP32/F64 CPU reference and a device scalar implementation. Include lane permutations, all-zero/negative/subnormal inputs, head/key masks, K tails, and repeated graph launches; run Compute Sanitizer `memcheck` and `synccheck`. Do not call GPU arithmetic correct because PTX compilation succeeded.
- Reuse this signed-off primitive in P2/P3 rather than writing independent untested MMA operand layouts. Record a small `tests/v100/test_mma884_fragment_contract.cu` plus emitted PTX/SASS samples in the PR.

### P1.2 Numerical strategy

For each FP32 query value `x`, prepare `q_hi = fp16(x)` and `q_lo = fp16(x - float(q_hi))`; retain the original FP16 key. For each 8×8 score tile, keep two **independent FP32 MMA accumulator sets**:

```text
D_hi += q_hi × idx_K
D_lo += q_lo × idx_K
D     = D_hi + D_lo  // FP32, after the independent chains finish
score += max(D, 0) × idx_weight
```

This adapts Strata-V100's improvement after it found that adding low-half products into an existing high-half chain lost precision. Separate chains cost instructions/registers; test both quality and throughput. Apply a numerically safe per-tensor or per-row scaling only as an **optional, verified** second experiment. Do not silently clamp infs, underflow or change the ReLU location.

**Important precision qualification:** SM70 FP32 accumulation from F16 products is not necessarily equivalent to a scalar FP32 dot; expect different reduction order. The reference is **output/decision parity**, not bitwise equality. Examine top-k changes especially when the score margin around ranks 512/513 is tiny.

### P1.3 Stepwise implementation

**P1.3.1 — Fragment and scorer microtest**

- Unit test `8×8×4` and multiple K steps against FP32 and FP64 reference, including zeros, negative per-head dots, non-finite test guards, tails and large/small FP32 queries.
- Start **one token** and 64 heads; use 8-head bands and 128-dot K reductions (`32` K steps of 4). Assign the four quad-pair MMAs to distinct 8-row tiles where practical (e.g. **32 compressed rows/warp × eight heads**, then advance over eight head groups). Experiment with **4 versus 8 warps per CTA**.
- Cache/reuse the query hi/lo fragments and 64 weights per block. Use coalesced, aligned K row loads; avoid a full `[n_rows,64]` intermediate in global memory.
- Finish each head-band dot, apply ReLU, update weighted FP32 row totals. Bound registers and shared memory; use `ptxas` reports and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

**P1.3.2 — Single-token decode integration**

- Add `idx_score_volta_kernel`, only for **64 heads, 128 index dimensions, valid FP16 keys**, with checked SM70 device dispatch.
- Retain the same output layout expected by **unchanged** `idx_select_kernel`.
- Compare 512, 513, 1,024, 8,192, 31,250 visible rows; exercise rows not divisible by 8/32/CTA span. Benchmark score, select and indexer-skip overhead separately. Compare prefill disabled and speculation disabled to isolate decode effect.

**P1.3.3 — Batched prefill integration**

- **Baseline matters:** existing FP32 batch kernel reuses 32 tokens × 32 compressed rows; the SM80 optimized batch kernel reuses 32 token queries and key fragments. A naive one-token/CTA SM70 version may lose badly due to redundant key loads.
- Implement a **correctness-first** batched kernel using P1a arithmetic, initially one token per work item, then a distinct **key-reuse** variant: tile e.g. 8/16/32 prompt tokens against 32/64/128 compressed rows and reuse the K fragments across tokens.
- Evaluate a token-major loop retaining loaded K registers or shared memory; double-buffer queries **only if measured worthwhile**. Use `p0`, token `t`, `ratio`, `T`, `ld` consistently. Handle early-visible-row skips and valid-row tails without reading unwritten score entries.
- Benchmark 256/1,024/4,096 prompt tokens against 2K/8K/32K/120K starting contexts, and isolate CSA layers from HCA/raw-only layers.

**P1.3.4 — Integration, selection and acceptance**

- New **proposed** `ABYSS_V100_IDX=auto|volta|cublas|f32` (the `cublas` option is a P1e batched scorer only). Preserve `STRATA_DS4_IDX=1` as an unconditional legacy-FP32 override. If a user *forces* `volta` on an unsupported device, fail with a clear diagnostic rather than launching a trap. `auto` chooses the new path only for **validated profitable shapes** (initially allow conservative fallback).
- Log the actual scorer selected, geometry, dynamic shared memory size, and why a fallback occurred.
- Run scorer→unchanged-top-k parity; record scores, absolute/relative score differences, chosen index IDs, the rank-512 margin, disagreement rate conditioned on margin, and downstream attention/logit impact.
- **Do not demand identical top-k IDs at arbitrarily near-tied margins** if the resulting numerical difference is explained; but investigate every mismatch with a meaningful margin. No undefined, out-of-range, future or duplicate index introduced by the port.

**P1.3.5 — FP32-scratch batched cuBLAS scorer (from 1Cat DeepSeek SM70 reference)**

A third candidate for repeated C4 compressed keys is **batched cuBLAS GEMM**, distinct from the direct fused FP32 baseline and the new inline-PTX MMA884 scorer. Assemble a per-head query matrix with `T × 64` rows, multiply by visible compressed keys and apply **ReLU separately to each head’s dot product**, then multiply by that head's learned weight and sum 64 heads in FP32. A tempting `relu(sum_h weight[h]*q[h]·k)` factorization is **invalid** for the trained model. For initial quality gates, store dot scores in **FP32**, not FP16: near rank-512 tie errors have been observed in external DeepSeek indexers.

Tile by key rows with a hard workspace allocation cap and causal/visibility masks for each token; account for query packing, cuBLAS call setup and reduction epilogue. Benchmark T=1,2,3,8,16,32,256 with 512,2K,8K and ~31K visible compressed rows, against the current FP32 32×32 shared-key baseline and the MMA884 scorer. If a wider GEMM is winning only at large T/visible rows, permit it **only in that dispatch regime**. The proposed `ABYSS_V100_IDX=cublas` is meaningful **only when implemented and a compatible shape is present**, otherwise fail clearly; keep legacy `STRATA_DS4_IDX=1` force-FP32 precedence.

**P1 exit (also governed by §6):** passes CUDA sanitizer, scorer and selection contract, teacher-forced logits and boundary regression, preserves sm80/HIP; demonstrates the kernel speed and overall effect on fixed benchmarks. Keep FP32 default for any regime in which Volta loses. *P1 should not be pursued for its own sake if the P0 trace shows its indexer occupies negligible critical-path time.*

---

## 4. P2 — Volta Tensor Core batched DeepSeek attention (detailed)
### P2.0 Semantics and scope

P2 covers **`attend_batch` (prompt prefill)** first; the separate decode `attend_split`/`attend_combine` kernels are deferred to P4. This distinction is essential because prefill has an 8-head×many-token workload and decode has a different launch/parallelization tradeoff.

Abyss's head dimensions are **64 heads × 512**, with K=V stored as FP16. For absolute token `p = p0 + t` and an attention layer of compression ratio `r`:

```text
n_raw = min(p + 1, sliding_window=128)
raw0  = p - n_raw + 1 - (p0 - n_hist)     // existing batch-buffer indexing

if r == 0:    n_comp = 0                           // raw window only
if r == 4:    n_comp = min(top_k=512, (p+1)/4)   // selected CSA rows
if r == 128:  n_comp = floor((p+1)/128)           // chronological HCA rows

key(j):
  j < n_raw       -> raw_keys[raw0+j, :]
  j >= n_raw, r=4 -> comp_keys[selected[t,j-n_raw], :]
  j >= n_raw,r=128-> comp_keys[j-n_raw, :]
```

The attention logit is `(q[h,:] · key[j,:]) * scale`. **Sink** `sinks[h]` is an extra **softmax logit with no value vector**. Equivalently, initialize the online-softmax max to `sinks[h]`, denominator to `1.0`, and numerator to zero. Update max, denominator and numerator on each tile with mathematically consistent FP32 rescaling. After normalization, write FP32 output and run **the existing inverse-RoPE transform on the final 64 dimensions**.

**Must preserve** masks/causality, raw-ring or batched-linear indexing as appropriate, CSA `sel[t,*]` ordering, HCA chronology, compression boundaries, sink treatment and scale. It is *not* interchangeable with Maya or Qwen QSA attention.

**P2.0.1 — Exact MLA latent-space contract and parity gate (mandatory, before integrating raw-only P2b).** The comment that generic MLA "up-projects compressed KV" is **not the operation in this pinned Abyss kernel**. Its actual [DeepSeek forward specification](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/docs/deepseek4/SPEC.md) and [`dsv4_fast.cu`](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/kernels/cuda/dsv4_fast.cu) implement shared **512-D latent K = V**, used by 64 independently normalized 512-D query heads. Preserve this exact identity and projection order:

```text
qr   = RMS(wq_a @ x, attn_q_a_norm)               # [1024]
q    = per_head_RMS(reshape(wq_b @ qr, [64,512]))
q    = RoPE_last64(q, query_position, layer_rope)
kv   = RMS(wkv @ x, attn_kv_a_norm)                # [512], K = V
kv   = RoPE_last64(kv, key_position, layer_rope)
# Existing raw/ratio-4/ratio-128 compressor and selection update rules.
logits[h,j] = dot(q[h,:], keys[j,:]) / sqrt(512)  # latent-space QK
softmax_h = softmax([logits[h,*], sinks[h]])       # sink contributes denominator, no V
z[h,:] = sum_j softmax_h[j] * keys[j,:]           # P @ same K = V
z[h,:] = inverse_RoPE_last64(z[h,:], query_position)
output = wo_b @ concat_g(wo_a[g]^T @ z[heads_in_g])
```

RoPE for ratio-0 uses the uncompressed-layer base (10,000), while compressed layers use the **specific YaRN-scaled base 160,000**; compressors rotate completed rows at block **start position** (`s0`), not the query's position. Never synthesize an extra per-head KV up-projection, reconstruct full-dimensional K/V from a nonexistent latent decoder, de-rotate prior to applying V=K, or move inverse RoPE past `wo_a`/`wo_b` without an explicit equivalence proof. Output scaling and sink logits must match the existing spec and source precisely; a different MLA engine's equations are not authoritative here.

`tests/v100/test_dsv4_mla_latent_parity.cu` must compare **(1) FP32 reference and existing Abyss** with **(2) isolated SM70 QK, softmax and PV**, **(3) fused raw-only attention**, **(4) CSA top-512 selected and HCA chronological gathers**, and **(5) post-inverse-RoPE grouped output projection**, on the **same cached raw/compressed rows**. Cover ratio=0/4/128, no keys/sink-only if reachable, one/partial/full 128-token windows, wrap, compressor completion at 4/128 boundaries, distinct query/key positions, YaRN context transitions, malformed/near-tie selection defenses and 512-dim edge values. Dump stage tensors `q`, `kv_raw`, `kv_comp`, `index_selection`, `QK_logits`, `softmax_den`, `PV_latent`, `inverse_RoPE_out` and `post_wo_out`, comparing max/relative error and final logits. Fail hard on causality, row identity/order, sink bookkeeping and projection placement; numerical thresholds remain calibrated under the Level A/B model gates. **No P2b integration until latent parity passes on a physical V100.**

### P2.1 Arithmetic and geometry feasibility study

Implement a standalone MMA884 attention math test **before** writing a large fused kernel.

- **QK:** 8 heads × 8 (or 32) key rows. Q is FP32, so prepare FP16 `Q_hi` and `Q_lo`, compute **independent** MMA chains, then combine F32 score fragments. QK's reduction length is **512** (128 steps of K=4); parallelize across dimension groups where useful, then reduce partial scores in a defined F32 order.
- **Softmax:** Compute score scaling, sink-aware maximum and denominator in FP32; respect masks for final incomplete tile; calculate exponentials in FP32.
- **PV:** Probabilities are FP32; represent as FP16 `P_hi` and `P_lo` and use **independent** MMA chains against the FP16 K=V rows; combine in F32. When the online-softmax max changes, rescale **both** running output chains before adding new contributions. Alternatively, combine chains at well-defined tile boundaries into a single F32 running output, after validating parity.
- **Packing:** Derive A/B fragments and C coordinates from the verified Strata-V100 four-quad-pair map. Unlike that implementation's 64-D partial rows, DeepSeek's outputs have 512 dimensions and no Qwen KV dequant scales; never copy Qwen-specific layouts or scale logic wholesale.
- **Candidate schedule:** CTA `(one token, eight heads)` with **4 or 8 warps**, `KEY_TILE=16 or 32`, dimension subtile=64 or 128, then optional 16-head geometry. `KEY_TILE=32` is a useful first comparator because Abyss's FP32 fallback already uses it.
- **Register/shared-memory design:** An 8-head × 512 **FP32** output is 16 KiB by itself. A 32-key ×512 **FP16** key tile is 32 KiB; 8-head ×512 FP16 hi+lo queries are another 16 KiB. Score/probability scratch and padding add several KiB, potentially taking a naive simultaneous allocation beyond ~64 KiB/block and limiting occupancy to one CTA per V100 SM. **Show a compile-time byte ledger for each layout**; reuse shared memory where lifetimes do not overlap, or keep per-warp 64-D output bands in registers. The hardware has ~96 KiB shared memory per SM, with per-block limits requiring opt-in above the default; query device properties and validate `cudaFuncSetAttribute` success. Keep register spills **zero** where practical; report occupancy, not assumed occupancy.
- Benchmark **SM70 scores**, **softmax**, **PV**, and combined tile kernel separately. A two-pass reference variant using explicit score scratch may be useful for debugging, but avoid a large global attention matrix in the final implementation. **Superfast comparison:** its streaming verifier uses D=256, GQA=6 and huge KV histories, while DeepSeek gathers at most 512+128 CSA or roughly `context/128 + 128` HCA keys per head. Do not predict similar benefit from a K/V streaming pipeline.

### P2.2 Stepwise delivery

**P2.2.1 — F32-compatible MMA primitives**

1. Implement QK and PV isolated parity kernels. Verify MMA register maps and hi/lo behavior for 512 dimensions and varied key counts. Record errors against F32 fallback and F64 synthetic reference.
2. Establish a stable softmax/online normalization implementation, including sinks, fully/partially masked key tiles and FP32 exponentials.
3. Validate resource feasibility using `ptxas --verbose`, CUDA function attributes, device smem opt-in and Nsight Compute. Reject designs with unexpected local-memory spills or unstable occupancy.
4. Do **not** add compressed-index gathering in the first arithmetic microtest; use contiguous keys with explicit masks.

**P2.2.2 — Raw-only (`ratio=0`) integration**

1. Implement `attend_batch_volta_kernel` (initially 8 heads/CTA) with the **real raw-window addressing**, QK → sink-softmax → PV → F32 output → inverse RoPE.
2. Compare every head/output dimension to existing `attend_batch_kernel` for 1, 7, 8, 15, 16, 31, 32, 33, 127, 128 and 129 visible raw tokens; test 128-token window wrap and partial prefill chunks.
3. Compare the first and last output token of the chunk and a chunk split into multiple calls; detect off-by-one in `raw0` and absolute positions.
4. Test numerics and GPU memory sanitization **before** measuring large-prompt throughput.

**P2.2.3 — CSA (`ratio=4`) and HCA (`ratio=128`)**

1. Add `sel[t,*]` keyed gathers for CSA after preserving **exact** top-512 ordering and eligibility at each token.
2. Add sequential HCA compressed keys with the existing `floor((p+1)/128)` visibility rule.
3. Test at token positions 0, 1, 3, 4, 127, 128, 129, 511, 512, 2,047, 2,048, 2,049; also near every compressed-block boundary, nonzero `p0`, `n_hist` and long-context tails.
4. Measure HCA, CSA and raw-only separately, because different selection and gather patterns can alter the winning tile layout.

**P2.2.4 — Tile tuning and final fused version**

1. For each candidate `(heads/CTA, warps/CTA, key_tile, dim_subtile, shared/register O)`, capture kernel µs/ms, setup/launch overhead, `ptxas` registers, spills, smem, achieved occupancy, effective bandwidth and output error.
2. Consider 16 heads/CTA, Q/P conversion caching, double-buffered K loads, and alternate shared-memory staging **only after** the 8-head baseline is correct. An optional scheduling comparison separates QK/PV warps and prefetches the next key tile, inspired by Superfast `fattn-verify-stream.cuh`, **without borrowing its GQA/causal-mask math**. Avoid `ldmatrix`, `cp.async`, BF16/TF32 and SM80 `mma16816` paths: not supported on Volta.
3. Keep F32 fallback for short/irregular/low-occupancy regimes, if faster. Superfast’s own sparse-attention Volta-MMA alternative is **off by default because slower** than its non-MMA sparse kernel in its tested setup; treat this as a caution, not a result for Abyss. Do not add data-dependent switches that change attention semantics.
4. If the complete P2 kernel cannot be made both correct and profitable, retain correct component kernels/microbenchmarks and **ship FP32**. Do not force a high-shared-memory tensor-core path simply to close the checklist.

**P2.2.5 — Dispatch and numeric gates**

- Add **implemented with PR P2c** `ABYSS_V100_ATTN=auto|volta|f32`; `STRATA_DS4_ATTN=1` remains the authoritative legacy-FP32 override. Forced SM70 on another GPU gives a diagnostic, not a trap.
- `auto` enables only on actual V100, valid `(ratio, n_head, D, tile, smem)` constraints, and measured successful shapes. Preserve SM80/HIP routing.
- For a fixed test matrix record relative L2, maximum absolute and worst relative error away from zero, p99 elementwise error, sink-denominator checks and long-context accumulation error versus FP32 and a limited FP64 reference.
- **Component-screening target, subordinate to the consolidated exact-math gate in §6:** normal-case attention relative L2 around `1e-5` or better where practical; investigate/fail ordinary cases exceeding `1e-4` pending a calibrated distribution. These are **not** sufficient by themselves: require stable teacher-forced logits, boundary/top-k effects, greedy-prefix/task tests and no quality regression beyond measured run-to-run variability.
- Include an indexer FP32 + attention SM70 configuration to isolate P2 from P1; then test both enabled.

**P2 exit (also governed by §6):** a correct SM70 batched-attention path (raw/CSA/HCA) with a documented fallback, a real performance win in at least one useful V100 prefill regime **or** a documented evidence-based reason to retain FP32, and no regressions outside SM70. **Appendix candidates C05/C06 are separate optional gates**, not prerequisites for the exact FP16-cache core implementation.

---



## 5. P3 — admit no more than three data-supported improvements

These are **candidate experiments, not commitments**. P0b must rank their expected *end-to-end* contribution; P1/P2 benchmarks establish the remaining work. Admission means one focused PR, its own A/B, numeric verification and a rollback path. If P0 finds only one meaningful problem, implement only one P3 item.

### P3.1 Quantize-once/scatter for real DeepSeek MMQ (conditional)

**Source:** upstream Strata [`c675f5b`](https://github.com/Niko1221/Strata/commit/c675f5b78725) quantizes a prompt token's routed activation once and scatters identical q8_1 blocks to the selected expert destinations, avoiding repeated quantization. In Abyss `src/prefill/moe_mmq.cu`, ordinary `mmq::quantize` uses `quantize_mmq_q8_1_cuda`, and fused GLU uses `turing_mma_available` gating. These facts **do not** establish whether the relevant DeepSeek MMQ GU path executes on Volta; §2.3 decides.

**Admission:** proof from **actual V100 trace** that (i) grouped ordinary MMQ runs on the chosen DeepSeek layers with q8_1 input quantization; (ii) repeated activation quantization is measurable; (iii) the vendored ggml supports a valid SM70 `quantize_scatter_mmq_q8_1_cuda` or a small equivalent. If absent, close N/A, do not build a new MMQ engine merely to make this optimization applicable. Keep MXFP4's dedicated FP4-activation route distinct, and show whether Q2_K down consumes scattered q8_1 or H re-quantization.

**Implementation if admitted:** prototype with byte-identical q8_1 activation blocks for T=1,2,3,8,16,128,512,1024,4096, each route multiplicity including duplicate/tail groups, no output overwrite, correct alignments, graph capture and memory sanitizer. Benchmark quantization-only, grouped-GU and full prefill/TTFT, including the GPU time spent writing K destination copies. Preserve current gather fallback. A microbenchmark improvement alone is insufficient; only choose default if the whole prompt improves repeatedly without quality drift.

### P3.2 Broadwell-EP exact IQ2_XXS singleton CPU expert rows (conditional)

**Source:** upstream [`8ee2005`](https://github.com/Niko1221/Strata/commit/8ee2005dae0b) and [`77ddc2f`](https://github.com/Niko1221/Strata/commit/77ddc2ff147d) offer AVX2-bit-exact one-token expert code for certain IQ types. Current Abyss `src/kernels/cpu/native_expert.cpp` already dispatches optimized AVX2 multi-token work and a specialized IQ3 single-token path; therefore the comparison must be incremental against **Abyss's real IQ2_XXS route** on this CPU, not against an unsupported AVX-512 path or generic scalar code.

**Admission:** CPU expert time contributes materially to decode/verify wall time, IQ2_XXS gate/up is actually executed on Broadwell CPU, and ggml-cpu baseline is an AVX2 variant with matchable dot arithmetic. Record exact dispatch and exclude nonmatching CPU ISA builds. Unit test GGML row-dot equivalence for extrema/zeros/random packed blocks and fixed DeepSeek 4096→640→4096 geometry. Compare T=1 and T=2–8 plus sockets/node-local vs QPI-crossing memory, with core/thread affinity and fixed clocks. Do not port AVX-512 into Broadwell. Close N/A when existing optimized path is equal/faster or little CPU work is on the critical path.

### P3.3 Online per-layer CPU/PCIe miss allocation (conditional)

**Source:** upstream [`04be241`](https://github.com/Niko1221/Strata/commit/04be241a5a7e) derives a changing split from measured GPU and CPU processing costs. Abyss already calibrates CPU-lane execution, DMA and mapped zero-copy transfer at startup in `src/core/glm_fast_path.cu` and may already overlap these. **Do not replace a functioning static plan without evidence.**

**Admission:** record static-plan end-of-stage waits per layer and whether they vary under cold→warm, route changes, long context, competing CPU load and different NUMA/page placements. Fit an optional bounded estimator from completed transfers and CPU worker times (no extra GPU sync in critical path); include shared DRAM contention and PCIe 3 transfer serialization. Bound update frequency and minimum sample count, enforce deterministic max CPU/GPU work counts, preserve same selected experts and exact results. A/B real 128/512/1024-token generations with repeated clean intervals; retain startup plan if adaptation adds overhead or thrashes.

**Priority rule:** admit P3.3 ahead of P3.1/P3.2 if P0 shows uncovered CPU/PCIe waits dominate token latency; no fixed pre-ranking is justified by Qwen results on other GPUs.

## 6. One authoritative acceptance and PR table

This section **replaces** v8's separate evidence A/B/C, v7-v8 additional gate lists, preliminary gates, stop-rule table, dashboard table and PR table. Every experiment has exactly one **gate → stop rule → artifact → PR** row. Details in §§2–5 and Appendix A are tests, not alternative admission criteria. Artifact paths are proposed outputs and must include SHA/model digest/compiler/settings, raw paired runs, selected kernel, safety/quality, flags and rollback; record `N/A` closures in the same manifest.

### 6.1 Numerical baselines, not relative cross-engine budgets

**Same-pack ΔKL definition:** For matched prompt positions, same GGUF bytes, tokenizer, seed, context and baseline ABI, let `p0` be the **Abyss F32** fallback distribution and `p1` the candidate implementation; report mean `KL(p0 || p1)` over a fixed teacher-forced corpus. Compare the same branch's F32 baseline to the candidate (v1.3.28 and v1.3.31 have separate baselines). The Abyss-vs-llama.cpp reported KL of approximately `0.008–0.009` is **not** the allowed additional implementation error.

- **Exact-math arithmetic paths** (P1/P2, non-lossy P3): target mean **ΔKL ≤ 0.0005**, with p99/worst-position tails, task quality and token/top-k changes disclosed. This is a **maximum admission screen**, not a license to alter logic: byte-identical changes such as activation-scatter or exact CPU dots must additionally pass **byte-for-byte output equivalence** on their supported shapes. Any nonfinite output, invalid cache index, sink treatment or unsafe race is a hard fail even at low aggregate KL.
- **Lossy experiments only** (Appendix KV Q8/IQ4, Q8 hyper-connection): maximum proposed mean **ΔKL ≤ 0.002** relative to same-pack Abyss F32, plus explicit near-tie/top-512, multilingual, code, long-context/needle and rollback checks. They remain off in the first release. This larger budget does not apply to a Volta MMA approximation that purports to preserve model mathematics.
- **Arithmetic components:** P1 raw score and top-512 margin distribution; P2 separate QK, sink-inclusive softmax, PV, inverse RoPE and grouped output stages. Use an FP64 reference for limited synthetic tests, FP32 Abyss as operational baseline, and report relative L2/p99/max error. The original component target ~`1e-5` and investigation threshold ~`1e-4` never overrides the distributional/semantic and quality gates.
- **Performance admission:** a sustained paired, interleaved real-model V100 improvement beyond run-to-run noise in the *affected* critical-path stage and **no end-to-end regression** in supported settings; do not borrow claimed speedups from RTX 4090, SM86, P100 or another quant. Record confidence intervals or per-repeat deltas and sample count, power/clock/thermal state, competing loads and peak memory.
- **Correctness hard stops:** wrong model semantics, stale GPU/CPU cache view, ABA slot reuse, compiler SASS incompatible with SM70, page/pin OOM, data race, or backend regression to other architectures. Output differences near exact ties must be traced, not hand-waved.

### 6.2 Single project admission, stop and PR matrix

| Experiment | Gate | Stop rule | Required artifact | PR |
|---|---|---|---|---|
| P0a selected base | Both SHAs build **and actually perform V100 full-model inference** or clear first failing stage | Cannot load/run after isolated triage → block remaining production integration | `p0/bases.json`, `p0/memory.csv`, `p0/model-manifest.json`, outputs/logit deltas | **P0a** |
| P0b bottlenecks/dispatch | Real MMQ/GLU/CPU paths, per-layer critical path, thermal/NUMA/PCIe measured | Unavailable hardware counters → use CUDA events/nsys; never fabricate occupancy counters | `p0/trace.jsonl`, `p0/dispatch.csv`, `p0/hardware.csv`, decision `yes-no-na.json` | **P0b** |
| P1 fragment primitive | Four quad-pair map and typed PTX `"r"`/`"+f"` constraints, emitted SM70 SASS, sanitizer, FP64 test | Incorrect map/instruction/error → no integration | `p1/mma884-parity.json`, `p1/sm70.sass`, microbench | **P1a** |
| P1 indexer | Scorer+unchanged select correct, ΔKL ≤0.0005, winning real V100 shapes | No repeatable model-critical-path win → retain FP32 | `p1/indexer-ab.jsonl`, selected-top512 margins, model logs | **P1b** |
| P2 latent primitive | 512-D K=V, sink denominator/no sink V, RoPE/inverse, raw/CSA/HCA reference and errors | Any semantic miss or unsafe resource use → retain FP32 | `p2/latent-parity.json`, `p2/mma884-qkpv.json` | **P2a** |
| P2 batched integration | Raw→CSA/HCA boundaries, model ΔKL ≤0.0005, resource envelope, real V100 A/B | No clear winning tile/shape → retain FP32 | `p2/attn-ab.jsonl`, spill/smem/occupancy and dispatch report | **P2b/P2c** |
| P3.1 MMQ scatter | Actual SM70 repeated q8_1 MMQ path **yes**; byte-identical scatter; beneficial prefill | `mmq=no`, fused-only unavailable, repeated prep immaterial or slower → **N/A** | `p3/scatter-eligibility.json`, `p3/scatter-ab.jsonl` | **P3.1** if admitted |
| P3.2 Broadwell IQ2 | Real IQ2_XXS CPU path; exact ggml AVX2 rows; repeatable end-to-end gain | Existing dot equal/faster, CPU off critical path → **N/A** | `p3/iq2-dispatch.json`, `p3/iq2-ab.jsonl` | **P3.2** if admitted |
| P3.3 PCIe balance | Static lane leaves measured waits; bounded online control lowers all-in decode | No imbalance, thrashing, +sync, NUMA regression → **N/A** | `p3/pcie-cost-model.json`, `p3/pcie-ab.jsonl` | **P3.3** if admitted |
| P4 release | Real V100 smoke/sanitizer/long-run; exact and optional routes isolated; other-arch compile pass | Incorrect image, model parity loss, instability, unsustained speed → do not release | `release/manifest.json`, `release/ci.json`, `release/benchmarks.jsonl` | **P4** |

**PR packaging:** isolate one behavior/format/dispatch decision per PR. P0a and P0b may use existing `STRATA_*` diagnostics; `ABYSS_V100_PROFILE` appears only with P0b. The indexer and attention controls appear only with their actual runtime dispatch PRs. Experimental controls are not built or documented early as inert toggles.

## 7. P4 — runtime safety, integration and release

- **Cache and repack lifetime:** retain canonical host bytes; distinguish `slot_epoch` (physical slot generation) from `layout_version` (encoded GPU format). Slots transition `empty → loading → [repacking] → ready → eviction`; publish a pointer or `ready` bit **only after the correct stream event**. Guard against ABA reuse, post-capture stale pointers, mismatched CPU/GPU layouts, cache reset and failed copies. If no repack experiment is admitted, do not add repack controls/code.
- **Architecture isolation:** inspect actual SM70 object images and architecture tags for invalid SM80/PTX instructions. Keep inline PTX MMA validation on the GPU, with compiler and driver hashes. Do not rely on whole-fatbin grep or compile success as runtime proof.
- **Memory safety:** monitor pinned RAM and THP (actual realized pages and compaction), VRAM allocations/fragmentation, PCIe 3 and QPI. Guard CPU worker pool against NUMA oversubscription; never assume all 128 GB is local to the V100.
- **Reliability:** full prompts, cancellation and restart, 1/2/4-token speculative verify, chunk boundaries, rollback, file-tier miss and recovery; do not publish partially loaded/repacked experts. Use CUDA sanitizer where feasible and a bounded background-work shutdown test.
- **Deployment and runner:** do not churn the existing `abyss-v100-runner`. Keep image/toolchain provenance, preserve the user model volume, mark PR code and actual-V100 results separately. Avoid sampling tools that require unsupported `NVGPUCTRPERM` grants when equivalent CUDA-events/nsys traces suffice.
- **Ship:** only useful SM70 paths chosen by `auto`, legacy FP32 fallback, one documented environment-control table, per-PR results, known limitations and repeatable benchmark commands. Never assert RTX5090 results as V100 expectations.



## Appendix A — candidates, not committed

**These are research opportunities, not scheduled PRs or additional acceptance tables.** Each remains a separate **C-ID** (without the stale v8 identifiers). Promotion requires (i) a P0b measured bottleneck, (ii) a candidate-specific feasibility note addressing DeepSeek-0731 shapes and SM70, (iii) the §6 numerical gate, and (iv) an explicit PR assignment then. Candidates do not get public environment variables before promotion.

| ID | Candidate | Trigger and decisive stop rule | Relevant source/evidence |
|---|---|---|---|
| **C01** | Xolotl exclusive per-layer VRAM LRU, disjoint GPU/host residency, warm-route profiles and hysteresis | Real cache misses/churn/duplication dominate after P0; compare same recorded expert routes; stop if Abyss already equivalent or gains disappear on topic switches | `tools/ds4/ds4_moe.cpp`, `docs/ds4/FINDINGS.md` |
| **C02** | Predicted expert prefetch / bounded lookahead | Prove it hides uncovered PCIe 3 waits, subtract wasted predictions and duplicated DMA; stop on copy contention, stale slot or no gain | Xolotl `ds4_moe.cpp`; Abyss `glm_fast_path.cu` |
| **C03** | Direct resident-slot pointer-table MMQ | Upstream supports IQ2_XXS but **not target Q2_K down or MXFP4**; require mixed direct/gather, slot guard bytes, per-format memory layout, exactness and amortization; stop without real format coverage | Strata [`7d9b50d`](https://github.com/Niko1221/Strata/commit/7d9b50dc1a5c), [`86f677b`](https://github.com/Niko1221/Strata/commit/86f677b4a976) |
| **C04** | Group sorting, shared activation preparation/fused GEMVs | Whole-tile MMQ route, exact partial-group behavior, shape-specific B/A; stop if groups differ or prefill already efficient | Strata [`62fc7c8`](https://github.com/Niko1221/Strata/commit/62fc7c8a3c98); Xolotl fusion findings |
| **C05** | FP16 compressed KV → Q8_0 / rotated IQ4_NL, separately indexer keys Q8 | A *filled* 32K–128K context must trade freed VRAM for enough additional expert slots to beat decode overhead. Distinguish comp vs icomp, raw 128 window and state; reapply Hadamard inverse; quantized ranking can differ | Xolotl `docs/ds4/FINDINGS.md`, §35 |
| **C06** | Host-mapped compressed cache / raw-window query blocking | Stop if explicit direct-gather baseline already avoids masked union, or PCIe penalty outweighs slots. Note Xolotl's reported host cache penalty on faster PCIe | Xolotl `ds4_dense.cpp` + findings; Abyss `dsv4_fast.cu` |
| **C07** | Static dense QPN SM70 repack / shape-routed quant GEMV | QPN supports Q4_K/Q5_K/Q6_K/IQ4*/Q3_K/Q8_0, **not** Q2_K, IQ2*, MXFP4; only target real hot static tensors and preserve all readers. Do not repack all expert tiers | Superfast `mmvq-qpn.cu/.cuh`, `mma.cuh` |
| **C08** | Independent Q2_K small-T MMA and MXFP4 E8M0/32 decoder | Separate two formats and reference oracles; FP4 must implement block-scales E8M0/32; no substitution of NVFP4 or Q2_0. Stop if hardware cost exceeds GGUF-native kernel | anyei `llamacpp-v100`; 1Cat-vLLM; detailed v8 research archive |
| **C09** | Q8_0 hyper-connection storage | Verify actual memory recovered vs FP16/BF16 and need for per-stream dequant scratch; **lossy** §6 gate, first release off | Strata [`b60aab1`](https://github.com/Niko1221/Strata/commit/b60aab1ba27c) |
| **C10** | Async expert exchange, selective pinned ranges, host exchange-buffer rotation | Only if memcpy/copy scheduling, pinned limits and cache turnover are measured bottlenecks; preserve epochs/aliases and account for extra RAM; source reports device-specific slowdowns | Strata [`661ff29`](https://github.com/Niko1221/Strata/commit/661ff2933941), [`ab6bce0`](https://github.com/Niko1221/Strata/commit/ab6bce05de44), `docs/EXCHANGE_ROTATION.md` |
| **C11** | Output-head VRAM alias for prompt buffers, graph/workspace reuse | Proof head or duplicated workspace prevents useful chunk/cache allocation; safe swap back before decode, count PCIe reload on V100; don't transplant ggml allocator internals | Strata [`640ac43`](https://github.com/Niko1221/Strata/commit/640ac438be9e); Xolotl `ds4_dense.cpp` |
| **C12** | Runtime NUMA/prefetch/PCIe micro-policies and cache service | Only when P0b shows spare DMA/CPU capacity; test local socket0 vs interleaved socket0+1, file mmap vs direct/streamed, per-miss payload vs CPU execution. Existing Abyss streaming/THP may cover most | Abyss v1.3.31 `glm_fast_path.cu`, `pinned.cu`; Strata memory changes |
| **C13** | Small-T output head, speculative acceptance-aware cost routing | Actual head GEMV/staging or verify work must dominate; no new trained drafter, preserve DeepSeek n-gram speculation and exact acceptance | Superfast conditional head; Abyss `dsv4_verify.cu` |
| **C14** | Other architectures, Turing/SM86, Qwen GDN, multi-GPU, lossy router pruning | **Out of scope** for first single-V100 DeepSeek release unless scope changes; cannot count as progress on SM70 path | Other upstream Strata feature branches |

**Historical detailed designs:** revision v8 contained longer descriptions of C01–C14 (including quant format lifetime, host mailbox, compression and numeric edge cases). This v9 intentionally keeps the committed plan short; retain v8 as a research/archive input, not a second set of requirements or toggles. In any promoted candidate, re-open the upstream source at a recorded SHA; never silently port patches written for Qwen3.8-Flash-Next into DeepSeek latent attention.

## Appendix B — source-confidence and reference ledger

**Confidence terminology:** **Verified in source** = source behavior or architectural statement was inspected directly; **Author-reported** = numerical benchmark or performance result appears in the project's documentation and has *not* been independently rerun on the user's hardware; **Unchecked** = external claim/benchmark not separately validated for this model and GPU. Multiple labels may appear in one row for different claims. Reviewer confirmations stated in the comments are distinguished from inspection performed while preparing this specification.

| Source | What it supports here | Source confidence | Measurement caveat |
|---|---|---|---|
| [Abyss current repository](https://github.com/zephel01/Abyss) [DeepSeek `PORT.md`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/docs/deepseek4/PORT.md) and [`SPEC.md`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/docs/deepseek4/SPEC.md) | DeepSeek latent math, build-supported/not-run V100, actual kernel design | **Verified in source** (reviewer confirmed) | RTX5090 measurements **author-reported**; not V100 |
| [Abyss MMQ](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/prefill/moe_mmq.cu) and [prefill](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/prefill/prefill.cpp) | Ordinary MMQ supported-format switch; **fused GLU has Turing gate**; unproven real SM70 dispatch | **Verified in source** | Actual V100 path **unchecked until P0b** |
| [Abyss CPU native experts](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/kernels/cpu/native_expert.cpp) | Existing AVX2, dedicated IQ3 singleton and ggml fallback | **Verified in source** | Which route is used on this Broadwell **unchecked until P0b** |
| [Abyss `glm_fast_path.cu`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/core/glm_fast_path.cu) and [latest changelog](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/CHANGELOG.md) | NUMA/THP, stream-fetch additions, startup CPU/PCIe calibration | **Verified in source** | Effect on this host **unchecked** |
| [Strata-V100](https://github.com/jmnargi/Strata-V100), [NVIDIA/Volta documentation](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/NVIDIA_V100.md) | SM70 MMA884 primitive, FP16 hi/lo separate accumulation and V100 feasibility | **Verified in source** (reviewer confirmed) | V100 **Qwen** benchmarks **author-reported**, not DeepSeek predictions |
| [Niko1221/Strata, pinned head](https://github.com/Niko1221/Strata/tree/477ac13cbe8e582c20484871ce459d20d82df09d), [MMQ scatter commit](https://github.com/Niko1221/Strata/commit/c675f5b78725), [CPU commit](https://github.com/Niko1221/Strata/commit/8ee2005dae0b), [PCIe commit](https://github.com/Niko1221/Strata/commit/04be241a5a7e) | Selected new source mechanisms; not a proof of shared GitHub fork ancestry | **Verified in source** (assistant inspection), **not reviewer-confirmed** | All throughput numbers **author-reported** and different Qwen workload |
| [Xolotl DS4](https://github.com/Indras-Mirror/Xolotl/tree/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16), [`FINDINGS.md`](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/docs/ds4/FINDINGS.md) | Cache/prefetch/compressed-KV/ggml graph ideas on DeepSeek-0731 | **Verified in selected source files** (assistant inspection), **not reviewer-confirmed** | Performance figures **author-reported**, RTX4090, different quant/CPU |
| [llama-superfast-v100](https://github.com/mylordmonkeyman/llama-superfast-v100), `mmvq-qpn.cu`, `mma.cuh` | SM70 MMA fragment checks and quant-repack concepts | **Unchecked** for this revision (reviewer did not confirm); earlier v8 source notes retained | Performance and applicability **author-reported / unchecked** for DeepSeek |
| [anyei/llamacpp-v100](https://github.com/anyei/llamacpp-v100) | Proposed Q2_K SM70 small-T path | **Unchecked** in this revision | No independent DeepSeek result |
| [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | Proposed DeepSeek SM70 lightning/MXFP4 concepts | **Unchecked** in this revision | No independent user hardware result |
| [mw00/project-maya](https://github.com/mw00/project-maya) | Older inherited Strata execution architecture | **Verified in Abyss ancestry documentation; independent project audit unchecked** | Not a V100 DeepSeek baseline |

**Do not promote an unchecked claim to "verified" merely because it has a URL.** Published commit metadata, source inspection, synthetic parity, author-run throughput and measured user-V100 numbers are separate evidence classes. Each future PR updates this table only for evidence it actually produces.

## Appendix C — version history and supersession

| Revision | Role |
|---|---|
| v1–v5 (17 October 2026) | Original Abyss V100 support plus Superfast and other low-level Volta research, quant/shape constraints |
| v6 (17 October) | Thermal/power, QPI/NUMA, exact 512-D latent MLA, PTX register contracts, SM70 image isolation, cache-slot epochs |
| v7 (18 October) | Xolotl cross-audit; optional caching, prefetch, compressed-KV and workspace experiments |
| v8 (18 October) | Post-October-10 Strata candidate audit: MMQ scatter, Broadwell IQ2, measured PCIe share, direct MMQ and memory lanes |
| **v9 (18 October)** | **One-page critical path, P0 base comparison/dispatch/memory worksheet, single acceptance table, three initial controls, explicit N/A paths, source confidence and repaired Markdown tables** |

**Supersession:** v9 is the sole forward implementation plan. Prior versions remain supporting research archives, not competing PR/task/gate lists. No repository code was modified and no V100 model tests were performed by writing this specification.
