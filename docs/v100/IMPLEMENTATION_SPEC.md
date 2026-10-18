# Abyss on Tesla V100 (sm_70): DeepSeek V4 Flash Performance Port

**Implementation specification — consolidated revision 8.0 (upstream Strata delta and Xolotl findings integrated; all v7 Volta/thermal/NUMA/MLA/CI gates retained) — 18 October 2026**

**Target:** [zephel01/Abyss](https://github.com/zephel01/Abyss), pinned to **`47001f31cbfba14bcbf4c083e3af07c50707ec0c` (v1.3.28)** for implementation and A/B testing.  
**Additional v8 upstream source:** [Niko1221/Strata](https://github.com/Niko1221/Strata/tree/477ac13cbe8e582c20484871ce459d20d82df09d), audited at **`477ac13cbe8e582c20484871ce459d20d82df09d`** (last upstream `main` commit: 16 October 2026). **Abyss-current differential reference:** [`7c893ac17e61ec45c934b3278356807fc4626c3c`](https://github.com/zephel01/Abyss/commit/7c893ac17e61ec45c934b3278356807fc4626c3c) (18 October 2026, Abyss v1.3.31); this does **not** supersede the pinned v1.3.28 build baseline without an explicit rebase and repeat of P0. Strata was compared by repository history, affected source and current feature presence—not by assuming that an unrelated GitHub repository retains shared fork ancestry.  
**Additional v7 source:** [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl/tree/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16), audited at **`f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16`** (2026-10-18), especially the *DeepSeek-V4-Flash-0731* `tools/ds4` implementation and `docs/ds4/FINDINGS.md` (not the unrelated 552B DeepSeek-V4.1 benchmark). Xolotl is a comparison/reference source, **not** the new base repository.  
**Reference implementations:** [mw00/project-maya](https://github.com/mw00/project-maya); [jmnargi/Strata-V100](https://github.com/jmnargi/Strata-V100); [mylordmonkeyman/llama-superfast-v100](https://github.com/mylordmonkeyman/llama-superfast-v100) (audited at **`70df8710401a32c4735cf0c7874bf626bbe6b493`**); [anyei/llamacpp-v100](https://github.com/anyei/llamacpp-v100) (SM70 Q2_K small-T kernels); [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) (DeepSeek-V4 SM70 indexer/MXFP4). New references are research inputs, **not** code incorporated into Abyss.  
**Machine:** Ubuntu 24.04 LTS; CUDA 12.8; Tesla V100-PCIe-32GB (compute capability 7.0, NUMA node 0); two Xeon E5-2697A v4 (32 physical cores, 64 logical CPUs total); 128 GB DDR4 across two sockets; NVMe.  
**Model:** DeepSeek-V4-Flash-0731, primarily Unsloth **`UD-Q2_K_XL`** (~96.8 GB GGUF, as described upstream).  
**Status:** The repository claims a working *build* for `sm_70` but does not claim **tested DeepSeek inference** on V100. Xolotl likewise does **not** establish full-model DeepSeek-0731 V100 throughput. This is an implementation plan, not a record of completed V100 DeepSeek tests.

## Executive decision

This version incorporates the earlier Volta arithmetic/NUMA corrections, detailed P1/P2 engineering plan, and source audit of `llama-superfast-v100`. It adds the **external V100 research candidates** (`anyei/llamacpp-v100` Q2_K, 1Cat-vLLM DeepSeek-specific indexer/MXFP4) and four **hardware-verification gates**: occupancy/spill analysis, accurately measured PCIe/CPU handoff, SM70-compatible profiling with roofline/counters, and Broadwell AVX2/FMA dispatch. It rejects unverified fixed occupancy/spill/PCIe thresholds as automatic disqualifiers. **Revision 6 further adds a sustained thermal/power attribution gate, QPI-aware NUMA placement experiments, model-exact latent-attention tests, inline-PTX operand-contract tests, CUDA architectural-manifest CI, legally acquired pinned profiling tools, and explicit GPU-cache-slot format epochs. Revision 7 adds a source-grounded Xolotl/Strata-DS4 review and conditional A/B tests for exact expert-cache replacement, predictive prefetch, disjoint GPU/RAM expert residency, compressed-KV storage, bounded prefill streaming, workspace reuse, persistent route profiles and per-layer launch fusion. These change *optimization ordering and measurement requirements*, not DeepSeek math or the starting repository. **Revision 8 audits upstream Strata changes after the Abyss repository was created, adds an explicitly source-verified quantize-once/scatter prefill test, a Broadwell IQ2_XXS exact-AVX2 CPU comparison, and measured online CPU/PCIe work allocation. It also adds strictly conditional direct-slot MMQ, Q8 hyper-connection, adaptive exchange/overlap and prefill-head alias studies, with model-format/VRAM/NUMA/quality preconditions and a check against newer Abyss changes.**

1. **P0:** Bring up **the shipped** DeepSeek GGUF on a V100, inventory weight types and tensor shapes, measure bottlenecks and establish accuracy/performance baselines.
2. **P1:** Use the independently documented Strata-V100 and Superfast SM70 **`mma.sync.m8n8k4`** layouts to accelerate DeepSeek’s lightning-indexer scoring if long-context prefill/decode is actually limited by it.
3. **P2:** Develop the model-correct **batched raw/CSA/HCA compressed-attention** kernel with tensor-core QK/PV, preserving sinks, K=V and inverse RoPE; profile against existing direct-gather FP32 fallback.
4. **P3:** Tune expert cache, CPU/PCIe/NUMA, then evaluate **Q2_K-capable** small-T MMA (anyei), **MXFP4-specific** Volta expert kernels (1Cat), and Superfast-derived format-aware MMA/repacking, activation sharing and grouped products, with strict type/shape/residency and register/occupancy gates. These are parallel candidates after P0, not prerequisites for P1/P2.
5. **P4:** Optimize decode/speculation, scheduling, optional Xolotl-inspired graph/workspace reuse and *upstream Strata measured online CPU/PCIe scheduling*; release only accuracy-validated whole-model improvements.

**Performance rule:** Neither published Strata-V100 results, Superfast’s Qwen-model speedups, nor Abyss’s RTX 5090 speedups predict V100 DeepSeek performance. A kernel improvement is accepted only with correctness evidence and an honest end-to-end measurement. There is **no guaranteed 2× throughput target**.

### Change-control and scope

- Work on `feature/deepseek-v100-perf` branched from the pinned Abyss commit. Rebase or cherry-pick later upstream changes **only after** the baseline is preserved.
- Keep existing CUDA `sm_80+` and HIP paths untouched. All new SM70 implementations must have an explicit, runtime-selectable FP32 fallback and an easy no-recompile A/B switch.
- Port **Volta optimization techniques**, **not** Qwen QSA/GDN, GLM model semantics, a Qwen-specific dense-product route table, or Superfast’s MTP model. New upstream Strata code is an **implementation reference**, not a supported import or a new default for DeepSeek.
- **Xolotl is an optimization reference, not a drop-in V100 target.** Do not replace Abyss's direct CUDA DeepSeek path with `ds4_generate`, reintroduce ggml-heavy paths into Abyss, or treat its RTX 4090 results as V100 predictions. Xolotl V4.1, Qwen and GLM features require independent model-specific justification.
- Do not add a new MTP/DSpark drafter, Windows/multi-GPU pipeline support, approximate MoE expert-skipping, or a replacement expert-offload framework in the first release.
- Follow the repositories' licenses and retain notices for copied code.
- Keep changes small, separately reviewable, with machine-readable benchmark artifacts.

## 0. Review findings and decisions

| Issue in prior drafts | Resolution in this specification | Why |
|---|---|---|
| Primary Volta primitive was `nvcuda::wmma` in one draft | **Use inline PTX `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` as the primary primitive**; keep WMMA as a contingency | Strata-V100 `qsa_prompt_attn.cu` supplies a measured, working V100 quad-pair/register layout |
| Single MMA accumulator mixed query/probability hi and lo halves | **Use independent FP32 accumulator chains for hi and lo products, then combine in FP32** | Strata-V100's measured change reduced 32K FP64-reference error ~5.08e-6 → ~2.33e-6 |
| Ambiguous architecture checks | Device-code guard `700 <= __CUDA_ARCH__ < 750` for the candidate kernel, **host dispatch only on actual cc 7.0** | Excludes Ampere/Turing/HIP paths, avoids illegal SM80 instructions |
| Over-prescriptive P1 geometry (one CTA / eight rows) | Start from 8×8 MMA tiles but **explicitly use the warp's four independent quad-pair MMA groups**; benchmark 32-row/warp and multwarp schemes | The individual 8×8 primitive does not imply an efficient CTA geometry |
| P1 batched kernel was treated as a simple extension of single-token | Preserve the existing baseline's **32-token × 32-row data reuse** as a comparator; explicitly design key reuse across tokens | One-token/CTA can be much slower in prefill because of repeated 128-D key loads |
| P2 described an ambitious fused tensor-core kernel with too little resource accounting | Develop QK/PV primitives first; design shared/register storage and **96-KB/SM occupancy** before fused implementation | Eight heads × 512 outputs, key staging, and separate hi/lo chains can exhaust registers/shared memory |
| Fixed accuracy targets looked more authoritative than evidence supports | Make **component parity, selection near ties, model logits and task behavior** independent gates; use provisional thresholds pending distributions | Different summation orders legitimately affect near-tied top-k entries and sampled/greedy outputs |
| “~97 GB expert payload” equated to full GGUF size | Measure **expert-only packed payload**, entire GGUF, VRAM residency, host pinned bytes, page cache and CUDA workspace separately | Published ~96.8 GB refers to the **whole file**, not a measured expert-only resident allocation |
| Forcing `numactl --membind=0` for a ~70+ GB RAM tier | Prefer Abyss's existing **two-socket interleaved NUMA pinning**; strict node-0 is a negative-control only | A single node may have only ~64 GB and cannot necessarily host the RAM tier |
| BF16 narrowing or conversion of all small GEMMs | Start with **lossless, cached BF16→FP32 widening** for profiled small GEMMs; FP16 is optional and numerically validated | Volta cannot multiply BF16 natively; arbitrary BF16→FP16 is not exact |
| Unnecessary `qsa_*` kernel porting | First verify which kernels DeepSeek launches; exclude unused GLM/Qwen kernels | Avoid unrelated work |
| A blanket claim that the default should always choose tensor cores | `auto` selects Volta only when **measured faster and validated at that workload** | Short contexts and tiny batches can favor FP32 |
| New: Superfast load-time-repacked quant kernels | **Add a conditional P3 lane, not an unconditional port** | Proven V100 ideas target Q3/Q4/Q5/Q6, IQ4 and Q8; **no shipped Q2_K/MXFP4 QPN kernel** and its selected shape routes are Qwen-specific |
| New: moving/repacked expert weights | **Canonical CPU/pinned/NVMe expert bytes remain untouched**; choose per-GPU-slot transformed copies only if amortized | Abyss actively promotes/evicts experts, unlike Superfast’s once-resident static dense weights |
| New: Superfast streaming/sparse attention | Reuse **warp division, direct gathers and occupancy heuristics as test ideas**; reject direct kernel drop-in | Qwen’s D=256/GQA=6 or 12, huge KV history and no DeepSeek sink/CSA semantics differ from DeepSeek D=512, K=V |
| New: Other SM70 code actually supports Q2_K and MXFP4 | Add separate Q2_K small-T and MXFP4 E8M0 kernels as **real candidates**, not inferred Superfast support | `anyei/mmsmt.cu` has Q2_K; 1Cat has SM70 MXFP4; different layouts and model/engine integration |
| New: Hardware occupancy rule | Require `ptxas`, occupancy and spill analysis; **50% occupancy and 32 B/thread spill are review triggers, not global pass/fail** | Volta occupancy trades against work reuse; actual runtime/roofline and correctness decide |
| New: PCIe assertion | **PCIe 3.0 ×16 ≈15.75 GB/s theoretical each direction**; measure achieved pinned H2D/D2H, zero-copy and remote-NUMA effects | A claimed 12.6 GB/s *bidirectional ceiling* is not correct; copy and CPU mailbox costs differ |
| New: Profiler tool compatibility | Use Nsight Systems plus **Volta-compatible Nsight Compute 2025.1/2025.2**, not the latest `ncu` | NVIDIA removed Volta support from Nsight Compute 2025.3; `nvprof` is legacy |
| New: CPU SIMD correctness | AVX2+FMA baseline with runtime feature detection; **no AVX-512 assumption** | E5-2697A v4 is Broadwell; validate compiler flags, active kernels and NUMA throughput |
| **v6: thermal/power stability** | Sample clocks, temperatures, power draw and **active NVML clock-limit reasons throughout** each A/B; classify thermal vs normal power-capped behavior | A single idle P-state snapshot cannot establish absence of thermal throttling; persist suspect data but exclude from clean comparisons |
| **v6: QPI topology** | Measure memory first-touch/NUMA placement, cross-socket QPI, CPU-local DRAM and GPU DMA separately | Host-to-CPU and host-to-GPU locality may favor different nodes; do not bind all 128 GB to node 0 |
| **v6: DeepSeek latent attention** | Test the exact 512-D K=V latent-space forward path, sink, RoPE/inverse RoPE and subsequent grouped output projections | Abyss has **no separate per-head KV up-projection inside `attend_batch`**; inventing one would change the model |
| **v6: PTX and fatbin isolation** | Compile SM70-specific device images, inspect per-image PTX/SASS/ELF tags, and run V100 launch tests | Whole-fatbin opcode grep confuses valid SM80 code with SM70 code; mnemonics vary between PTX and SASS |
| **v6: profiler preservation** | Require a **verified, checksum-pinned Volta-compatible `ncu` tool** installed in the test runner by license-compliant means | Do not put proprietary binaries in public git or assume NVIDIA permits redistribution |
| **v6: GPU repack slot state** | Separate generation (`slot_epoch`) from encoded-format version (`layout_version`), publish only completed repacks, retain canonical host bytes | Detect ABA reuse and eliminate stale-pointer or mixed-layout CPU/GPU reads |
| **v7: Xolotl / DS4 memory runtime** | Keep Abyss baseline; trace cache, useful prefetch, duplicate VRAM/RAM residency, context costs before coding; test exact per-layer LRU/hysteresis and disjoint copies | Source reports are 4090/other quant and cannot establish V100 benefit; source code overlaps existing Abyss service |
| **v7: compressed cache and workspace** | Optional FP16→Q8_0/rotated IQ4_NL storage with separate indexer-key lane; profile graph arena and chunking only when they exist in Abyss | Lossy KV needs explicit quality gate; ggml graph-allocation bottleneck may not exist in Abyss direct-CUDA path |

## 0A. New source audit: `llama-superfast-v100` applicability to Abyss

**Audited source:** [repository](https://github.com/mylordmonkeyman/llama-superfast-v100/tree/70df8710401a32c4735cf0c7874bf626bbe6b493), commit `70df871` (2026-10-09), especially [README-27B.md](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/README-27B.md), [`mmvq-qpn.cu`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-qpn.cu), [`mmvq-qpn.cuh`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-qpn.cuh), [`mmvq-tc.cu`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-tc.cu), [`mma.cuh`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mma.cuh), [`fattn-verify-stream.cuh`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/fattn-verify-stream.cuh) and [`qsa-attn.cu`](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/qsa-attn.cu).

Superfast is a **different engine/model** (llama.cpp-derived Qwen3.8-27B and Qwen3.8-Flash-Next). Its README reports approximately **924–935 tok/s cold 16K prompt** and **738–742 tok/s cold 64K prompt** on its V100 for the 27B, but those figures reflect different weights, architecture, runtime and input. **They are not DeepSeek targets or speedup ratios**. Its per-round QPN routing table supplies narrower evidence: on measured Qwen shapes, e.g. Q5_K FFN gate/up (`N=17408,K=5120`) 6.32→4.50 ms and Q4_K same shape 2.29→1.44 ms; other projections **lost** with QPN and are deliberately excluded.

| Source implementation / evidence | Implemented or reported in that source | Relevance to DeepSeek Abyss | Decision |
|---|---|---|---|
| `mmvq-qpn.cu`, `mmvq-qpn.cuh` | Static K-quant/IQ4/Q8 matrices repacked at load into **32 output rows × 256 input columns** superblocks in SM70 MMA884 fragment order; 16-byte lane/coalesced reads, 32-row-per-warp, split-K, no shared-memory weight staging; same packed byte count | **Potentially high** for eligible *static quantized dense* matrices, and maybe stable/hot GPU-resident expert slots | **P3 pilot**, after actual GGUF type/shape inventory; no blind “repack all” |
| QPN supported types | **Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q3_K, Q8_0**; **not** Q2_K, IQ2*, IQ3*, MXFP4 | DeepSeek UD-Q2_K_XL may use unsupported types extensively; two layers are documented with MXFP4 expert-down rows | **First measure bytes, calls and time by real type**; adding new decoders is a separate gated R&D item |
| QPN arithmetic | Quantized 4/5/6-bit codes unpacked in registers using bit operations/FP16 half2 arithmetic; scale/min terms applied at FP32 superblock boundaries; activations optionally power-of-two scaled, with **nonfinite fallback** | Could reduce repeated quantized row loads/dequant work in 1–16-token GEMV; precision differs from q8_1 dot | **Prototype with native quant format and independent FP32/FP64 tests**; do not assume exact-output identity |
| QPN dispatch | Chosen by **type and specific `(N,K)`**, run width and input layout; QPN up to **64 tokens**, passes of 8 or 16, then dequant-F16/cuBLAS | The important lesson is **route narrow winners only**, not transplant Qwen shape tables | **Shape/quant/width-level A/B routing** on Abyss actual workload |
| QPN input reuse / grouped products | Prepares FP16 fragment activations once, shares them across same-input projections, groups two or three compatible GEMVs in one launch | DeepSeek has same-input projections and gate/up components, but actual layout/lifetimes differ | **Second-stage P3 candidate** after single-product validation, never group across distinct activations |
| `mmvq-tc.cu` non-repacked path | Dedicated **Q5_K**, **3–8 tokens**, output-head-like **≥65,536 rows** only; *smaller* products on V100 measured slower | May matter for **large quantized DeepSeek vocabulary head during verification**, **if** its actual type/shape matches | **Low-cost targeted microbenchmark**; not a general expert/decode replacement |
| `mma.cuh` | Exposes documented Volta four-quad-pair MMA layouts with `I_MAJOR_MIRRORED` input layout | Independent register-map cross-check for Strata's MMA884 work | **Use as a second reference** in P1 and P2, not as a wholesale GGML dependency |
| `fattn-verify-stream.cuh` | Specialized **D=256**, Qwen GQA=6, 1–16-token verify: QK warps load keys, PV warps compute previous tile concurrently; K staging ahead and separate QK/PV warp groups | **Scheduling technique only**. DeepSeek D=512, shared latent K=V, sinks, raw/CSA/HCA gathers, shorter *visible* attention budget | **Optional P2/P4 experiment after the correct simpler kernel**; no direct port or assumed long-history win |
| `qsa-attn.cu` | Direct reads from selected KV slots (no gathered cache copy); splits small workloads to enough CTAs; split-softmax combine | Abyss *already* gathers CSA keys directly and decode attention already uses split/combine | **Retain Abyss design**; test adaptive splits/CTA count only if underoccupied |
| Superfast QSA MMA selector | `LLAMA_QSA_ATTN_MMA=1` enables an alternative Volta MMA prompt path, explicitly **off by default because slower on tested Qwen setup** | Strong caution about P2's expected payoff | P2 must earn `auto` routing **against existing FP32 direct-gather kernel** |
| Attention mask on GPU | Superfast reports **1.5 ms/verify round** improvement by avoiding a host→GPU mask copy | Abyss has different mask/selection construction | Inventory current Abyss H2D transfers; optimize **only if repeated mask/metadata copies exist** |
| Speculative model and cadence | MTP/DFlash2, trained/ranked draft vocabulary, rejection sampling, adaptive 7-token window and reusable verification | Current Abyss DeepSeek GGUF **has no MTP/DSpark head** and uses n-gram speculation with a shorter verify width | **Do not port drafter/vocabulary**. Use the **acceptance-conditioned cost model** as a P4 profiling/selection idea |
| GDN, fused recurrence, multigpu, IOMMU, `GGML_CUDA_CUB_3DOT2` | Qwen GDN-specific kernels; P2P setting for two GPUs; CUB build compatibility option | Not a DeepSeek V100 arithmetic optimization on this single-GPU host | **Out of scope** for the first release |

**Most consequential caveat:** QPN *repacking* is **not a drop-in kernel choice**. Superfast tags repacked GGML weights with `GGML_TENSOR_FLAG_BACKEND_LAYOUT` and diverts **every later reader** to QPN or a repack-aware dequant-to-FP16 path; other reads are forbidden. Abyss's dynamic expert tiers, independent CPU lane, regular prefill MMQ, staging buffers and eviction/promotion policies currently expect native GGUF-packed bytes. Repacking whole expert tensors and then copying them to pinned RAM/NVMe could corrupt other consumers, increase total memory, or make churn disproportionately expensive. The first P3 candidate is **static dense weights only**; expert copies require the explicit **dual-format/slot-lifetime plan** below.

**Decision on scheduling:** P0 must identify whether (a) **long-context indexer**, (b) **attention**, (c) **quantized projection/GEMV**, or (d) **expert/PCIe/CPU stalls** dominates. P1 and P3 prototypes may run in parallel after P0; P2 remains a measured, optional optimization. Superfast's high end-to-end performance is not evidence that QPN is the right first port for a 284B offloaded DeepSeek model.

---

### Source-specific starting point

The following are supported by the Abyss v1.3.28 code and repository documentation; exact line numbers can drift after rebases.

| Area | Existing behavior on V100 | Primary source |
|---|---|---|
| Build/device admission | CUDA 12.x and compute capability 7.0 are permitted; V100 runtime is untested by upstream | `CMakeLists.txt`, `src/core/device.cu`, `docs/MANUAL.md` |
| Single-token lightning indexer | `idx_score_kernel` (FP32) instead of `idx_score_mma_kernel` (SM80) | `src/kernels/cuda/dsv4_fast.cu` |
| Batched lightning indexer | `idx_score_batch_kernel`: FP32, **32 tokens × 32 compressed rows**, 64 heads; SM80 uses `idx_score_batch_mma_kernel` | `src/kernels/cuda/dsv4_fast.cu` |
| Early decode-indexer skip | No score/top-k kernel when all visible compressed rows fit in top-512, unless explicitly overridden | `src/core/dsv4_fast_path.cu`; `STRATA_DS4_IDX_ALL` |
| Batched attention | `attend_batch_kernel` FP32 (8 heads/block, 32 keys/tile); SM80 uses `attend_batch_mma_kernel` (16 heads/block) | `src/kernels/cuda/dsv4_fast.cu` |
| Decode attention | `attend_split_kernel` + `attend_combine_kernel` FP32 on all architectures | `src/kernels/cuda/dsv4_fast.cu` |
| Expert compute | Quantized decode kernels and batched GGML MMQ; pre-Blackwell MXFP4 activation route uses Q8_1 | `src/kernels/cuda/glm_fast.cu`, `src/prefill/moe_mmq.cu` |
| BF16-weight prompt GEMM | Some small paths widen BF16 to F32 per call; large projections dequantize to F16 and use cuBLAS; indexer projection weight has an F32 cached copy | `src/core/glm_prefill.cu` |
| Memory/offload | GPU experts, pinned RAM, NVMe, CPU lane, background promotion/prestage, two-socket NUMA behavior | `src/core/glm_fast_path.cu`, `docs/MANUAL.md` |

**Important evidence distinctions:** Abyss README reports 11,008 routed experts warm in approximately **23 GB VRAM + 70 GB pinned RAM** on its reference system, and a 3,000-token logits comparison to llama.cpp with KL ~0.009 and top-1 agreement ~94.5%. These are **external reference-machine observations**, not measured V100 acceptance results. For long-context attention, Abyss's RTX 5090 timings likewise provide a motivation for optimization, **not** a transfer speedup estimate.

---

## 0B. External SM70 research retained in the actionable plan

These are implementation references for **exact model math or the missing weight types**, rather than evidence of single-V100 Abyss throughput. The accompanying external-optimization research addendum contains further citations and rejected candidates; the operative requirements below are self-contained.

| Reference | What exists | Abyss task |
|---|---|---|
| [`anyei/llamacpp-v100` `mmsmt.cu`](https://github.com/anyei/llamacpp-v100/blob/master/ggml/src/ggml-cuda/mmsmt.cu) and [design notes](https://github.com/anyei/llamacpp-v100/blob/master/docs/mmsmt-implementation.md) | SM70 `m8n8k4` small-token MMA including **Q2_K**, plus source-reported numerical hazards involving all-zero superblocks and `dmin/d` | **P3.5d**, compare with native `glm_fast` on actual Q2_K `(N,K,T)`; add zero-scale quality cases |
| [`1Cat-vLLM` DeepSeek indexer](https://github.com/1CatAI/1Cat-vLLM/blob/main/vllm/models/deepseek_v4/sm70/indexer.py) | Exact headwise ReLU, fused decode scoring, shared key gather plus cuBLAS batch score in selected shapes | **P1e**, separate MMA884/FP32/cuBLAS batched routes; keep FP32 score scratch at cutoff |
| [`1Cat-vLLM` MXFP4](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu), [grouped MoE](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu) | SM70 native MXFP4 E2M1 and E8M0 scale-per-32, grouped/skinny MoE MMA | **P3.5e**, *only* on matching MXFP4 resident rows after pack audit; neither NVFP4-only nor Q2_K decoder is interchangeable |
| [`1Cat` fixed-shape FP16 GEMV study](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/sm70_deepseek_v4_fp16_gemv.md) | Single-row GEMV shape tuning, graph-level verification; large apparent isolated wins sometimes small in end-to-end results | **P3.5f**, compare actual Abyss dense/router/compressor shapes only if profiling shows critical time |
| [`anyei` GPU/CPU MoE doorbell](https://github.com/anyei/llamacpp-v100/blob/master/src/llama-moe-doorbell.h) | Pinned mailbox posts input, expert IDs, weights and partial results while GPU processes hot experts | **P4g**, study handoff synchronization, not a wholesale replacement of Abyss's working CPU lane |
| [Volta sparse attention implementations](https://github.com/jmnargi/Strata-V100/blob/main/src/kernels/cuda/qsa_prompt_attn.cu) and [Superfast streaming verify](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/main/ggml/src/ggml-cuda/fattn-verify-stream.cuh) | MMA884 quadpairs, separated QK/PV warps, online softmax, split/reduce | **P2** optional schedule only after DeepSeek 512-D/sink/r0/r4/r128 reference passes |

Do not port an absent DeepSeek drafter (DSpark/MTP), Qwen-specific GDN, multi-GPU TP, or offload engines in the initial scope. Reproducing source-reported performance on their specific machines is not a substitute for same-pack Abyss A/B.

---

## 0C. Xolotl / Strata-DS4 cross-audit (v7): what to test, what NOT to port

**Source and provenance.** Xolotl (revision `f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16`) contains an experimental [DeepSeek-V4-Flash-0731 engine](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/README.ds4.md) implemented in [`tools/ds4/ds4_dense.cpp`](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_dense.cpp), [`ds4_moe.cpp`](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_moe.cpp), and [`ds4_generate.cpp`](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_generate.cpp); experiment evidence is in [`docs/ds4/FINDINGS.md`](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/docs/ds4/FINDINGS.md), especially §§12, 27–37. Its measured hardware was **RTX 4090 24 GB, Ryzen 7 5700X, ~90 GB DDR4**, and its `DeepSeek-V4-Flash-Q2-0731.gguf` was **80.8 GiB**. Abyss is pinned to a different **~96.8 GB Unsloth UD-Q2_K_XL GGUF** on **V100 32 GB + two Broadwell CPUs**, with *slower PCIe 3.0*. Xolotl and Abyss share the underlying **DeepSeek-V4-Flash-0731 architecture**, but their quant types, runtime design and hardware differ. **No Xolotl V100 whole-model decode/prefill result was established in this audit.** Xolotl's general engine offers *experimental* CUDA `sm_70` compilation (`STRATA_EXPERIMENTAL_SM60`), not proof that the DS4 0731 engine runs correctly or quickly on V100.

### Source-to-Abyss decision matrix

| Xolotl finding (its tested machine and workload) | Abyss overlap or missing opportunity | Decision and gate |
|---|---|---|
| **Exclusive layer-local VRAM LRU** raised a source-reported cache hit rate ~70→84%; cache hit reductions corresponded to substantial decode improvements | Abyss already has **per-layer victim partitions**, background promotion, LFU/adaptation and pinned service ring; a replacement policy alone could help, but Xolotl's speed gain cannot be assumed | **P3.2c**, benchmark admission+victim policy using same routes and fixed slot budget; ensure slot epochs, no in-flight victim and exact results |
| **Predictive prefetch** of likely misses in a second CUDA stream during current-layer attention; Xolotl reports ~62–70% candidate recall and a nonmonotonic prefetch optimum | Abyss already has **prestage/promotion**, GPU-owned routing and a CPU service thread; copying Xolotl's predictor can introduce *extra* CPU readbacks and duplicate DMA | **P3.2d**, capture routing prediction precision, issued/useful/wasted bytes, overlap and PCIe/DRAM contention; evaluate prediction only if an available earlier hidden state genuinely predicts current route |
| `--arena-skip-resident` places GPU-seeded experts outside the pinned host arena, freeing RAM; `--arena-adapt` lazily admits experts from SSD | Abyss RAM/VMM/canonical-source accounting differs; 128 GB host total is not equal to pinned pool; removing host canonical copies may prevent immediate CPU execution after a GPU slot eviction | **P3.2c**, compare disjoint vs duplicated **resident** copies; preserve GGUF canonical bytes and enforce miss fallback, NUMA, pin budget and page-cache impact |
| **Quantized CSA/HCA compressed KV**: Q8_0 and rotated IQ4_NL + indexer Q8 yielded source-reported ~0.54% and ~0.60% *perplexity increases vs FP32*, while reducing compressed KV bytes to ~27% and ~14% of Xolotl's FP32 baseline | Abyss **already uses FP16** compressed caches; expected incremental savings, decode cost and quality **differ**; `comp` attention values and `icomp` lightning-indexer keys have distinct consumers | **P2f**, staged FP16→Q8_0 then optionally rotated IQ4_NL; independent attention/indexer cache formats; precise top-k/near-tie and RoPE parity; long-context cost/slot gains, never claim numerical exactness for lossy caches |
| At 256K `--comp-host` (mapped pinned host reads) gave ~62 extra slots but **~3 ms attention cost/token** in source tests | Over **PCIe 3.0**, mapped host gathers may be even less attractive; copy/gather strategy and CPU DRAM contention matter | **P2f only at true VRAM budget pressure**, compare resident quantized cache against host/mixed sparse gather; never enable simply to save bytes |
| **Fused lightning indexer** replaced ~9 ggml operations and avoided two `[cap,n,64]` copies; optimized sorting only selected top-512 rows | Abyss already has **native fused indexer score/selection** and **direct selected-row attention**; Xolotl's eliminated ggml intermediates are not necessarily in Abyss | **P0 audit first; no blind port**. Check only whether Abyss has an O(cap) sort/mask/allocation or excessive scratch at long context; retain exact headwise ReLU and indexer top-k order |
| **Chunked prefill** streams a layer's experts once per chunk, overlaps next-layer staging, uses MMQ; experiment includes query-blocked raw SWA window | Abyss already has a dedicated batched DeepSeek prefill path, quantized MMQ and staged experts; Xolotl's graph implementation is not drop-in | **P3.10/P2g**, run chunk/token-width, staging and actual quant-route sweeps; borrow raw-window *query-block* idea only if Abyss processes a larger masked union than necessary |
| **Shared graph compute arena** and graph reuse saved source-reported ~850 MiB and ~120 expert slots; Xolotl has ggml-specific graph UIDs/allocator epochs | Abyss uses custom CUDA streams/kernels and its own allocation layout; an identical ggml allocator memory leak/cost may **not exist** | **P4h**, inventory actual graphs and workspaces; implement pointer-stable reuse only if a measurable lifetime conflict or overhead exists; stress graph recapture and memory epochs |
| Xolotl's Nsight Systems ledger identifies **~4,300 tiny kernels/token**, but only ~2–3 ms of dense-side launch/sync gaps; same-input `q8_1` conversions repeatedly executed | Abyss fast decode path documents **~11 fused launches/layer**, not the Xolotl ggml operation count; launch fusion could be a regression on V100 if it increases register pressure | **P3.11**, profile actual Abyss small-operation and repeated activation-quantization cost; reuse a valid activation only where bit representation, lifetime and layout match |
| **REAP hard/soft expert pruning, cache-aware route bias, skip-miss** | Alters true routed expert selection or availability; source itself reports material multilingual perplexity failures in hard-prune trials | **Defer lossy methods to a separately authorized quality-focused project**; do not mix with exact baseline or attribute their gains to V100 kernels |
| **MTP, DSpark and 552B DeepSeek-V4.1 features** | Different GGUF assets, verifier semantics, model family or unverified learned-draft weights | **Out of scope** for this DeepSeek-0731 exact-quality port; keep Abyss n-gram speculation first |

**Important Xolotl evidence qualification:** The source's **`--pos-offset` 128K/256K “decode” probes run over unpopulated caches**, with short-context expert hit rates, and are **upper bounds** on what a genuinely filled long document delivers. Do not use them as real-document tokens/s targets. In Xolotl's own 128K real-document runs, expert hit rate and long-context prefill were considerably worse than the empty-cache probe implied. Its reported per-model perplexity variations also depend on CPU/GPU quantized-activation choices. Benchmarks of altered caches or schedules must hold *both weights and expert execution placement* constant to isolate numerics, or separately report the unavoidable lane effect.

### Ordering and ownership

- **Do not restart the engine port from Xolotl.** Keep Abyss pinned base and canonical model, existing CPU service path, `dsv4_fast` attention kernels, and V100 fallback. Xolotl additions are **independent optional experiment lanes**; no dependency on Xolotl's `ggml` allocator internals or the RTX 4090 CUDA architecture.
- **Perform the cheap no-code diagnostics first:** existing resident GPU/host expert duplication; cache misses/promotions by layer; prefetch usefulness and traffic; duplicate activation conversions; actual long-context indexer scratch/selected-key access; graph workspace bytes; and chunked prefill copy/gemm overlap.
- **Priority is data-driven:** on a memory/PCIe-bound V100, avoiding one expert transfer or regaining a useful resident slot may beat an isolated Tensor Core speedup. Conversely, if existing Abyss caching is already efficient, do not spend P3 months reproducing Xolotl's different allocator/cache design.

---

## 0D. Post-Abyss upstream Strata delta review (v8): actionable changes and scope exclusions

**Provenance and boundary.** Abyss was published as a new, non-GitHub-fork repository on **10 October 2026**; its imported initial `CMakeLists.txt` identifies Strata **0.1.27**, but the histories are not a common-ancestor compare on GitHub. The upstream Strata compare from `cf09057a867d5be85389ecf459afe4a758cdc873` (10 October) to `477ac13cbe8e582c20484871ce459d20d82df09d` (16 October) returned **650 commits** (including merges). This is a source-selection interval, **not proof that every commit is absent from Abyss**. The implementation baseline for this plan remains Abyss `47001f31` v1.3.28; use current Abyss `7c893ac` v1.3.31 only as a **separate, more recent feature-check** and rebase candidate. At each PR, recheck source against the actual working branch and reject duplicate work. No upstream Strata commits or GPU benchmarks were executed as part of this review.

**Decision matrix (the exact reference features are upstream-only unless Abyss's selected branch independently implements them):**

| Upstream change / pinned reference | Source-supported mechanism | Specific applicability to DeepSeek-V4-Flash UD-Q2_K_XL on SM70 | Decision |
|---|---|---|---|
| [14 Oct `c675f5b`](https://github.com/Niko1221/Strata/commit/c675f5b78725): `quantize_scatter_mmq_q8_1_cuda` | Quantize each token's MoE activation **once**, scatter identical `q8_1` bytes to the token/expert destination rows; upstream test reports byte-identical layouts | Abyss [`src/prefill/moe_mmq.cu`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/prefill/moe_mmq.cu) still invokes the indexed `quantize_mmq_q8_1_cuda`. Highest-confidence independent prefill experiment, after checking vendored ggml API, actual repeated activations and MXFP4 path | **P3.12, early, exact-output target** |
| [14 Oct `8ee2005`](https://github.com/Niko1221/Strata/commit/8ee2005dae0b), [`77ddc2f`](https://github.com/Niko1221/Strata/commit/77ddc2ff147d) | ggml-AVX2-bit-exact **singleton** CPU expert rows with IQ2_XXS/IQ2_S coverage, CPU ISA and token-width guards | The target CPU has AVX2/FMA, not AVX-512; the main DeepSeek pack uses IQ2_XXS gate/up. Abyss has a **different** exact-singleton optimization mainly for IQ3; do not assume this upstream kernel wins or can be directly substituted | **P3.2f, high, compare only** |
| [14 Oct `04be241`](https://github.com/Niko1221/Strata/commit/04be241a5a7e): `STRATA_PCIE_BALANCE=1` | Fit measured per-layer GPU PCIe service and CPU pool time to choose the miss split online; accounts for CPU/PCIe contention | Abyss already **startup-calibrates** CPU workers, DMA and mapped zero-copy. Test whether dynamic changes in occupancy/NUMA/CPU load make that calibration stale; the Qwen implementation and precise cost model are not drop-in | **P4i, high after stable P0** |
| [14 Oct `7d9b50d`](https://github.com/Niko1221/Strata/commit/7d9b50dc1a5c), [`86f677b`](https://github.com/Niko1221/Strata/commit/86f677b4a976): `STRATA_MMQ_DIRECT` | Pointer-table MMQ reads **already-resident cache-slot** expert blobs without a contiguous weight gather, with per-slot guard/tail/lifetime checks | Supported upstream table types include IQ2_XXS but **do not include Q2_K or MXFP4**; target model has Q2_K down and some MXFP4 layers. Need format-by-format partial-gather/partial-direct gate, no unguarded overread | **P3.13, optional and late** |
| [14 Oct `c675f5b`](https://github.com/Niko1221/Strata/commit/c675f5b78725), [`62fc7c8`](https://github.com/Niko1221/Strata/commit/62fc7c8a3c98) | Separate prefill MMQ group ordering to reduce padding, preserves partial-group order/bitwise results under limited shapes | Possible only when real target geometry, launch tiles, group-size and SM count pass upstream's whole-tile predicate. Avoid gratuitous reordering of hash layers or changing route weights | **P3.13 appendix, optional** |
| [12 Oct `b60aab1`](https://github.com/Niko1221/Strata/commit/b60aab1ba27c): `STRATA_HC_Q8` | Store selected dense hyper-connection projection weights in Q8_0 and dequantize into per-stream scratch when needed | Only if target tensor census shows significant **separately resident** BF16/F16 HC weights and the code consumes them in an equivalent form. Quantization is lossy; old BF16/FP16 expansion may have its own error already | **P3.14, conditional/opt-in** |
| [12–13 Oct `661ff29`](https://github.com/Niko1221/Strata/commit/661ff2933941), [`ab6bce0`](https://github.com/Niko1221/Strata/commit/ab6bce05de44), [`f087f78`](https://github.com/Niko1221/Strata/commit/f087f780ed85) | Async GPU↔RAM expert swaps; selective pinned ranges; pinned expert-profile ranking | Relevant *ideas* for bounded DMA overlap, RAM pin capacity, and workload dynamics, but exact Strata source treats a different Qwen cache/expert residency model; overlap had **device-dependent regressions**, and Linux 128-GB baseline may already pin enough | **P4j, only after measured gaps** |
| [`EXCHANGE_ROTATION.md`](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/EXCHANGE_ROTATION.md) | Swap host buffer ownership instead of copying demoted expert bytes back into a resident slot | Useful only where Abyss **actually** performs that host commit memcpy under mutable host-resident complement semantics; needs pinned capacity, pointer/alias lifetime and mixed-size slots | **P4j, conditional** |
| [13 Oct `640ac43`](https://github.com/Niko1221/Strata/commit/640ac438be9e): `--prefill-head-alias` | Temporarily lend output-head VRAM to prompt buffers while reading a prompt, restore head before next decode | Only when P0 shows output-head memory causes a smaller chunk or fewer cache slots; on single V100, reload bytes over PCIe 3 and TTFT can outweigh larger chunks | **P4k, optional** |
| [12 Oct `6b39557`](https://github.com/Niko1221/Strata/commit/6b395578a222): conditional THP | Avoid hugepage request if available 2-MiB free blocks are inadequate, preventing costly direct memory compaction | **Already addressed in latest Abyss** `STRATA_GLM_RAM_THP` / `numa_pinned` in `src/core/glm_fast_path.cu`, but not necessarily in pinned v1.3.28. Check/cherry-pick latest Abyss independently; **do not reimplement older upstream THP patch** | **P0 upstream-delta baseline check** |
| [11 Oct `003ea8c`](https://github.com/Niko1221/Strata/commit/003ea8c678): NaN guard | Abort/recover from nonfinite head results rather than poison conversation cache | Error-handling principle applies if DeepSeek rollback/snapshot can be corrupted; requires explicit DeepSeek snapshot and exact forward rebuild tests, no performance assertion | **P4 safety audit, optional** |

**Negative scope/duplication decisions.** Upstream's `STRATA_PROMPT_ATTN_OLD`, `STRATA_BF16_TC`, `STRATA_SM70_TABLE` and corresponding measured Volta Qwen3.8 kernels **predate the October 10 repository creation** and already inform P1/P2/P3. They do not directly implement the distinct **DeepSeek-0731 latent K=V, CSA/HCA sinks and lightning indexer**; continue model-specific tests. Strata's native NVFP4 and Qwen3.6/GDN work must not be mistaken for **MXFP4 E2M1 + E8M0 scales/32**, and its Q2_0 experts are not **Q2_K**. Turing `sm_75` optimizations, CUDA `sm_86` tables, AMD, Metal, SYCL, multi-GPU, vision and new model support do **not** warrant ports for this one-V100 engine. Upstream `STRATA_ROUTE_TAIL_SKIP` / router bias are **lossy** and stay out of the exact-quality default.

**Version-bounded safety note.** Recent *Abyss*, not upstream Strata, added `STRATA_GLM_RAM_THP` (hugepage NUMA residency) and `STRATA_GLM_STREAM_FETCH` (streaming disk-landed expert fetch) in its October 18 commits. In the implementation branch, record whether these features exist and are enabled **before** P3 or P4 experiments. Incorporate only via an explicit Abyss-base update/change review, a second P0 baseline and the v6 page-placement and thermal gates. Never claim those newer Abyss features are present in pinned v1.3.28 merely because they appear on current `main`.

**Audit boundary:** The GitHub commit-compare count is a history count, not an exhaustively audited list of all 650 changes and not an admissible causal speed attribution. This table identifies source-inspected candidates; expected gains are **unknown** on the requested V100 DeepSeek workload. Stop a redundant or inapplicable candidate with a short evidence note, rather than modifying code to fulfill the checklist.


## P0 — Working V100 inference, runtime correctness and bottleneck ledger

**Gate:** Establish a stable, correct **actual V100** fallback before merging P1/P2 into the production branch. Work on isolated arithmetic microtests may proceed in parallel; production integration cannot bypass this gate.

### P0.1 Reproducible build

Fetch Abyss and its pinned third-party `llama.cpp` dependency (the setup wrapper may populate it; ensure the required checkout exists before manual CMake). Check the actual CUDA compiler at `/usr/local/cuda-12.8/bin/nvcc`, or substitute its detected path.

```bash
git checkout 47001f31cbfba14bcbf4c083e3af07c50707ec0c

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

Alternative: `./setup-deepseek.sh --nvcc /path/to/cuda12/nvcc --arch 70`. The setup-generated configuration may refer to `build/strata`; if using a separate build directory, correct its executable (`exe`) field. **Do not run a CUDA-13 `nvcc` for SM70.** Inspect `cuobjdump`/`nvdisasm` on new V100 kernels before first launch.

### P0.2 Required bring-up checks

1. Record `git rev-parse HEAD`, `nvcc --version`, driver version, device compute capability, `nvidia-smi -q` (actual PCIe negotiated width and generation), `nvidia-smi topo -m`, `lscpu`, and `numactl -H` (free bytes by node).
2. Startup with the **shipped** UD-Q2_K_XL pack, no re-quantization. Log actual GPU dense/expert buffers, pinned RAM allocation, host memory pressure, cache and workspaces, and any NVMe misses. **New:** export a weight-role/type/shape inventory for every GPU-called matrix (indexer and Q/O projections, router, shared expert, vocabulary output, routed expert gate/up/down), with quant format (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, IQ types, MXFP4, BF16, etc.), bytes, `(N,K)`, token widths, calls/token, residency source and existing decode/prefill kernel. The model label `UD-Q2_K_XL` is **not** proof of a particular per-tensor type distribution.
3. Complete one-token, 256-token and 1,024-token generations; small and large prompt prefills. No illegal-instruction, pointer/memory fault, OOM, incorrect shape or missing CUDA symbol.
4. Verify **MXFP4-down rows** on both affected layers in **decode and batched prefill**. Confirm the pre-Blackwell/Q8_1 activation path is linked and produces correct results.
5. Use Nsight Systems to list launched kernels. Confirm DeepSeek uses the FP32 `idx_score*` / `attend_batch*` branches on SM70. Test whether any `qsa_*` kernels are genuinely launched before considering them for V100 work.
6. Reference parity using the same model and input token IDs against Abyss's pinned llama.cpp (`tools/dsv4_logits_dump.cpp` or current equivalent); where possible, also compare a known-good Abyss GPU run. Include boundary tokens for raw window 128, compression 4 and 128, top-512 visibility cutoff (approximately 2,048 source tokens), chunk boundaries and speculative rollback.
7. Verify cold start, warm start, long prompt then immediate decode, and reused-cache session correctness. Record background promotion and CPU-lane behavior separately.

### P0.2c — Baseline ancestry/delta control (new v8 acceptance prerequisite)

Produce `results/p0_upstream_delta.json` with (a) working Abyss commit / `git describe`; (b) tested Strata source SHA `477ac13`; (c) CUDA / ggml vendor SHA and toolchain; (d) exact `UD-Q2_K_XL` shard SHA/quant inventory; (e) for each candidate, `present|absent|different|inapplicable` on **the branch actually built**, with source file/symbol; and (f) separately note whether later Abyss `STRATA_GLM_RAM_THP` and `STRATA_GLM_STREAM_FETCH` have been included. When changing the Abyss base after v1.3.28, run the full P0 parity and timing suite anew; do not compare across moving baselines. Inventory actual expert gate/up/down formats and mixed MXFP4 layers; do not infer every layer's types from the quant name.

**P0 targeted measurements:** Count duplicate quantization of identical input rows per token × routed expert for *decode and prefill separately*; measure existing `glm_prefill.cu` MMQ quantization duration/bytes; isolate CPU IQ2_XXS `ms/expert` against default ggml dot on the Broadwell CPU; and compare startup-calibrated CPU/PCIe completion predictions with **observed per-layer** CPU completion, PCIe and GPU-wait time after changing load/route distribution. Record `results/p0_strata_delta_candidates.csv`. No new GPU kernels or adaptive policies until the candidate exists in the shipped path and is shown to matter.

### P0.3 Instrumentation

**Proposed new** `ABYSS_V100_PROFILE=1` wraps or extends existing `STRATA_GLM_PREFILL_PROF=1` and `STRATA_GLM_GAP_PROF=1`. Produce per-stage time and count metrics:

- Attention-side: Q projections, indexer Q preparation, indexer score, top-k selection, raw/CSA/HCA batch attention, compressor writes, normalization and output projection. For every matrix operation log **weight type `(N,K)`, token width, route and calls**, so QPN applicability can be screened without conjecture.
- MoE-side: router, shared expert, routed GPU GEMV/MMQ, CPU-lane work, host↔device copies, prefetch/prestage, expert promotions/demotions, SSD misses and waits; add **expert-slot residence lifetime / promotions per 1,000 tokens** for amortization calculations.
- Host/device coordination: CUDA stream events, CPU wall, synchronization, time between tokens, speculative acceptance and verification.
- **Keep GPU kernel milliseconds, host wall milliseconds, prefill tokens/s, decode tokens/s and time-to-first-token distinct**. Event timing across concurrent CPU/GPU work is not simply additive; use a critical-path trace, not only stage sums.

**P0.3a — Mandatory Volta hardware-counter profile, toolchain and bottleneck classification.** Execute before deciding whether P1, P2 or P3 merits first engineering effort.

1. **Tool versions:** Record `ncu --version`, `nsys --version`, `nvcc --version`. For V100 **pin Nsight Compute 2025.1 (CUDA 12.8 bundle) or verified 2025.2**: NVIDIA [2025.1 support](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1) includes Volta, while [2025.3 explicitly drops it](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3). Do not use a newer `ncu` and assume it can collect sm_70 counters. Use **Nsight Systems** for CPU/GPU concurrency and launches; `nvprof` is deprecated legacy, not the prescribed P0 profiler. If an isolated kernel cannot be profiled, use `ptxas` + CUDA events + architecture-specific supported metrics and **mark missing hardware evidence**.
2. **Profiler provenance and retention:** Before any hardware-counter campaign, verify that the selected **2025.1/2025.2** `ncu` executable actually profiles a trivial **SM70** kernel on the target driver; record its exact version, executable/container digest, SHA-256, source/license, compatible driver and access mechanism in `results/toolchain_manifest.json`. Keep an approved copy in a **private, access-controlled binary artifact store**, or build a runner image from **legitimately acquired vendor toolkit media when the license permits**. Do **not** commit redistributability-uncleared NVIDIA tools to public source control or assume older binaries can be downloaded anonymously in CI. The GPU runner must fail loudly rather than silently substituting a 2025.3+ build. Compile-only CI should still run without `ncu`; its absence is recorded as a **GPU-counter-gate blocker**, not a compilation failure.
3. **GPU metrics:** For representative isolated **decode T=1**, **verify T=2/3**, and **prefill batch** shapes collect kernel duration, registers/thread, static/dynamic shared bytes, configured and achieved active warps, issued/active instruction throughput (including HMMA where supported), local-memory load/store transactions, HBM2 DRAM bytes/bandwidth, L2 bytes/hit rate, stall reasons and memory/scheduler statistics. Use version-supported metric names discovered by `ncu --query-metrics` or compatible `--section` queries; **do not hardcode Ampere-only counters**. Save complete reports with command line and kernel-selection filter.
4. **CPU/PCIe metrics:** Measure actual negotiated PCIe gen/width, host-mapped reads, pageable and pinned H2D/D2H, memcpy setup/latency at **sub-64-KiB** and MB sizes, pinned host read bandwidth, per-NUMA-node DRAM read rates, CPU lane worker utilization and SM idle gaps. Isolate the GPU and CPU expert lanes rather than summing overlapping work. Include cold promotion and warm pinned expert runs.
5. **Bottleneck classification:** Label each important stage **HBM2 bandwidth / PCIe-transfer-or-mapped-access / CPU-DRAM / SM compute-or-instruction / register-or-shared-occupancy / launch-or-sync / mixed**. A roofline showing high DRAM saturation **rules out compute-only rewrites that cannot change data movement**, but **does not** rule out a smaller/fused/reused-weight kernel that moves fewer bytes, nor HBM optimizations that overlap waits. Record `bytes/token`, realized GB/s against **measured sustainable bandwidth**, and whether the stage sits on the overlapped critical path. Never infer 'memory-bound' solely from low tensor-core utilization.
6. **Resource budget:** SM70 [hardware](https://docs.nvidia.com/cuda/volta-tuning-guide/) has 65,536 × 32-bit registers/SM, 255 registers/thread maximum, up to 64 warps/SM and 96 KiB shared memory/SM. Dynamic shared memory above the usual 48-KiB per-block static compatibility limit requires a supported opt-in and an occupancy check. Compare two CTA sizes and at least two tile/split-K geometries when resources are limiting; record local-memory **traffic**, not just compilation-reported spill slots.

Example **commands to adapt to the selected benchmark**, not verbatim production launch settings:

```bash
ncu --version    # use Volta-compatible 2025.1/2025.2 only
ncu --query-metrics > results/sm70-metrics-available.txt
ncu --set full --target-processes all -o results/p0_v100_ncu -- <isolated-kernel-benchmark> 
nsys profile --trace=cuda,nvtx,osrt -o results/p0_v100_nsys -- <baseline-run-command>
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
```

Counter replay changes execution; **measure production wall-time outside profiling**. Nsight Compute counter collection, especially multi-pass replay, cannot substitute for synchronized end-to-end timing.

**P0.3b — Sustained GPU thermal/power/clock control (mandatory measurement-quality gate).** The passively cooled 250-W V100-PCIe relies on adequate chassis airflow. Before performance attribution, measure clock/thermal behavior **for an entire warmed, sustained run**; do not infer throttling from a single P-state, a single instantaneous clock reading, or the statement that V100s are "notorious" for it.

1. Record GPU index/UUID, model, power cap, requested/application clocks (if configured), actual SM/memory clocks, GPU temperature, power draw, fan/airflow context if known and **NVML clock-event/throttle reasons**. On this driver, discover exact available query fields with `nvidia-smi --help-query-gpu`. Sample continuously (e.g. every 200–500 ms via `nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv -lms 500`, after checking property support), plus `nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE` or NVML/DCGM counters for *active* hardware/software thermal slowdown and power limit reasons. Persist time-stamped `results/p0_gpu_clock_power_thermal.csv` and throttle-event records; align with NVTX stages and kernel timings.
2. Use an A/B/A or ABBA interleaved run order, identical power limit/application clock settings, repeat after reaching a comparable thermal steady state, avoid simultaneous unrelated GPU workloads, and report observed **time distribution** of clocks/power/temperature alongside latency, not only averages. Sample warm/cold separately. If authorized, set **stable legal application clocks** only after confirming V100 support and matching thermal headroom; never change GPU power limits/thermal protections to manufacture a speedup.
3. **Classify, do not silently discard:** `HW Thermal Slowdown` or `SW Thermal Slowdown` during measured windows, or unexplained large clock collapse, invalidates a *clean performance attribution*; retain the raw run with `thermal_flag=true`, fix airflow/thermal state and repeat the matched pair. `SW Power Cap` may describe **normal** operation under a consistent power limit: retain as a separate power-limited regime, compare at matched power/clock conditions and report it; idle clocks/P0–P2 transitions outside timed kernels alone are not thermal proof. Maintain a thermal-flagged ledger instead of deleting results.
4. **Admission:** performance claims require a clean matched baseline/candidate pair or a clearly labeled matched power-limited pair, with a stated uncertainty/noise margin. A 20% regression cannot be blamed on throttling without an observed clock/temperature/reason trace. A kernel that genuinely raises power and triggers thermal limiting in the intended enclosure also has a **real sustained-service penalty**; it must not be promoted based solely on artificial locked-clock microbenchmarks.

Quick operational sketch (adjust fields for the installed driver, export timestamped data, stop the sampler after the run):

```bash
nvidia-smi --help-query-gpu | grep -Ei 'clock|temperature|power|pstate|throttle|event'
nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv -lms 500 \
  > results/p0_gpu_clock_power_thermal.csv
# A separate interval sample or NVML/DCGM collector is required for throttle-reason flags.
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE
```

### P0.4 Benchmark matrix and acceptance

- **Quick gate:** 2K, 8K and 32K token prompts; 256 output tokens. Repeat ≥5 warm times, record median and range. 64K/125K prompts and 1,024 output tokens belong to the extended suite after reliability is established.
- Cold startup: ≥3 independent restarts. Distinguish cache-cold, warmed resident experts, and full-context reuse.
- Keep identical GGUF hash, quantization, prompt IDs, context, CPU threads, NUMA, GPU thermals/clocks, power, speculation setting and background-memory policy across an A/B pair; alternate A/B order to reduce thermal drift.
- Evaluate logits using matching teacher-forced tokens. Reproduce README reference KL/top-1 numbers only if the **same document and sampling/scoring protocol** can be reconstructed. Otherwise document a new controlled reference. **Do not convert the README's 0.009 KL into an unconditional V100 pass/fail limit.**
- No performance improvement target for P0. Deliver `results/p0_v100_baseline.jsonl`, at least one annotated `nsys` trace, logit comparisons, memory ledger and the **top 5 critical-path components**. Also deliver **`results/p0_upstream_delta.json`**, **`results/p0_strata_delta_candidates.csv`**, **`results/toolchain_manifest.json`**, **`results/p0_gpu_clock_power_thermal.csv`**, active throttle-reason trace and **`results/p0_qpi_numa_placement.csv`** (or a documented hardware-counter-unavailable report). The ability to reproduce *clean sustained A/B conditions* is a P0 output, not an optional release-time note.

**Exit gate:** The real V100 completes correct DeepSeek generation and batch prefill, the baseline is repeatable, and the critical-path bottlenecks have a **source-tagged classification (compute, HBM, PCIe, CPU DRAM, launch/sync or mixed)**. Save `results/p0_tensor_inventory.csv` (`role,layer,type,N,K,bytes,token_width,invocations,device_ms,VRAM/RAM/NVMe,hotness,existing_kernel`), `results/p0_kernel_counters.csv`, `results/p0_pcie_numa.csv`, and both the wall-time and profiler runs. If counter collection was unavailable, explicitly mark the gate partial; do not invent a bottleneck label. Any fallback correctness failure must be isolated before shipping Tensor Core alternatives.

---

### P0.4a — Xolotl-inspired comparator ledger and controlled A/B fixtures

Add **source-independent Abyss measurements** *before* implementing new cache policies, quantized caches, prefetch, or graph reuse:

1. `results/p0_expert_trace.jsonl`: per `(layer, expert, token, context, CPU/GPU)` route ID, routing weight, GPU hit/miss, predicted/not-predicted, prefetch issue/completion/use, resident victim, promotion source, packed byte count, SSD fallback, slot epoch and CUDA/CPU timing; sample or aggregate without perturbing hot-path cost. Measure hit rate **both per layer and weighted by actual packed bytes/critical-path time**, including layer-local imbalance and long-document route drift. Retain privacy-safe synthetic/public benchmark prompts.
2. `results/p0_cache_residency.csv`: bytes in **GPU resident**, **CPU/pinned resident**, **host page cache / mapped GGUF**, **SSD file**, duplicated canonical/transformed copies, stage buffers, and CUDA graphs per context. “VRAM resident but not pinned RAM resident” does **not** mean inaccessible from canonical source: account for file read/page-cache cost and CPU fallback. Record promotion/eviction counts and lifetime before attempting repacking or disjoint placement.
3. `results/p0_prefetch.csv`: predictor lead time, eligible prediction source, precision/recall, `issued_MB`, `useful_MB`, `wasted_MB`, H2D throughput **alone vs concurrently with CPU expert reads**, stream overlap, CPU-readback latency and decode critical path. An oracle next-layer route trace can bound potential, but must **not** be used for live decisions.
4. `results/p0_graph_memory.csv`: peak/stable workspace allocations by purpose, number of CUDA graph variants and captures/recaptures, stable address requirements, fragmentation and changes in available **expert** VRAM slots as context grows. If the number of such ggml graph variants is zero in Abyss, mark the Xolotl-specific 850-MiB hypothesis **inapplicable**.
5. Paired workloads: 600-token short context, 8K, 32K, 64K and 128K **filled** real document where affordable; 1, 2–3 and 7–32 token verification widths where supported; cold and warmed expert pools; an **A→B→A topic-switch** route-reuse test. A `--pos-offset`-style empty-cache probe is permitted **only for kernel shape/attention-cost diagnosis**, never quality or total long-context throughput. Begin with short smoke/parity runs before full 128K profiling.
6. Every candidate gets the same **quantized model hash, context tokens, residency budget, CPU thread topology, prefetch state, ambient clocks/thermal conditions and number of generated tokens**. Run *exact routing and speculation off* for isolated tests. For intentionally lossy KV experiments, compare quality against identical full-precision/FP16-cache runs and account for differing placement (which may itself shift logits).

**P0.4a exit gate:** deliver a ranked **uncovered critical-path milliseconds** and a memory-savings→additional-slot **measured conversion**, not imported percentages from Xolotl's RTX 4090. P1/P2/P3 proposals must cite the trace row that justifies implementing them. Update the stop-rule table when P0 changes the bottleneck ordering.

## P1 — Volta Tensor Core lightning-indexer scoring (detailed)

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

**P1.1a — Inline PTX register/operand contract (hard arithmetic gate).** For `mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`, the packed pairs of F16 A/B operands are 32-bit bit-pattern carriers and should use an inline-assembly **`"r"` 32-bit register constraint** (two carriers per operand fragment), while each of the eight FP32 accumulator/destination registers uses **`"+f"`**. PTX `.f16x2` fragment storage may be represented in appropriately sized 32-bit containers; never substitute a float value for the packed half-bit-pattern or reinterpret a PTX `.f32` accumulator as `uint32_t` without a defined bitcast. Follow the [PTX `mma` fragment specification](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html) and [inline assembly constraint documentation](https://docs.nvidia.com/cuda/archive/12.2.2/inline-ptx-assembly/index.html), not an assumed register map from a newer MMA shape.

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

**P1a — Fragment and scorer microtest**

- Unit test `8×8×4` and multiple K steps against FP32 and FP64 reference, including zeros, negative per-head dots, non-finite test guards, tails and large/small FP32 queries.
- Start **one token** and 64 heads; use 8-head bands and 128-dot K reductions (`32` K steps of 4). Assign the four quad-pair MMAs to distinct 8-row tiles where practical (e.g. **32 compressed rows/warp × eight heads**, then advance over eight head groups). Experiment with **4 versus 8 warps per CTA**.
- Cache/reuse the query hi/lo fragments and 64 weights per block. Use coalesced, aligned K row loads; avoid a full `[n_rows,64]` intermediate in global memory.
- Finish each head-band dot, apply ReLU, update weighted FP32 row totals. Bound registers and shared memory; use `ptxas` reports and `cudaOccupancyMaxActiveBlocksPerMultiprocessor`.

**P1b — Single-token decode integration**

- Add `idx_score_volta_kernel`, only for **64 heads, 128 index dimensions, valid FP16 keys**, with checked SM70 device dispatch.
- Retain the same output layout expected by **unchanged** `idx_select_kernel`.
- Compare 512, 513, 1,024, 8,192, 31,250 visible rows; exercise rows not divisible by 8/32/CTA span. Benchmark score, select and indexer-skip overhead separately. Compare prefill disabled and speculation disabled to isolate decode effect.

**P1c — Batched prefill integration**

- **Baseline matters:** existing FP32 batch kernel reuses 32 tokens × 32 compressed rows; the SM80 optimized batch kernel reuses 32 token queries and key fragments. A naive one-token/CTA SM70 version may lose badly due to redundant key loads.
- Implement a **correctness-first** batched kernel using P1a arithmetic, initially one token per work item, then a distinct **key-reuse** variant: tile e.g. 8/16/32 prompt tokens against 32/64/128 compressed rows and reuse the K fragments across tokens.
- Evaluate a token-major loop retaining loaded K registers or shared memory; double-buffer queries **only if measured worthwhile**. Use `p0`, token `t`, `ratio`, `T`, `ld` consistently. Handle early-visible-row skips and valid-row tails without reading unwritten score entries.
- Benchmark 256/1,024/4,096 prompt tokens against 2K/8K/32K/120K starting contexts, and isolate CSA layers from HCA/raw-only layers.

**P1d — Integration, selection and acceptance**

- New **proposed** `ABYSS_V100_IDX=auto|volta|cublas|f32` (the `cublas` option is a P1e batched scorer only). Preserve `STRATA_DS4_IDX=1` as an unconditional legacy-FP32 override. If a user *forces* `volta` on an unsupported device, fail with a clear diagnostic rather than launching a trap. `auto` chooses the new path only for **validated profitable shapes** (initially allow conservative fallback).
- Log the actual scorer selected, geometry, dynamic shared memory size, and why a fallback occurred.
- Run scorer→unchanged-top-k parity; record scores, absolute/relative score differences, chosen index IDs, the rank-512 margin, disagreement rate conditioned on margin, and downstream attention/logit impact.
- **Do not demand identical top-k IDs at arbitrarily near-tied margins** if the resulting numerical difference is explained; but investigate every mismatch with a meaningful margin. No undefined, out-of-range, future or duplicate index introduced by the port.

**P1e — FP32-scratch batched cuBLAS scorer (from 1Cat DeepSeek SM70 reference)**

A third candidate for repeated C4 compressed keys is **batched cuBLAS GEMM**, distinct from the direct fused FP32 baseline and the new inline-PTX MMA884 scorer. Assemble a per-head query matrix with `T × 64` rows, multiply by visible compressed keys and apply **ReLU separately to each head’s dot product**, then multiply by that head's learned weight and sum 64 heads in FP32. A tempting `relu(sum_h weight[h]*q[h]·k)` factorization is **invalid** for the trained model. For initial quality gates, store dot scores in **FP32**, not FP16: near rank-512 tie errors have been observed in external DeepSeek indexers.

Tile by key rows with a hard workspace allocation cap and causal/visibility masks for each token; account for query packing, cuBLAS call setup and reduction epilogue. Benchmark T=1,2,3,8,16,32,256 with 512,2K,8K and ~31K visible compressed rows, against the current FP32 32×32 shared-key baseline and the MMA884 scorer. If a wider GEMM is winning only at large T/visible rows, permit it **only in that dispatch regime**. The proposed `ABYSS_V100_IDX=cublas` is meaningful **only when implemented and a compatible shape is present**, otherwise fail clearly; keep legacy `STRATA_DS4_IDX=1` force-FP32 precedence.

**P1 acceptance:** passes CUDA sanitizer, scorer and selection contract, teacher-forced logits and boundary regression, preserves sm80/HIP; demonstrates the kernel speed and overall effect on fixed benchmarks. Keep FP32 default for any regime in which Volta loses. *P1 should not be pursued for its own sake if the P0 trace shows its indexer occupies negligible critical-path time.*

---

## P2 — Volta Tensor Core batched DeepSeek attention (detailed)

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

**P2.0a — Exact MLA latent-space contract and parity gate (mandatory, before integrating raw-only P2b).** The comment that generic MLA "up-projects compressed KV" is **not the operation in this pinned Abyss kernel**. Its actual [DeepSeek forward specification](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/docs/deepseek4/SPEC.md) and [`dsv4_fast.cu`](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/kernels/cuda/dsv4_fast.cu) implement shared **512-D latent K = V**, used by 64 independently normalized 512-D query heads. Preserve this exact identity and projection order:

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

**P2a — F32-compatible MMA primitives**

1. Implement QK and PV isolated parity kernels. Verify MMA register maps and hi/lo behavior for 512 dimensions and varied key counts. Record errors against F32 fallback and F64 synthetic reference.
2. Establish a stable softmax/online normalization implementation, including sinks, fully/partially masked key tiles and FP32 exponentials.
3. Validate resource feasibility using `ptxas --verbose`, CUDA function attributes, device smem opt-in and Nsight Compute. Reject designs with unexpected local-memory spills or unstable occupancy.
4. Do **not** add compressed-index gathering in the first arithmetic microtest; use contiguous keys with explicit masks.

**P2b — Raw-only (`ratio=0`) integration**

1. Implement `attend_batch_volta_kernel` (initially 8 heads/CTA) with the **real raw-window addressing**, QK → sink-softmax → PV → F32 output → inverse RoPE.
2. Compare every head/output dimension to existing `attend_batch_kernel` for 1, 7, 8, 15, 16, 31, 32, 33, 127, 128 and 129 visible raw tokens; test 128-token window wrap and partial prefill chunks.
3. Compare the first and last output token of the chunk and a chunk split into multiple calls; detect off-by-one in `raw0` and absolute positions.
4. Test numerics and GPU memory sanitization **before** measuring large-prompt throughput.

**P2c — CSA (`ratio=4`) and HCA (`ratio=128`)**

1. Add `sel[t,*]` keyed gathers for CSA after preserving **exact** top-512 ordering and eligibility at each token.
2. Add sequential HCA compressed keys with the existing `floor((p+1)/128)` visibility rule.
3. Test at token positions 0, 1, 3, 4, 127, 128, 129, 511, 512, 2,047, 2,048, 2,049; also near every compressed-block boundary, nonzero `p0`, `n_hist` and long-context tails.
4. Measure HCA, CSA and raw-only separately, because different selection and gather patterns can alter the winning tile layout.

**P2d — Tile tuning and final fused version**

1. For each candidate `(heads/CTA, warps/CTA, key_tile, dim_subtile, shared/register O)`, capture kernel µs/ms, setup/launch overhead, `ptxas` registers, spills, smem, achieved occupancy, effective bandwidth and output error.
2. Consider 16 heads/CTA, Q/P conversion caching, double-buffered K loads, and alternate shared-memory staging **only after** the 8-head baseline is correct. An optional scheduling comparison separates QK/PV warps and prefetches the next key tile, inspired by Superfast `fattn-verify-stream.cuh`, **without borrowing its GQA/causal-mask math**. Avoid `ldmatrix`, `cp.async`, BF16/TF32 and SM80 `mma16816` paths: not supported on Volta.
3. Keep F32 fallback for short/irregular/low-occupancy regimes, if faster. Superfast’s own sparse-attention Volta-MMA alternative is **off by default because slower** than its non-MMA sparse kernel in its tested setup; treat this as a caution, not a result for Abyss. Do not add data-dependent switches that change attention semantics.
4. If the complete P2 kernel cannot be made both correct and profitable, retain correct component kernels/microbenchmarks and **ship FP32**. Do not force a high-shared-memory tensor-core path simply to close the checklist.

**P2e — Dispatch and numeric gates**

- Add **proposed** `ABYSS_V100_ATTN=auto|volta|f32`; `STRATA_DS4_ATTN=1` remains the authoritative legacy-FP32 override. Forced SM70 on another GPU gives a diagnostic, not a trap.
- `auto` enables only on actual V100, valid `(ratio, n_head, D, tile, smem)` constraints, and measured successful shapes. Preserve SM80/HIP routing.
- For a fixed test matrix record relative L2, maximum absolute and worst relative error away from zero, p99 elementwise error, sink-denominator checks and long-context accumulation error versus FP32 and a limited FP64 reference.
- **Provisional engineering target:** normal-case attention relative L2 around `1e-5` or better where practical; investigate/fail ordinary cases exceeding `1e-4` pending a calibrated distribution. These are **not** sufficient by themselves: require stable teacher-forced logits, boundary/top-k effects, greedy-prefix/task tests and no quality regression beyond measured run-to-run variability.
- Include an indexer FP32 + attention SM70 configuration to isolate P2 from P1; then test both enabled.

**P2 core acceptance:** a correct SM70 batched-attention path (raw/CSA/HCA) with a documented fallback, a real performance win in at least one useful V100 prefill regime **or** a documented evidence-based reason to retain FP32, and no regressions outside SM70. **P2f/P2g are separate optional gates**, not prerequisites for the exact FP16-cache core implementation.

---

### P2f — Xolotl-inspired compressed-KV storage ladder (OPTIONAL; baseline remains FP16)

**Goal:** Return meaningful V100 VRAM to routed-expert slots at medium/long context **without materially worsening** model quality or indexed-key selection. Unlike Xolotl's **FP32** storage baseline, Abyss currently stores its DeepSeek compressed keys as **FP16**, and the two engines use different GGUF packs. Therefore Xolotl's reported ~0.54% Q8_0 and ~0.60% rotated IQ4_NL perplexity changes and its 0.27×/0.14× memory ratios **are evidence of feasibility only**, not Abyss predictions.

1. **Census exact current cache semantics:** read `dsv4_alloc`, `dsv4_bind`, `prep/compress`, `attend_batch`, `idx_score_batch`, and decode `attend_split`; inventory **raw window `raw`**, CSA compressed attention rows `comp`, HCA compressed rows `comp`, indexer compressed keys `icomp`/index cache and compressor **state rings**. Distinguish the resident cache from temporary projection/workspaces and from transient index score buffers. Quantize `comp` and `icomp` **independently**, never mistakenly quantize the 128-token raw window or compressor state without a new accuracy study.
2. Implement **`comp: FP16 → Q8_0`** first, using legal aligned block packing with a validated SM70 path for **both attention QK and V=K PV**, plus compressed-cache append/read, chunked prefill and speculative rollback. Compare bytes, per-key dequant µs, total attention ms, saved expert slots, promotion traffic and cold/filled context. Keep the original FP16 cache allocation option.
3. Only if Q8 offers measurable *net* benefit, trial **`comp: rotated IQ4_NL`**. Walsh–Hadamard rotation is an *orthonormal coding transform*, distinct from model **RoPE / inverse RoPE**. Apply the same inverse transform on read or use a mathematically equivalent transformed-dot/value route; preserve 512-D K=V semantics and 64-D RoPE tail. Define quant scaling/packing precisely and test zero, saturating, near-tie and boundary rows. A rotation without the matching inverse can silently change the model even if perplexity on one prompt looks acceptable.
4. **Indexer `icomp`: FP16→Q8_0** is a separate third candidate. The lightning indexer must still implement `sum_h w_h*ReLU(q_h·k_h)` with correct RoPE, visibility and exact **top-512 selection policy**. Quantization is lossy: some near-cutoff rank flips are expected; record margin histograms, number/type of swaps, top-k overlap, logit/KL and task quality. Never describe quantized-key ranking as bit-identical to FP16.
5. **Host-mapped cache / hybrid:** only when measured cache VRAM competes with expert residency, trial per-format contiguous GPU cache vs pinned-host sparse gather (prefer explicit batched/coalesced row staging where link latency dominates). Xolotl measured **~3 ms/token penalty at 256K** despite +62 resident slots on PCIe 4.0; a V100 on PCIe 3.0 **requires independent measurement** before any host-mapped cache is enabled. Account for QPI if pinned pages live on socket 1.
6. **Safety:** cache slot/generation, append-only compressor key identity, valid-published row flag, zero/uninitialized rows, graph/pointer stability, rollback and context reset; no stale quant-format view. For every stored quantized row, decode it and compare to reference *before* running attention. Retain `P2.0` exact latent-space oracle and its sinks/YaRN/inverse-RoPE tests.

**Acceptance:** paired true filled 32K/64K/128K (256K only when safe) benchmarks demonstrate positive **net** decode/prefill latency or a quantifiable VRAM→expert cache benefit that outweighs added dequant/PCIe time, with calibrated quality changes disclosed and no hard semantic faults. Quantized caches are **off by default** in the initial exact-quality V100 release; an explicit opt-in may be accepted after Level B task checks.

### P2g — Query-blocked prefill raw-window audit (conditional)

Xolotl identified a **specific ggml prefill inefficiency**: for a chunk of 4,096 queries, a masked union raw window can contain `128 + 4096 - 1` keys even though **each query attends to only 128 raw keys**. Query-blocking to 16–128 queries reduced wasted raw-key visits in that engine. Abyss's [`attend_batch_kernel`](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/kernels/cuda/dsv4_fast.cu) already computes each query's `n_raw=min(p+1,n_swa)` and uses **per-query raw0 + direct selected-row addressing**; **the Xolotl masked-union overhead may be absent**. Inspect actual SM70 launched kernels, queried raw keys and memory throughput before implementing anything. **If** another Abyss prompt route builds/re-reads an oversized shared raw union, compare query-block sizes 1/8/16/32/64/128 and preserve raw-window visibility, ratio 4/128, compressor boundaries and chunk-position parity. Otherwise close **N/A** and continue P2's more relevant QK/PV and occupancy work.

## P3 — Expert memory tiers, CPU lanes, quantization and host performance

### P3.1 Measured memory budget (mandatory)

Before changing anything, report **separately**:

| Resource | What to measure | Why |
|---|---|---|
| Model file | Exact bytes and manifest by tensor type/role; expert-only packed bytes versus dense/embeddings | The published “96.8 GB” file is **not** automatically “97 GB of experts” |
| VRAM | Dense resident bytes; FP16 KV + GDN/aux state where applicable; CUDA context/modules; GEMM workspaces; prefill staging; GPU expert cache; reserve/free | 32 GB is a total GPU budget, not expert capacity |
| RAM | RAM-tier packed/pinned bytes; NUMA node residency; CPU-lane buffers; file/page cache; helper processes; free memory | Cannot assume 128 GB all available or pin-able |
| NVMe | Cold expert reads; warm-cache misses; storage read rate; initialization page-cache pressure | Determine whether “all experts warm” is achieved |
| Transport | Negotiated PCIe width/generation and effective device DMA throughput, mapped host accesses | V100 PCIe latency and link width can dominate swaps |

The upstream reference run reports about **23 GB GPU expert residency + 70 GB host pinned experts**. Treat those as **reference configuration observations**; the actual V100 distribution requires measurement. Keep model residency, OS reserves and auxiliary apps in the ledger. Avoid OOM and swap thrash.

### P3.2 Use the existing engine's NUMA/offload system first

- The target host is dual-socket NUMA. Abyss has **`STRATA_GLM_NUMA`**, two-node pinned/interleaved memory and **`STRATA_GLM_CPU_PIN`**; test these against an explicitly recorded fallback, **not** immediate global `numactl --membind=0`.
- `--membind=0` could constrain a 70+ GB RAM tier to one node with roughly 64 GB physical RAM. If used, treat it as an intentional negative-control experiment with available-node memory checked first. Prefer `--cpunodebind=0` *without* strict memory binding only where its interaction with engine thread pinning is understood.
- Verify existing **`STRATA_GLM_DROP_CACHE`** (avoids file cache consuming one NUMA node), and benchmark with it separately. Record node-local versus interleaved pinned-bandwidth.
- Profile `STRATA_GLM_PCIE_SHARE` calibration and CPU versus PCIe expert lanes on the Broadwell AVX2 cores: try physical-core-only workers before SMT; compare PCIe-stage, CPU-execute and hybrid. Confirm CPU i-quant and Q2 paths are actually selected for the shipped GGUF.
- Keep recent upstream fixes **`STRATA_GLM_PRESTAGE`**, `STRATA_GLM_BG_PACE_KB` (default 1024), `STRATA_GLM_PROMOTE_RATIO` / `_MARGIN`, `STRATA_GLM_ZERO_COPY`, and prefill chunk/copy limits. Measure “prompt → first decode token” and later decode separately; DMA pacing can be more important than compute speed.
- Benchmark **unfused MMQ** versus dequant-to-F16 + cuBLAS HMMA **only if** profiling identifies the relevant expert GEMM hotspot; 1-token GEMV and large prompt GEMM may have opposite winners. Never assume dequantization wins at small batches.

### P3.2a Broadwell CPU expert lane: dispatch and vectorization gate

**Specific host:** Intel Xeon E5-2697A v4 (Broadwell-EP) supports **AVX2 and FMA3**, not AVX-512. Inspect `lscpu`/`/proc/cpuinfo` flags for `avx2` and `fma` and the *actual built binaries* for dispatched ggml/Abyss CPU expert kernel variants. Use `-march=broadwell` or a portable AVX2/FMA-specific translation unit (`-mavx2 -mfma`) for the expert CPU implementation; **keep baseline and non-AVX2 portability paths separate** rather than enabling `-march=native` uncritically for cross-built binaries. Verify optimized paths run, e.g. inspect emitted `vfmadd*`/AVX2 instructions and counters; do not label scalar code AVX2 just because the CPU supports it.

Benchmark: per-quant (`Q2_K`, `Q3_K`, IQ types, MXFP4 if CPU-supported) reference versus AVX2/FMA kernel with one/few token inputs, 1/4/8 expert misses, one socket's physical workers (starting around 12 **only as a sweep point**), 16/24/28/32 worker sweeps and SMT-on variants; use node-local versus interleaved pinned memory and actual memory-intensive cache churn. Collect `ms/expert`, **bytes from DRAM**, sustained GB/s, cache miss rate where available, AVX frequency, numerical parity (SIMD reduction order may change outputs), and the end-to-end CPU-lane completion time. Broadwell's dual-socket DDR4 bandwidth and intersocket QPI contention may matter more than single-core FLOPs. If already optimized, **do not rewrite the kernels**; tune dispatch/worker placement first. Never require AVX-512 in baseline build or introduce code that traps on Broadwell.

### P3.2b QPI/NUMA topology and intersocket bandwidth gate

The **E5-v4 uses QPI, not UPI**. The V100 is reported attached to node 0, whereas a substantial resident expert tier may occupy both sockets. CPU read-locality and GPU-DMA-locality are *distinct optimization objectives*:

- **Measure, do not assume:** map GPU PCIe bus ID→root CPU/NUMA (`nvidia-smi topo -m`, `lspci -vv`, `numactl --hardware`, `lscpu -e=CPU,NODE,SOCKET,CORE`, `/sys/bus/pci/devices/.../numa_node`), verify the exact PCIe association, and record pinning/first-touch. Use `numastat -p <PID>`, `/proc/<PID>/numa_maps` and sampled page migration to establish **where the large host allocations actually reside**, not merely the requested `numactl` policy.
- **QPI telemetry:** use compatible **Intel PCM** (`pcm-memory`, `pcm-numa`, Intel PCM QPI/uncore counters) or platform-exposed `perf` uncore events to record intersocket bytes/direction, local vs remote DRAM reads, per-socket utilized memory channels and CPU worker bandwidth. Query event availability instead of relying on a counter name universal to all BIOSes; record when uncore permission, kernel, virtualized platform or BIOS configuration makes QPI counters unavailable. Never label a stall "QPI saturated" from low AVX utilization alone. [Intel PCM Xeon E5/QPI information](https://www.intel.com/content/www/us/en/developer/articles/tool/performance-counter-monitor.html).
- **Controlled four-arm placement sweep:** (A) node0 workers+node0 resident data (when the data fits); (B) node1 workers+node1 data; (C) socket-local workers over a **sharded** expert bank where practical; (D) Abyss's current two-node interleaved pinned pool with its actual CPU pinning. For each, measure CPU expert completion, GPU DMA to/from the same buffers, QPI bytes, occupancy, achieved PCIe bandwidth, and model output rate. **The 128-GB model tier may not fit in node0 alone**; never force global strict node0 memory binding as a production recommendation.
- **NUMA admission:** maintain per-allocation ownership (expert shard/node, pinned host pool, work buffers), and bind **the reader CPU workers near their expert pages**, unless a measured CPU/GPU critical-path tradeoff favors another placement. For node1-pinned data, node0 GPU DMA may still traverse QPI even when CPU workers are local to node1. Investigate socket-local staging or modest data replication only with explicit RAM budget and measured benefit; avoid an unmeasured blanket "node0 GPU and CPU" policy. Export `results/p0_qpi_numa_placement.csv` and classify uncontended CPU-only, GPU-DMA-only and combined contention cases.

### P3.2c — Xolotl cache-policy / disjoint-residency experiment (exact-quality, conditional)

**First inspect existing Abyss behavior:** `src/core/glm_fast_path.cu` documents **GPU-resident device lookup**, **per-layer slot victim partition**, **mapped host route ring**, **CPU service**, and per-layer DMA when misses occur. Its slot replacement cannot safely overwrite an expert being read by that layer. Its LFU/adaptive residency and promotion controls may already outperform Xolotl's simpler per-layer LRU. **Do not replace the service system or make a global cross-layer cache by default.**

1. **Replay harness:** consume the **same P0 expert route stream** and fixed physical slot budget for (A) unchanged Abyss policy; (B) exact per-layer **exclusive LRU victim**; (C) existing LFU/decayed count with candidate admission/hysteresis; (D) only if measured per-layer imbalance dominates, **opt-in cross-layer elastic slots**, which requires a new proven-safe ownership/lifetime design. Xolotl's own source explains that its “global slots” behavior was *not* a true cross-layer evicting LRU; do not cite its old global cache comparisons as evidence against a genuine global LRU.
2. Compare **cold-to-warm and topic-change curves**, not only aggregate hit rate. Record hit/miss count, weighted missed **bytes**, source (host CPU/GPU/NVMe), prefetch-useful hit, evictions per 1K tokens, slot residence duration and total critical-path milliseconds. An LRU policy can raise hit count while evicting large or expensive experts and **increase** actual wall time.
3. **Disjoint placement:** Xolotl's `--arena-skip-resident` avoids keeping a second pinned copy of a GPU-seeded expert. On Abyss this is only safe if the original immutable GGUF/file/page-cache copy remains addressable, if CPU access to evicted GPU experts has a timely source, and if the **GPU promoted** bytes and CPU-lane fallback are already accounted for. Compare full pinned duplication vs selective disjoint placement at the **same 128GB host limit**. Xolotl explicitly warns **not** to combine its `--arena-skip-resident` short-decode preset with the large chunked-prefill preset that requires the full host arena: test Abyss *prefill → first decode* and not just warmed single-token decode. Quantify SSD reads and latency on a cold or topic-switch run; do not confuse saved pinned RAM with saved VRAM or guaranteed performance.
4. **Safety:** preserve per-layer victim constraints unless an explicit cross-layer allocator is designed. Integrate v6 `slot_epoch`, `layout_version`, `repacking-in-progress`, completion event, and rollback guard. Confirm predictive prefetch and CPU workers cannot read a victim during DMA/repack; retain canonical native weight bytes for Q2_K/MXFP4 and separate CPU quantization. No routing-score changes: the route/expert IDs and weights are **identical** under an exact caching experiment.
5. **Stop:** if the existing Abyss hit/byte/critical-path ledger is at least as good, close the borrowed policy as unnecessary. Adopt only if **real model wall time** improves reproducibly at fixed quant, precision and host resource limits, with correct outputs and safe recovery from SSD.

### P3.2d — Xolotl predicted-prefetch scheduling comparison (no duplicate DMA)

Xolotl's DS4 engine uses `predict(layer)` from an available earlier state, stages predicted misses in a **separate CUDA stream under attention**, then computes correctly routed experts after actual router output. It reports roughly **62–70% candidate recall**, but excessive prefetches crowded PCIe. Abyss already has prestage/promotion; this task tests the **gap**, not an assumption of missing functionality.

1. Record a *causally valid* prediction source in Abyss (e.g., a prior hidden state already on the GPU without a new sync), and compare it with baseline `STRATA_GLM_PRESTAGE` behavior and an oracle next-route trace used **only as an unattainable upper bound**.
2. For each `(layer, context, warm/cold)` log `predicted IDs`, actual routed IDs, lead-time distribution, **useful-on-arrival** fraction, discarded/stale issue count, extra `readback`/CUDA launches, DMA bytes from node0/node1, CPU DRAM read contention and uncovered critical-path time. Successful prefetch can *still regress* if it displaces a hot expert or starves the CPU lane of bandwidth.
3. Sweep budget **0, 1, 2, 4, 6 candidate expert prefetched rows per eligible layer**, or normalized bytes if expert sizes vary. Sweep staging stream priority and timing only after output parity. Never blindly import Xolotl's numeric `--pf-b=0.7` or `--pcie=0.55`: its PCIe, CPU and quant distribution are different.
4. **Cancel/invalidate correctly** after an unexpected actual route, context reset, cache eviction, speculative rollback, GPU-slot repack, or in-flight DMA source disappearance. Use queue limits, `slot_epoch` and staged completion events; do not block the actual CPU/GPU route behind unnecessary predicted reads.
5. **Stop:** require positive measured **net wall** improvement with identical chosen experts and logits. If gains are attributable only to a larger cache rather than prediction, report them under P3.2c, not prefetch.

### P3.2e — Warm expert profiles, hysteresis and topic-switch stability

Xolotl's engineering notes distinguish static route profiles, dynamic `--arena-adapt`, and source-reported REAP pruning. For **exact** Abyss, retain the *true router output* and explore only the exact-memory component: decayed/hysteretic admission to GPU/RAM tiers, optional persisted hot-expert counters across *the same model hash* and task-specific warm-start, **without suppressing or biasing router-selected experts**. A persisted cache must be model/version/quant/format checked, bounded, atomically written, and recover safely after interrupted updates; otherwise ignore it. Compare **fresh cold**, warm same task, model reload, and A→B→A domain switches; report startup cost, early-token latency, SSD activity and steady decode. Do not promote cold experts on one incidental miss if it thrashes a useful current working set. Exact-model logits must remain unchanged by profile policy except already-characterized CPU/GPU kernel round-off; **REAP hard/soft pruning, route bias and skip-miss remain expressly out of scope.**

### P3.2f — Upstream Strata exact singleton IQ2_XXS CPU lane (HIGH, compare before port)

**Source-specific gap:** Strata's [`STRATA_CPU_IQ_ALL_EXACT1`](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/kernels/cpu/native_expert.cpp) adds AVX2 one-token expert rows (including IQ2_XXS and IQ2_S) that deliberately reproduce *ggml-cpu's AVX2 reduction order*. Abyss [`src/kernels/cpu/native_expert.cpp`](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/kernels/cpu/native_expert.cpp) already optimizes one-token IQ3_XXS/IQ3_S and requires a per-format exactness check. This is **not** permission to replace those existing paths wholesale.

1. **Format + dispatch audit:** from P0 census identify actual IQ2_XXS gate/up and Q2_K down rows and the contiguous packed representation. Gate any borrowed path on *both* CPU `avx2 + fma` and **the ggml-cpu binary's actual AVX2 backend**; upstream explicitly excludes an AVX-512 or low-ISA ggml dot and preserves the original singleton/multitoken threshold. `-march=broadwell` is appropriate for a host-only build, but keep runtime CPU feature guards in shared builds.
2. **Parity microtests before speed:** compare each candidate row against exactly the same ggml-cpu configuration using zero, scale extremes, sign patterns, randomized rows, real model blobs and fixed per-token activation bytes. Use bytewise parity when promised by the chosen implementation and report real reduction-order differences otherwise; compare composite expert output and teacher-forced logits. Test nt=1 and 2/3/4 thresholds explicitly. IQ2_XXS != IQ2_S != Q2_K: only implement a confirmed supported format; do not accidentally dispatch the IQ2 singleton kernel for Q2_K down.
3. **Cold-memory speed on Broadwell:** isolate one- and two-socket CPU worker pools, node-local/interleaved expert placement, 1/4/8 CPU misses and simultaneous GPU mapped/DMA traffic. Report per-format `ms/expert`, sustained DRAM GB/s, QPI costs, frequency and actual **decode critical-path reduction**. Xolotl/Strata other-CPU speedups are not target estimates.
4. **Admission:** add an independently gated IQ2 singleton path only where parity and repeated A/B show an improvement that exceeds noise; otherwise retain existing Abyss dot. Keep CPU canonical quant bytes unchanged and keep AVX-512 completely out of the executable's chosen call graph on Broadwell.

Deliver `results/p3_cpu_iq2_singleton.csv` and format-dispatch log. Proposed control: `ABYSS_V100_CPU_IQ2_EXACT1=0|1` (not an existing Abyss variable); use `0` as explicit A/B fallback.


### P3.3 Dense quantization and MXFP4

- Abyss's `tools/requant_dense.py` offers an independently testable **Q8_0 → Q6_K / Q5_K** change to the attention projections. Upstream documents approximate three-projection sizes: **4.28 GiB (Q8_0), 3.31 GiB (Q6_K), 2.77 GiB (Q5_K)**. Verify the actual pack layout and compare at identical input lengths.
- Each quant candidate is a **separate GGUF/pack** with its own hash, quality, VRAM, prefill and decode measurements; never quietly change quant in P1/P2 comparisons.
- Validate pre-Blackwell MXFP4 decode/prefill. Optional pack-time retyping of MXFP4 expert rows is permitted only with quantified added bytes, quality tests and a revert-to-shipped-pack path.

### P3.4 Small BF16 paths

- **No native BF16 GEMM on V100.** Some small BF16 matrices in `glm_prefill.cu` are widened per call and run through FP32 SGEMM; the indexer projection weight `Ly.idx_proj_f` is already cached widened.
- If profiler shows conversion overhead, **cache BF16→FP32** converted weights at load time or use reusable scratch, subject to measured memory cost. This is numerically lossless for finite BF16 values.
- Optionally try a **validated** BF16→FP16 conversion plus F16 Tensor Core GEMM, but check exponent-range overflow/underflow, actual round-off and logits. There is **no universal exact BF16→FP16 conversion**. Keep FP32 fallback for router and hyperconnection-sensitive paths.

### P3.5 Superfast-derived small-batch quantized MMA: feasibility gate and shape routing

**High-value new experiment, not a pre-approved port.** Superfast's most general new strategy is **format-aware** MMA884 for small matrix products and selective load-time repacking. Implement in this order, only after P0 has enumerated Abyss tensor shapes and the **real** model's quant types:

**P3.5a — Inventory and baseline**

1. Join `results/p0_tensor_inventory.csv` with the actual decode/prefill critical path. Aggregate **time**, bytes and invocation count by `(tensor role, weight type, N, K, token width, residency)`. Explicitly distinguish **GPU-resident dense**, **GPU-resident shared expert**, **hot GPU routed expert**, **RAM/CPU lane routed expert**, and **SSD source**.
2. For **Superfast-derived QPN only**, require its supported type (Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL, Q8_0), `K % 256 == 0`, `N % 32 == 0`, and contiguous row-major GGUF layout. **Its QPN does not support Q2_K, IQ2_*, IQ3_* or MXFP4.** Those are **not necessarily new research problems**: Q2_K has a working Volta small-T reference in [anyei `mmsmt.cu`](https://github.com/anyei/llamacpp-v100/blob/master/ggml/src/ggml-cuda/mmsmt.cu); MXFP4 has references in [1Cat](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu). Select the first experiment by **measured Abyss role/type/time share**, not which source was read first.
3. Preserve the baseline type-specific `mv_kernel_t`, split-row (`STRATA_GLM_MV_SPLIT`), GGML MMQ and cuBLAS paths; compare **one token**, **two-token verify**, **3/4/8-token groups**, then 16, 32, 64, 128, 256+ token prompt tiles. Superfast's QPN uses `<=64` and 8/16-token passes, but **these are tunable starting points, not Abyss defaults**.

**P3.5b — Standalone arithmetic benchmark**

1. Build a dedicated SM70 test harness under `tests/v100/` (e.g. `test_ds4_qpn.cu`) using Superfast’s `mma.cuh` and `mmvq-qpn.cu` as design references, **without importing GGML's backend registry/graph scheduler**. The candidate API is `GEMV/GEMM(weight_type, N, K, n_tok, packed_weight, f32_activation, f32_output)` plus a separate source-layout pack/unpack interface.
2. Use 32 output rows/warp and the four independent quad-pair 8×8×4 MMA groups; evaluate split-K across **2, 4 and 8 warps** and shared-memory FP32 reduction only after per-warp accumulation. Candidate superblock: **32 rows × 256 columns**, one coalesced **16-byte per-lane** read for each packed 512-byte chunk, with quant headers/scales in the layout. **For every variant**, report `ptxas` registers/thread, spill loads and stores, actual local-memory traffic, dynamic/static shared memory, theoretical and achieved warp occupancy, per-call launch time, resident blocks/SM and effective packed-byte bandwidth. Use the explicit P3.5g gate below, not an untested fixed occupancy threshold.
3. Compare **QPN compact-format register unpack** with the existing `q8_1` activation dot and with **dequantize-to-FP16 + cuBLAS**; do not silently change the model's quantized weights. Track per-superblock scale/min corrections in FP32 and test 256-element tail rules.
4. Prepare FP32 activations as F16 fragments, optionally using exact power-of-two exponent scaling per 256 elements to keep representable range; retain a **nonfinite/extreme-value fallback** that reproduces a defined reference route. Compare using q8_1 activation conversion versus F16-hi/lo correction. Analyze correctness at tiny magnitudes, overflow/underflow and saturating integer paths. A higher-precision route is not automatically more faithful to the unchanged Abyss model.
5. **Workload matrix:** `(N,K)` from the actual inventory; `n_tok = 1,2,3,4,8,16,32,64`; temperature/clock controlled. Report `(repack_ms, prep_us, kernel_us, output_error, bytes_allocated)` separately, as well as all-in per-graph latency. Kill an experiment that is slower after overhead even if its MMA kernel alone is faster.

**P3.5c — Sparse use and dispatch**

- Define a new **proposed** `ABYSS_V100_QPN=off|dense|hot-expert|auto` (`off` default throughout development). `dense` can repack explicitly eligible **static** matrices; `hot-expert` is separate and requires P3.6 safety checks; `auto` is later configured with empirically winning tensor roles/types/shapes and reuse threshold. Never repack all matrices by type alone.
- Shape table belongs to **DeepSeek's** actual model and workload, not Qwen's `17408×5120` etc. Store benchmark rationale per entry; require >A/B noise end-to-end benefit for automatic use.
- Proposed `ABYSS_V100_QPN_WIDTH`, `_GROUP`, `_PREP_SHARE` allow independent trial of token-pass width, compatible projection grouping and reuse of one prepared activation. Clearly mark them **new**, not shipped Abyss settings or Superfast environment variables.
- For any QPN path, inspect that residual adds, bias, output views, grouped gate/up activation and prefill dequant operators are **not reading an incompatible packed layout**. If read semantics cannot be guaranteed, leave the original tensor untouched and use a separate, explicitly budgeted copy.

### P3.5d Q2_K-capable small-T MMA — independent from Superfast QPN

The [anyei `mmsmt.cu`](https://github.com/anyei/llamacpp-v100/blob/master/ggml/src/ggml-cuda/mmsmt.cu) directly implements **Q2_K**, with Volta MMA884 and per-block scale/min handling; it is **not** proof of an immediate Abyss performance gain. Port only the appropriate integer unpack/half2 path and tile schedule into a standalone Abyss experiment for the actual `Q2_K` tensors in the UD-Q2_K_XL pack. Test T=1,2,3,4,8,16 and varied K/N. Explicitly compare to native Abyss `mv_kernel_t`/`moe_mmq.cu` inclusive of activation conversion, launch and repack costs.

**Correctness vetoes:** real GGUF **all-zero Q2_K blocks (`d=dmin=0`)**, division/ratio guarding before `dmin/d`, tiny scales, high activations, correct minima and scale application in FP32, nonfinite/overflow fallback, tails, cross-warp reductions and full target-model logit tests. The reference project documented a whole-model quality failure despite passing micro-op tests. Keep Q2_K and non-Q2 quant error ledgers separate. Use no automatic route unless both bandwidth/occupancy gate and model parity pass.

### P3.5e Native MXFP4 E8M0/32 — a separate decoder, not Q2_K or NVFP4

The [1Cat SM70 MXFP4 source](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu) and [skinny grouped-MoE source](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu) implement packed E2M1 elements with **E8M0 scales per 32** using in-register FP16 reconstruction and MMA884. This is **not** NVFP4 with FP8 per-16 scaling; the reconstitution layout, bias and dynamic range must match *Abyss's actual MXFP4 rows* exactly. First validate a standalone M=1 resident-expert kernel; then M=2/3 and prompt-batch paths only if the T=1 gate succeeds. Compare against the current V100 Q8_1 + generic GGML MMQ route with identical packed bytes, outputs and scratch. Do not assume the source's grouped experts share Abyss's layout or transform the RAM-tier canonical copy. On a 32-GB V100, repack and extra resident bytes can erase the speed gain.

### P3.5f Small fixed-shape FP16/BF16 GEMV

Examine [1Cat's DeepSeek-specific GEMV study](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/sm70_deepseek_v4_fp16_gemv.md) for a shape-tuned, no-unnecessary-split-K approach on narrow routers and compressors. Port only if Abyss's own **actual type/shape/critical-path** supports it. Measure the **joined multi-stream envelope**, not summed isolated kernel time; retain separate CPU/GPU correctness of router top-k. No preemptive router fusion: the source documented that impressive isolated gains did not meet the full-graph saving gate.

### P3.5g Mandatory register, spills, occupancy and measured-performance admission

For **every P1, P2, P3.5b, P3.5d, P3.5e or P3.5f** SM70 candidate, report `ptxas -v` registers/thread, spill-load/store counts, `cudaFuncGetAttributes` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` results with **the actual dynamic shared memory**, theoretical and achieved occupancy, local-memory bytes/instructions from **Volta-compatible `ncu`**, SM/tensor throughput, HBM/L2 traffic and all-in stage latency. Run `compute-sanitizer --tool memcheck` and appropriate racecheck/synccheck for kernels that exchange data across warps. Never assume `__syncthreads` is safe under divergent warp control flow.

**Screening rule:** ~50% theoretical warp occupancy, ~40% achieved occupancy, and ~32 B/thread compiler-reported spill *allocation* are **diagnostic review flags**, not independent stop/ship criteria. Occupancy can be low for beneficial high-reuse MMA; theoretical occupancy may be materially higher than actual if the grid is too small. Conversely 1–2 spill slots may trigger costly local-memory reads inside every unrolled loop. Investigate severe spills and any fall below these review figures; sweep CTA tile size, split-K, warp count, register count, and alternative scalar/`half2` fallback. Reject a **candidate** when it is slower after repack/prep/launch/wait or fails quality/safety gates, regardless of nominal occupancy. Do not reject the *entire P3 research avenue* just because one kernel configuration is below 40% occupancy. Document the observed winning tile or fallback.

---

### P3.6 Critical constraint: moving experts and dual-format weight lifetimes

Abyss's experts migrate among VRAM, pinned RAM and NVMe; the CPU lane computes expert rows directly from the **native source layout**. Superfast's QPN relies on weights repacked *once* at model load and a backend tensor flag that prevents ordinary consumers from reading them. **These memory models differ.**

1. **Default:** original packed experts remain canonical in the GGUF-backed file, pinned-RAM pool and CPU-lane buffers. Do not repack RAM-tier experts, change on-disk shards, or mutate a slot that a CPU worker may read.
2. **Static dense experiment:** for resident dense weight W, compare (a) **original W + per-use conversion**, (b) a **separate QPN-packed W** (additional VRAM!) and (c) **in-place QPN-packed W only if *every* consumer**, including prefill and debug reference, has a repack-aware reader. Track **source format, ownership, memory bytes and unpack path**.
3. **Hot routed-expert experiment (conditional):** attach layout metadata to an *individual GPU cache slot* (quant type, geometry, epoch/generation, canonical/repacked flag). When a canonical expert is promoted, optionally transform it **once after DMA** into an owned GPU slot, before publishing its device pointer. Never expose a partly transformed slot; on eviction simply discard that representation, leaving the canonical host copy intact. Synchronize with in-flight kernels and speculative rollback.
4. **Amortization:** explicitly evaluate `benefit = (uses_during_residency × [old_product_us − new_product_us]) − repack_us − extra_DMA_us − eviction_wait_us − extra_prefill_conversion_us`. Measure benefit for each hotness decile, prompt-heavy and decode-heavy traffic, warm and cold starts. Reject a global repack policy if churn negates savings.
5. **Prefill/MMQ coexistence:** current `src/prefill/moe_mmq.cu` reads native ggml K/I/MXFP4 blocks. A repacked expert used by it must either be **unpacked**, routed to a correct repack-aware MMQ/dequant kernel, or read from the canonical copy; no caller may use the wrong stride or scale-header position. **Unit tests must intentionally trigger** prompt after GPU promotion, expert eviction then reload, CPU↔GPU lane changes, and packed slot reuse at identical address.
6. **VRAM policy:** never allocate a second large population of expert weights by default. If copying static weights, account for dense resident bytes, 32-GB total VRAM and buffer fragmentation before enabling. Do not sacrifice a larger, performance-critical expert hot cache for a marginal GEMV gain.

**P3.6a — Explicit dual-format slot lifecycle and ABA prevention.** Upgrade the informal metadata in item 3 above to the following testable contract; implement only if repacked GPU hot experts are actually enabled:

```text
Host canonical expert storage: immutable native packed layout; never in-place repacked.
GPU slot metadata (per slot): {slot_id, expert_id, slot_epoch, layout_version,
                                quant_type, shape, state, ready_event,
                                in_flight_reader_count or retained completion events}
States: EMPTY -> DMA_LOADING -> [GPU_REPACKING] -> READY -> EVICTING -> EMPTY.
```

- `slot_epoch`/generation increases on **every reuse of a physical cache slot** (avoids ABA if the pointer is identical); `layout_version` identifies canonical/MMA-repacked encoding and packing ABI, **independent** of the generation. CPU-visible metadata is advisory until synchronization makes the GPU bytes ready; correct publication requires CUDA stream/event dependencies, coherent visibility and appropriate host/device release-acquire semantics or equivalent explicitly proven barriers. An `atomic` store alone is *not* a substitute for the completion of asynchronous DMA/repack.
- Any GPU consumer pins or snapshots `(slot_epoch, expert_id, layout_version)` and waits for the corresponding ready event before reading; no concurrent eviction/repack may invalidate its pointer until in-flight kernels finish. CPU workers **normally use only the immutable canonical host copy**, and therefore do **not** need to probe every GPU slot. A CPU caller that consults shared GPU slot metadata must reject `DMA_LOADING`, `GPU_REPACKING`, generation mismatch or wrong layout and either read canonical host bytes or wait according to its independently verified lane policy—**never reinterpret repacked bytes as GGUF**.
- On completion of a promotion, publish `READY` **only** after the whole GPU layout and all scale headers are valid. On eviction, first retire publication, then synchronize readers before reusing the slot and increment epoch. Failed/aborted DMA/repack and speculative rollback cannot leave `READY` stale. Test remap/eviction at same address, one CPU miss concurrent with GPU repack, changes of prefill/decode operators, pointer/version ABA, asynchronous stream ordering and fallback correctness under `compute-sanitizer` where device races are detectable.
- Add a negative test intentionally passing the wrong layout version to each reader: the route must reject or redirect it with **no silent output**. Keep source-format identity and repack cost ledger required above. Proposed `ABYSS_V100_SLOT_FORMAT_DEBUG=1` is a **new diagnostic**, not an existing Abyss flag.

**P3.6 gate:** no concurrent-reader corruption, no mixed-format CPU/GPU reads, correct prefill path and unchanged canonical on-disk pack; quality validated; **positive end-to-end amortized gain** over realistic expert residency windows. A successful isolated QPN microbenchmark does **not** authorize general expert repacking.

### P3.7 Fusing compatible GEMVs and sharing input preparation

Superfast's QPN paths provide **one-launch pairs/triples** and reuse of prepared input between projections. Apply only to model-verified same-input relationships (e.g., actual DeepSeek gate/up branches and appropriate adjacent projections):

- Draw a dataflow graph for each layer, including mHC mixer, norm/RoPE, router, `gu` gate/up, shared expert and `dn`. List whether the inputs are bitwise identical and simultaneously available; **never** group operations across an intervening transform, nonlinear activation, distinct expert ID or stream dependency.
- First share **activation preparation** (e.g. F32→F16 fragment/scaled representation or `q8_1`) between independently launched quantized projections, with unchanged output layout. Then test two kernels in one launch and optionally three. Separate **saved launch/preparation overhead** from MMA occupancy loss, changed register footprint or worsened coalescing.
- Guard by `batch width`, `(type,N,K)`, device residency and output stride. The fast expert launch already batches work (`MvBatch`), so measure whether fusion removes an actual launch or merely adds branching to an existing multi-job launch.
- Differential test each output tensor independently, *then* full model logits and latency. Retain a separate-launch fallback even on SM70.

### P3.8 Large vocabulary/output-head targeted experiment

Superfast's **unrepacked** `mmvq-tc.cu` is a narrow precedent: **Q5_K, at least 65,536 output rows, three to eight verify tokens**, and its author reports smaller projections slower. Determine Abyss's *actual* output-head type, `N`, `K`, placement and verification width. If it matches (or can be adapted without reformatting), benchmark the output head alone, with full logits; otherwise **close as inapplicable**. Avoid creating a generalized Q5_K expert replacement from this special-purpose kernel.

### P3.9 Sparse-attention / scheduling optimizations: only where Abyss lacks them

- **Already present:** Abyss's `attend_batch_kernel` selects compressed CSA rows by index in-place (no gathered KV copy) and the decode `attend_split_kernel` + `attend_combine_kernel` split/merge over keys. Do not duplicate these.
- Superfast's QSA dispatch scales `n_split` by GPU SM count and token/kv-head CTA count to fill the device. Compare **dynamic split count and tile length** against Abyss's current decode split and P2 attention paths, especially **one-/two-token verify** and HCA at 64K/125K; benchmark split scratch/merge cost. Change only if the trace shows idle SMs and a net win.
- Superfast's `fattn-verify-stream.cuh` has distinct QK/PV warp groups, 64-key double-buffered tiles and staged loads. Because it is **D=256/GQA6** and DeepSeek is **D=512/sink/CSA/HCA**, implement a separate DeepSeek prototype (P2d/P4) only if a long-visible-key or verify workload is actually memory/warp-stall bound.
- Test whether Abyss copies any attention mask, selected-key metadata or compressor state **from the CPU every iteration**. If no such copy exists, Superfast's **device-mask win** is inapplicable.

### P3.10 — Xolotl-inspired bounded/chunked prefill stream A/B (retain Abyss's engine)

Abyss already contains the **DeepSeek batched prompt path** in `glm_prefill.cu`, including GPU expert MMQ, compressor/indexer batch kernels and promotion/prestage. Xolotl's `--prefill-chunk 4096 --chunk-mmq --chunk-prestage` is **not an Abyss launch recipe**; its report that larger chunks amortize expert copies motivates a *bounded* sweep on V100 PCIe 3.0:

- Hold the **unchanged GGUF, precision and tensor roles** fixed; sweep Abyss's supported chunk sizes **128, 256, 512, 1024, 2048, 4096** and optional sizes beyond only when VRAM staging allows. Record per-layer **expert bytes transferred per chunk**, number of times an expert weight is copied, GPU F16/MMQ utilization by actual quant type, CPU expert occupancy, double-buffer staging bytes, NUMA locality, peak VRAM, prefill wall and decoder *after* the prefill. **Short 600/3K, 8K, 32K and 64K+** can have distinct optima.
- Optional next-layer **prestaging** must not evict current useful experts or induce *additional* same-layer demand copies. Compare none, existing Abyss prestage and a bounded two-buffer pipeline; log overlap and actual PCIe+CPU contention. A DMA overhead that was apparently hidden on RTX 4090 may be critical on PCIe 3.0; do not extrapolate.
- **Do not use Xolotl's RTX 4090 integer Tensor Core MMQ figures** to predict V100: inspect which quantized paths Abyss actually launches for IQ2_XXS, Q2_K and MXFP4 and which HMMA/DP4A features those paths use on SM70. Shape-specific kernel selection belongs to P3.5.
- Compare **latency to first generated token**, total *prefill + 128-token decode*, end-to-end accuracy with compressor boundary states and complete source-document context. No useful prefill win if it produces a worse seeded cache or delays the first decode disproportionately.

### P3.11 — Small-op fusion and shared activation preparation (Abyss path only)

Xolotl's ggml trace counted ~4,300 short kernels/token, and its source notes repeated q8_1 activation quantization across seven same-input projections. **Abyss fast decode is not that ggml path**: the initial code audit documents ~11 fused launches/layer, so copying its graph-op fusion is not justified without proof. Instrument Abyss's **actual** same-input `q_a`, `kv`, compressor-KV/gate, indexer projection and routed gate/up input groups; determine whether a quantized activation can be computed once and legally shared with *identical* scale/format/lifetime. Preserve reference rounding or explicitly qualify the changed logits. Integrate with existing **P3.7** activation sharing and P3.5 quantified small-T groups; do not create a second competing implementation. Add a per-layer saved-launch and saved-byte estimate, `ptxas` resource/occupancy gate, and interleaved end-to-end A/B. **Stop** when fusion raises register spills, destroys overlap or offers only isolated kernel changes.

### P3.12 — Post-fork Strata quantize-once/scatter MMQ input (HIGH; first code candidate after P0)

Abyss's `src/prefill/moe_mmq.cu` currently calls the ggml indexed `quantize_mmq_q8_1_cuda(...)` for quantized expert input. Upstream commit [`c675f5b`](https://github.com/Niko1221/Strata/commit/c675f5b78725) instead uses `quantize_scatter_mmq_q8_1_cuda(...)` to quantize each **token** once, then distribute the *same* packed Q8_1 bytes to its `top-k` selected expert rows. It reports byte-identical Q8_1 output under its own supported layouts (`D4`, `DS4`, `D2S6`), **not** a verified Abyss V100 gain. This is a narrower, more credible first patch than a Q2_K MMA rewrite.

1. **Preconditions:** inspect the exact vendored `third_party/llama.cpp`/ggml pin and exported quantization functions; prove the `quantize_scatter_mmq_q8_1_cuda` API exists or vendor a small, clearly tested matching wrapper. Confirm whether the actual Abyss DeepSeek prefill passes the *same pre-MoE activation* into multiple routed experts, that the `ids`, `slot`, `src`, `ld`, `rows`, `n_tokens` and `top-k` semantics are as expected, and whether any 0/1/2-token path already quantizes only once. Do **not** change a correct one-quantization path merely to match upstream naming.
2. **Semantic guard:** build and compare indexed gather vs quantize-once/scatter on *identical source float bytes* with token widths {1, 2, 3, 8, 31, 64, 512, 2048} and varied routed IDs, duplicate IDs (if representation allows), partial groups, 43-layer routing and zero/NaN/edge inputs. Compare packed `q8_1` bytes **bit for bit** for every destination row, including padding, sentinel guards and any permutation. Preserve expert results, compressor/indexer state, final hidden state and full-model logits. `q8_1` quantization must use the same block scale and saturation/rounding as the reference.
3. **Format gate:** this path is relevant to the **Q8_1 activation route** for formats such as IQ2_XXS and Q2_K when supported by the actual prefill MMQ. Treat native Blackwell MXFP4 FP4-activation encoding as a distinct path; a V100 must not launch a Blackwell-only `quantize_mmq_fp4_cuda` route. Do not claim all MXFP4 layers are covered by this change without a real dispatch trace.
4. **SM70 performance:** measure quantization calls, memory bytes and intermediate buffers per token, kernel launch count, V100 `ptxas`/SASS, prefill 600/3K/8K/32K and real post-prefill decode with the same GGUF, chunk size and expert cache. The upstream observation of twice as fast a quantizer on RTX 5090 is **not** a V100 prediction. If quantization is under a small fraction of P0 prefill time, this may be deferred despite exactness.

**Exit:** byte-parity fixture, no new sync or transient VRAM regression, and positive **end-to-end** prefill or prompt-to-first-token gain. Deliver `results/p3_mmq_scatter_ab.csv` and log `quantize_calls/token`, `q8_bytes_written`, `prefill_ms`, `TTFT` and `all_in_ms`. Start with `ABYSS_V100_MMQ_ACT_SCATTER=0` fallback; `=1` is the experimental route, not a previously supported Abyss flag.

### P3.13 — Direct resident-slot pointer-table MMQ and group ordering (OPTIONAL)

Upstream's [`STRATA_MMQ_DIRECT`](https://github.com/Niko1221/Strata/commit/7d9b50dc1a5c) avoids gathering some GPU-resident expert weights into a temporary contiguous group; its 16-pointer table is verified for nine quant formats **including IQ2_XXS but not Q2_K or MXFP4** (`src/prefill/moe_mmq_tab.hpp`). Because DeepSeek-V4-Flash has different gate/up, down and exceptional FP4 roles, **measure those roles separately**. First establish how much of Abyss's actual MMQ wall and memory traffic comes from redundant weight copies (versus CPU/PCIe streaming), and count direct-eligible expert bytes; do not extrapolate from unrelated Strata Qwen benchmarks.

If justified, keep a mixed path: only gate/up/down matrix types and shapes independently verified by `mmq_tab_case` use in-place cache pointers; incompatible matrices or streamed experts use the original gather. Allocate and maintain slot-tail overread guards per **format/shape** and account for their VRAM cost. An expert must not be evicted, repacked or reused while MMQ holds its slot; use P3.6's **slot epoch, format version, CUDA-event lifetime** and canonical-copy fallback. Validate bytewise outputs with 1–16 experts per group and differing row counts, stress last/partial group flush and exact prompt caches. Do not treat upstream's own `256 B` slot guard as sufficient for **Q2_K** until an actual target kernel's worst-case overread is proved. The upstream build option `STRATA_MMQ_TAB=ON` is source-local and **not** automatically part of Abyss; pin/fingerprint a vendored ggml scheduler copy if one is necessary, preserve attribution/licenses, and measure CUDA-module VRAM overhead. **Exit:** net speed/VRAM benefit after guard bytes, slot-count loss, and recapture/cross-branch cost, not just fewer copies.

Upstream `STRATA_MMQ_GROUP_SORT` sorts only safe **complete 16-expert groups**, leaving the final partial group intact because different stream-K splits can change FP bits. This is separate from direct pointers and worth considering only when P0 shows MMQ padding dominant and the exact target GGUF shapes/SM70 launch-grid pass a whole-tile check. No semantic router/rank bias. Both are *late* experiments; do not block basic decode/prefill on them.

### P3.14 — Optional Q8_0 hyper-connection weight storage (VRAM tradeoff)

Upstream [`STRATA_HC_Q8`](https://github.com/Niko1221/Strata/commit/b60aab1ba27c) stores some hyper-connection down/up matrices in Q8_0 and dequantizes into stream-local scratch rather than retaining a BF16 copy. **This is not the compressed-KV quantization in P2f.** Assess actual DeepSeek mHC tensors, original pack types, on-device converted copies and consumer precision from the P0 census. Only if HC weights are a meaningful resident VRAM cost, trial per-role Q8_0 with independently measured additional scratch/dequant kernels and newly freed routed-expert slots. Maintain FP16/BF16 baseline, test mHC normalization/mix parity over history/verification and disclose teacher-forced KL drift; it is a **lossy** weight representation and stays default-off. Even if a higher-precision source rounds into Q8 better on a non-V100 machine, no benefit is assumed here.

**Updated P3 priority order (v8):** after full P0, run low-risk, exact-output tests first: (1) P3.12 activation quantize-once/scatter if repeated work exists; (2) P3.2f IQ2_XXS single-token AVX2 comparison where CPU timing dominates; (3) the highest measured DeepSeek-specific SM70 Q2_K/MXFP4/indexer/attention opportunity; (4) Xolotl exact caching/disjoint-residency and bounded prefetch where cache misses dominate; (5) P3.13 direct MMQ only if resident GPU weight gathers dominate; (6) P3.14 Q8 HC only when VRAM saved can yield net benefit. This overrides the relative order described as "v7" below where they conflict; all other v7 correctness/resource requirements continue.


**Earlier P3 priority context (v7, superseded by v8 where different):** (0) audit cache/prefetch/duplicate resident bytes and exact V100 bottlenecks; (1) quantify dominant quant types/shapes/PCIe/CPU stalls; (1a) independently test an exact cache/prefetch or residency change when it can recover critical bytes/time; (2) **test whichever of Q2_K small-T, a Superfast-supported static dense matrix, or MXFP4 has the highest trace-supported potential**; (3) test same-input prep/launch sharing; (4) conditionally repack individual GPU expert slots with amortization and safety gate; (5) trial fixed-shape GEMV only when CPU/GPU sync and compute traces make it worthwhile. Q2_K and MXFP4 are **supported in the newly researched reference implementations**, although **not** Superfast's QPN.

**P3 acceptance:** each intervention reports wall-time change, kernel time, VRAM/RAM/PCIe delta, quality and a reversible control. Keep only demonstrated gains; do not combine independent cache, NUMA, quant and thread experiments into an uninterpretable result.

---

## P4 — Decode, speculation, integration and release

1. Test baseline **`attend_split_kernel` / `attend_combine_kernel`** on V100. Their FP32 decode is already tuned for concurrency; an SM70 Tensor Core decode kernel is **not** a prerequisite. Inspect if decode attention or index selection is on the critical path first.
2. Run `STRATA_DS4_SPEC=0` while isolating P1/P2/**P3-QPN**. Separately re-enable Abyss’s **n-gram** speculative decoder: report accepted-token rate, verification latency by width (1 vs 2/3 proposed), rollback correctness and net tokens/s. Superfast’s 7-token rejection-sampling/MTP performance is **not** applicable without DeepSeek’s missing MTP drafter; a **cost-calibrated acceptance gate** is the transferable idea. Benchmark actual current DeepSeek draft widths before changing them; do not introduce a learned drafter in this release.
3. Audit avoidable launches, CPU/device sync, host-mapped PCIe reads, expert promotion stalls, mask/index metadata copies, and prompt-to-decode transition. Profile **two-token verification** separately from single-token decode: grouped small-batch quantized products and attention schedule can change the winning kernel. Port only hot, relevant tricks from Strata-V100; do **not** import Qwen GDN/KV/PLE-specific code blindly.
4. Build automated SM70 **compile-only CI** (works without a GPU) and real V100 run/parity benchmarks (GPU required) as distinct gates. Verify the fatbin's **per-image SM architecture tags, PTX target, and target-specific SASS**, not an unscoped grep of a multiarch binary. The toolchain's correct `sm_70` targeting usually makes SM80+ instructions in a true SM70 cubin impossible; the real failure modes are **wrong code object selection, unguarded PTX JIT, unsafe dispatch, and incompatible third-party libraries**. See P4a below.
5. Confirm no changes to Ampere+, Turing or HIP behavior; record build success and, where hardware exists, smoke/parity tests.
6. Publish a documented V100 profile and fallback/rollback commands in `docs/V100_DEEPSEEK.md`; preserve default exact-quality routing. **Do not enable** approximate `STRATA_GLM_DEFER` or `STRATA_GLM_TAIL_SKIP` by default.

### P4a Static SM70 fatbin/PTX/SASS isolation and runtime CI

**Compile-only CI** (works on any host that has the compatible CUDA toolchain, without executing GPU code):

1. Pin the supported CUDA 12.8 compiler/toolkit and use an explicit **`sm_70`/`compute_70`** compile route, e.g. `-DCMAKE_CUDA_ARCHITECTURES=70` for the isolated SM70 target or `nvcc -gencode arch=compute_70,code=sm_70` for a one-arch cubin; inspect resulting CMake/nvcc command lines and all objects. Maintain **separate** Ampere/Turing builds if the project normally bundles several image architectures. Do not require compile-only CI to contain licensed `ncu` (P0.3a profiler gate belongs to GPU-run CI).
2. For every new device library/target run `cuobjdump --list-elf <binary>` and `cuobjdump --list-ptx <binary>` (or documented short equivalents) and retain **an image manifest** of compiled and embedded PTX targets. For the SM70 target, use `cuobjdump -sass -arch sm_70 <binary>` where supported; alternatively extract the `sm_70` cubin (`cuobjdump -xelf`) and disassemble with `nvdisasm`. **Assert the selected image tag is SM70**, and verify the actual code object includes expected primitive symbols; do not scan the unrelated SM75/SM80/SM90 cubins as if they were SM70 code.
3. **PTX check vs SASS check:** PTX-source checks target syntax such as `ldmatrix`, `cp.async`, `mma.sync.*m16n8k16`, `wgmma`, `mbarrier`/new async constructs where target-incompatible; SASS uses different mnemonics such as `LDSM`, `LDGSTS` and `HMMA` variants, so `grep 'mma.m16n8k16'` over **SASS** cannot enforce the claimed policy. Use an **architecture-specific tested allowlist/blacklist plus image-target metadata**, not a fragile list including undefined names like `tg4`. Fail if unexpected opcodes/symbols appear in an **SM70 image**; classify ambiguous changes for manual disassembly review. Check inline-PTX compiler/assembler diagnostic output separately.
4. Inspect PTX-only fallback images and JIT behavior: any `compute_80+` PTX cannot be substituted for an SM70 kernel, and any `compute_70` PTX intended to JIT on SM70 must have a validated instruction set. Detect missing `sm_70` implementation and missing fallback **at the host dispatch layer**. Source `#if __CUDA_ARCH__` is not the same as checking a runtime GPU's capability. Audit third-party libraries' supported arch sets.
5. **Real V100 CI** (separate job, necessary for release): execute selected direct kernel launches and end-to-end DeepSeek fallbacks with CUDA error checking, `compute-sanitizer` memory/synchronization checks, and the P1 PTX fragment contract, P2 MLA latent parity and P3 cache-epoch race suite. Require **no `cudaErrorInvalidDeviceFunction`, `cudaErrorNoKernelImageForDevice` or illegal instruction**. Record GPU UUID, build SHA, driver, SM70 binary hash and data hashes. Compile-only inspection cannot prove runtime dispatch or mathematical correctness.

Recommended commands, to be adapted to actual built binaries and installed toolkit options:

```bash
nvcc --version
cuobjdump --list-elf ./path/to/binary-or-library > results/sm70_elf_manifest.txt
cuobjdump --list-ptx ./path/to/binary-or-library > results/sm70_ptx_manifest.txt
cuobjdump --dump-sass --gpu-architecture sm_70 ./path/to/binary-or-library > results/sm70_sass.txt
# Audit PTX for SM70 *after selecting the correct section/image*:
cuobjdump --dump-ptx ./path/to/binary-or-library > results/all_ptx_sections.txt
# No grep of *all* fatbin SASS should be interpreted as a failure for SM70.
```

**CI artifacts:** `results/sm70_arch_manifest.json`, a target-scoped SASS review, `results/sm70_ptx_compatibility.txt`, compiler/driver/tool versions, a real-GPU launch log, and sanitizer reports. Only committed code and license-permitted source artifacts belong in public git. The installed `ncu` executable is tracked by checksum/digest in a private artifact store or a license-compliant image, not pushed into the project by default. [NVIDIA CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html), [PTX architecture restrictions](https://docs.nvidia.com/cuda/archive/11.4.1/parallel-thread-execution/index.html).

### P4g CPU-expert handoff, PCIe 3.0 and NUMA budget — conditional optimization

**Audit existing Abyss synchronization before implementing a new mailbox.** A [GPU-driven MoE doorbell](https://github.com/anyei/llamacpp-v100/blob/master/src/llama-moe-doorbell.h) is only a useful pattern if an `nsys` timeline shows that CPU/GPU stream fences or layer-wise launch gaps are *on the critical path*. Abyss already has pinned mapped buffers and an expert CPU lane, so duplicate metadata handoffs are a risk.

**Two different payload classes must remain separate:**

- **CPU-resident expert compute:** expert weight matrices stay in host DRAM and are consumed by AVX2/FMA CPU workers; a correct mailbox transmits **input activation/hidden vector** (e.g. 4096 F32 values = 16 KiB per token), selected expert IDs, route weights, epoch and readiness state **and** receives an accumulated output of similar size. These are **small transactions**, often latency/coherency/synchronization-limited. The CPU need not copy all expert matrices across PCIe for each miss.
- **GPU promotion/fetch:** when an expert is brought from RAM into GPU cache, its **whole packed expert payload** travels across PCIe. This is a **separate larger transfer**, with different scheduling, page-pinning and overlap/amortization tradeoffs. Do not charge GPU promotion bytes to every CPU-expert miss or assume promotion is free.

**Link check:** the E5 v4 platform supports PCIe 3.0, but measure current generation and link width: **PCIe Gen3 ×16 theoretical payload ≈15.754 GB/s in *each* direction**. A ~11–13 GB/s unidirectional pinned-copy observation is a **starting comparison**, not a measured ceiling on this system; read and write paths, linked buffers, mapped polling, host PCIe addressability and NUMA placement have different behavior. Never label ~12.6 GB/s the combined bidirectional hardware limit. Verify `nvidia-smi -q` / `topo -m` and performance at 4/16/64 KiB, 1/8/32 MiB, with simultaneous CPU expert work and background GPU tier promotions. Record node-0-local vs node-1-remote allocation and intersocket/QPI overhead.

**Critical-path model:** measure `T_mailbox ≈ T_post + T_cpu_compute + T_join + T_required_fences`, but account for true overlap with GPU-resident expert execution (total critical time is closer to `max(GPU_hot_path, CPU_path)` *plus uncovered barriers*, not a naïve sum). For promotion, measure `T_promotion ≈ T_copy_start + bytes / measured_H2D_bandwidth + T_repack + T_publish`, minus overlap outside the critical path. A mailbox's success criterion is **reduced step wall time and CPU/GPU waiting**, not meeting an arbitrary per-layer byte count. When bandwidth dominates, optimize bytes/reuse/overlap or keep CPU computation; when launch/host-mapped latency dominates, evaluate a posted GPU work ring with stable epochs.

**Safety:** step ID and slot epoch, producer/consumer memory ordering, visibility of mapped pinned buffers, aborted requests and speculative rollback, no premature slot reuse, no in-flight DMA source reallocation, bounded CPU-worker spin and release-time synchronization. Preserve existing exact expert weights/results; an optional GPU+CPU overlap change must match full-model logits. Gate by **layer/role + miss-count + packet sizes + measured PCIe/NUMA profile**, rather than enabling globally by whether PCIe is 'below 12 GB/s.'

### P4h — Graph/workspace memory and capture reuse (Xolotl idea; NOT ggml allocator port)

Xolotl attributed roughly **850 MiB** of VRAM and **~120 expert slots** (on *its* 4090/quant/layout) to reusing one graph compute arena for many ggml attention variants. Its code uses graph UIDs, pointer/allocator epochs and a shared allocator, but it also describes **dangling pointers after buffer growth** unless stale graph bindings are invalidated. Abyss is primarily direct CUDA kernels/service streams, not the identical ggml graph runtime. **First measure whether an analogous problem exists.**

- Inventory peak/baseline VRAM for CUDA context/modules, persistent weights, attention raw/comp/indexer caches, MMQ/cuBLAS workspaces, captured graphs and temp buffers *at* 0/8K/32K/64K/128K context. Count graph variants/captures/recaptures, scratch lifetimes, alignment, allocator fragmentation and resulting **additional routed-expert slots** possible after a real reduction.
- If the same temporaries are allocated per graph variant and lifetimes do not overlap, test a **stable-address scratch arena** or VMM-backed shared reserve. Explicitly record graph ownership, pointer-stability, allocation-generation checks, capture/launch synchronization and reference counting. A newly moved/reallocated workspace **must invalidate/recapture** all dependent graph executions; do not ever invoke a captured kernel with stale pointers. Do not bypass v6 `slot_epoch`/`layout_version` for expert slots.
- No universal assumption that “workspace bytes saved = expert slots gained”: include fragmentation, resident-dense reservations and any per-layer expert slot size differences. Benchmark warmed end-to-end performance, captured graph memory, recapture overhead, context growth, repeated prompt reuse, abort and speculative rollback with `compute-sanitizer` when feasible.
- If direct CUDA Abyss has no duplicated graph compute buffers and allocation is stable, mark the Xolotl ggml-specific optimization **not applicable**. Still retain the P0 ledger for future regressions.

### P4i — Online measured PCIe versus Broadwell CPU work split (HIGH, after P0)

Upstream [`STRATA_PCIE_BALANCE=1`](https://github.com/Niko1221/Strata/commit/04be241a5a7e) estimates how much of the **per-layer expert miss set** the PCIe/GPU and CPU lanes should take, based on recent measured completion. Abyss `glm_fast_path.cu` already runs startup experiments for **DMA**, GPU mapped-zero-copy and CPU expert work, and builds a fixed split plan. **Do not replace that baseline without testing whether its predictions diverge from the real per-layer wall**.

1. **Model and logging:** for each supported miss count and layer/format, log `n_cpu`, `n_pcie`, `cpu_time_us`, `gpu_expert_time_us`, `copy_engine_time_us`, `gpu_wait_for_cpu_us`, `cpu_wait_for_gpu_us`, `pcie_bytes`, CPU threads/node, and thermal/PCIe conditions. Observe overlap: minimizing `CPU_time + PCIe_time` can be wrong when the true service time is `max(CPU_lane, GPU_lane) + barriers`. Distinguish zero-copy GPU work from H2D promotion. Detect material shared-memory and QPI contention from concurrent CPU + GPU reads.
2. **Conservative online estimator:** start from Abyss's calibrated plan and update with bounded/forgetful per-layer or per-format cost estimates **only when enough samples exist**. Test simple bounded discrete miss-count allocation against upstream's fitted `a + c*n_cpu + d*n_pcie` / `g0 + g*n_pcie` approach. No unbounded negative slopes, rapid flip/flop, oscillation in idle cases, or route-dependent *identity* changes. Hardware-based requirements: one V100, exact expert outputs, no speculative committed work reordering, correct time-of-use slot generations.
3. **Workloads:** compare fixed baseline, conservative adaptive and forced CPU/PCIe extremes on short/8K/32K/filled long prompts, warm→cold expert transitions, high/low host contention and socket-local/interleaved placement; assess P0's thermal and QPI clean-run gate. A/B/A ≥5 warm repeats plus a topic-switch sequence. Measure actual end-to-end decode, not only service-loop prediction error. Assess estimator cost/CPU overhead, stability, fallback when counters or runtime timings are missing.
4. **Reject/rollback:** if adaptive allocation saves one microbenchmark but adds PCIe bytes, stalls CPU RAM, regresses overall decode, or fluctuates unpredictably, revert to the startup plan. No static upstream PCIe fractions transplanted from a different CPU/GPU. Upstream reported that the strategy could **save nothing** on DRAM-bound host work even when PCIe counts changed; remeasure on dual Broadwell/PCIe 3.

Deliver `results/p4_cpu_pcie_balance.jsonl`, the per-layer fixed versus adaptive policy ledger, and an explicit zero-extra-accuracy-cost parity test. Proposed A/B control: `ABYSS_V100_PCIE_ADAPT=0|1`, default `0` until measured and qualified.

### P4j — Conditional asynchronous swaps, pinned ranges and exchange rotation

Upstream `STRATA_ADAPT_OVERLAP`, `STRATA_PIN_BY_NEED` and `STRATA_EXCHANGE_ROTATE` were developed for Strata's resident RAM/cache model. They are **not** independent improvements until we verify their underlying bytes and epochs occur in Abyss. Upstream reported device-dependent regression for `STRATA_ADAPT_OVERLAP`; do not enable it indiscriminately on V100.

- **Async overlap:** identify copies that block the **GPU decode critical path** rather than merely staging ahead of it (Abyss already has nonblocking streams and a service thread). If an exact demand/copy/commit barrier can be advanced, use CUDA events to sequence `source valid → copy enqueued → copy done → format/slot metadata published → readers resume`. Stress aborted requests, speculation rollback, CPU readers, cross-thread closure, same-layer victim in use and mixed quantization. Keep `ABYSS_V100_TIER_OVERLAP=0` baseline.
- **Selective host pinning:** when pin-registering the entire RAM tier fails or constrains OS memory, rank which *specific* expert ranges benefit from GPU mapping and CPU access; register page-aligned ranges with ownership metadata, preserving address stability. Upstream's `STRATA_PIN_BY_NEED` primarily addressed WDDM pin-cap failures and is **not inherently better than Abyss's working Linux NUMA pinned tier**. With 128-GB host RAM and 70+ GB resident experts, avoid creating extra duplicate full-sized pinned copies. Document mapping fractions and host page faults. No change unless measured pinning problem.
- **Exchange rotation:** only if a captured profile shows a host **commit memcpy** of an evicted expert into an occupied/old resident RAM slot. Ownership rotation may omit that memcpy; it is **not** a reduction of PCIe promotion traffic. Need size-class-matched scratch buffers, publication epochs and GPU mapped pointer/alias consistency, and enough additional pinned memory. Preserve P3.6 canonical bytes. Compare memcpy saved with reserve/fragmentation and NUMA placement costs.

**Stop:** if Abyss has no analogous synchronous host commit path or sufficient available pinning/VRAM, mark *not applicable*. Do not build a second expert cache or violate the invariants of Xolotl-derived disjoint host/GPU residency.

### P4k — Optional prefill-head VRAM alias and exact server recovery

Upstream [`--prefill-head-alias`](https://github.com/Niko1221/Strata/commit/640ac438be9e) temporarily lends output-head VRAM during prefill and restores it before decode. On your 32-GB V100, this can help **only** if P0 demonstrates a real head-versus-prompt-workspace overlap limit and reclaiming it increases useful chunk width/cache room. The head may cost one or more PCIe 3 transfers, prompt→decode TTFT and pointer/graph invalidation; quantify all of these. Never evict/reload a head or shift the live weight owner without epoch fencing and model hash checks. On no measurable V100 prefill allocation pressure, **do not implement**.

Optional service safety: upstream's nonfinite-logit recovery (`STRATA_NAN_GUARD`) is a *correctness idea*, not a DeepSeek-specific plug-in. Compare exact snapshot/restore state (CSA/HCA append positions, raw rings, host and GPU aliases) before adding any recovery; nonfinite logits should fail closed with a diagnostic rather than silently leave a corrupted cache. No performance claim.


### Proposed controls and precedence

| Variable | Status | Values / effect |
|---|---|---|
| `ABYSS_V100_IDX` | **NEW**, proposed | `auto` (default), `volta`, `cublas` (batch only), `f32`: DeepSeek lightning-indexer scorer |
| `ABYSS_V100_ATTN` | **NEW**, proposed | `auto` (default), `volta`, `f32`: DeepSeek batched attention |
| `ABYSS_V100_PROFILE` | **NEW**, proposed | `0` (default) or `1`: combined stage/critical-path ledger |
| `ABYSS_V100_MMQ_ACT_SCATTER` | **NEW**, proposed | `0` baseline / `1` experimental; once-per-token Q8_1 activation quantization followed by exact scatter (P3.12); do not affect unrelated dense/indexer paths |
| `ABYSS_V100_CPU_IQ2_EXACT1` | **NEW**, proposed | `0` baseline / `1` selectively enable verified Broadwell AVX2 singleton IQ2_XXS row implementation (P3.2f); no dispatch for Q2_K |
| `ABYSS_V100_PCIE_ADAPT` | **NEW**, proposed | `0` fixed startup-calibrated split / `1` bounded per-layer online CPU/PCIe split (P4i); honor explicit existing `STRATA_GLM_PCIE_SHARE`/`STRATA_GLM_CPU_PLAN` controls when set |


| `ABYSS_V100_QPN` | **NEW**, proposed | `off` (development default), `dense`, `hot-expert`, `auto`; quant-aware repack/MMA routing subject to P3.5–3.6 gates |
| `ABYSS_V100_QPN_WIDTH` / `_GROUP` / `_PREP_SHARE` | **NEW**, proposed | Independently tune token pass width, compatible GEMV fusion and same-input activation preparation |
| `ABYSS_V100_Q2_MMA` | **NEW**, proposed | `off` initially; `on` for Q2_K-only isolated trial and later role/width-tested `auto` |
| `ABYSS_V100_MXFP4_MMA` | **NEW**, proposed | `off` initially; `on` for MXFP4 E8M0/32 resident expert MMA, later tested `auto` |
| `ABYSS_V100_CPU_MAILBOX` | **NEW**, proposed | `0` default; `1` enables audited pinned work-ring after P4g gate |
| `ABYSS_V100_SLOT_FORMAT_DEBUG` | **NEW**, proposed | `0` default; `1` checks GPU slot generation, format/version and safe publication during P3.6a stress tests (not a production timing baseline) |
| `ABYSS_V100_HW_PROFILE` | **NEW**, proposed | `0` default; `1` emits lightweight trace tags/counters; hardware `ncu` collection is external and **not** on production hot path |
| `ABYSS_V100_CACHE_POLICY` | **NEW**, proposed | `baseline` default; `layer-lru`, `lfu-hysteresis`, optional `cross-layer` only after slot-ownership redesign; P3.2c |
| `ABYSS_V100_RAM_DISJOINT` | **NEW**, proposed | `0` default; `1` opt-in no redundant pinned host copy of GPU-seeded experts with guaranteed file/CPU fallback |
| `ABYSS_V100_PREFETCH_BUDGET` | **NEW**, proposed | `baseline` default; `0`, or eligible expert count/byte budget after P3.2d; **must not change route** |
| `ABYSS_V100_COMP_KV` | **NEW**, proposed | `f16` default; `q8_0`, `iq4_nl` conditional P2f; explicitly **not** Xolotl's CLI `--comp-type` |
| `ABYSS_V100_IDX_KV` | **NEW**, proposed | `f16` default; `q8_0` conditional P2f; separate from compressed attention KV |
| `ABYSS_V100_COMP_HOST` | **NEW**, proposed | `0` default; `1` experimental pinned-host/mixed gather with PCIe/NUMA audit |
| `ABYSS_V100_MMQ_DIRECT` | **NEW**, proposed, OPTIONAL | `0` baseline / `1` pointer-table MMQ only for audited compatible quant formats + guarded slots (P3.13); not Q2_K/MXFP4 by inference |
| `ABYSS_V100_HC_Q8` | **NEW**, proposed, OPTIONAL | `0` baseline / `1` lossy Q8 HC weight cache with separate numerical and VRAM validation (P3.14) |
| `ABYSS_V100_TIER_OVERLAP` | **NEW**, proposed, OPTIONAL | `0` baseline / `1` experimentally overlapped expert-cache transfers with epoch checks (P4j) |


| `ABYSS_V100_GRAPH_ARENA` | **NEW**, proposed | `baseline` default; `reuse` only after pointer-stability and capture-lifetime proof; P4h |
| `ABYSS_V100_ROUTE_WARMSTART` | **NEW**, proposed | `off` default; `read`, `readwrite` guarded by model hash/version and atomic persistence; exact expert selection preserved |
| `ABYSS_V100_ATTN_SPLIT` | **NEW**, proposed | Optional occupancy-aware decode/verify split-count A/B, baseline otherwise unchanged |
| `STRATA_DS4_IDX=1` | Existing | Legacy scorer FP32 force, **highest precedence** |
| `STRATA_DS4_ATTN=1` | Existing | Legacy batched-attention FP32 force, **highest precedence** |
| `STRATA_DS4_IDX_ALL=1` | Existing | Enables indexer even when visible rows fit the top-k; useful for tests only |
| `STRATA_DS4_SPEC=0` | Existing | Disable speculation for deterministic kernel benchmark |
| `STRATA_GLM_PREFILL_PROF=1` | Existing | Stage-level prefill timing |
| `STRATA_GLM_GAP_PROF=1` | Existing | Decode intertoken gap diagnostics |
| `STRATA_GLM_BG_PACE_KB` | Existing | Controls incremental expert-cache refill/promotion copy pacing |

**New controls are proposals, not present in pinned Abyss.** Upstream-only `STRATA_QUANT_GATHER`, `STRATA_CPU_IQ_ALL_EXACT1`, `STRATA_PCIE_BALANCE`, `STRATA_MMQ_DIRECT`, `STRATA_MMQ_TAB`, `STRATA_HC_Q8`, `STRATA_ADAPT_OVERLAP`, `STRATA_PIN_BY_NEED`, `STRATA_EXCHANGE_ROTATE` and `--prefill-head-alias` are **not Abyss launch switches**. New Abyss flags require explicit implemented code; existing Abyss `STRATA_GLM_PCIE_SHARE` / `STRATA_GLM_CPU_PLAN` forced settings take precedence over optional adaptive balancing. Never assume enabling an environment variable without implementing corresponding dispatch changes a kernel. Ensure the new Q2/MXFP4 controls are mutually consistent with `ABYSS_V100_QPN` and with `STRATA_DS4_*` existing overrides; compare unmodified baseline by keeping all proposed controls off.

**Xolotl-only variables are not Abyss features:** `--vram-lru`, `--arena-skip-resident`, `--pf-b`, `--comp-type`, `--icomp-q8`, `--comp-host`, `DS4_FUSED_IDX`, `DS4_RAW_BLOCK`, `DS4_GRAPH_REUSE`, and `--pos-offset` come from the **reference DS4 program**, not pinned Abyss. The proposed `ABYSS_V100_*` controls above require an explicit implementation; never use Xolotl flags as if they affected Abyss. All exact-quality defaults remain the current Abyss semantics.  

**Superfast-only variables are not Abyss features:** `LLAMA_MMVQ_QPN`, `LLAMA_QPN_MAX_TOKENS`, `LLAMA_QPN_W16`, `LLAMA_QPN_GROUP`, `LLAMA_QPN_SHARE`, `LLAMA_MMVQ_TC`, `LLAMA_FATTN_VERIFY_STREAM`, and `LLAMA_QSA_ATTN_MMA` appear in the reference fork, **not** in the current Abyss baseline. Do not put them in Abyss launch configurations as if already supported. A new control needs an explicit implementation and test.

**Forced-path semantics:** If an existing `STRATA_DS4_* =1` FP32 override is set, it wins. If `volta` is requested on unsupported hardware or a shape the kernel cannot process, issue a descriptive error; **never launch an architectural trap**. `auto` may choose FP32 based on shape or calibrated speed. Log dispatch decisions once per kernel shape/device. Benchmark and release defaults must not silently change quant, thread count or expert-policy behavior.

---

## Validation, benchmark protocol and project gates

### Evidence levels

**Level A — Isolated kernel correctness:** Synthetic FP64 reference where feasible, F32 baseline, sanitizer, edge masks, **repack→unpack exact byte check for format transforms**, overflow/underflow/nonfinite tests, actual SM70 launch. Explicit error distribution, never just one mean.

**Level B — Model parity:** Same GGUF/pack, same token IDs, same context and speculation setting. Teacher-forced next-token logits, mean/p95 KL or other justified distributions, top-1 agreement, top-k cutoff margins and borderline selections. Include compressed-boundary probes, long histories, generated output verification and memory safety.

**Level C — End-to-end performance:** Report separately **repack startup milliseconds, per-GPU-expert promotion repack microseconds, extra VRAM, expert residency/churn and same-input-prep overhead**, not just the arithmetic kernel. Same hardware, clocks, power, NUMA, pack, threads and cache state, A/B interleaved, ≥5 repeats. Report startup, TTFT, prefill tok/s, decode tok/s, warm and cold, stage ms, CPU, PCIe, RAM/VRAM and temperature. Performance results must state whether speculation is on. **Thermal/power gate:** Every timed pair must include timestamped `nvidia-smi`/NVML thermal, clock and power samples, the active throttle reasons, the applied power policy and an explicitly assessed sustained thermal state. Flag and retain thermally limited/unmatched pairs but do **not** cite them as clean kernel-regression evidence; power-cap-limited matched pairs are a separate documented regime. Re-run after stabilizing temperature/airflow when necessary. Record chip and HBM temperature if the device/driver exposes both; an unavailable HBM sensor is marked unavailable, not invented. **NUMA/QPI gate:** record worker affinity, actual page placement, intersocket traffic availability, and separate CPU lane vs GPU DMA critical-path costs. **Mathematical gate:** all P2 results include the MLA latent parity test and P3 repacked-expert runs include slot-epoch/pointer reuse tests.

### v7 Xolotl-derived validation gates (additional to A/B/C above)

| Feature | Hard correctness gate | Performance/resource gate | Evidence artifact |
|---|---|---|---|
| Expert LRU/admission | Same routed expert IDs, weights, output; safe per-layer victim with `slot_epoch` and no ABA | Weighted missed bytes, GPU/CPU/SSD critical path, stable slot budget, repeat topic switch | `results/p3_cache_ab.jsonl` |
| Disjoint VRAM/RAM residency | CPU/file fallback for evicted expert; canonical source; correct source hash | Pinned RAM/VRAM bytes, GPU slot count, cold NVMe/page-cache cost, CPU wait | `results/p3_disjoint_ab.csv` |
| Prediction/prefetch | Actual router unchanged; no stale stage DMA publication, rollback safe | Useful vs wasted prefetch bytes, measured overlap, CPU memory/PCIe competition, net wall | `results/p3_prefetch_ab.jsonl` |
| Q8_0/IQ4_NL compressed caches | K=V, RoPE/inverse RoPE, sinks, visibility, `icomp` headwise ReLU, top-512 near-ties; quantify *lossy* KL/task drift | Comp+indexer bytes, decode dequant time, added expert slots, 32K–128K filled prefill/decode | `results/p2_kv_quant_quality.csv` |
| Host cache | Consistent compressor/gather state and page lifetime | PCIe 3 ×16, NUMA/QPI and CPU contention; savings minus per-row gather cost | `results/p2_comp_host_ab.csv` |
| Bounded prefill / graph reuse | Identical completed cache/forward state across chunk and graph rebuild; no stale pointers | TTFT, bytes per chunk, steady decode, peak arena/VRAM, captured variant counts | `results/p3_prefill_chunk_ab.csv`, `results/p4_graph_ab.csv` |
| Warm routes / hysteresis | Hash/quant/schema check, no route bias/drop, atomic profile update | First-N-token latency, A→B→A recovery, actual SSD misses and warm reuse | `results/p3_warmstart_ab.csv` |

**Metric distinctions:** report `(measured source result)`, `(Abyss hypothesis)`, and `(measured V100 result)` separately. Benchmarks on different quant files or RTX 4090/5090 are not **apples-to-apples**. Xolotl's empty-cache long-context `--pos-offset` probe cannot fulfill Level C **filled-context** validation. Isolated GPU cache efficiency does not establish model quality; cache quantization is an intentional numerical intervention and requires separate quality sign-off even if all visible key locations are valid.

### v8 post-fork Strata admission gates (additional to v7; preserve P0–P4 order)

| Candidate | Must show at Level A/B | Must show at Level C | Artifact |
|---|---|---|---|
| P3.12 scatter Q8_1 activation | Bit-identical destination activation blocks, aligned/scattered row indices, duplicate/partial group and DeepSeek whole-model parity | Saved real quantized bytes/launches, prefill and TTFT, unchanged peak VRAM | `results/p3_mmq_scatter_ab.csv` |
| P3.2f exact IQ2 AVX2 singleton | CPUID/runtime ggml backend correct, per-row/reference and complete expert output parity, not dispatched on Q2_K | Broadwell CPU `ms/expert`, node-local/interleaved, QPI, critical-path improvement | `results/p3_cpu_iq2_singleton.csv` |
| P4i adaptive CPU/PCIe split | Identical routed IDs, weight outputs and correct host/device timeline; bounded estimates and deterministic forced fallback | Measured CPU and PCIe wait, percent of windows improved, stable full-model decode over changing cache and host load | `results/p4_cpu_pcie_balance.jsonl` |
| P3.13 direct-cache MMQ | Supported GGUF types **only**; no slot over-read, stale pointer or eviction race; every Q2_K/MXFP4 fallback exercised | Gather bytes saved minus guards, extra modules VRAM, realistic group and overall prompt speed | `results/p3_mmq_direct_ab.jsonl` |
| P3.14 Q8 HC | mHC tensors correct, explicit precision tradeoff, teacher-forced logits and task scores | VRAM saved and *actual* additional expert slots; dequant overhead amortized | `results/p3_hc_q8.csv` |
| P4j/k other conditional runtime changes | Event/epoch/mapped-alias safety and complete rollback; no unqualified lossy route changes | Measurable critical-path benefit, pin budget, slot budget, load/reload overhead and TTFT | `results/p4_tier_head_ab.jsonl` |

**Cross-branch regression control:** Because Abyss has newer NUMA/hugepage/SSD-fetch changes beyond the pinned baseline, take separate baseline measurements for each selected code lineage. Do not attribute a gain from changing Abyss commits to the new Strata-derived optimization. Upstream reported performance and parity are **hypothesis support**, not substitute Level C V100 evidence.


### Preliminary numeric gates (calibrate in P0)

- Target kernel **relative L2 ~1e-5 where feasible**, with `1e-4` as a **provisional investigate/fail point for ordinary data**, not a blanket promise that all edge cases meet it.
- Check mean KL over sampled teacher-forced positions **and** per-position outliers; use the same baseline pack, document sampling and zero/near-zero probabilities. A suggested incremental budget of **0.002 mean KL over P0** is a **working review threshold, not a substitute for task quality**; revise after measuring baseline run-to-run variation and sensitivity.
- Any meaningful top-k disagreement away from a near tie, wrong causal visibility, incorrect sink normalization or invalid CUDA memory behavior is a **hard failure**, regardless of L2/KL.
- No required numerical throughput uplift until P0 identifies where runtime is actually spent. If P1/P2 do not improve representative **total** prefill, keep them experimental and select F32 by default.

### Updated experimental sequence and stop rules (post-P0)

The following integrates the external-research order with the user's hardware constraints. **No path is removed solely because of one 40%/50% occupancy sample; the stop rule is controlled model-correct wall-time evidence.** P1 and P3 can run in parallel once P0 is complete; priority within them follows measured percentage of decode or prefill critical time.

| Order | Experiment | Required measurements | Decision / stop rule |
|---|---|---|---|
| **0 — mandatory** | P0 ancestry/source-feature map, source-vs-runtime comparison, type/timing census, Volta counters, **sustained GPU thermals/clocks**, PCIe/QPI/NUMA, SIMD and model parity | `(N,K,type,T)` census; HBM/L2, SM/HMMA, spill/occupancy; measured QPI/local-vs-remote memory, clocks/throttle reasons; critical path and baseline quality | Block kernel promotion if reference forward fails or clean matched thermal/perf attribution cannot be established; reconsider compute-only kernels when the bottleneck is proved to be HBM/PCIe |
| **1A — first exact prefill candidate** | P3.12 quantize-once/scatter if P0 finds repeated per-expert activation quantization | Format/API compatibility, byte-identical Q8_1 destinations, P0 quantizer time, end-to-end prefill+TTFT | Close N/A if no redundant quantization; do not merge if padding/indices differ or measured all-in gain absent |
| **1B — first CPU candidate** | P3.2f exact single-token IQ2_XXS AVX2 on Broadwell | Correct ggml AVX2 backend, exact ggml and expert parity, node-local/remote timings and end-to-end CPU lane | Retain Abyss existing singleton kernel for formats/widths where faster; no Q2_K misdispatch |
| **1C — after stable fixed split** | P4i bounded online per-layer CPU/PCIe miss allocation | `n_cpu/n_pcie`, uncovered waits, DRAM/PCIe/QPI, changing workload, quality and no oscillation | Return to calibrated fixed split if net decode slows, overlap/CPU memory worsens, or estimate unstable |
| **1 — conditional** | P3.5d Q2_K 2–3-token MMA vs native kernel | Real type/time share, scale/zero cases, registers, local spill traffic, all-in milliseconds, token logits | Retain native path when slower or numerical error; low occupancy alone is not a kill condition |
| **2 — conditional** | P1 MMA884 scorer vs shared-key FP32-scratch cuBLAS and FP32 baseline | **PTX operand/lane-map gate**; C4 visible rows/128-D heads, 1–32/256-token workloads, top-512 margins, scratch | Reject wrong fragment coordinates, rank/visibility errors or exhausting VRAM; dispatch by winning `(T,visible_rows)` only |
| **3 — conditional** | P3.5e MXFP4 E8M0/32 resident expert M=1,2,3 | Precise E2M1/scale semantics, effective affected layers, pack and promotion cost, register/occupancy, FP32 logits | Stop unless stage and model benefit amortizes pack and preserves CPU/MMQ consumers |
| **4 — conditional** | P2 raw/CSA/HCA MMA884 and producer/consumer attention | **MLA latent K=V parity** with exact RoPE/YaRN/compressor/sink/`wo_a`/`wo_b` stages, shared memory, long-context prefill | Reject memory-safety errors, inaccurate latent math, spill-heavy **slower** path, or no end-to-end benefit |
| **5 — conditional** | P3.5f fixed-shape GEMV + P4g CPU mailbox handoff | Broadwell AVX2/FMA, actual pinned page node/worker node, **QPI utilization**, PCIe traffic, waits and critical path | Stop when **uncovered** DMA/QPI/sync overhead exceeds savings; do not charge full expert bytes to a CPU miss |
| **6A — early alongside P1/P3** | Xolotl-inspired layer-local LRU/admission, disjoint residency, bounded predictive prefetch | P0 expert/slot/byte trace, exact route parity, warm/cold/topic switch, NUMA PCIe/QPI contention | Stop if existing Abyss policy is already better; adopt only measurable net model latency benefit at same budget |
| **6B — long-context conditional** | Compressed FP16→Q8_0 and optional rotated IQ4_NL, separate indexer-key Q8 | 32K–128K **filled** context, selection margins/quality, dequant time, saved GPU expert slots, CPU/GPU traffic | Keep FP16 unless quant saved slots pay for new work **and** lossy-quality review passes |
| **6C — low-overhead optional** | Prefill chunk/staging, **P3.12 activation scatter**, graph arena and warm profiles | Same context pack, TTFT, full prefill+decode, context graph recapture, SIMD and epochs | Close as not applicable if Abyss's existing direct CUDA path already avoids ggml overhead |
| **6D — late, evidence-gated** | P3.13 direct-cache MMQ and group sort, P3.14 Q8 HC, P4j asynchronous swap/pin/rotation, P4k prefill-head alias | Actual format compatibility (Q2_K/MXFP4 fallbacks), exact epochs, extra VRAM/pin, lossy HC quality, TTFT/reload | Skip any feature without a measured underlying staging/memcpy/workspace bottleneck; keep individual switches and original host/data owner |

| **7 — independent** | NUMA/cache/quant/prestage/pacing sweeps | VRAM, RAM, NVMe, speculation, end-to-end A/B and exact input hash | Retain only reproducible improvements with specified memory/quality costs |

**Scope of stop rules:** A poor result stops that implementation/shape, not the entire phase. Roofline-bound kernels can still benefit from **moving fewer bytes**, higher reuse or overlapping PCIe; only a no-change-in-bytes/no-change-in-overlap *compute-only* rewrite lacks a credible benefit. Use predeclared statistical repeat and noise margins rather than one threshold from a different engine.

**Revised first implementation batch:** P0 branch-delta feature audit and a clean GPU/CPU baseline → P3.12 exact Q8_1 scatter microtest → P3.2f exact IQ2_XXS AVX2 test → P4i adaptive CPU/PCIe A/B only if observed fixed-plan misallocation. P1/P2 V100 mathematical parity work and existing Q2_K/MXFP4 trials proceed by measured hotspot priority. The following v7 experiment rows remain conditional and are **not** automatically completed by adopting source ideas.


### Dashboard

| Gate | Measure | Passing condition |
|---|---|---|
| P0 runtime | Real UD-Q2_K_XL full forward, prompt and decode | Complete, no illegal launches/oom, baseline reference and critical-path ledger |
| **P0 hardware counters/tool** | Volta-supported profiler version, checksum/license provenance, HBM/L2/SM/HMMA, spill, occupancy, CPU/PCIe and NUMA | Source-verified counters with missing data explicitly reported; never silently substitute unsupported `ncu` |
| **P0 thermal/power** | Time-aligned NVML thermal throttling reasons, clocks, power, matched sustained-state A/B | Clean or explicitly matched power-cap-limited pair; thermal events flagged, preserved and rerun |
| **P3 QPI/NUMA** | Worker/page locality, Intel PCM/uncore QPI if available, remote DRAM, PCIe DMA | No unmeasured cross-socket saturation; per-allocation policy grounded in end-to-end timing |
| **P1 PTX register** | PTX `r` packed F16 and `+f` accumulators, four quadpair lane maps and real V100 test | Exact matrix fragment orientation/coordinate coverage and correct warp-uniform MMA |
| **P2 MLA latent** | 512-D shared K=V, two RoPE modes, compressor boundaries, sink, inverse RoPE and `wo_a`/`wo_b` | Paired reference and SM70 stage parity; no fabricated up-projection |
| **P3 cache epoch** | `slot_epoch`, `layout_version`, state+event, CPU fallback and ABA race tests | No premature publish, stale reader, mixed-format view or incorrect pointer reuse |
| **P4 SM70 CI** | Image manifest, targeted SM70 SASS/PTX scan, real-device dispatch, numerical and sanitizer tests | No unsupported opcode in SM70 image, no runtime wrong-code-object, compile/runtime gates separated |
| **P3/P2 occupancy & spills** | Registers/thread, spill loads/stores, achieved/theoretical occupancy, local traffic, CTA residency | Review <50% theoretical / <40% achieved / ≥32 B/thread spill; **pass/fail by correctness and all-in runtime**, not occupancy alone |
| **P4g host/PCIe handoff** | Two payload categories, measured pinned H2D/D2H, CPU AVX2/FMA, barriers and overlap | Net reduced decode critical path with exact CPU expert contribution and safe epochs |
| P1 arithmetic | Per-head dots, score, top-512 and margins | Correct semantics, acceptable numerical drift, safe boundaries |
| P1 performance | Score/select latency, P1+model prefill/decode | Measured speed tradeoffs; no default regression |
| P2 arithmetic | QK, sink softmax, PV, inverse RoPE for r=0/4/128 | Component and model parity; no masks/selection errors |
| P2 performance | Attention ms and *end-to-end* prefill, occupancy | Prefer SM70 only for measured winning shapes |
| P3 memory/offload | Expert residency, CPU/PCIe, memory pressure, quality | Reproducible reversible benefit and no resource failure |
| **P3-QPN static dense** | Real type/shape/width, 1–64 tokens, pack/prep/core and overall time, numerical errors | Strict route table; no slower automatic path; preservation of all tensor consumers |
| **P3-QPN hot experts** | Promotion/eviction/CPU-lane/MMQ interactions, amortized reuse, VRAM | Canonical source intact, slot epochs safe, no increase in critical-path time or unintended CPU read |
| P4 release | Cross-arch builds, real SM70 tests, docs, repeats | Safe fallback, reproducible result, known limitations |

### Pull request / task breakdown

| PR | Deliverable | Exit evidence |
|---|---|---|
| **P0a** | CUDA12/SM70 build, load/forward fixes and memory ledger | Complete response, initial long-context parity |
| **P0b** | Profiler, performance harness, sampled logits, thermal/QPI telemetry and kernel inventory | Baseline JSONL, Nsight trace, Volta-profiler version manifest, sustained clocks/throttle, NUMA/QPI and ranked bottlenecks |
| **P0c** | Xolotl comparator cache/prefetch/VRAM/graph ledger and filled-context fixtures | `p0_expert_trace`, per-layer weighted misses, predictor waste, duplicated residency, context memory, source-vs-V100 evidence labels |
| **P1a** | MMA884 fragment map, typed inline-PTX constraints and single-token indexer | `r`/`+f` operand test, emitted SM70 SASS, four quadpair lane-map parity, index scores/top-512 |
| **P1b** | Batched indexer with a key-reuse variant and runtime dispatch | Multiple shapes, register/smem ledger, P1 model parity |
| **P2a** | MMA884 QK + sink-softmax + PV and **MLA 512-D latent parity harness** | Numerics, RoPE/inverse-RoPE, compressor boundary and grouped `wo_a/wo_b` reference stages |
| **P2b** | Raw-only r=0 batch attention | Short window, wrap/chunk and inverse-RoPE parity |
| **P2c** | CSA/HCA gathers, masks, selection, adaptive fallback | Ratio-4/128 full boundary parity and prefill benchmark |
| **P2f (optional)** | Compressed FP16 KV→Q8_0 / rotated IQ4_NL and indexer-key Q8 | Exact structural semantics, documented quality drift, real 32K–128K filled context, added GPU expert slots, explicit fallback |
| **P2g (conditional)** | Raw-window query-block and masked-union audit | If Abyss already directly gathers only actual 128-key window, record **N/A** without code changes |
| **P3a** | Measured NUMA/PCIe/expert/CPU-lane tuning | End-to-end controlled A/B report |
| **P3a-xolotl (conditional)** | Expert LRU/admission + disjoint residency + bounded prefetch, warm profile/hysteresis | Replay with same actual routes and slots, cold/warm/topic switch, epochs, all-in decode and file fallback; no lossy router edits |
| **P3a-prefill (conditional)** | Chunk/prestage A/B and repeated-activation quant-prep audit | MMQ quant type/SM70 legality, critical bytes/launches, peak VRAM, TTFT and post-prefill decode; stop if no actual opportunity |
| **P3-strata-act (HIGH)** | Upstream-style quantize once + Q8_1 scatter in real DeepSeek MMQ path, with format guards | Byte-identical full rows, model parity, saved quantizer work, end-to-end V100 prefill / TTFT A/B |
| **P3-strata-iq2 (HIGH)** | IQ2_XXS exact AVX2 singleton versus existing Abyss CPU dot on Broadwell | Identical reference/row bytes, verified real CPU ISA dispatch, per-socket/QPI throughput and full decode benefit |
| **P4-strata-pcie (HIGH, conditional)** | Bounded online per-layer CPU/PCIe miss allocation against startup plan | Exact expert results, estimator consistency, A/B critical path and stable multi-context decode |
| **P3-strata-mmq (OPTIONAL)** | Verified pointer-table MMQ direct-cache path and partial group-safe sorting | Format whitelist excludes unsupported Q2_K/MXFP4, guard bytes, cache epochs, in-place vs gathered all-in benefit |
| **P3-strata-hc (OPTIONAL)** | Q8 HC-weight residency with independent DeepSeek mHC parity and scratch ledger | VRAM→expert slots, lossy numerical sign-off, decode/prefill gain and correct rollback |
| **P4-strata-tier (OPTIONAL)** | Async tier overlap / selective pin / exchange rotation and head alias only if P0 identifies a bottleneck | Correct alias/epoch ownership, no NUMA/pin regression, proven V100 latency and memory benefit |


| **P3b** | Weight/quant/shape inventory and **static dense QPN MMA884 test** | Type-specific kernel microbench, source-layout/unpack parity, DeepSeek end-to-end A/B |
| **P3c** | Optional same-input prep sharing / compatible GEMV fusion | Correct dependency graph, per-call and whole-model gain |
| **P3d (conditional)** | Hot-expert GPU-slot repack, explicit format-version/slot-epoch transitions and routing | Immutable CPU canonical copy, race/ABA/eviction fallback tests, repack amortization, no VRAM regression |
| **P3e** | Independent **Q2_K** small-T MMA from anyei reference and **MXFP4 E8M0/32** skinny expert trials | Q2_K zero-scale and MXFP4 decode unit harness, occupancy/spill, real DeepSeek weight/activation logit and all-in A/B |
| **P3f (optional)** | Independent quant or small fixed-shape GEMV variants | Separate packs, KL, source-specific role and fully overlapped graph speed |
| **P4g (conditional)** | PCIe/NUMA-aware CPU expert pinned work-ring with AVX2/FMA dispatch audit | No full-weight per-miss transfers, measured mailbox/promotion time, safe abort/epochs, end-to-end wall improvement |
| **P4h (conditional)** | Stable-address graph/workspace arena and optional variant reuse | Actual duplicated allocation first; lifetime/recapture/epoch and sanitizer tests; measured regained GPU expert slots |
| **P4** | Spec/decode tuning, separate **fatbin/SM70 SASS compile-only CI** + real V100 runtime/quality CI, docs | Arch-tag/PTX image manifest, target-scoped disassembly, no wrong dispatch, stable branch, flags/rollback |

For each PR include fixed git SHAs, model hash, compiler/driver, GPU clocks and topology, build arguments, selection flags, benchmark command, raw results, numerical trace and *exactly which fallback path remains viable*. Minimize changes to self-hosted runners and preserve the working baseline build between candidate tests.

---

## Primary source references

These locations are **implementation sources**, not evidence that DeepSeek V4 has been run on the target V100.

1. [Abyss pinned base commit 47001f3](https://github.com/zephel01/Abyss/commit/47001f31cbfba14bcbf4c083e3af07c50707ec0c)
2. [Abyss README](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/README.md) and [manual](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/docs/MANUAL.md)
3. [DeepSeek architecture contract](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/docs/deepseek4/SPEC.md), [port notes](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/docs/deepseek4/PORT.md)
4. [Abyss DeepSeek scorer and attention source](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/kernels/cuda/dsv4_fast.cu)
5. [Abyss DeepSeek decode/prefill fast path](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/core/dsv4_fast_path.cu) and [batch mixer](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/core/glm_prefill.cu)
6. [Abyss expert memory tiers and NUMA](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/core/glm_fast_path.cu)
7. [Strata-V100 Volta `mma884` implementation](https://github.com/jmnargi/Strata-V100/blob/main/src/kernels/cuda/qsa_prompt_attn.cu) and [PR #600 benchmark note](https://github.com/jmnargi/Strata-V100/blob/main/benchmarks/v100-q2_0-pr600-2026-10-03.md)
8. [Strata-V100 hi/lo accumulator microbenchmark evidence (source comments)](https://github.com/jmnargi/Strata-V100/blob/main/src/kernels/cuda/qsa_prompt_attn.cu)
9. [Maya Volta attention comparison](https://github.com/mw00/project-maya/blob/main/src/kernels/cuda/glm_batch.cu)
10. [Superfast repository and benchmark methodology](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/README-27B.md)
11. [Superfast fragment-layout quant GEMV, repack and conditional routing](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-qpn.cu), [contract and flags](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-qpn.cuh), [Volta MMA tile implementation](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mma.cuh)
12. [Superfast non-repacked Q5_K output-head route](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/mmvq-tc.cu), [GPU correctness test](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/tests/test-mmvq-tc.cpp)
13. [Superfast streaming QK/PV attention](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/fattn-verify-stream.cuh), [sparse gathered attention and split-routing](https://github.com/mylordmonkeyman/llama-superfast-v100/blob/70df8710401a32c4735cf0c7874bf626bbe6b493/ggml/src/ggml-cuda/qsa-attn.cu)
14. [anyei/llamacpp-v100 small-T SM70 `mmsmt.cu` (includes Q2_K)](https://github.com/anyei/llamacpp-v100/blob/master/ggml/src/ggml-cuda/mmsmt.cu) and [implementation/numerical lessons](https://github.com/anyei/llamacpp-v100/blob/master/docs/mmsmt-implementation.md)
15. [1Cat-vLLM DeepSeek-V4 SM70 indexer](https://github.com/1CatAI/1Cat-vLLM/blob/main/vllm/models/deepseek_v4/sm70/indexer.py), [FP16 GEMV](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/design/sm70_deepseek_v4_fp16_gemv.md), [MXFP4 QPN](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu) and [MoE QPN](https://github.com/1CatAI/1Cat-vLLM/blob/main/csrc/sm70_turbomind/ops/skinny_moe_qpn_sm70.cu)
16. [anyei CUDA-graph CPU expert doorbell](https://github.com/anyei/llamacpp-v100/blob/master/src/llama-moe-doorbell.h)
17. [NVIDIA Volta Tuning Guide — registers/occupancy/smem](https://docs.nvidia.com/cuda/volta-tuning-guide/) and [NVIDIA's PCIe Gen3 ×16 payload-rate example](https://github.com/NVIDIA/skills/blob/main/skills/deepstream-profile-pipeline/references/hw-ceiling-formulas.md)
18. [NVIDIA Nsight Compute 2025.1 (Volta supported)](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_1), [2025.2 (Volta supported)](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_2), [2025.3 (Volta removed)](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_3)
19. [Intel Xeon E5-2697A v4 reference — 16 cores, 40 MB L3/socket, DDR4, PCIe 3.0, AVX2](https://www.intel.com/content/www/us/en/products/compare.html?productIds=91768)

20. [NVIDIA NVML clock event/throttle-reason definitions](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html) and [NVIDIA `nvidia-smi` CLI property queries](https://docs.nvidia.com/deploy/nvidia-smi/)
21. [NVIDIA PTX ISA `mma.sync` fragment mapping and architecture restrictions](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html) and [CUDA inline PTX operand constraints](https://docs.nvidia.com/cuda/archive/12.2.2/inline-ptx-assembly/index.html)
22. [NVIDIA `cuobjdump`/`nvdisasm` GPU binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
23. [Intel PCM Xeon E5 QPI measurements and limitations](https://www.intel.com/content/www/us/en/developer/articles/tool/performance-counter-monitor.html)
24. [Abyss model math as implemented in `dsv4_fast.cu` and `SPEC.md`](https://github.com/zephel01/Abyss/blob/47001f31cbfba14bcbf4c083e3af07c50707ec0c/src/kernels/cuda/dsv4_fast.cu)

**Final evidence caution:** These source inspections justify a *technical implementation strategy*, not predicted throughput. No Abyss/DeepSeek V4 V100 kernel was compiled, run or timed to prepare this document. Superfast’s measurements were reported by its authors on **Qwen**, not reproduced here; some design choices and priority rankings are **our proposed extrapolations** from their code and Abyss architecture. This specification does not assert that any Superfast-specific kernel is already integrated into Abyss.

---

### Additional Xolotl source references (v7; accessed 18 October 2026)

Xolotl source revision: [`f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16`](https://github.com/Indras-Mirror/Xolotl/tree/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16). Links are intentionally pinned so future Xolotl `main` changes do not silently modify the engineering evidence.

- [README.ds4.md — DeepSeek-V4-0731 experimental source-reported 4090 results, feature flags and caveats](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/README.ds4.md)
- [docs/ds4/FINDINGS.md — source ablations; §§12, 27–37 (prefetch, LRU, cache bytes, indexer, KV precision, profiler, REAP)](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/docs/ds4/FINDINGS.md)
- [docs/ds4/ENGINE_MOE.md — routed expert tier semantics, `ds4_moe` CPU/GPU source invariants](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/docs/ds4/ENGINE_MOE.md)
- [tools/ds4/ds4_moe.cpp — host arena, prefetch, source fallback and optional lazy fill](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_moe.cpp)
- [tools/ds4/ds4_generate.cpp — stage ordering, chunked prefill, cache options and documented speculative restrictions](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_generate.cpp)
- [tools/ds4/ds4_dense.cpp — indexed-cache graph, graph UID and allocation-epoch handling](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/tools/ds4/ds4_dense.cpp)
- [docs/ds4/RESEARCH_2026-10-09.md — independently explored upstream state/admission/elastic-KV ideas; proposals not measurements of Abyss](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/docs/ds4/RESEARCH_2026-10-09.md)
- [Xolotl CMakeLists.txt — `STRATA_EXPERIMENTAL_SM60`; experimental SM70 build is not verified 0731 V100 runtime](https://github.com/Indras-Mirror/Xolotl/blob/f7c561b7d4e1ca4ae9c7303adf253cc52fc28e16/CMakeLists.txt)

**Revision discipline:** v7 is **one consolidated artifact**, not a detached memo. All P0–P4 and v6 hardware/thermal/PTX/MLA/slot-epoch gates remain operative. No repository edits or V100 measurements are claimed by this specification.

---

### Additional upstream Strata sources (v8; source-audited 18 October 2026)

Pinned upstream tree: [`Niko1221/Strata@477ac13cbe8e582c20484871ce459d20d82df09d`](https://github.com/Niko1221/Strata/tree/477ac13cbe8e582c20484871ce459d20d82df09d). Change interval inspected: [`cf09057a...477ac13`](https://github.com/Niko1221/Strata/compare/cf09057a867d5be85389ecf459afe4a758cdc873...477ac13cbe8e582c20484871ce459d20d82df09d), **650 commits** (includes merge commits; do not treat the platform's truncated changed-file list as exhaustive). Compare candidate source with the **actual Abyss implementation branch**, not just present-day GitHub `main`.

- [14 Oct: Quantize-once/scatter for Q8_1 routed experts (`c675f5b`)](https://github.com/Niko1221/Strata/commit/c675f5b78725); [upstream `moe_mmq.cu`](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/prefill/moe_mmq.cu); [current Abyss MMQ adapter](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/prefill/moe_mmq.cu).
- [14 Oct: Exact AVX2 singleton rows (`8ee2005`)](https://github.com/Niko1221/Strata/commit/8ee2005dae0b); [IQ2_S and width guard (`77ddc2f`)](https://github.com/Niko1221/Strata/commit/77ddc2ff147d); [upstream CPU expert dispatch](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/kernels/cpu/native_expert.cpp); [Abyss existing native CPU expert](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/kernels/cpu/native_expert.cpp).
- [14 Oct: Online PCIe versus CPU balancing (`04be241`)](https://github.com/Niko1221/Strata/commit/04be241a5a7e); [Abyss startup calibration](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/core/glm_fast_path.cu).
- [14 Oct: Direct resident GPU expert MMQ (`7d9b50d`)](https://github.com/Niko1221/Strata/commit/7d9b50dc1a5c); [pointer-table MMQ instantiation/type whitelist (`moe_mmq_tab.hpp`)](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/src/prefill/moe_mmq_tab.hpp); [guard/fingerprint and parity (`86f677b`)](https://github.com/Niko1221/Strata/commit/86f677b4a976); [complete-group sorting (`62fc7c8`)](https://github.com/Niko1221/Strata/commit/62fc7c8a3c98).
- [12 Oct: Q8 HC weights (`b60aab1`)](https://github.com/Niko1221/Strata/commit/b60aab1ba27c); [upstream format and quality survey](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/QUANTS.md).
- [12 Oct: Adaptive expert overlap mixed-hardware caveat (`661ff29`)](https://github.com/Niko1221/Strata/commit/661ff2933941); [13 Oct: Pin by need (`ab6bce0`)](https://github.com/Niko1221/Strata/commit/ab6bce05de44); [resident exchange rotation and ownership invariants](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/EXCHANGE_ROTATION.md).
- [13 Oct: Head-as-prefill VRAM (`640ac43`)](https://github.com/Niko1221/Strata/commit/640ac438be9e); [12 Oct: Hugepage fragmentation/compaction guard (`6b39557`)](https://github.com/Niko1221/Strata/commit/6b395578a222); [new Abyss NUMA/THP/stream-fetched expert source](https://github.com/zephel01/Abyss/blob/7c893ac17e61ec45c934b3278356807fc4626c3c/src/core/glm_fast_path.cu).
- [Upstream source-tested V100/older GPUs (historical, **pre-fork**, not new to v8)](https://github.com/Niko1221/Strata/blob/477ac13cbe8e582c20484871ce459d20d82df09d/docs/NVIDIA_V100.md).

**v8 status / source classification:** This document is a **design and benchmark contract**. Statements about upstream Strata's outcomes describe the source authors' reported results; the corresponding Abyss candidates are explicitly hypotheses. No current Abyss or V100 code has been changed, benchmarked or validated as part of preparing v8; flags prefixed `ABYSS_V100_` are implementation **proposals** unless the chosen branch is subsequently modified to support them.
