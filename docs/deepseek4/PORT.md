# DeepSeek V4 Flash in Abyss (the `deepseek4` port)

Abyss is a copy of Maya (project-maya, the GLM-5.3-Flash fork of Strata). DeepSeek V4 Flash
(llama.cpp arch `deepseek4`) runs on the `Glm5Model` fast path as a second architecture: the mHC residual, the
router + expert tiers (VRAM / pinned RAM / disk, CPU lane), the shared expert, the server and the prompt path are
Maya's; the attention mixer, the routing function, the head and the packaging are new. The math is in
[SPEC.md](SPEC.md) (transcribed from llama.cpp `src/models/deepseek4.cpp`).

## What is new

| Piece | Where |
| --- | --- |
| Geometry from `deepseek4.*` metadata (`Glm5Geometry::arch == 1`, `from_gguf_dsv4`) | `src/core/glm_model.cu`, `include/strata/core/glm_model.hpp` |
| Token path attention: q_a/kv/compressor projections, norms, partial RoPE (YaRN for compressed layers), the 128-token window ring, CSA (ratio 4, overlapping windows) / HCA (ratio 128) compressors, the lightning indexer top-512, attention with sinks, de-rope, grouped output LoRA | `src/kernels/cuda/dsv4_fast.cu`, `src/core/dsv4_fast_path.cu` |
| The gated hyper-connection head (`output_hc_*`) | `dsv4_fast.cu` `head_prep` |
| Router: sqrt(softplus) scoring, hash-routed layers 0-2 (`ffn_gate_tid2eid`) | `glm_fast.cu` `moe_route`, `glm_batch.cu` `route` |
| MXFP4 expert rows (2 layers of unsloth UD-Q2_K_XL): decode dot product; prompt-path MMQ with llama.cpp's FP4 activations (needs `CMAKE_CUDA_ARCHITECTURES=120a`) | `glm_fast.cu` `dw<39>`, `src/prefill/moe_mmq.cu` |
| Prompt path (batched): window keys in a linear buffer, batched compressors, per-token indexer, 8-heads-per-block online-softmax attention | `dsv4_fast.cu` (`*_batch`), `glm_prefill.cu` `prefill_ds4_mixer` |
| Conversation snapshot (window + compressor rings; the compressed caches are append-only by position) | `dsv4_fast_path.cu` `dsv4_snap_copy`. File slots: not implemented (off) |
| Packer: `deepseek4` arch, I32 hash tables, hc fn matrices rounded to BF16 (`--compat-bf16`) | `tools/iq_pack.py`, `tools/gguf_reader.py` |
| Tokenizer: the `joyai-llm` pre-tokenizer (three regexes applied in sequence) | `tools/strata_tokenizer.py` |
| Server: DeepSeek's chat template (thinking / reasoning_effort, `bos_token`, `from_json`), DSML tool calls, `<｜end▁of▁sentence｜>` stop | `serve/frontend.py`, `serve/server.py` |

The Hadamard rotations llama.cpp applies to quantized caches and to the indexer keys are orthonormal on both sides of
every dot product and are omitted (caches are FP16, llama.cpp's default).

## Validation (unsloth `DeepSeek-V4-Flash-0731` UD-Q2_K_XL, RTX 5090 32 GB, 94 GB RAM)

Reference: llama.cpp `bed0a8566` on the same GGUF (`tools/dsv4_logits_dump.cpp`, batch mode on the GPU).

| Test | Result |
| --- | --- |
| 10-token chat prompt, token path | KL mean 0.008 (max 0.017), top-1 equal 9/9 |
| 3000-token document, token path | KL mean 0.0093, top-1 equal 94.5%, ppl 6.36; next-token top-1 60.35% (llama.cpp: 60.35%). No drift by position: [0,128) 0.015, [128,512) 0.012, [512,1024) 0.011, [1024,2048) 0.0095, [2048,3000) 0.0065 |
| 2499 tokens through the prompt path, then 500 by the token path | KL mean 0.0059, top-1 equal 95.2% |
| Tokenizer vs llama-tokenize | equal on a 5578-token document and the chat prompt |

The window ring wrap (p >= 128), HCA compression (every 128) and the indexer's top-512 selection (p >= 2048) are all
inside the 3000-token test.

## Speed (same machine; all 11008 experts fit VRAM 23 GB + pinned RAM 70 GB, warm-up 21 s)

| | Abyss | llama.cpp (`-ot exps=CPU`) |
| --- | --- | --- |
| Decode, answers of ~1,500 tokens | ~73 tok/s with the iGPU lane (v1.1: 56-57, v1.0: 48) | 5-7 tok/s |
| Decode, 20 coding prompts (up to 800 tokens) | ~66 tok/s, ~78 with `STRATA_DS4_SPEC=1` (v1.1: 51) | 5-7 tok/s |
| Prompt, 5.6K tokens (server) | ~1,000-1,090 tok/s (v1.0: ~690) | 3-6 tok/s (short prompts) |
| Prompt, 18K tokens (server) | ~1,700 tok/s | - |

v1.1.0's speed work (CHANGELOG): the token-path attention split over key chunks (flash-decoding), a lighter route
kernel and CPU-answer read, the prompt path's CPU rows by the AVX-512 i-quant kernels, and an adaptive prestage. The
validation above was repeated after it: 3000 tokens KL 0.0091 (CPU lane on), 2499 + 500 KL 0.0054.

v1.2.0 (CHANGELOG): the RAM-tier experts on the Ryzen AI Max+ 395's integrated GPU (the iGPU lane, gfx1151 through
ROCm, `src/igpu/`) instead of its CPU threads - 0.046 against 0.107 ms an expert, and the same numbers as the RTX 5090's
kernels (relative L2 0.00000 between them; 0.0137 against a double reference, the CPU lane's 0.0265). 3000 tokens one
by one: KL 0.0093 against llama.cpp, 14.8 ms a token (CPU lane: 20.9). And speculative decoding with drafts from the
text itself (`STRATA_DS4_SPEC`, `src/core/dsv4_verify.cu`): a window of 1 + k tokens runs as k + 1 staggered streams
(a token's attention of a layer overlaps the previous token's experts of that layer); the window and the token path
give the same logits (KL 0.000000 with every expert on the GPU, `STRATA_GLM_SCORE_VERIFY=<T>`).

## Build and pack

```sh
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
.venv/bin/python tools/iq_pack.py --gguf <shard 1>.gguf --out ../Abyss-data/models/<repo>/<quant>/pack-dsv4 --compat-bf16
```

## Not done yet

- The MTP / DSpark draft (the GGUF has no MTP block; DSpark needs its own model) - speculative decoding drafts from
  the prompt and the answer instead (n-gram lookup, `STRATA_DS4_SPEC`).
- File conversation slots (`STRATA_GLM_SLOTS`) - off for deepseek4; the live snapshot works.
- Images.
- Further prompt-path tuning: on PCIe Gen4 x4 the expert copies still take most of a prompt's time.
