# Abyss

**Run DeepSeek V4 Flash on one consumer GPU.**

Abyss runs [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (a 284 B-parameter
mixture-of-experts model with 256 experts per layer) from a GGUF on a single NVIDIA GPU. The engine keeps the most-used
experts in VRAM, the next ones in pinned RAM and the rest on the NVMe SSD, and moves them as the conversation needs
them. It serves a browser chat and OpenAI- / Anthropic-compatible APIs, with thinking and tool calls.

Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), which grew out of
[Strata](https://github.com/Niko1221/Strata) (MIT). Maya's GLM-5.3-Flash runner, expert tiers, prompt path and server
are kept; DeepSeek V4 (llama.cpp's `deepseek4` architecture) is added as a second architecture next to GLM-5.3-Flash.
Maya's own README is kept as [README.maya.md](README.maya.md).

## What it does on one RTX 5090

Measured with unsloth's [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
`UD-Q2_K_XL` (96.8 GB) on an RTX 5090 (32 GB) with 94 GB of RAM:

| | Abyss | llama.cpp (experts on the CPU) |
| --- | --- | --- |
| Decode | 36-62 tok/s | 5-7 tok/s |
| Prompt (5.6K tokens) | ~600 tok/s | - |

All 11,008 routed experts fit in VRAM (23 GB) plus pinned RAM (70 GB); the tiers are filled at start (about 20 s).

## Correctness

Checked against llama.cpp on the same GGUF (teacher-forced logits, `tools/dsv4_logits_dump.cpp`):

- 3,000-token document: KL 0.009 on average, the same top token 94.5% of the time, next-token accuracy 60.35% for both
  engines, no drift with position (the sliding window, both compressors and the indexer's top-512 are all exercised);
- the batched prompt path: KL 0.006 on the positions after a 2,499-token prompt;
- the tokenizer (`joyai-llm`) produces the same ids as `llama-tokenize`.

Details: [docs/deepseek4/PORT.md](docs/deepseek4/PORT.md). The forward pass it implements:
[docs/deepseek4/SPEC.md](docs/deepseek4/SPEC.md).

## Build

Linux, CUDA 13.3 (the version tested), an RTX 50-series GPU (the MXFP4 expert rows use Blackwell's FP4 MMA; build for
`120a`). Python 3.12 in `.venv` with the packages Maya's installer uses (`maya.py --setup`).

```sh
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

## Run

1. Download the three shards of `UD-Q2_K_XL/` from
   [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) into one folder.
2. Make the pack (dense tensors, tokenizer, expert table) next to the shards:

   ```sh
   .venv/bin/python tools/iq_pack.py --gguf <dir>/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
       --out <dir>/pack-dsv4 --compat-bf16
   ```

3. Write a run config (`strata-dsv4.json`):

   ```json
   {
    "exe": "<abyss>/build/strata",
    "args": ["--glm-pack", "<dir>/pack-dsv4", "--max-context", "65536", "--prefill", "auto"],
    "cwd": "<abyss>",
    "tokenizer": "<dir>/pack-dsv4/tokenizer",
    "model_name": "deepseek-v4-flash",
    "sampling": {"temperature": 1.0, "top_p": 0.95},
    "port": 8082
   }
   ```

4. Start the server and open `http://127.0.0.1:8082/`:

   ```sh
   .venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
   ```

## Not yet

- Speculative decoding (the GGUF has no MTP block; DSpark is a separate drafter model).
- Conversation slots on disk (the live snapshot of the last conversation works).
- Images.

## License

MIT, as Maya and Strata: see [LICENSE](LICENSE). Model weights are under their own licenses (DeepSeek V4: MIT).
