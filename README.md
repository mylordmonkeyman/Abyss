# Abyss

**Run DeepSeek V4 Flash on one consumer GPU.**

Abyss runs [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (a 284 B-parameter
mixture-of-experts model with 256 experts per layer) from a GGUF on a single NVIDIA GPU. The engine keeps the most-used
experts in VRAM, the next ones in pinned RAM and the rest on the NVMe SSD, and moves them as the conversation needs
them. It serves a browser chat and OpenAI- / Anthropic-compatible APIs, with thinking and tool calls.

Abyss is a copy (fork) of [Project Maya](https://github.com/mw00/project-maya) (MIT, by mw00 and contributors), which
grew out of [Strata](https://github.com/Niko1221/Strata) (MIT, by Niko1221 and the Strata contributors). Maya's
GLM-5.3-Flash runner, expert tiers, prompt path, server and dashboard are kept as they are; DeepSeek V4 (llama.cpp's
`deepseek4` architecture) is added as a second architecture next to GLM-5.3-Flash. Maya's own README and documentation
are kept in [docs/maya/](docs/maya/README.md). Thanks to both projects.

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

## Getting started

The manual has every step - build, the model, the pack, the config, the API, the settings and troubleshooting:
**[docs/MANUAL.md](docs/MANUAL.md)** (日本語: [docs/MANUAL.ja.md](docs/MANUAL.ja.md)). In short:

```sh
# build (CUDA 13.x, an RTX 50 GPU: arch 120a; llama.cpp's source in third_party/llama.cpp - see the manual)
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16

# the model (unsloth UD-Q2_K_XL, 96.8 GB) and its pack, next to the shards
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF --include "UD-Q2_K_XL/*" --local-dir ../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF
.venv/bin/python tools/iq_pack.py --gguf ../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out ../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4 --compat-bf16

# a config (see the manual) and the server: http://127.0.0.1:8082/
.venv/bin/python serve/server.py --engine strata --config abyss-dsv4.json --port 8082
```

`./abyss.sh` / `START-ABYSS.bat` are Maya's installer, renamed: they still set up Maya's GLM-5.3-Flash models
([docs/maya/README.md](docs/maya/README.md)); DeepSeek V4 is set up as above.

## Documentation

| | |
| --- | --- |
| [docs/MANUAL.md](docs/MANUAL.md), [docs/MANUAL.ja.md](docs/MANUAL.ja.md) | the manual for DeepSeek V4 |
| [docs/deepseek4/PORT.md](docs/deepseek4/PORT.md) | what the port adds, how it was checked against llama.cpp |
| [docs/deepseek4/SPEC.md](docs/deepseek4/SPEC.md) | DeepSeek V4's forward pass as implemented |
| [docs/maya/](docs/maya/README.md) | Maya's documentation, kept as it was: GLM-5.3-Flash, the installer, tuning, AMD, several GPUs |

## Not yet

- Speculative decoding (the GGUF has no MTP block; DSpark is a separate drafter model).
- Conversation slots on disk (the live snapshot of the last conversation works).
- Images.
- The installer (`./abyss.sh`) for DeepSeek V4; one GPU tested (Maya's layer split is untested with DeepSeek V4).

## License

MIT, as Maya and Strata: see [LICENSE](LICENSE). Model weights are under their own licenses (DeepSeek V4: MIT).
