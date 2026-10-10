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

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model    # engine, model (96.8 GB), pack, run config, ../Abyss-tools
../Abyss-tools/abyss.sh start           # then http://127.0.0.1:8082/
```

The setup builds the engine for your GPU, downloads unsloth's UD-Q2_K_XL into `../Abyss-data`, checks it, makes the
pack, writes the run config and creates **Abyss-tools** beside Abyss (`abyss.sh start | stop | status | update | log |
models`). Every step by hand, the settings, the API and troubleshooting: **[docs/MANUAL.md](docs/MANUAL.md)**
(日本語: [docs/MANUAL.ja.md](docs/MANUAL.ja.md)).

`./abyss.sh` / `START-ABYSS.bat` are Maya's installer, renamed: they still set up Maya's GLM-5.3-Flash models
([docs/maya/README.md](docs/maya/README.md)).

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
- Windows and several GPUs for DeepSeek V4 (one GPU on Linux tested; Maya's layer split is untested with it).

## License

MIT, as Maya and Strata: see [LICENSE](LICENSE). Model weights are under their own licenses (DeepSeek V4: MIT).
