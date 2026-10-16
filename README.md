<div align="center">

# Abyss

**Run DeepSeek V4 Flash — a 284B mixture-of-experts model — on a single consumer GPU.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)
![ROCm](https://img.shields.io/badge/ROCm-7%20(experimental)-ED1C24?logo=amd&logoColor=white)
![GPU](https://img.shields.io/badge/tested-RTX%205090%20%C2%B7%20RTX%203090%20%C2%B7%20Radeon%208060S-76B900)
![Model](https://img.shields.io/badge/model-DeepSeek--V4--Flash-4D6BFE)
[![Release](https://img.shields.io/github/v/release/zephel01/Abyss?label=release)](https://github.com/zephel01/Abyss/releases)

**English** · [日本語](README.ja.md)

[Quick start](#-quick-start) · [Performance](#-performance) · [How it works](#-how-it-works) ·
[Manual](docs/MANUAL.md) · [Changelog](CHANGELOG.md) · [Credits](#-credits)

</div>

---

Abyss runs [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) from a GGUF on one NVIDIA
GPU. The engine keeps the most-used experts in VRAM, the next ones in pinned RAM and the rest on the NVMe SSD, and
moves them as the conversation needs them. You get a browser chat and OpenAI- / Anthropic-compatible APIs, with
thinking and tool calls — all on your own machine.

> [!NOTE]
> Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya), which grew out of
> [Strata](https://github.com/Niko1221/Strata). It adds DeepSeek V4 to Maya's engine. See [Credits](#-credits).

## ✨ Highlights

- **284B MoE on one card** — 256 experts per layer spread over VRAM, pinned RAM and SSD, filled from your own usage.
- **~3x llama.cpp's decode speed and ~5x its prompt speed** on the same GGUF and machine (llama.cpp tuned with
  `-ncmoe`; 10x+ against its experts on the CPU). Prompts at ~1,600–3,300 tok/s, decode at ~92–95 tok/s.
- **Long contexts stay fast** — the lightning indexer and the prompt attention run on tensor cores: a 125K-token
  prompt reads at ~2,800 tok/s, and decode holds ~65 tok/s at 125K.
- **An AMD APU's integrated GPU as a second lane** — on a Ryzen AI Max the Radeon iGPU computes the RAM-tier experts
  in place, for decode and for prompts (ROCm; picked at start when it is faster than the CPU).
- **Speculative decoding without a draft model** — n-gram drafts from the prompt and the answer, verified in one pass;
  the same output, +10–13% on code. On by default.
- **Checked against llama.cpp** — same next-token accuracy over 3,000 tokens (KL 0.009).
- **Full DeepSeek V4 attention** — sliding window, CSA/HCA compressed KV, the lightning indexer's top-512, sinks.
- **Thinking and tool calls** — reasoning levels `none` / `low` / `high` / `max`, DeepSeek's DSML tool calls, parallel calls.
- **OpenAI and Anthropic APIs** plus a browser dashboard with a live view of the expert tiers.
- **Several GPUs, AMD, V100** — a layer split across NVIDIA cards, an experimental HIP build for Radeon, and an sm_70
  build with CUDA 12.
- **One-command setup** — engine, model, pack, config and the `Abyss-tools` helper scripts.

## 📊 Performance

unsloth's [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) `UD-Q2_K_XL`
(96.8 GB), RTX 5090 (32 GB, PCIe Gen4 x4) + Ryzen AI Max+ 395 with 94 GB RAM (iGPU lane on), Linux, Abyss v1.3.16.
llama.cpp (`bed0a8566`) on the same machine and file, at its best settings (`-ngl 99 -ncmoe 32`):

| | Abyss | llama.cpp |
| --- | :---: | :---: |
| Decode (1,024-token answers, prose / code) | **~92–95 tok/s** | 28–30 tok/s |
| Decode (16 swe-bench coding tasks x 3, end to end) | **~100 tok/s** | – |
| Decode at a 125K-token context | **~65 tok/s** | – |
| Prompt (8K tokens) | **~1,630–1,760 tok/s** | 78 (ubatch 512) – 660 (ubatch 8192) |
| Prompt (32K tokens) | **~3,100–3,270 tok/s** | – |
| Prompt (125K tokens) | **~2,800 tok/s** | – |
| Start-up (weights + expert warm-up) | ~1 min | ~50 s |

All 11,008 routed experts fit in VRAM (23 GB) + pinned RAM (70 GB) — nothing is read from the SSD once warm. Decode is
measured with the attention projections re-quantized to Q5_K (the lighter dense pack in the manual: +8% over the GGUF as
published at the same swe-bench quality). With the
CPU lane instead of the iGPU (any NVIDIA box with enough RAM), decode is ~55–60 tok/s. Every number is in the
[changelog](CHANGELOG.md) with how it was measured.

<details>
<summary><b>Other setups measured</b></summary>

| Setup | Decode | Prompt 8K / 32K |
| --- | :---: | :---: |
| RTX 3090 (USB4 enclosure, ~3.8 GB/s) + iGPU lane | ~49 tok/s | ~690 / ~1,010 tok/s |
| RTX 5090 + RTX 3090 (layer split 27/16) + iGPU lane | ~55–57 tok/s | ~1,500 / ~1,990 tok/s |
| Radeon 8060S alone (HIP build, experts partly from the SSD) | – | 7K prompt ~150 tok/s |

A second, slower card does not yet beat one RTX 5090 with the iGPU lane: the hop between the cards and the slow link
cost more than the RAM-tier experts it saves (speculative decoding runs in a split from v1.3.26).

</details>

## 🚀 Quick start

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model     # engine, model (96.8 GB), pack, config, ../Abyss-tools
../Abyss-tools/abyss.sh start            # → http://127.0.0.1:8082/
```

Then talk to it:

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
```

<details>
<summary><b>What the setup does</b></summary>

1. checks the PC (NVIDIA GPU, RAM)
2. installs the Python packages into `.venv` and fetches llama.cpp's source at the pinned commit
3. compiles the engine for your GPU (`120a` on RTX 50)
4. downloads the model to `../Abyss-data` (asks first without `--download-model`) and checks every SHA-256
5. builds the pack next to the shards
6. writes `abyss-<name>.json` and `run-abyss-<name>.sh`
7. creates `../Abyss-tools` — `abyss.sh start | stop | restart | status | update | build | log | models`

Every step is skipped when done, so it is safe to run again. Options: `--quant`, `--context`, `--port`, `--gpus 0,1`
(a layer split), `--backend hip` (AMD), `--nvcc` (a CUDA 12 toolkit for a V100), `--tools-only`, and more in the
[manual](docs/MANUAL.md#setup-in-one-command).

</details>

### Requirements

| | |
| --- | --- |
| OS | Linux (tested on Ubuntu) |
| GPU | NVIDIA: RTX 5090 32 GB and RTX 3090 measured; a V100 builds with CUDA 12 (not run yet). AMD Radeon through HIP, experimental: Radeon 8060S run, discrete cards untested |
| Memory | VRAM + RAM ≳ 100 GB to keep every expert in memory (less works, reading from the SSD) |
| Disk | NVMe SSD with ~100 GB free |
| Toolchain | CUDA 13.x (`nvcc`; 12.9 for a V100), g++, Python 3.10+; ROCm 7 (`hipcc`) for the iGPU lane on an AMD APU or for an AMD build |

## 🧠 How it works

```mermaid
flowchart LR
    T[token] --> A["attention<br/>window + compressed KV<br/>indexer top-512"]
    A --> R{"router<br/>sqrt(softplus)"}
    R --> V["VRAM tier<br/>hot experts"]
    R --> P["pinned RAM tier<br/>iGPU / CPU lane / PCIe"]
    R --> D[("NVMe SSD<br/>cold experts")]
    V --> O[next layer]
    P --> O
    D --> O
```

Each layer's attention runs on the GPU; its router picks 6 of 256 experts. Experts in VRAM run in place. RAM-tier
ones are computed where they are — by the APU's integrated GPU when there is one, else by the CPU threads — or copied
over PCIe (the split is measured at start), and the rest are read from the SSD and promoted as you use them. Long
prompts go through a batched path with expert GEMMs (MMQ) and tensor-core attention. When the answer repeats
something from the text, an n-gram draft is verified together with the next token.
Details: [docs/deepseek4/PORT.md](docs/deepseek4/PORT.md), the forward pass: [docs/deepseek4/SPEC.md](docs/deepseek4/SPEC.md).

## ✅ Correctness

Teacher-forced logits against llama.cpp on the same GGUF (`tools/dsv4_logits_dump.cpp`):

| Test | Result |
| --- | --- |
| 3,000-token document | KL 0.009 · same top token 94.5% · next-token accuracy 60.35% (llama.cpp: 60.35%) · no drift with position |
| Batched prompt (2,499 tokens) then 500 tokens | KL 0.006 · same top token 95.2% |
| Tokenizer (`joyai-llm`) | identical ids to `llama-tokenize` |

Every speed change is re-checked the same way (the KL per release is in the [changelog](CHANGELOG.md)); the tensor-core
kernels match the FP32 ones to ~2e-6, and speculative decoding gives the token path's logits exactly.

## 📚 Documentation

| | |
| --- | --- |
| [Manual](docs/MANUAL.md) · [マニュアル](docs/MANUAL.ja.md) | setup, configuration, API, several GPUs, AMD, settings, troubleshooting |
| [Changelog](CHANGELOG.md) | every release with its measurements |
| [Port notes](docs/deepseek4/PORT.md) | what Abyss adds and how it was validated |
| [Spec](docs/deepseek4/SPEC.md) | DeepSeek V4's forward pass as implemented |
| [Maya docs](docs/maya/README.md) | Maya's documentation, unchanged: GLM-5.3-Flash, its installer (`./abyss.sh`), tuning, AMD, multi-GPU |

## 🗺️ Roadmap

- [x] DeepSeek V4 Flash: token path, batched prompt path, conversation reuse
- [x] Thinking levels, DSML tool calls, OpenAI / Anthropic APIs
- [x] One-command setup with `Abyss-tools`
- [x] The iGPU lane (Ryzen AI Max) for decode and prompts
- [x] Speculative decoding from the text itself (n-gram), on by default
- [x] Long contexts: the indexer and the prompt attention on tensor cores (125K tested)
- [x] Several NVIDIA GPUs (layer split; RTX 5090 + RTX 3090 measured)
- [x] AMD (HIP, experimental): runs on the Radeon 8060S with WMMA kernels
- [x] V100 (sm_70) builds with CUDA 12
- [ ] A DSpark drafter (this GGUF has no MTP block)
- [ ] Conversation slots on disk
- [ ] Windows for DeepSeek V4 (Linux tested)
- [ ] Speed with several cards: a split's prompts (30% behind one RTX 5090 here); a run on a V100; discrete Radeon cards
- [ ] Prompt-path tuning at short lengths (staging RAM-tier experts still dominates an 8K prompt)

## 🙏 Credits

Abyss is a copy (fork) of **[Project Maya](https://github.com/mw00/project-maya)** (MIT, by mw00 and contributors),
which grew out of **[Strata](https://github.com/Niko1221/Strata)** (MIT, by Niko1221 and the Strata contributors).
Maya's GLM-5.3-Flash runner, expert tiers, prompt path, server and dashboard are kept as they are; DeepSeek V4
(llama.cpp's `deepseek4` architecture) is added next to GLM-5.3-Flash. Maya's own README and documentation are kept in
[docs/maya/](docs/maya/README.md). The DeepSeek V4 math follows [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
implementation, and the quantized model is [unsloth](https://huggingface.co/unsloth)'s. Thanks to all of them.

## 📄 License

MIT, as Maya and Strata — see [LICENSE](LICENSE). Model weights are under their own licenses (DeepSeek V4: MIT).
