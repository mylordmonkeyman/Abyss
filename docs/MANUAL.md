# Abyss manual

How to build Abyss, set up DeepSeek V4 Flash, run it and tune it. Japanese: [MANUAL.ja.md](MANUAL.ja.md).

Abyss is a fork of [Project Maya](https://github.com/mw00/project-maya) (MIT), itself grown out of
[Strata](https://github.com/Niko1221/Strata) (MIT). The engine, the expert tiers, the server and the dashboard are
Maya's; DeepSeek V4 support is what Abyss adds. Maya's own documentation (GLM-5.3-Flash, AMD, multi-GPU, the
installer) is kept under [docs/maya/](maya/README.md).

- [What you need](#what-you-need)
- [Setup in one command](#setup-in-one-command)
- [Build](#build)
- [Get the model](#get-the-model)
- [Make the pack](#make-the-pack)
- [Configure and start](#configure-and-start)
- [Using it](#using-it)
- [Memory and speed](#memory-and-speed)
- [Settings](#settings)
- [Checking the output against llama.cpp](#checking-the-output-against-llamacpp)
- [Something went wrong?](#something-went-wrong)
- [Limitations](#limitations)

## What you need

| | Tested | Notes |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows is not tested for DeepSeek V4 |
| GPU | RTX 5090 (32 GB) | build for your GPU's architecture (`120a` for RTX 50). Other NVIDIA GPUs build but are untested with DeepSeek V4; on pre-Blackwell GPUs the MXFP4 expert rows run through the q8_1 path instead of FP4 MMA |
| RAM | 94 GB | the model is 96.8 GB: with VRAM + RAM above ~100 GB every expert stays in memory; with less, the rest are read from the SSD (works, slower) |
| Disk | NVMe SSD, ~100 GB free | the GGUF files; the pack adds 0.2 GB |
| CUDA | 13.3 | with `nvcc` |
| Compiler | g++ 13 | a version your CUDA accepts |
| Python | 3.12 | a private `.venv` in the Abyss folder |

`./abyss.sh` (Maya's installer, renamed) still installs Maya's GLM-5.3-Flash models. DeepSeek V4 is set up with
`./setup-deepseek.sh` (next section), or by hand with the steps after it.

## Setup in one command

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

`setup-deepseek.sh` makes the `.venv`, then runs `tools/abyss_setup.py`. Each step is skipped when it is already done,
so it can be run again at any time:

1. checks the PC (an NVIDIA GPU, RAM);
2. Python packages, llama.cpp's source at the pinned commit;
3. compiles the engine for the GPU (`120a` on RTX 50);
4. downloads the model to `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>` - it asks first unless
   `--download-model` is given - and checks every file against Hugging Face's SHA-256;
5. makes the pack next to the shards;
6. writes the run config `abyss-<name>.json` and the start script `run-abyss-<name>.sh` in the Abyss folder;
7. creates the **Abyss-tools** folder beside Abyss: `abyss.sh start | stop | restart | status | update | build | log |
   models`, from the templates in `ops/abyss-tools/`. Running the setup again brings `abyss.sh` / `common.sh` up to date
   with the templates (the old ones kept as `.bak`) and keeps your own `README.md` there.

```sh
../Abyss-tools/abyss.sh start      # then http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| Option | |
| --- | --- |
| `--quant UD-IQ3_XXS` | another quant of unsloth/DeepSeek-V4-Flash-0731-GGUF (default `UD-Q2_K_XL`, the tested one) |
| `--name NAME` | the model's name in `abyss-<name>.json` (default `dsv4-<quant>`, e.g. `dsv4-q2_k_xl`) |
| `--context N`, `--port N` | the context (default 65536) and the port (default 8082) |
| `--data-dir DIR`, `--tools-dir DIR` | other places for Abyss-data and Abyss-tools |
| `--gpu N`, `--arch A` | the GPU, the CUDA architecture to compile for |
| `--download-model`, `--yes` | download without asking; the defaults for every question |
| `--rewrite-config` | write `abyss-<name>.json` again (otherwise an existing one is kept) |
| `--tools-only` | only create / update Abyss-tools |

The sections below are the same steps by hand.

## Build

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# the private Python environment
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# llama.cpp's source at the commit the engine is pinned to (setup.py LLAMA_CPP_COMMIT): ggml, the MMQ kernels, gguf-py
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# the engine: build/strata
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

The first build takes a few minutes; after a change only what changed is compiled again. Use `120a`, not `120`, on
RTX 50: the MXFP4 kernels use Blackwell's FP4 MMA instructions, which `120` cannot compile.

## Get the model

unsloth's [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF), quant
`UD-Q2_K_XL` (3 files, 96.8 GB), with the Hugging Face CLI (`pip install huggingface_hub` gives `hf`):

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss-data beside Abyss (any NVMe folder works)
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| File | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

If a download stops and does not resume, `curl -L -C - -o <file> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<file>`
continues it.

Other quants of the same repository should work if their expert types are among the decode kernels' (IQ1_S, IQ1_M,
IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K-Q6_K, MXFP4; IQ1_M has no prompt-path kernel, so such a quant reads
prompts token by token); only UD-Q2_K_XL is tested.

## Make the pack

The pack is the engine's index of the GGUF files: the small tensors (dense.bin), the expert table and the tokenizer
with the chat template. It goes next to the shards:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` rounds the hyper-connection matrices (stored F32 in the GGUF) to BF16, the form the engine reads. A
few seconds; the experts stay in the GGUF files.

## Configure and start

A config file (any name; `strata-dsv4.json` here) in the Abyss folder:

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/strata",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4/tokenizer",
 "model_name": "deepseek-v4-flash",
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/strata-dsv4.log"
}
```

| Field | |
| --- | --- |
| `exe` | the engine (`build/strata`) |
| `args` | the engine's arguments: `--glm-pack` the pack (the engine runs DeepSeek V4 through the same entry as GLM), `--max-context` the context in tokens, `--prefill auto` the prompt chunk size |
| `tokenizer` | the pack's `tokenizer` folder (its chat template is DeepSeek's own) |
| `model_name` | the name the API reports |
| `sampling` | defaults for requests that do not set them (DeepSeek's recommendation: temperature 1.0, top_p 0.95) |
| `reasoning_effort` | optional default thinking level (`none`, `low`, `high`, `max`) |
| `thinking_budget` | optional cap on a thinking block in tokens (default 32768; 0 = none) |
| `env` | optional `{"STRATA_GLM_...": "..."}` settings for the engine (see [Settings](#settings)) |
| `log` | the engine's log |

Start the server:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

It prints `ready: http://127.0.0.1:8082/v1` after loading. A start takes about a minute: the dense weights go to the
GPU, then every expert is read into the VRAM and RAM tiers (about 20 s at 4 GB/s from an NVMe). Ctrl+C stops it.

Server options: `--host 0.0.0.0` to accept other machines (always with `--api-key <secret>`), `--open` opens the page
in the browser, `--gpu 0` picks the GPU.

## Using it

- **Browser:** `http://127.0.0.1:8082/` - chat, a live monitor of the expert tiers and the GPU, settings.
- **OpenAI-compatible API:** base URL `http://127.0.0.1:8082/v1`, any model name.

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  The answer is in `content`, the thinking in `reasoning_content`.
- **Anthropic-compatible API:** `http://127.0.0.1:8082/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`).
- **Thinking:** DeepSeek V4 thinks unless asked not to. `reasoning_effort` (or the chat's setting):

  | Level | What DeepSeek gets |
  | --- | --- |
  | not set | thinking on |
  | `none` | thinking off (the prompt ends with `</think>`) |
  | `low` | thinking on |
  | `high` | thinking on + DeepSeek's "Reasoning Effort: Absolute maximum" instruction |
  | `max` | thinking on + its "Beyond maximum" instruction |

  The server always leaves the answer at least 1024 tokens (or a quarter of `max_tokens`): with a small `max_tokens`
  (a few hundred) the thinking is closed almost at once. Give thinking requests a `max_tokens` of a few thousand.
- **Tool calls:** OpenAI `tools` / Anthropic `tools` work, including several calls in one turn. The model writes them
  in DeepSeek's DSML form; the server turns them into `tool_calls` and renders past calls and `tool` results back into
  the prompt.
- **Conversation reuse:** a request that continues the previous one (the same messages plus new ones) reads only the
  new tokens.
- **One request at a time;** others wait in line.

### With Strata-tools' strata.sh

If you use the `strata.sh` helper (start / stop / log), put a run script and the config into the Abyss folder under
its naming (`run-<name>.sh`, `strata-<name>.json`) and point it there: `STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <name>`.
The run script is the start command above (`exec .venv/bin/python serve/server.py --engine strata "--config" "<config>" "--port" "8082" "$@"`).

## Memory and speed

The engine fills the GPU's free VRAM with experts (after the dense weights, 6.2 GB for UD-Q2_K_XL, the caches and the
prompt buffers), sizes a pinned RAM tier from the free RAM, and reads the rest from the SSD when needed. It records
which experts you use (`<pack>/expert_usage.txt`) and loads those first next time.

Measured on one RTX 5090 + 94 GB RAM (UD-Q2_K_XL, context 65536, every expert in VRAM 23 GB + RAM 70 GB):

| | |
| --- | --- |
| Decode, short answers | 36-62 tok/s |
| Prompt, 5.6K tokens | ~600 tok/s |
| Start-up (weights + expert warm-up) | ~1 min |
| Attention caches at 65536 tokens | 0.44 GB |

The first prompts after a start, or with less memory than the model, read experts from the SSD and are slower. The
CPU computes part of the RAM-tier experts while the GPU copies others over PCIe; the split is measured at start.

## Settings

Environment variables for the engine, in the config's `"env"` block (all of Maya's are listed in
[docs/maya/README.md](maya/README.md#tuning); these matter most here):

| Setting | Default | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | on | `0`: no warm-up at start (the tiers fill as experts are used; the first answers are slower) |
| `STRATA_GLM_RAM_GB` | from free RAM | a fixed size of the pinned RAM tier in GB |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM left free for the system when the RAM tier is sized |
| `STRATA_GLM_RESERVE_MB` | 700 | VRAM left free beside the expert pool |
| `STRATA_GLM_CPU_LANE` | one thread per physical core | CPU threads computing RAM-tier experts; `0` = off |
| `STRATA_GLM_PCIE_SHARE` | measured at start | the share of RAM-tier experts copied over PCIe instead of computed on the CPU (`0` = all on the CPU, `1` = none) |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | the prompt chunk: `auto`, a number of tokens, or `0` (token by token) |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | your expert usage between starts; `0` = off |
| `STRATA_GLM_NO_REUSE` | off | `1`: no conversation reuse |
| `STRATA_GLM_THINK_RESERVE` | 1024 | (server) tokens a thinking block always leaves for the answer |

`--max-context` sets the context. DeepSeek V4 is trained for 1M tokens; 65536 is tested. The attention caches are small
(0.44 GB at 65536), but the prompt path's buffers grow with the context.

## Checking the output against llama.cpp

`tools/dsv4_logits_dump.cpp` writes llama.cpp's logits for a token sequence (build it against a llama.cpp build that
supports `deepseek4`); the engine's `STRATA_GLM_SCORE` mode compares its own logits with them. See
[docs/deepseek4/PORT.md](deepseek4/PORT.md) for the commands and the results: on UD-Q2_K_XL the two engines agree to
KL 0.009 over 3000 tokens, with the same next-token accuracy.

## Something went wrong?

- **The build stops at `mma with block scale not supported on .target sm_120`** - configure with
  `-DCMAKE_CUDA_ARCHITECTURES=120a`.
- **`the engine exited before it was ready`** - read the engine log (`log` in the config). Out of memory while pinning
  the RAM tier: lower `STRATA_GLM_RAM_GB` or raise `STRATA_GLM_RAM_HEADROOM_GB`.
- **`refers to missing files` / `pack:` errors** - the pack must sit next to the three shards it was made from; make it
  again after moving them.
- **The thinking stops after a word and the answer starts** - `max_tokens` is too small (see Thinking above).
- **Slow, the SSD busy** - not enough RAM + VRAM for the experts; close programs or add RAM.
- **Port in use** - another server runs on it; use `--port`.

## Limitations

- No speculative decoding: this GGUF has no MTP block, and DSpark (DeepSeek's drafter) is not supported.
- Conversation slots on disk (`STRATA_GLM_SLOTS`) are off for DeepSeek V4; the live reuse of the last conversation works.
- No images (DeepSeek V4 Flash is text-only here).
- One GPU tested; Maya's layer split across GPUs is untested with DeepSeek V4.
- `./abyss.sh` (Maya's installer) is for GLM; DeepSeek V4 is set up with `./setup-deepseek.sh`.
