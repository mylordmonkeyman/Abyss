# DeepSeek V4.1 Flash (1017) on Abyss

From v1.3.29 Abyss runs [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e10`) on the same fast path as DeepSeek V4: one GPU, the RAM tier (computed by the CPU or an APU's
iGPU), and the SSD for the experts that fit neither.

## What is different from V4

- 40 layers, hidden 5120, 384 routed experts top-6 + 1 shared, expert width 2304.
- Attention: four shared KV slots written by the KV-source layers (2 / 8 / 14 / 20) - ratio-2 (pair-pooled) or
  ratio-1 compressed rows plus index keys; the index-source layers (2, 8, 14, 20, 24, 28, 32, 36) pick the top-512
  rows and the layers between them reuse that choice; layer 20's candidate blocks narrow layers 24-36 past 16K rows.
- The hyper-connection pre is lagged by one sublayer; no per-head q norm.
- **Engram**: two hashed n-gram embedding tables (layers 1 and 14, ~100 GB each) read row by row from the GGUF -
  48 rows of 264 bytes a token; never loaded whole. Keep the GGUF on a fast local SSD.

## GGUFs that load

The packer (`tools/iq_pack.py`) reads both V4.1 layouts:

| GGUF | layout | experts | notes |
|---|---|---|---|
| [nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF](https://huggingface.co/nazeshinjite/DeepSeek-V4.1-Flash-1017-ds4-GGUF) `Q2` | ds4 (antirez) | IQ2_XXS gate/up, Q2_K down, 1017 imatrix (9.95 MB an expert) | Q8_0 dense, FP8 Engram. **The fast one.** 366 GB |
| [mrfakename/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF) `Q2_K` | llama.cpp PR | Q2_K gate/up, Q3_K down (12.8 MB) | dense and Engram tables are Q2_K too - low quality alone (teacher KL 0.64) |

The `deepseek41` GGUFs from kyuz0 / antirez named "v4.1" without `1017` are the September checkpoint
(`general.source.revision` `df42c109`); check the revision before trusting a repo name. Lucebox's ROCMFP files use
their own expert types (107 / 104), which Abyss does not read; their dense tensors and Engram tables do (a hybrid,
below).

```sh
.venv/bin/python tools/iq_pack.py --gguf <DeepSeek-V4.1-Flash-1017-Q2.gguf> --out <dir>/pack --compat-bf16
```

The pack keeps only an index, ~2.5 GB of converted small tensors and the Engram table offsets; the experts and the
Engram rows are read from the GGUF. Copy the 1017 chat template (`tokenizer/chat_template.jinja` from a GGUF that has
one, e.g. mrfakename's) into the pack if the GGUF carries none - the ds4-layout files do not.

## Quality and speed (RTX 5090 + Radeon 8060S lane + 93 GB RAM + NVMe)

Quality is measured against the unquantized 1017's top-256 log-probs on Lucebox's 8 held-out sequences
(8 x 1,023 positions, `eval/` of [Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF](https://huggingface.co/Lucebox/DeepSeek-V4.1-Flash-1017-ROCMFP-GGUF)):

| | decode | KL | top-1 agreement | ppl |
|---|---|---|---|---|
| nazeshinjite Q2 | 56.6 ms/token (17.7 tok/s) | 0.229 | 79.2% | 9.69 |
| hybrid: Lucebox Q8_0 dense + 8-bit Engram, mrfakename Q2_K/Q3_K experts | 118 ms/token (8.5 tok/s) | 0.168 | 83.4% | 9.25 |
| (Lucebox ROCMFP2S-IM1017, their numbers) | - | 0.253 | 78.7% | 9.78 |

On the same GGUF Abyss matches llama.cpp's V4.1 pull request (teacher KL 0.639 against 0.636). The prompt path
(batched MoE, the attention a token at a time) agrees with the token path at KL 0.013; a chat's next turn reuses the
previous state (1,132-token turn: 44 s -> 4.7 s of prompt, identical text).

The hybrid shard set is shard 1 = the dense and Engram tensors fetched from Lucebox's GGUF by HTTP range, shards
2..11 = symlinks to mrfakename's shards, packed with `iq_pack.py --first-wins`.

## Options for this model

| variable | effect |
|---|---|
| `STRATA_GLM_TAIL_SKIP=4 STRATA_GLM_TAIL_SKIP_DISK=1` | leave out a route's 5th-6th experts when they are only on disk: nazeshinjite Q2 61 -> 41 ms/token, KL 0.246 -> 0.249 on two sequences (lossy, off by default) |
| `STRATA_GLM_DISK_LANE=1` | a disk read is computed by the CPU / iGPU lane from its RAM landing instead of being pulled into VRAM (-5% on the Q2_K/Q3_K experts; none on the Q2) |
| `STRATA_DS41_ENGRAM_CACHE_MB` / `_THREADS` | the Engram raw-row cache (default 256 MB) and read threads (default 15 + caller) |
| `STRATA_GLM_RAM_GB=60` | the RAM tier; on a 93 GB box keep ~25 GB for the page cache (the SSD experts and the Engram rows go through it) |

Not yet for V4.1: speculative decoding / the verify window, conversation slots (files), a layer split, and a batched
prompt attention (the prompt's attention runs a token at a time, ~5 ms a token).
