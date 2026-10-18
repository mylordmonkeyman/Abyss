# V100 implementation: P0 bring-up

Branch: `feature/deepseek-v100-perf`, based on upstream
`47001f31cbfba14bcbf4c083e3af07c50707ec0c` (v1.3.28).
The [revision-6 specification](IMPLEMENTATION_SPEC.md) defines the remaining gates.

## Implemented in this first increment

- Release and diagnostics CMake presets for CUDA SM70 and a portable AVX2 CPU baseline.
  Existing AVX-512 translation units stay isolated and runtime-dispatched; the preset
  does not make Broadwell execute AVX-512 or claim CPU throughput qualification.
- `STRATA_V100_DIAGNOSTICS=ON`: CUDA line info, ptxas register/resource reports and
  spill warnings. The default remains off. Explicit SM70 with CUDA 13 is rejected
  (compiler identification itself can reject it before the project diagnostic).
- `ABYSS_V100_PROFILE=1`: on DeepSeek only, enables the existing decode CUDA-event
  stage timers (`ds_proj_mv`, `ds_prep`, `ds_q`, `ds_index`, `ds_attend`, `ds_out_mv`,
  shared/router/expert stages), decode host-gap diagnostics and prefill phase timers.
  Legacy controls keep their previous semantics, including `STRATA_GLM_PROF=<skip>`.
  This is an initial profiling alias, not the full per-call critical-path ledger.
- A standalone standard-library P0 capture tool with model/input/binary SHA-256,
  tool/topology snapshots, static tensor census, continuous selected-GPU XML sampling,
  clock/power/thermal CSV and JSONL, command log and a machine-readable outcome.
- GPU-free tests and host-build CI. No candidate kernels or automatic dispatch changes.

## Build on the V100 host

Install Ninja and a CUDA **12.x** toolkit; use 12.8 for the specified host. The presets
require CMake 3.24+. The pinned GGML dependency is fetched by CMake unless supplied
with `STRATA_GGML_DIR`; preserve its baseline checkout for A/B comparisons.

```bash
cmake --preset v100-release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build --preset v100-release

cmake --preset v100-diagnostics \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build --preset v100-diagnostics 2>&1 | tee ptxas-v100.log
```

Executables are `build-v100/strata` and `build-v100-diagnostics/strata` respectively.
Use a fresh build directory when changing compiler/toolkit. Adjust the launch config's
`exe` field when using a preset instead of setup's default `build/strata`.

## Capture a baseline

First obtain the shipped GGUF/pack and prepare a **finite** generation or benchmark
command with fixed token IDs, context, threads, NUMA policy, speculation and cache state.
`capture` runs exactly the command provided, with no shell interpolation. It does not
prepare a model, start an HTTP benchmark, choose generation flags, change hardware settings,
stop other workloads, or automatically run five repetitions.

```bash
python tools/v100/p0.py inventory /path/to/model-00001-of-00004.gguf \
  --output tensor-inventory.csv

# Replace these paths with the real finite workload's script, tokens and config.
python tools/v100/p0.py capture \
  --output results/v100/baseline-A1 --gpu 0 --label baseline-A1 \
  --cache-state 'process restart; model files hashed before launch; expert cache as configured' \
  --model /path/to/model-00001-of-00004.gguf \
  --input /path/to/fixed-token-ids.json --input /path/to/engine-config.json \
  --binary build-v100/strata \
  -- bash /path/to/finite-baseline-workload.sh
```

Split GGUF siblings are expanded and missing shards rejected. All model shards are
hashed **before** the timed command and sampler start: this warms the page cache, so
these runs cannot be called cache-cold without an independent controlled procedure.
Inventory `bytes` is packed tensor payload, separate from complete file size. Expert
N/K describe one matrix, while `shape` preserves expert/group axes. Unknown quant types
retain unknown byte counts. Header-only files with short known payloads are rejected.

`p0_tensor_inventory.csv` is **metadata-only**: runtime token width, invocations,
residency, kernel and device timing columns remain blank. Join actual runtime/counter
measurements later; tensor presence alone does not prove a GPU launch.

For an instrumented run, use another output directory and add `--profile`. For an
Nsight trace, use another capture command with `nsys profile ...` as the workload and
hash the engine through `--binary`. Environment variables are inherited, so explicitly
unset all profiling controls for the production wall-time baseline. Instrumented timing
can change graph use and synchronization; keep it separate from unprofiled performance.

Every capture refuses to overwrite an existing directory and requires an observable
cc 7.0 GPU. It records failed launches, command failures and timeouts; timeout/interruption
stops the workload process group. Logs and evidence remain available after failure.
Command wall milliseconds include startup and teardown. Prefill/decode rates and TTFT
remain unmeasured in the JSONL; obtain them from the engine's own declared intervals.

Telemetry uses `nvidia-smi -q -x -i <device>` every 500 ms, including active clock-limit
reasons in either driver XML spelling. Samples bracket the command and use UTC plus
host monotonic timestamps. This is interval sampling, not proof that no short throttle
event occurred; external sampling also has overhead. Software power-cap activity is
separate from thermal slowdown. Missing reasons, failed samples or insufficient coverage
remain unassessed. Even without observed thermal flags, a capture alone never admits a
performance claim: inspect clock distributions, comparable sustained state and matched
A/B conditions, including intervals actually timed by the engine.

`toolchain_manifest.json` records profiler executable hashes and versions. Counter gate
status remains blocked until a real SM70 smoke profile, license/source provenance and
counter reports exist. It accepts only the specified 2025.1/2025.2 ncu version family
as version-compatible; it does not install/substitute/redistribute a profiler or invoke
counter replay during wall-time measurement. NUMA topology and affinity are snapshots;
QPI counters, actual page placement and CPU-vs-GPU critical paths remain unmeasured.

## Next work and gates

P0 remains **partial** until the real V100 builds and completes the shipped-model prompt,
decode, MXFP4 affected-layer and boundary/parity suites. Capture the quick 2K/8K/32K
matrix with at least five warm repeats and matched interleaved A/B conditions, plus the
independent cold-start suite. Add the runtime matrix census, GPU/pinned/workspace memory
ledger, sampled logits, supported counter reports and annotated Nsight Systems trace.
Rank measured critical-path components before selecting P1/P2/P3 production work.

P1 arithmetic microtests may be developed independently, but no MMA/scorer/attention or
expert-repack path is enabled by this increment. `ABYSS_V100_IDX`, `ABYSS_V100_ATTN` and
other proposed optimization switches remain unimplemented; setting them has no effect.
CUDA compilation, per-image PTX/SASS inspection, GPU launches and sanitizer runs are
separate pending checks. Host-only CI must not be reported as passing those gates.

Local verification:

```bash
python -m unittest discover -s tools -p test_v100_p0.py -v
cmake --list-presets
cmake -S . -B build-p0-cpu -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=OFF
cmake --build build-p0-cpu --parallel 4
```

## GitHub V100 runner workflow

[`.github/workflows/v100.yml`](../../.github/workflows/v100.yml) builds the full engine
and runs the small real-device self-test under memcheck and synccheck. It queues on
`self-hosted, Linux, X64, v100, sm70, cuda-12-8`; the runner must be registered with
**this Abyss repository** (a runner registered only with NInfer cannot take this job).
Pushes affecting the build on this feature branch/main trigger it; manual dispatch
is also declared. GitHub's Actions UI exposes manual dispatch once the workflow is
present on the repository's default branch.

It uses the existing runner environment and retains the diagnostics build directory.
Optional repository variables: `ABYSS_NVCC` (default `/usr/local/cuda-12.8/bin/nvcc`),
`ABYSS_GPU` (default `0`, exactly one SM70 GPU), and `ABYSS_BUILD_JOBS` (default `32`).
The runner needs Git, Python 3, CMake 3.24+, Ninja, a host C++ compiler, CUDA 12.8
including cuobjdump and Compute Sanitizer, and network access to the pinned GGML
checkout. No profiler installation or model download occurs. Logs, tool/binary hashes,
code-object listings and sanitizer output are uploaded even when a step fails.
Passing this workflow establishes build/device-smoke evidence, not full-model P0 parity.

## Model-free DeepSeek and MMA gates

`-DSTRATA_V100_TESTS=ON` adds `dsv4_fallback_parity` and
`test_mma884_fragment_contract` without enabling the unrelated global test suite.
Both executables reject non-SM70 GPUs. The workflow runs default and forced-FP32
DeepSeek component cases, plus memcheck/synccheck. The isolated MMA contract also
produces target-scoped PTX/SASS and checks for the expected MMA884/HMMA instructions.
The primitive is not connected to production scoring or attention. See
[STATUS.md](STATUS.md) for actual run evidence and remaining full-model gates.
