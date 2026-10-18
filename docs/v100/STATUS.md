# Implementation status

First increment: P0 infrastructure, based on `47001f3`.

Completed and checked locally:

- Nine GPU-free P0 tests: pass (including real child-process success, error and timeout).
- CMake preset parsing and complete host-only build with CUDA/native GGML/tests disabled: pass.
- Inventory of the repository's synthetic `data/glm5-synth.gguf`: 184 tensors,
  11,731,328 packed payload bytes. This is a tool smoke check, not DeepSeek evidence.
- Python compilation and `git diff --check`: pass.

Existing failure reproduced on both the unmodified pinned base and this branch:
`python tools/check_hip_compat.py` reports unmapped `cudaMemcpyKind` at
`src/core/glm_fast_state.hpp:421`. No HIP implementation was changed in this increment.
The inherited CI job running this checker may remain red for that baseline issue.

Not performed: CUDA compilation, actual V100 inference, profiler/counter collection,
model logit parity, sanitizer checks, memory-budget qualification or performance A/B.
The implementation workspace has no CUDA toolkit, GPU or DeepSeek model. P0 is not
passed, and no speedup is claimed. The next implementation should consume actual target-host
build/runtime evidence and complete the P0 runtime inventory/ledger before changing
production arithmetic. See [bring-up instructions](README.md) and the [specification](IMPLEMENTATION_SPEC.md).
