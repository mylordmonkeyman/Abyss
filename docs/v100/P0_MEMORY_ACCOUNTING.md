# P0 memory accounting collection

Status: physical collection passed for both immutable upstream binaries in run
39194149121. See [accounting results](P0_ACCOUNTING_RESULTS.md) and the conserved
per-SHA budget in `p0-memory-accounting-2026-10-19.json`. Fine source-derived
attribution inside the borrowed prefill arena is now reconciled in
`p0-prefill-layout-2026-10-19.json`; union views overlap and add no GPU capacity.
See `P0_COMPLETION.md` for final gate dispositions.

`tools/v100/cuda_alloc_observer.cpp` observes synchronous CUDA runtime allocation,
registration, free and host/device alias calls without changing the engine binary.
It is opt-in through `LD_PRELOAD` and `ABYSS_CUDA_ALLOC_LOG`. Build separately:

```sh
c++ -shared -fPIC -std=c++17 -O2 -pthread \
  tools/v100/cuda_alloc_observer.cpp -ldl -o /tmp/abyss-cuda-alloc-observer.so
```

Only use it in a separate accounting run **after** the active unprofiled campaign.
Before launching, inspect `ldd`/`readelf` for dynamically linked cudart and validate
actual symbol interception on the physical runner. A statically linked runtime or
noninterposable internal calls invalidate completeness; zero events is an error,
not zero memory. Do not alter the immutable cached binaries or compare this run's
throughput with the unprofiled baseline. Use one fresh log per process; appended
sessions, missing sequence numbers and events after the exit marker are rejected.
A killed process may lack its exit marker: retain it as an incomplete observation.

Set both environment variables only for the profiled child, preserving all other
baseline affinity, model and environment controls:

```sh
ABYSS_CUDA_ALLOC_LOG=/absolute/fresh/allocations.jsonl \
LD_PRELOAD=/tmp/abyss-cuda-alloc-observer.so /path/to/immutable/strata <baseline arguments>
python tools/v100/allocation_ledger.py /absolute/fresh/allocations.jsonl \
  --page-size "$(getconf PAGESIZE)" --output /absolute/fresh/allocation-ledger.json
```

Use the existing interactive harness to collect a contemporaneous worksheet while
the child is alive, after model startup and at filled-context checkpoints:

```sh
python tools/v100/memory_worksheet.py --pid "$engine_pid" \
  --output /absolute/fresh/memory-worksheet.json
```

The observer records allocation calls in successful return order, monotonic
 timestamps, integer pointers, requested bytes, flags and CUDA results. Nested
runtime calls are suppressed to avoid counting `cudaMallocHost` implemented by
`cudaHostAlloc` twice. Device aliases add zero host bytes. The reconstruction
reports live ranges, high-water marks and a host virtual-page union. It is **not**
a physical residency counter. CUDA driver/context/modules, vendor internals and
unobserved calls remain outside coverage; reconcile against contemporaneous NVML
and preserve a labeled residual. There is no async-allocation claim: none is used
by the inspected current engine source, and those APIs are not wrapped here.

The physical host worksheet separates anonymous pages, shmem/tmpfs, file cache
excluding shmem, buffers, slab, stacks, page tables, free pages and an explicit
unassigned residual. It preserves raw `/proc/meminfo` and `smaps_rollup` with a
bounded read-time interval. It does not add RSS, PSS, registration, THP, Cached or
MemAvailable together. Pinned/registered allocations are subsets of the relevant
resident host pages; CUDA registration is not Linux `mlock` and cannot be inferred
from VmLck/Mlocked. File-backed application RSS overlaps the file cache.

Remaining collection must attribute arena subcomponents (dense/context/KV/indexer/
workspace/staging/expert tier), resolve registered-range residency and aliasing,
report OS/container/swap/pressure safety margins, and distinguish NUMA placement
from unavailable QPI traffic. Keep the exact observations and unknowns separate.

Validation: `python -m unittest discover -s tools -p 'test_v100_*.py'` passes
27 tests locally. The new interposition test compiles a shared fake CUDA runtime
and observer, executes a real dynamically linked process, checks nested API
suppression, failed allocations, alias accounting and final releases. This is
host-tool validation, not CUDA-driver or physical-memory validation.
