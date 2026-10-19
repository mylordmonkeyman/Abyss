# P0 dispatch ledger

Status: CUDA compilation and physical collection passed in run 39194149121.
See [accounting/dispatch results](P0_ACCOUNTING_RESULTS.md) and
`p0-dispatch-2026-10-19.json`. No dropped calls or unresolved decode joins.
Detailed per-token cache/route and exact per-call input-duplication counts remain
open; source presence or a successful capture does not close every P0 gate.

Enable only in a separate single-GPU diagnostic process with
`ABYSS_V100_DISPATCH=1 ABYSS_V100_PROFILE=1`. Production defaults leave the ledger
off. The ledger also disables the optional KDA host graph capture/replay path so
its host-call counts cannot silently describe capture once instead of replay.
This has no effect on a process where the diagnostic is disabled. Do not use its
timing as unprofiled throughput evidence. No arithmetic or dispatch eligibility
rule is changed.

The bounded process-local ledger emits `V100_DISPATCH` JSON records and one
`V100_DISPATCH_END` record to stderr after normal process shutdown. It aggregates
calls by phase, layer, position/chunk anchor, request sequence, weight type,
actual token count and chosen path. It caps unique keys at 131072; any dropped
calls invalidate completeness. Use short decode samples (e.g. 16–32 output
positions), and a separate process per context fixture. Never silently keep only
an arbitrary prefix of an overflowed or killed capture.

```sh
python tools/v100/dispatch_ledger.py /absolute/profile/engine.stderr \
  --output /absolute/profile/dispatch-ledger.json
```

Recorded paths include:

- MMQ built/support checks and unsupported prompt fallback; selected versus
  unselected fused GLU; Q8_1 quantization with/without gather; Blackwell-only FP4
  quantization if actually reached; `Context::run`, `run_glu`, native gather.
- CPU gate/up and down rows at the actual selected function branch: IQ512,
  IQ256, single-token IQ256 rows1, packed K-quants, direct IQ512 prefill, IQ4NL256
  and GGML `vec_dot` fallback. An `iq512` label describes the selected function;
  `ggml_vec_dot` does not claim an ISA that has not been verified in that build.
- T=1 versus T=2–8 and larger batches, using each expert's actual `nt`, rather
  than the prompt length or number of CPU pool threads.

CPU worker scopes explicitly copy the caller's context across pool threads.
Setup/calibration remains phase 0, prefill phase 1, decode phase 2. Prefill's
`position` is the chunk's first position, and `chunk_tokens` its full length;
gathered expert rows can represent noncontiguous tokens. **Individual prefill
row-to-token attribution is not claimed by this collector.** Decode issue uses
its actual position and host request sequence; service work is joined to that
position using `(layer,request_seq)`. Unresolved/ambiguous joins invalidate that
part of the evidence. Single-GPU only; multi-part sequence spaces need a device
identifier before sharing a capture.

`rows` and `logical_weight_bytes` describe matrix ranges presented to a host
call (gate/up counts both matrices). They do not multiply by `nt` to invent
physical traffic and are not hardware DRAM/PCIe counters. Device launches must
still be corroborated with a CUDA timeline. MMQ token count denotes product or
quantizer rows, which are routed expert rows rather than distinct prompt tokens.

Remaining P0 collection must preserve the selected CPU build/ISA metadata,
actual cache/wait/copy outcomes, per-layer dispatch totals and the long-context
critical-path timeline. A zero GLU execution count plus supported ordinary MMQ
is an expected SM70 observation, not a global MMQ failure. Results must be
reported separately for both exact upstream bases if making a base comparison;
these hooks initially instrument the working fork, not the immutable binaries.

Validation: 29 local V100 tooling tests pass. The ledger host test compiles two
translation units, aggregates concurrent workers into one process ledger, checks
scope restoration and default-off behavior. Parser tests cover service joins,
actual token grouping, truncation and dropped-call rejection. This does not
substitute for the queued CUDA build or physical V100 execution.
