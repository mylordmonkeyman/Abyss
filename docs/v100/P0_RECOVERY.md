# P0 interrupted campaign recovery

The original campaign completed the 108-request short/output matrix and one
historical long arm (32K/64K/128K, 1024 outputs, two requests each). The current
base's first long arm stopped after both 32K requests. Read-only inspection run
39193892390 preserved the partial data, found no surviving engine, and reported
no current cgroup OOM events. Pre-interruption samples retained >44 GiB available
memory and zero cgroup OOM events. The last surviving sample is 03:01:24Z, while
GitHub marked failure at 03:10:58Z with no finalized job log. This suggests an
external runner interruption but does not prove its cause. No unchanged engine
failure is being blindly retried and no completed arm is deleted.

`v100-p0-ledgers.yml` now collects separate allocation and actual-dispatch
observations. It verifies dynamic CUDA imports, uses the immutable binaries for
memory and the newly built working fork for opt-in dispatch, preserves active
host worksheets and process mapping/caller evidence, and excludes diagnostic
throughput from comparisons. Its 32K/128K requests repeat only to measure missing
accounting evidence, not to relabel timing samples as new performance repeats.

`v100-p0-remaining.yml` uses a new persistent `completion-2` checkpoint directory.
Each long context/base arm is its own bounded job and new process, with an
excluded first request and a second warm request. Both bases receive the same
fresh-process-per-case controls. This replaces the interrupted campaign's
cross-case residency pattern for the new long comparison; do not pool the two
patterns into one speedup estimate. The original evidence remains useful as a
separate baseline observation. Each unit uploads immediately, so an external
interruption loses at most one unit rather than a whole phase.

The fixed plan is 18 long units (three planned interleaved pairs per 32K/64K/128K),
three chunk-setting units (six process arms each for 512/1024/4096), then two
rootless timeline units (32K and 128K with 16 output positions). Matrix jobs run
one at a time on the existing runner, under the shared GPU concurrency lock.
Actual start timestamps must confirm process order before claiming interleaved
A/B controls; matrix scheduling alone is not proof of order. Completed units are
reused only with the frozen model/pack/binary/fixture/control identity. A partial
unit is preserved and stops for review. No container recreation, GPU reset,
privilege change, clock change, global cache drop or forced socket0 placement.

The long timeline uses immutable historical-base inference, a first unrecorded
warmup, then one full genuine-history request with reuse/speculation off. Nsight
CPU sampling/context-switch collection and hardware counters remain disabled.
Raw reports and SQLite stay on the runner; only bounded identities, activity
summaries, telemetry and logs upload. The analyzer reports a disjoint main
compute-stream wall-lane budget and its top three kernel categories. Gaps remain
unassigned without CPU/API/I/O correlation. Overlapping copy-stream activity is
not added to the main-stream wall budget; it is not a full causal dependency DAG.

Retain the 16 GiB pre-arm and 8 GiB runtime host-memory guards. Before each job,
refuse a leftover compute process instead of killing unrelated work. On a failed
unit, inspect its bounded artifact and fix the cause before retrying that unit.
P0 completion still requires reconciliation and final decisions; successful
collection alone does not close every gate.
