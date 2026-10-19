#!/usr/bin/env python3
"""P0 evidence capture. Standard library only; never changes clocks, power, affinity or weights."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from gguf_reader import GGUFFile, BLOCK_GEOMETRY  # noqa: E402

BASE = "47001f31cbfba14bcbf4c083e3af07c50707ec0c"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(argv, timeout=20):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": argv, "returncode": None, "error": str(e)}


def shards(paths):
    """Expand conventional split filenames, deduplicate paths and refuse missing siblings."""
    found = set()
    for p in paths:
        p = Path(p).resolve()
        m = re.fullmatch(r"(.*)-(\d{5})-of-(\d{5})\.gguf", p.name)
        if m:
            count = int(m[3])
            if not 1 <= int(m[2]) <= count <= 99999:
                raise ValueError(f"invalid shard numbering: {p}")
            found.update(p.with_name(f"{m[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1))
        else:
            found.add(p)
    for p in sorted(found):
        if not p.is_file():
            raise ValueError(f"missing model shard: {p}")
    return sorted(found)


def role(name):
    if "ffn_gate_exps" in name: return "routed_expert_gate"
    if "ffn_up_exps" in name: return "routed_expert_up"
    if "ffn_down_exps" in name: return "routed_expert_down"
    if "_shexp" in name: return "shared_expert"
    if "ffn_gate_inp" in name: return "router"
    if "indexer" in name: return "indexer"
    if "compressor" in name: return "compressor"
    if name == "output.weight": return "vocabulary_output"
    if name == "token_embd.weight": return "embedding"
    if "attn_q_" in name or "attn_kv" in name: return "q_kv_projection_or_norm"
    if "attn_output" in name: return "output_projection"
    return "other"


def inventory(paths, output):
    """Static metadata is not a runtime call, residency or kernel census. Leave those fields empty."""
    rows, seen = [], set()
    model_paths = shards(paths)
    split_numbers = set()
    for path in model_paths:
        gg = GGUFFile(path)
        if "split.count" in gg.metadata:
            if gg.metadata["split.count"] != len(model_paths):
                raise ValueError(f"split.count disagrees with supplied model: {path}")
            number = gg.metadata.get("split.no")
            if not isinstance(number, int) or not 0 <= number < len(model_paths) or number in split_numbers:
                raise ValueError(f"invalid or duplicate split.no: {path}")
            split_numbers.add(number)
        size = path.stat().st_size
        ordered = sorted(gg.tensors, key=lambda t: t.offset)
        for i, t in enumerate(ordered):
            if t.name in seen:
                raise ValueError(f"duplicate tensor across shards: {t.name}")
            seen.add(t.name)
            if not t.shape or len(t.shape) > 4 or any(d <= 0 for d in t.shape):
                raise ValueError(f"invalid shape: {t.name}")
            geom = BLOCK_GEOMETRY.get(t.type_name)
            if geom and t.shape[0] % geom[0]:
                raise ValueError(f"quant block does not divide row: {t.name}")
            nbytes = t.expected_bytes()
            bound = ordered[i + 1].offset if i + 1 < len(ordered) else size - gg.data_start
            if t.offset % gg.alignment or t.offset < 0 or bound < t.offset or gg.data_start + bound > size:
                raise ValueError(f"invalid tensor offset: {path.name}: {t.name}")
            if nbytes is not None and t.offset + nbytes > bound:
                raise ValueError(f"short/overlapping tensor payload: {path.name}: {t.name}")
            layer = re.match(r"blk\.(\d+)\.", t.name)
            rows.append(dict(role=role(t.name), layer=layer[1] if layer else "", name=t.name,
                             type=t.type_name, N=t.shape[1] if len(t.shape) >= 2 else "",
                             K=t.shape[0] if len(t.shape) >= 2 else "", shape=json.dumps(t.shape),
                             bytes=nbytes if nbytes is not None else "", shard=str(path),
                             token_width="", invocations="", device_ms="", residency="", hotness="",
                             existing_kernel="", evidence="GGUF_metadata_only"))
    if split_numbers and len(split_numbers) != len(model_paths):
        raise ValueError("mixed split and non-split model files")
    if not rows:
        raise ValueError("model contains no tensors")
    with Path(output).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compatible_ncu(version):
    return re.search(r"\b2025\.[12](?:\.|\b)", version) is not None


def manifest(gpu):
    commands = {
        "git_head": ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        "git_status": ["git", "-C", str(ROOT), "status", "--porcelain"],
        "nvcc": ["nvcc", "--version"], "ncu": ["ncu", "--version"], "nsys": ["nsys", "--version"],
        "gpu_identity": ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,name,compute_cap", "--format=csv,noheader"],
        "gpu_details": ["nvidia-smi", "-i", gpu, "-q"],
        "gpu_topology": ["nvidia-smi", "topo", "-m"], "cpu": ["lscpu"], "numa": ["numactl", "-H"],
    }
    results = {k: command(v) for k, v in commands.items()}
    tools = {}
    for name in ("nvcc", "ncu", "nsys"):
        path = shutil.which(name)
        tools[name] = {"path": path, "sha256": sha256(path) if path else None}
    ncu = results["ncu"]
    supported = ncu["returncode"] == 0 and compatible_ncu(ncu.get("stdout", "") + ncu.get("stderr", ""))
    return dict(schema_version=1, timestamp_utc=now(), pinned_base=BASE, commands=results, tools=tools,
                ncu_version_compatible=supported,
                counter_gate={"status": "blocked", "reason": "SM70 profiling smoke test, tool license/source provenance and counter reports required"},
                qpi_gate={"status": "unmeasured", "reason": "QPI counters and actual worker/page placement require a target-host experiment"},
                cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                environment={k: v for k, v in os.environ.items() if k.startswith(("ABYSS_V100_", "STRATA_", "OMP_"))})


def xml_sample(xml):
    root = ET.fromstring(xml)
    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise ValueError("sampler requires exactly one selected GPU")
    g = gpus[0]
    # Driver releases use both spellings. Keep every reason, not just an aggregate bit.
    reasons = g.find("clocks_event_reasons")
    if reasons is None:
        reasons = g.find("clocks_throttle_reasons")
    flags = {e.tag: e.text.strip() if e.text else "" for e in reasons} if reasons is not None else {}
    thermal_values = [v for k, v in flags.items() if "thermal" in k]
    thermal_known = len(thermal_values) >= 2 and all(v in ("Active", "Not Active") for v in thermal_values)
    power_values = [v for k, v in flags.items() if "sw_power_cap" in k]
    return dict(gpu_uuid=g.findtext("uuid"), temperature=g.findtext("temperature/gpu_temp"),
                memory_temperature=g.findtext("temperature/memory_temp"),
                power_draw=g.findtext("gpu_power_readings/instant_power_draw") or
                g.findtext("gpu_power_readings/power_draw") or g.findtext("power_readings/power_draw"),
                power_limit=g.findtext("gpu_power_readings/current_power_limit") or
                g.findtext("gpu_power_readings/power_limit") or g.findtext("power_readings/power_limit"),
                sm_clock=g.findtext("clocks/sm_clock"), memory_clock=g.findtext("clocks/mem_clock"),
                pstate=g.findtext("performance_state"), reasons=flags,
                thermal_flag=any(v == "Active" for v in thermal_values), thermal_reasons_known=thermal_known,
                power_cap_active=any(v == "Active" for v in power_values))


def thermal_assessment(samples):
    good = [s for s in samples if "error" not in s]
    flagged = any(s["thermal_flag"] for s in good)
    complete = len(good) >= 2 and len(good) == len(samples) and all(s["thermal_reasons_known"] for s in good)
    return dict(samples=len(samples), valid_samples=len(good), thermal_flag=flagged,
                reason_coverage_complete=complete,
                regime="thermal_flagged" if flagged else "unassessed" if not complete else
                       "power_limited_observed" if any(s["power_cap_active"] for s in good) else "no_thermal_reason_observed",
                performance_claim_admitted=False,
                note="Sampled reasons alone do not establish clock stability, sustained state or matched A/B conditions")


def capture(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite or silently append to a previous experiment.
    meta = manifest(args.gpu)
    write_json(out / "toolchain_manifest.json", meta)
    identity = meta["commands"]["gpu_identity"]
    identities = [line for line in identity.get("stdout", "").splitlines() if line.strip()]
    if identity["returncode"] != 0 or len(identities) != 1 or not re.search(r",\s*7\.0\s*$", identities[0]):
        raise ValueError(f"SM70 GPU preflight failed; see {out / 'toolchain_manifest.json'}")
    model_files = shards(args.model)
    inventory(model_files, out / "p0_tensor_inventory.csv")
    # Hash before sampling/timing: reading ~97 GB can materially affect the page-cache state.
    inputs = [dict(kind=kind, path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha256(p))
              for kind, paths in (("model", model_files), ("input", [Path(p) for p in args.input])) for p in paths]
    write_json(out / "inputs.json", inputs)
    argv = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not argv:
        raise ValueError("capture requires a command after --")
    binary_paths = list(getattr(args, "binary", []))
    executable = shutil.which(argv[0])
    if executable:
        binary_paths.append(executable)
    write_json(out / "binary_hashes.json", [dict(path=str(Path(p).resolve()), sha256=sha256(p))
               for p in sorted(set(binary_paths))])
    env = os.environ.copy()
    if args.profile:
        env["ABYSS_V100_PROFILE"] = "1"
    legacy_profile = {k: env[k] for k in ("STRATA_GLM_PROF", "STRATA_GLM_PREFILL_PROF", "STRATA_GLM_GAP_PROF") if k in env}
    profiled = env.get("ABYSS_V100_PROFILE") == "1" or bool(legacy_profile)
    write_json(out / "run_config.json", dict(argv=argv, label=args.label, cache_state=args.cache_state,
               profile=profiled, requested_profile=args.profile, legacy_profile_controls=legacy_profile,
               gpu=args.gpu, interval_ms=args.interval_ms, timeout_s=args.timeout,
               cwd=str(Path.cwd()), effective_profile=env.get("ABYSS_V100_PROFILE"),
               hash_cache_effect="All model/input hashes read before run; this is NOT a verified cache-cold run"))
    samples, stop = [], threading.Event()

    def sample():
        r = command(["nvidia-smi", "-i", args.gpu, "-q", "-x"], timeout=10)
        stamp = dict(timestamp_utc=now(), monotonic_ns=time.monotonic_ns())
        try:
            if r["returncode"] != 0:
                raise ValueError(r.get("error") or r.get("stderr", "nvidia-smi failed"))
            stamp.update(xml_sample(r["stdout"]))
        except (ValueError, ET.ParseError) as e:
            stamp["error"] = str(e)
        samples.append(stamp)
        with (out / "p0_gpu_clock_power_thermal.jsonl").open("a") as f:
            f.write(json.dumps(stamp) + "\n")

    def sampler():
        while not stop.wait(args.interval_ms / 1000):
            sample()

    sample()
    thread = threading.Thread(target=sampler, daemon=True)
    thread.start()
    run_started = now()
    start = time.monotonic_ns()
    status, rc, error = "launch_failed", None, None
    proc = None
    try:
        with (out / "command.log").open("w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            status = "running"
            try:
                rc = proc.wait(timeout=args.timeout)
                status = "completed" if rc == 0 else "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
    except (OSError, KeyboardInterrupt) as e:
        error = str(e)
        status = "interrupted" if isinstance(e, KeyboardInterrupt) else "launch_failed"
    finally:
        # Stop an entire timed-out/interrupted command tree, including profiler children.
        if proc is not None and (status in ("timeout", "interrupted") or proc.poll() is None):
            import signal
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                pass
        end = time.monotonic_ns()
        stop.set()
        thread.join(timeout=15)
        sample()
        with (out / "p0_gpu_clock_power_thermal.csv").open("w", newline="") as f:
            fields = ["timestamp_utc", "monotonic_ns", "gpu_uuid", "temperature", "memory_temperature",
                      "power_draw", "power_limit", "sm_clock", "memory_clock", "pstate",
                      "thermal_flag", "thermal_reasons_known", "power_cap_active", "reasons", "error"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for sample_row in samples:
                writer.writerow({**sample_row, "reasons": json.dumps(sample_row.get("reasons", {}))})
        assessment = thermal_assessment(samples)
        row = dict(schema_version=1, label=args.label, timestamp_utc=run_started, status=status, returncode=rc,
                   error=error, command_wall_ms=(end - start) / 1e6, monotonic_start_ns=start, monotonic_end_ns=end,
                   profile=profiled, cache_state=args.cache_state, thermal=assessment,
                   prefill_tokens_per_s=None, decode_tokens_per_s=None, ttft_ms=None,
                   metrics_note="Command wall includes load/teardown; engine rates remain in command.log, never inferred from this wall time",
                   p0_gate="partial: correctness, counters, memory ledger, critical path, sustained matched A/B still required")
        (out / "p0_v100_baseline.jsonl").write_text(json.dumps(row) + "\n")
    return 0 if status == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    inv = sub.add_parser("inventory", help="static GGUF census; runtime fields remain unmeasured")
    inv.add_argument("model", nargs="+")
    inv.add_argument("--output", required=True)
    cap = sub.add_parser("capture", help="capture one real SM70 workload and its continuous telemetry")
    cap.add_argument("--output", required=True, help="new experiment directory")
    cap.add_argument("--gpu", default="0", help="GPU index or UUID; must resolve to one cc 7.0 device")
    cap.add_argument("--label", required=True)
    cap.add_argument("--cache-state", required=True, help="describe actual warm/reuse/restart conditions")
    cap.add_argument("--model", action="append", required=True, help="GGUF path; split siblings discovered")
    cap.add_argument("--input", action="append", required=True, help="prompt token IDs/config/workload file; repeat to hash all inputs")
    cap.add_argument("--binary", action="append", default=[], help="additional engine/library file to hash, e.g. when argv starts with nsys")
    cap.add_argument("--profile", action="store_true", help="enable instrumented stages; collect wall baseline separately")
    cap.add_argument("--interval-ms", type=int, default=500)
    cap.add_argument("--timeout", type=float, default=3600)
    cap.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.action == "inventory":
            inventory(args.model, args.output)
            return 0
        if not args.command or args.command == ["--"]:
            parser.error("capture requires a finite command after --")
        if args.interval_ms < 200 or args.timeout <= 0:
            parser.error("interval must be >=200 ms and timeout must be positive")
        return capture(args)
    except (ValueError, OSError, KeyError, EOFError, struct.error) as e:
        parser.exit(1, f"P0: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
