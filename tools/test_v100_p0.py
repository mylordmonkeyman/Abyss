"""P0 admission/measurement tests: synthetic GGUFs, captured driver XML, real child processes; no GPU."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("v100_p0", Path(__file__).parent / "v100" / "p0.py")
p0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p0)


def gguf(path, tensors):
    blob = bytearray(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), 0))
    offset, payload = 0, bytearray()
    for name, shape, ty, size in tensors:
        name = name.encode()
        blob += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(shape))
        blob += struct.pack(f"<{len(shape)}Q", *shape) + struct.pack("<IQ", ty, offset)
        payload += bytes(size)
        payload += bytes(-len(payload) % 32)
        offset = len(payload)
    blob += bytes(-len(blob) % 32)
    path.write_bytes(blob + payload)


def driver_xml(thermal="Not Active", power="Not Active", tag="clocks_event_reasons"):
    return f'''<nvidia_smi_log><gpu><uuid>GPU-test</uuid><temperature><gpu_temp>72 C</gpu_temp></temperature>
<clocks><sm_clock>1200 MHz</sm_clock><mem_clock>877 MHz</mem_clock></clocks>
<{tag}><clocks_event_reason_sw_thermal_slowdown>{thermal}</clocks_event_reason_sw_thermal_slowdown>
<clocks_event_reason_hw_thermal_slowdown>Not Active</clocks_event_reason_hw_thermal_slowdown>
<clocks_event_reason_sw_power_cap>{power}</clocks_event_reason_sw_power_cap></{tag}></gpu></nvidia_smi_log>'''


class Inventory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_real_types_roles_and_expert_axis(self):
        path = self.root / "m.gguf"
        gguf(path, [("blk.2.ffn_down_exps.weight", [256, 4, 2], 10, 84 * 8),
                    ("blk.2.indexer.attn_q_b.weight", [32, 8], 39, 17 * 8),
                    ("output.weight", [2, 3], 30, 12)])
        rows = p0.inventory([path], self.root / "out.csv")
        self.assertEqual((rows[0]["type"], rows[0]["bytes"], rows[0]["N"], rows[0]["K"]), ("Q2_K", 672, 4, 256))
        self.assertEqual(rows[0]["role"], "routed_expert_down")
        self.assertEqual(rows[1]["type"], "MXFP4")
        self.assertTrue(all(r["invocations"] == "" and r["existing_kernel"] == "" for r in rows))
        with (self.root / "out.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_split_expansion_and_missing_shard(self):
        a, b = (self.root / f"m-{i:05d}-of-00002.gguf" for i in (1, 2))
        gguf(a, [("a", [8], 0, 32)])
        with self.assertRaisesRegex(ValueError, "missing model shard"):
            p0.inventory([a], self.root / "out.csv")
        gguf(b, [("b", [8], 0, 32)])
        self.assertEqual(len(p0.inventory([a, b], self.root / "out.csv")), 2)

    def test_duplicate_names_and_truncated_payload(self):
        a, b = self.root / "a.gguf", self.root / "b.gguf"
        for f in (a, b): gguf(f, [("same", [8], 0, 32)])
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            p0.inventory([a, b], self.root / "out.csv")
        a.write_bytes(a.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "short/overlapping"):
            p0.inventory([a], self.root / "out.csv")

    def test_unknown_bytes_and_invalid_quant_row(self):
        a = self.root / "a.gguf"
        gguf(a, [("unknown", [256, 2], 999, 32)])
        self.assertEqual(p0.inventory([a], self.root / "out.csv")[0]["bytes"], "")
        # Total elements divisible by the block is insufficient: each row must be divisible.
        gguf(a, [("bad", [128, 2], 10, 84)])
        with self.assertRaisesRegex(ValueError, "does not divide row"):
            p0.inventory([a], self.root / "out.csv")


class Telemetry(unittest.TestCase):
    def test_version_gate(self):
        for ver in ("Version 2025.1.0.0", "Version 2025.2.1.0"):
            self.assertTrue(p0.compatible_ncu(ver))
        for ver in ("Version 2025.3.0.0", "Version 2026.1", "Version 2025.12"):
            self.assertFalse(p0.compatible_ncu(ver))

    def test_thermal_and_power_are_distinct(self):
        for tag in ("clocks_event_reasons", "clocks_throttle_reasons"):
            s = p0.xml_sample(driver_xml(power="Active", tag=tag))
            result = p0.thermal_assessment([s, s])
            self.assertFalse(result["thermal_flag"])
            self.assertEqual(result["regime"], "power_limited_observed")
            self.assertFalse(result["performance_claim_admitted"])
            hot = p0.xml_sample(driver_xml(thermal="Active", tag=tag))
            self.assertTrue(p0.thermal_assessment([hot, s])["thermal_flag"])

    def test_driver_580_instant_power_fields(self):
        # The target's 580.178.04 XML uses these tags; average power is N/A.
        xml = driver_xml().replace('</gpu>', '<gpu_power_readings>\n<average_power_draw>N/A</average_power_draw><instant_power_draw>44.90 W</instant_power_draw>\n<current_power_limit>250.00 W</current_power_limit></gpu_power_readings></gpu>')
        sample = p0.xml_sample(xml)
        self.assertEqual(sample['power_draw'], '44.90 W')
        self.assertEqual(sample['power_limit'], '250.00 W')
        legacy = driver_xml().replace('</gpu>', '<power_readings><power_draw>120 W</power_draw>'
                                     '<power_limit>250 W</power_limit></power_readings></gpu>')
        self.assertEqual(p0.xml_sample(legacy)['power_draw'], '120 W')

    def test_missing_reasons_or_samples_never_pass(self):
        unknown = p0.xml_sample("<nvidia_smi_log><gpu><uuid>x</uuid></gpu></nvidia_smi_log>")
        self.assertEqual(p0.thermal_assessment([unknown, unknown])["regime"], "unassessed")
        s = p0.xml_sample(driver_xml())
        self.assertEqual(p0.thermal_assessment([s, {"error": "lost driver"}])["regime"], "unassessed")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            p0.xml_sample("<nvidia_smi_log><gpu/><gpu/></nvidia_smi_log>")


class Capture(unittest.TestCase):
    def test_success_failure_and_timeout_keep_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, inputs = root / "m.gguf", root / "tokens.json"
            gguf(model, [("output.weight", [2, 2], 0, 16)])
            inputs.write_text("[1, 2, 3]")
            fake_meta = {"commands": {"gpu_identity": {"returncode": 0, "stdout": "GPU-test, Tesla V100, 7.0"}}}
            for label, argv, expected in (
                ("ok", [sys.executable, "-c", "print('measured command')"], "completed"),
                ("bad", [sys.executable, "-c", "raise SystemExit(7)"], "failed"),
                ("timeout", [sys.executable, "-c", "import time; time.sleep(30)"], "timeout"),
                ("missing", [str(root / "does-not-exist")], "launch_failed"),
            ):
                args = argparse.Namespace(output=str(root / label), gpu="0", model=[model], input=[inputs],
                    command=argv, profile=False, label=label, cache_state="process_restart", timeout=0.1,
                    interval_ms=200)
                with patch.object(p0, "manifest", return_value=fake_meta), patch.object(p0, "command", return_value={
                    "returncode": 0, "stdout": driver_xml()}):
                    code = p0.capture(args)
                self.assertEqual(code, 0 if expected == "completed" else 1)
                row = json.loads((root / label / "p0_v100_baseline.jsonl").read_text())
                self.assertEqual(row["status"], expected)
                self.assertIsNone(row["decode_tokens_per_s"])
                self.assertEqual(len(json.loads((root / label / "inputs.json").read_text())), 2)
                self.assertFalse(row["thermal"]["performance_claim_admitted"])
                with self.assertRaises(FileExistsError):
                    p0.capture(args)

    def test_no_gpu_is_a_blocker_not_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = argparse.Namespace(output=str(Path(tmp) / "run"), gpu="0")
            with patch.object(p0, "manifest", return_value={"commands": {"gpu_identity": {"returncode": None}}}):
                with self.assertRaisesRegex(ValueError, "preflight failed"):
                    p0.capture(args)
            self.assertTrue((Path(args.output) / "toolchain_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
