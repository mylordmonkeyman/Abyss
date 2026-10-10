"""tools/abyss_setup.py's Abyss-tools folder: made from ops/abyss-tools with Abyss's path, the scripts follow the
template (a changed one kept as .bak), the README is the user's once written.  No GPU, no network.

    python -m unittest tools.test_abyss_setup
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abyss_setup as A  # noqa: E402


class AbyssTools(unittest.TestCase):
    def test_made_from_the_templates(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            for name in ("abyss.sh", "common.sh", "README.md"):
                self.assertTrue((t / name).exists(), name)
            self.assertTrue((t / "logs").is_dir())
            common = (t / "common.sh").read_text(encoding="utf-8")
            self.assertIn(f'ABYSS="${{ABYSS_DIR:-{A.ROOT}}}"', common)
            self.assertNotIn("@ABYSS@", common)
            self.assertTrue((t / "abyss.sh").stat().st_mode & 0o111)

    def test_again_updates_the_scripts_and_keeps_the_readme(self):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "Abyss-tools"
            A.make_tools(t)
            (t / "README.md").write_text("my notes\n", encoding="utf-8")
            (t / "abyss.sh").write_text("old\n", encoding="utf-8")
            A.make_tools(t)
            self.assertEqual((t / "README.md").read_text(encoding="utf-8"), "my notes\n")
            self.assertEqual((t / "abyss.sh.bak").read_text(encoding="utf-8"), "old\n")
            self.assertEqual((t / "abyss.sh").read_text(encoding="utf-8"),
                             (A.TOOL_TEMPLATES / "abyss.sh").read_text(encoding="utf-8"))
            self.assertFalse((t / "common.sh.bak").exists())   # unchanged: no backup

    def test_the_tested_quant_is_known_offline(self):
        files = A.KNOWN["UD-Q2_K_XL"]
        self.assertEqual(len(files), 3)
        self.assertTrue(all(len(sha) == 64 for _, _, sha in files))


if __name__ == "__main__":
    unittest.main()
