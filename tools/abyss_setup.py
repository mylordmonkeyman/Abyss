#!/usr/bin/env python3
"""Abyss - set up DeepSeek V4 Flash (run through ./setup-deepseek.sh, which makes the .venv first).

    ./setup-deepseek.sh                      everything below; each step is skipped when it is already done
    ./setup-deepseek.sh --quant UD-IQ3_XXS   another quant of unsloth/DeepSeek-V4-Flash-0731-GGUF
    ./setup-deepseek.sh --tools-only         only (re)create the Abyss-tools folder

What it does:

  1. checks the PC: an NVIDIA GPU, nvcc, RAM
  2. Python packages into .venv; llama.cpp's source at the pinned commit (third_party/llama.cpp)
  3. compiles the engine (build/strata) for the GPU - arch 120a on RTX 50, whose MXFP4 kernels need it
  4. the model: unsloth's DeepSeek-V4-Flash-0731-GGUF quant into <data>/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<quant>
     (asks first, or --download-model), every file checked against Hugging Face's SHA-256
  5. the pack next to the shards (tools/iq_pack.py --compat-bf16)
  6. the run config abyss-<name>.json and the start script run-abyss-<name>.sh in the Abyss folder
  7. the Abyss-tools folder (abyss.sh start / stop / status / update / log / models) from ops/abyss-tools/

The data folder defaults to Abyss-data beside Abyss, the tools folder to Abyss-tools beside it.  The helpers (the PC
checks, pip, llama.cpp's source, resumable downloads) are setup.py's, inherited from Maya and Strata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402

REPO = "unsloth/DeepSeek-V4-Flash-0731-GGUF"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{path}"
HF_TREE = "https://huggingface.co/api/models/{repo}/tree/main/{folder}"
# the quant this setup was tested with; its files are known even without the Hugging Face API
KNOWN = {
    "UD-Q2_K_XL": [
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf", 5257664,
         "de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b"),
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf", 49437013568,
         "c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327"),
        ("UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf", 47390237120,
         "a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4"),
    ],
}
TOOL_TEMPLATES = ROOT / "ops" / "abyss-tools"
MANAGED = ("abyss.sh", "common.sh")        # replaced when the template changes (the old one kept as .bak)
ONCE = ("README.md",)                      # written only when missing: the user's own notes go there


def quant_files(quant: str) -> list[tuple[str, int, str]]:
    """(path in the repo, size, sha256) of a quant's GGUF files, from Hugging Face's file list."""
    try:
        req = urllib.request.Request(HF_TREE.format(repo=REPO, folder=quant), headers={"User-Agent": "abyss-setup"})
        items = json.load(urllib.request.urlopen(req, timeout=60))
        files = sorted((i["path"], int(i["size"]), (i.get("lfs") or {}).get("oid", ""))
                       for i in items if i.get("type") == "file" and i["path"].endswith(".gguf"))
        if files:
            return files
    except OSError as e:
        if quant not in KNOWN:
            S.fail(f"cannot read the file list of {REPO}/{quant} ({e})", "check the internet connection, or the quant's name")
    if quant in KNOWN:
        return KNOWN[quant]
    S.fail(f"{REPO} has no quant {quant!r}", "names are the folders on https://huggingface.co/" + REPO)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verified(path: Path, sha: str) -> bool:
    """The file matches Hugging Face's SHA-256 (a <name>.sha256 mark keeps the answer while the file is unchanged)."""
    mark = path.with_name(path.name + ".sha256")
    st = path.stat()
    key = f"{sha} {st.st_size} {int(st.st_mtime)}"
    if mark.exists() and mark.read_text().strip() == key:
        return True
    S.say(f"  checking {path.name} ({st.st_size / 1e9:.1f} GB) ...")
    if sha256_of(path) != sha:
        return False
    mark.write_text(key)
    return True


def build_engine(arch: str | None, gpu) -> Path:
    exe = ROOT / "build" / "strata"
    cmake = ROOT / ".venv" / "bin" / "cmake"
    ninja = ROOT / ".venv" / "bin" / "ninja"
    if not (ROOT / "build" / "build.ninja").exists():
        nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
        if not Path(nvcc).exists():
            S.fail("nvcc (the CUDA toolkit) was not found", "install the CUDA toolkit (12.8 or newer for RTX 50)")
        cc = arch or gpu["arch"]
        if cc == "120":
            cc = "120a"                     # RTX 50: the MXFP4 MMQ kernels use Blackwell's FP4 MMA
        S.say(f"  Configuring for {gpu['name']} (CUDA arch {cc}) ...")
        S.run([cmake, "-S", ROOT, "-B", ROOT / "build", "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}",
               "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cc}",
               "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_COMPILER={nvcc}",
               f"-DSTRATA_GGML_DIR={ROOT / 'third_party' / 'llama.cpp'}"], quiet=True)
    S.say("  Compiling the engine (a few minutes the first time, only what changed later) ...")
    S.run([cmake, "--build", ROOT / "build", "--target", "strata", "-j", str(max(2, (os.cpu_count() or 4)))], quiet=True)
    S.ok(f"engine: {exe}")
    return exe


def write_config(name: str, pack: Path, ctx: int, port: int, rewrite: bool) -> tuple[Path, Path]:
    cfg = ROOT / f"abyss-{name}.json"
    run = ROOT / f"run-abyss-{name}.sh"
    if cfg.exists() and not rewrite:
        S.ok(f"{cfg.name} kept (--rewrite-config writes it again)")
    else:
        cfg.write_text(json.dumps({
            "exe": str(ROOT / "build" / "strata"),
            "args": ["--glm-pack", str(pack), "--max-context", str(ctx), "--prefill", "auto"],
            "cwd": str(ROOT),
            "tokenizer": str(pack / "tokenizer"),
            "model_name": "deepseek-v4-flash",
            "sampling": {"temperature": 1.0, "top_p": 0.95},
            "port": port,
            "log": str(ROOT / f"abyss-{name}.log"),
        }, indent=1) + "\n", encoding="utf-8")
        S.ok(f"{cfg.name} written (context {ctx}, port {port})")
    py = ROOT / ".venv" / "bin" / "python"
    run.write_text("#!/bin/sh\n"
                   f"# DeepSeek V4 Flash ({name}) on Abyss: the dashboard and API (written by setup-deepseek.sh)\n"
                   f'cd "{ROOT}" || exit 1\n'
                   f'exec "{py}" "{ROOT / "serve" / "server.py"}" --engine strata "--config" "{cfg}" '
                   f'"--port" "{json.loads(cfg.read_text())["port"]}" "$@"\n', encoding="utf-8")
    run.chmod(0o755)
    S.ok(f"{run.name} written")
    return cfg, run


def make_tools(tools: Path) -> None:
    """The Abyss-tools folder from ops/abyss-tools: abyss.sh and common.sh follow the template (a changed one is kept
    as .bak), README.md is written once."""
    tools.mkdir(parents=True, exist_ok=True)
    (tools / "logs").mkdir(exist_ok=True)
    for name in MANAGED + ONCE:
        text = (TOOL_TEMPLATES / name).read_text(encoding="utf-8").replace("@ABYSS@", str(ROOT))
        dst = tools / name
        if dst.exists():
            old = dst.read_text(encoding="utf-8")
            if old == text:
                continue
            if name in ONCE:
                S.ok(f"{dst} kept (yours)")
                continue
            dst.with_name(name + ".bak").write_text(old, encoding="utf-8")
            S.warn(f"{dst} updated from the template (the previous one: {name}.bak)")
        dst.write_text(text, encoding="utf-8")
        if name.endswith(".sh"):
            dst.chmod(0o755)
    S.ok(f"tools: {tools}  ({tools / 'abyss.sh'} start | stop | status | update | log | models)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Set up DeepSeek V4 Flash on Abyss")
    ap.add_argument("--quant", default="UD-Q2_K_XL", help="a folder of " + REPO + " (default UD-Q2_K_XL, the tested one)")
    ap.add_argument("--name", help="the model's name in abyss-<name>.json / run-abyss-<name>.sh (default dsv4-<quant>)")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "Abyss-data"), help="where the models go (default: Abyss-data beside Abyss)")
    ap.add_argument("--tools-dir", default=str(ROOT.parent / "Abyss-tools"), help="the tools folder (default: Abyss-tools beside Abyss)")
    ap.add_argument("--context", type=int, default=65536, help="the context in tokens (default 65536)")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--gpu", type=int, help="the GPU, as nvidia-smi numbers it (default: the one with the most VRAM)")
    ap.add_argument("--arch", help="the CUDA architecture to compile for (default: the GPU's; 120a on RTX 50)")
    ap.add_argument("--download-model", action="store_true", help="download the model without asking")
    ap.add_argument("--yes", action="store_true", help="the defaults for every question (the download still needs --download-model)")
    ap.add_argument("--rewrite-config", action="store_true", help="write abyss-<name>.json again")
    ap.add_argument("--tools-only", action="store_true", help="only create / update the Abyss-tools folder")
    a = ap.parse_args()
    tools = Path(a.tools_dir).expanduser().resolve()
    if a.tools_only:
        make_tools(tools)
        return 0
    quant = a.quant
    name = a.name or "dsv4-" + quant.lower().removeprefix("ud-")
    data = Path(a.data_dir).expanduser().resolve()
    mdir = data / "models" / REPO / quant

    S.step(1, "the PC")
    gpu = S.gpu_info(a.gpu)
    if gpu is None:
        S.fail("no NVIDIA GPU found (nvidia-smi)", "Abyss's DeepSeek V4 path runs on NVIDIA GPUs")
    S.ok(f"GPU {gpu['index']}: {gpu['name']}, {gpu['vram_gb']:.0f} GB, compute capability {gpu['arch']}")
    S.ok(f"RAM {S.ram_gb():.0f} GB (with VRAM, ~100 GB holds every expert of the 97 GB quant; less reads from the SSD)")

    S.step(2, "Python packages and llama.cpp's source")
    S.pip_install(S.PY_PACKAGES, ", ".join(S.PY_PACKAGES))
    S.ok(f"llama.cpp: {S.get_llama_cpp()}")

    S.step(3, "the engine")
    build_engine(a.arch, gpu)

    S.step(4, f"the model: {REPO} {quant}")
    files = quant_files(quant)
    total = sum(f[1] for f in files)
    S.say(f"  {len(files)} files, {total / 1e9:.1f} GB -> {mdir}")
    missing = [f for f in files if not (mdir / Path(f[0]).name).exists()]
    if missing:
        free = shutil.disk_usage(mdir.parent if mdir.parent.exists() else data.parent).free
        S.say(f"  to download: {sum(f[1] for f in missing) / 1e9:.1f} GB (free on that disk: {free / 1e9:.0f} GB)")
        if not a.download_model and S.ask("  Download it now?", ["y", "n"], "n", a.yes) != "y":
            S.fail("the model is not downloaded", "run again with --download-model, or put the files into " + str(mdir))
    for path, size, sha in files:
        dst = mdir / Path(path).name
        S.download(HF_FILE.format(repo=REPO, path=path), dst, dst.name)
        if sha and not verified(dst, sha):
            S.fail(f"{dst.name} does not match Hugging Face's SHA-256", "delete it and run again")
    S.check_shards([mdir / Path(f[0]).name for f in files])
    S.ok("every file checked")

    S.step(5, "the pack")
    pack = mdir / "pack-dsv4"
    if (pack / "native_experts.txt").exists() and (pack / "tokenizer" / "chat_template.jinja").exists():
        S.ok(f"pack: {pack} (already made)")
    else:
        first = mdir / Path(files[0][0]).name
        S.run([sys.executable, ROOT / "tools" / "iq_pack.py", "--gguf", first, "--out", pack, "--compat-bf16"], quiet=True)
        S.ok(f"pack: {pack}")

    S.step(6, "the run config")
    write_config(name, pack, a.context, a.port, a.rewrite_config)

    S.step(7, "Abyss-tools")
    make_tools(tools)

    S.say()
    S.say(f"Done. Start it with:  {tools / 'abyss.sh'} start {name}")
    S.say(f"then open http://127.0.0.1:{a.port}/  (the manual: {ROOT / 'docs' / 'MANUAL.md'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
