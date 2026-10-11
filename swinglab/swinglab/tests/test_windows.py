"""Windows-specific regressions (found setting up on Windows 11, Python 3.12, 2026-10-09).

1. Text files read/written without an encoding use the ANSI code page (cp1252)
   on Windows: prompts with "—" reach Gemini garbled, and SPEC.md ("❌", byte
   0x9D/0x8F) crashed scripts/build_uml.py with UnicodeDecodeError.
2. Output piped to a file or another process is cp1252 too: printing "→" or an
   emoji raised UnicodeEncodeError.
3. scripts/export_demo.py ran pytest with "python3", which on Windows is the
   Microsoft Store stub / a different interpreter without our packages.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted([*(ROOT / "swinglab").rglob("*.py"), *(ROOT / "scripts").glob("*.py"), *(ROOT / "tests").glob("*.py"),
                  *(ROOT / "api").glob("*.py")])


def _calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def test_every_text_file_access_names_an_encoding():
    bad = []
    for p in SOURCES:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for c in _calls(tree):
            kw = {k.arg for k in c.keywords}
            name = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
            if name in ("read_text", "write_text") and "encoding" not in kw:
                bad.append(f"{p.name}:{c.lineno} {name}")
            if name == "open" and isinstance(c.func, ast.Name):   # builtin open, text mode
                mode = c.args[1].value if len(c.args) > 1 and isinstance(c.args[1], ast.Constant) else \
                    next((k.value.value for k in c.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)), "r")
                if "b" not in mode and "encoding" not in kw:
                    bad.append(f"{p.name}:{c.lineno} open({mode!r})")
    assert not bad, bad


def test_no_hardcoded_python3_in_subprocesses():
    bad = [f"{p.name}:{i}" for p in SOURCES if p.name != Path(__file__).name
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if '"python3"' in line]
    assert not bad, f"use sys.executable: {bad}"


def test_prompts_are_read_as_utf8():
    from swinglab.coach import gemini
    txt = gemini._prompt("report")
    assert "—" in txt and "â€" not in txt


def test_piped_output_survives_non_ascii():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    r = subprocess.run([sys.executable, "-c", "import swinglab; print('→ ✓ ❌ 72°')"], capture_output=True, env=env)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert r.stdout.decode("utf-8").strip() == "→ ✓ ❌ 72°"


def test_paths_in_index_use_forward_slashes(tmp_path):
    import json
    from swinglab import pipeline
    pipeline.save_result({"id": "a", "analysis": {"scores": {"overall": 1}}}, "s 1", out_root=tmp_path)
    idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert idx["swings"][0]["path"] == "swings/s 1/a.json"
