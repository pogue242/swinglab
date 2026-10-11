"""Build the static demo site for Vercel: the viewer plus Matteo's and Wilson's swings.

  python scripts\\build_site.py
  python scripts\\build_site.py --profiles matteo wilson --out site

Copies each swing's annotated video into site/media and writes site/index.html with the swing
records embedded (window.SWINGLAB_DATA). Night-test swings (file names that do not start with the
hitter's name) are left out. Absolute paths from this computer are removed before anything is written.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from swinglab import config, profiles  # noqa: E402

REPO_URL = "https://github.com/pogue242/baseball-batting-analyze"
ZIP_URL = REPO_URL + "/archive/refs/heads/main.zip"
RECORDING_URL = REPO_URL + "/blob/main/README.md#run-it-on-your-own-swings"
MARKER = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
LOOSE_NUM = re.compile(r"([:,\[]\s*)-?(?:NaN|Infinity)(?=\s*[,\]}])")


def load_loose(path: Path) -> dict:
    """Read a swing file. The pipeline writes NaN for a missing number; JSON does not allow that."""
    text = LOOSE_NUM.sub(r"\1null", path.read_text(encoding="utf-8"))
    return json.loads(text)


def is_local_path(value: str) -> bool:
    """True for a Windows or macOS path from the machine that analyzed the swing."""
    slash = value.replace("\\", "/")
    return bool(re.search(r"[A-Za-z]:[/\\]", value)) or slash.startswith("//") or "/Users/" in slash or slash.startswith("/home/")


def scrub(value):
    """Replace absolute paths with null. Leaves every other value as it is."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, str) and is_local_path(value):
        return None
    return value


def slug(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", Path(name).stem).strip("-").lower()
    return stem or "swing"


def swing_files(pid: str, name: str) -> list[Path]:
    """`<Name>__*.json` only: the night-test files and the `_summary` / `_recording` files stay out."""
    prefix = f"{name}__"
    folder = profiles.swings_dir(pid)
    return sorted(p for p in folder.glob("*.json") if p.name.startswith(prefix) and not p.name.startswith("_"))


def take_swing(rec: dict, pid: str, media: Path, used: set[str]) -> dict | None:
    """Copy the annotated video and point the record at it. None when the swing has no video."""
    overlay = rec.get("overlay") or {}
    rel = overlay.get("path")
    src = (config.USERDATA / rel) if rel else None
    if src is None or not src.is_file():
        print(f"  skip {rec.get('id')}: no video")
        return None
    name = slug(rec.get("id") or src.stem)
    if name in used:
        raise SystemExit(f"two swings would both be named {name}")
    used.add(name)
    dest_rel = f"swings/{pid}/{name}.mp4"
    dest = media / dest_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    rec = scrub(rec)
    rec["overlay"]["path"] = dest_rel
    return rec


def trend_of(swings: list[dict]) -> list:
    return [s["analysis"]["scores"].get("overall") for s in swings if (s.get("analysis") or {}).get("scores", {}).get("overall") is not None][-12:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profiles", nargs="+", default=["matteo", "wilson"])
    ap.add_argument("--out", default="site")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    media = out / "media"
    if media.exists():
        shutil.rmtree(media)
    media.mkdir(parents=True)

    app = json.loads((config.WEB_DATA / "app.json").read_text(encoding="utf-8"))
    profs: list[dict] = []
    swings: dict[str, list] = {}
    n_bytes = 0
    for pid in args.profiles:
        prof = profiles.get(pid)
        if prof is None or prof.get("synthetic"):
            raise SystemExit(f"no such hitter: {pid}")
        taken: list[dict] = []
        used: set[str] = set()
        files = swing_files(pid, prof["name"])
        files.sort(key=lambda p: p.name)
        for f in files:
            rec = load_loose(f)
            kept = take_swing(rec, pid, media, used)
            if kept is not None:
                taken.append(kept)
        if not taken:
            raise SystemExit(f"{prof['name']} has no swings with a video")
        taken.sort(key=lambda r: (r.get("recorded_at") or "", r.get("id") or ""))
        trend = trend_of(taken)
        profs.append({**prof, "n_swings": len(taken), "latest_overall": trend[-1] if trend else None, "trend": trend})
        swings[pid] = taken
        n_bytes += sum(p.stat().st_size for p in (media / "swings" / pid).glob("*.mp4"))
        print(f"{prof['name']}: {len(taken)} swings")

    data = {"app": app, "profiles": profs, "swings": swings, "media_root": "media/",
            "repo_url": REPO_URL, "zip_url": ZIP_URL, "recording_url": RECORDING_URL}
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False)
    for bad in (":\\\\", ":\\", "/Users/", "C:/Users"):
        if bad in blob:
            raise SystemExit(f"a local path is still in the demo data ({bad})")

    src = (config.ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if MARKER not in src:
        raise SystemExit("viewer page has no three.js script tag to attach the demo data to")
    page = src.replace(MARKER, f"<script>window.SWINGLAB_DATA={blob};</script>\n{MARKER}", 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "vercel.json").write_text(json.dumps({
        "headers": [{"source": "/media/(.*)", "headers": [
            {"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]}]}, indent=2) + "\n",
        encoding="utf-8")
    print(f"{out}")
    print(f"{sum(len(v) for v in swings.values())} swings, videos {n_bytes / 1e6:.0f} MB, page { (out / 'index.html').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
