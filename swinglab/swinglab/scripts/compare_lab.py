"""Phone-measured features vs. the Driveline lab ranges, one row per feature.

Reads analyzed swing JSONs (userdata/swings/<profile>/*.json, skipping _summary_ files) and the lab
feature table computed at phone rate (data/processed/features_30fps.csv). Prints a markdown table:
lab p10 / median / p90, then each swing's value, with values outside the lab p5–p95 marked.

  .venv\\Scripts\\python scripts\\compare_lab.py                      # all TestSet2 swings
  .venv\\Scripts\\python scripts\\compare_lab.py --glob "wilson/Wilson*"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from swinglab import features  # noqa: E402

DEFAULT_GLOBS = ["matteo/Matteo__*.json", "wilson/Wilson__*.json"]


def short_name(sid: str) -> str:
    """'Wilson__angle 1- lidar 1-02' -> 'W1-2'; 'Matteo__angle 1- lidar 6' -> 'M6'."""
    m = re.match(r"(\w)\w*__.*lidar (\d+)(?:-0?(\d+))?", sid)
    if not m:
        return sid
    return f"{m[1]}{m[2]}" + (f"-{m[3]}" if m[3] else "")


def fmt(v: float, unit: str) -> str:
    if not np.isfinite(v):
        return "–"
    if unit == "ratio":
        return f"{v:.2f}"
    if unit == "bool":
        return "yes" if v >= 0.5 else "no"
    if unit == "m/s":
        return f"{v:.1f}"
    return f"{v:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", help="pattern under userdata/swings/ (repeatable)")
    ap.add_argument("--root", default=str(ROOT / "userdata/swings"), help="folder holding <profile>/<id>.json")
    ap.add_argument("--lab", default=str(ROOT / "data/processed/features_30fps.csv"))
    args = ap.parse_args()

    files = sorted(f for g in (args.glob or DEFAULT_GLOBS) for f in Path(args.root).glob(g)
                   if not f.name.startswith("_summary_"))
    swings = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    lab = pd.read_csv(args.lab)
    names = [short_name(s["id"]) for s in swings]

    print("| Swing | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for row, key in [("bats", lambda s: s.get("bats")), ("mode", lambda s: s.get("mode")),
                     ("form", lambda s: s["analysis"]["scores"]["form"]),
                     ("P / S / Po", lambda s: "/".join(f"{s['analysis']['scores'][g]:.0f}" for g in features.GROUPS))]:
        print(f"| {row} | " + " | ".join(str(key(s)) for s in swings) + " |")
    print()

    print("| Feature | unit | lab p10 | lab median | lab p90 | " + " | ".join(names) + " |")
    print("|---|---|---|---|---|" + "---|" * len(names))
    for f, (grp, label, unit) in features.FEATURES.items():
        col = lab[f].dropna()
        p5, p10, p50, p90, p95 = np.percentile(col, [5, 10, 50, 90, 95])
        cells = []
        for s in swings:
            v = float(s["features"].get(f, np.nan) if s["features"].get(f) is not None else np.nan)
            out = np.isfinite(v) and unit != "bool" and not (p5 <= v <= p95)
            cells.append(f"**{fmt(v, unit)}**" if out else fmt(v, unit))
        lab_cells = [fmt(x, unit) for x in (p10, p50, p90)] if unit != "bool" else ["", f"{col.mean():.0%} yes", ""]
        print(f"| {label} ({grp[:4]}) | {unit} | " + " | ".join(lab_cells) + " | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
