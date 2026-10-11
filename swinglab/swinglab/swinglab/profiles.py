"""Hitter profiles and their swing history, stored OUTSIDE the app code.

  <userdata>/profiles.json             [{id, name, bats, height_m, mass_kg, age, level, synthetic, created}]
  <userdata>/swings/<profile id>/      one JSON (+ annotated video) per analyzed swing,
                                       _summary_<recording>.json for recordings with 2+ swings
  <userdata>/trash/                    deleted profiles are moved here, never destroyed

<userdata> is config.USERDATA (env SWINGLAB_USERDATA, default <repo>/userdata, git-ignored), so a
fresh clone has no one's swings; the "Synthetic" profile (Driveline lab swings) is written there by
scripts/export_demo.py and can be shipped or not.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from . import config

FIELDS = {"name": str, "bats": str, "height_m": float, "mass_kg": float, "age": float, "level": str}
LEVELS = ("high_school", "college", "independent", "milb")


def root(r: str | Path | None = None) -> Path:
    return Path(r or config.USERDATA)


def _file(r) -> Path:
    return root(r) / "profiles.json"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-") or "hitter"


def list_profiles(r=None) -> list[dict]:
    f = _file(r)
    return json.loads(f.read_text(encoding="utf-8")).get("profiles", []) if f.exists() else []


def _save(profiles: list[dict], r=None) -> None:
    f = _file(r)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps({"profiles": profiles}, indent=1), encoding="utf-8")
    tmp.replace(f)


def get(pid: str, r=None) -> dict | None:
    return next((p for p in list_profiles(r) if p["id"] == pid), None)


def by_name(name: str, r=None) -> dict | None:
    n = (name or "").strip().lower()
    return next((p for p in list_profiles(r) if p["name"].strip().lower() == n), None)


def clean(fields: dict) -> dict:
    """Validated profile fields (unknown keys dropped)."""
    out = {}
    for k, typ in FIELDS.items():
        v = fields.get(k)
        if v in (None, ""):
            continue
        out[k] = typ(v) if typ is not str else str(v).strip()
    if "bats" in out and out["bats"].upper() not in ("R", "L"):
        raise ValueError("bats must be R or L")
    if "bats" in out:
        out["bats"] = out["bats"].upper()
    if "level" in out and out["level"] not in LEVELS:
        raise ValueError(f"level must be one of {', '.join(LEVELS)}")
    if "height_m" in out and not 1.0 <= out["height_m"] <= 2.3:
        raise ValueError("height_m should be in meters (e.g. 1.78)")
    if "mass_kg" in out and not 25 <= out["mass_kg"] <= 200:
        raise ValueError("mass_kg should be in kilograms (e.g. 75)")
    return out


def create(fields: dict, r=None, *, synthetic: bool = False, pid: str | None = None) -> dict:
    f = clean(fields)
    if not f.get("name"):
        raise ValueError("a profile needs a name")
    profiles = list_profiles(r)
    if any(p["name"].strip().lower() == f["name"].lower() for p in profiles):
        raise ValueError(f"a profile named {f['name']} already exists")
    base = pid or slug(f["name"])
    ids = {p["id"] for p in profiles}
    pid, k = base, 2
    while pid in ids:
        pid, k = f"{base}-{k}", k + 1
    p = {"id": pid, "bats": "R", "height_m": 1.78, "level": "college", **f, "synthetic": synthetic,
         "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(profiles + [p], r)
    (root(r) / "swings" / pid).mkdir(parents=True, exist_ok=True)
    if not synthetic:
        drop_dir(p["name"], r).mkdir(parents=True, exist_ok=True)
    return p


def drop_dir(name: str, r=None) -> Path:
    """<userdata>/drop/<Name>/: put a hitter's downloaded recordings here and the server analyzes them."""
    return root(r) / "drop" / name


def update(pid: str, fields: dict, r=None) -> dict:
    profiles = list_profiles(r)
    p = next((x for x in profiles if x["id"] == pid), None)
    if p is None:
        raise KeyError(pid)
    f = clean(fields)
    if "name" in f and any(x["id"] != pid and x["name"].strip().lower() == f["name"].lower() for x in profiles):
        raise ValueError(f"a profile named {f['name']} already exists")
    p.update(f)
    _save(profiles, r)
    return p


def delete(pid: str, r=None) -> Path | None:
    """Remove the profile; its swings move to <userdata>/trash/<id>-<time> (recoverable)."""
    profiles = list_profiles(r)
    if not any(p["id"] == pid for p in profiles):
        raise KeyError(pid)
    _save([p for p in profiles if p["id"] != pid], r)
    src = root(r) / "swings" / pid
    if src.exists():
        dst = root(r) / "trash" / f"{pid}-{time.strftime('%Y%m%d-%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return dst
    return None


def hitter(p: dict) -> dict:
    """Profile -> the hitter dict the pipeline takes."""
    return {"name": p["name"], "bats": p.get("bats", "R"), "height_m": p.get("height_m", 1.78),
            "mass_kg": p.get("mass_kg"), "age": p.get("age"), "level": p.get("level", "college")}


def ensure(name: str, defaults: dict | None = None, r=None) -> dict:
    """The profile with this name, created from ``defaults`` (e.g. a hitters.csv row) if missing."""
    return by_name(name, r) or create({**(defaults or {}), "name": name}, r)


def swings_dir(pid: str, r=None) -> Path:
    return root(r) / "swings" / pid


def swing_summaries(pid: str, r=None) -> list[dict]:
    """One line per analyzed swing of a profile, oldest first (for the history slider and trends)."""
    out = []
    for f in swings_dir(pid, r).glob("*.json"):
        if f.name.startswith("_"):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        an = rec.get("analysis") or {}
        sc = an.get("scores") or {}
        out.append({"id": rec.get("id"), "file": f.name, "recorded_at": rec.get("recorded_at") or rec.get("analyzed_at"),
                    "mode": rec.get("mode", "dry"), "scores": sc,
                    "bat_speed_predicted": (an.get("predicted") or {}).get("bat_speed"),
                    "bat_speed_measured": (rec.get("measured") or {}).get("bat_speed_mph"),
                    "exit_speed": (rec.get("measured") or {}).get("exit_velo_mph"),
                    "headline": (rec.get("report") or {}).get("headline"), "has_video": bool(rec.get("overlay")),
                    "quality_ok": not any("varies" in w or "placed from bone length" in w for w in rec.get("warnings", []))})
    return sorted(out, key=lambda x: (x["recorded_at"] or "", x["id"] or ""))


def recordings(pid: str, r=None) -> list[dict]:
    """A profile's swings grouped by the recording they came from (the whole-recording view), oldest
    first. Each: {"recording", "n_swings", "recorded_at", "summary" (file name of _summary_<rec>.json
    or None), "headline", "form" (mean/best/worst), "video" ({"file", "t0"} of _recording_<rec>.mp4:
    the whole recording, or None), "swings": [{"id", "file", "video_file" (annotated swing video),
    "t" ({start, peak, end}: seconds into the whole-recording video), "partial", "mode", "scores",
    "focus" (label)}]}. Swings analyzed before recordings were tracked (no "recording" field) are
    grouped under their own id."""
    d = swings_dir(pid, r)
    groups: dict[str, list[dict]] = {}
    for f in sorted(d.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = rec.get("recording") or rec.get("id") or f.stem
        w = rec.get("window") or {}
        t0 = rec.get("recording_t0")
        an = rec.get("analysis") or {}
        groups.setdefault(name, []).append({
            "id": rec.get("id"), "file": f.name,
            "video_file": Path(rec["overlay"]["path"]).name if (rec.get("overlay") or {}).get("path") else None,
            "t": {k: round(w[k] - t0, 3) for k in ("start", "peak", "end")} if w and t0 is not None else None,
            "partial": bool(w.get("partial")), "mode": rec.get("mode", "dry"), "scores": an.get("scores") or {},
            "focus": (an.get("focus") or {}).get("label"),
            "recorded_at": rec.get("recorded_at") or rec.get("analyzed_at")})
    out = []
    for name, sw in groups.items():
        sw.sort(key=lambda x: ((x["t"] or {}).get("peak", 0.0), x["id"] or ""))
        stem = Path(name).stem
        summ = d / f"_summary_{stem}.json"
        sm = {}
        if summ.exists():
            try:
                sm = json.loads(summ.read_text(encoding="utf-8"))
            except Exception:
                sm = {}
        rv = sm.get("recording_video") or {}
        video = d / f"_recording_{stem}.mp4"
        out.append({"recording": name, "n_swings": len(sw), "recorded_at": min((x["recorded_at"] or "") for x in sw) or None,
                    "summary": summ.name if summ.exists() else None,
                    "headline": (sm.get("report") or {}).get("headline"), "form": sm.get("form_score"),
                    "video": {"file": video.name, "t0": rv.get("t0")} if video.exists() else None,
                    "swings": [{k: v for k, v in x.items() if k != "recorded_at"} for x in sw]})
    return sorted(out, key=lambda x: (x["recorded_at"] or "", x["recording"]))
