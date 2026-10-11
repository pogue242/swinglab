"""Shared-folder inbox: both phones upload to one synced folder; the laptop pairs
and analyzes everything new.

Layout (e.g. a Google Drive folder synced to the laptop by Google Drive for desktop):

  SwingLab/
    hitters.csv                  name,bats,height_m,mass_kg,age,level
    sessions/
      2026-10-10_field/
        lidar/                   Stray Scanner folders (or .zip of them) from the iPhone
        slowmo/                  240 fps videos from the second phone
        log.csv                  optional: swing,hitter,set,lidar,slowmo,contact,notes

Pairing a LiDAR recording with its slow-motion video(s):
  1. log.csv rows naming both files (exact, preferred); several rows naming the same LiDAR
     recording = a long session with one slow-motion clip per swing
  2. by time, when both are known (Record3D names carry the start time, iPhone videos their
     creation time): each clip goes to the LiDAR recording that was running when it was filmed
  3. one LiDAR recording and several clips: all clips belong to it (a batting-cage session)
  4. otherwise by recording order (sorted by modification time, then name);
     counts must match or the session is flagged and pairs are marked "guessed".
Clips of a long session are matched to its swings later, once the swings are found
(session.match_clips: by when the ball left the tee, else by order).
The hitter comes from log.csv, else the session's only hitter in hitters.csv,
else the first row.
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

VIDEO_EXT = {".mov", ".mp4", ".m4v"}


@dataclass
class Pair:
    session: str
    swing: str
    lidar: str | None
    slowmo: str | None
    hitter: str | None
    set: str = ""
    method: str = "order"   # log | order | lidar_only | slowmo_only
    notes: list[str] = field(default_factory=list)
    yardstick: str | None = None   # the session's scale clip (see is_yardstick) for the slow-motion camera
    slowmo_clips: list[str] = field(default_factory=list)   # several clips for one long LiDAR recording


def is_yardstick(p: Path) -> bool:
    """A slow-motion file of the yardstick at the tee: name contains "yardstick" or "calib"."""
    n = p.stem.lower()
    return "yardstick" in n or "calib" in n


def _yardstick_for(video: str | None, sticks: list[Path]) -> str | None:
    """The latest yardstick clip recorded before this video (re-recorded whenever the camera moved),
    else the first one."""
    if not sticks:
        return None
    if video:
        t = Path(video).stat().st_mtime
        before = [y for y in sticks if y.stat().st_mtime <= t]
        if before:
            return str(max(before, key=lambda y: y.stat().st_mtime))
    return str(sticks[0])


def _is_stray(p: Path) -> bool:
    """Any LiDAR recording: Stray Scanner folder, Record3D .r3d file or extracted Record3D folder."""
    from .capture import is_recording
    return is_recording(p)


def unzip_new(folder: Path) -> None:
    """Expand zipped Stray Scanner exports in place (phones often zip folders)."""
    for z in folder.glob("*.zip"):  # (.r3d files are zips too, but are read directly)
        target = folder / z.stem
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)


def find_lidar(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    unzip_new(folder)
    found = [p for p in folder.rglob("*") if _is_stray(p) and not p.name.endswith("_r3d")]
    return sorted(set(found), key=lambda p: (p.stat().st_mtime, p.name))


def find_videos(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    vids = [p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXT and not _is_stray(p.parent)]
    return sorted(vids, key=lambda p: (p.stat().st_mtime, p.name))


def read_hitters(root: Path) -> dict[str, dict]:
    p = root / "hitters.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {r["name"].strip(): {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()} for r in csv.DictReader(f)}


def pair_session(sdir: Path, default_hitter: str | None) -> list[Pair]:
    """Pairs in one folder: lidar/ and slowmo/ subfolders, or everything loose in the folder itself
    (the drop-folder layout: .r3d files and slow-motion clips side by side)."""
    lid = find_lidar(sdir / "lidar" if (sdir / "lidar").is_dir() else sdir)
    vids = find_videos(sdir / "slowmo" if (sdir / "slowmo").is_dir() else sdir)
    sticks = [v for v in vids if is_yardstick(v)]
    vids = [v for v in vids if not is_yardstick(v)]
    pairs: list[Pair] = []
    log = sdir / "log.csv"
    if log.exists():
        by_name_l = {p.name: p for p in lid} | {p.stem: p for p in lid}
        by_name_v = {p.name: p for p in vids} | {p.stem: p for p in vids}
        with open(log, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:   # calibration rows name the yardstick clip ("set" calibration/yardstick, or a note)
            if ((r.get("set") or "").strip().lower() in ("calibration", "yardstick")
                    or "yardstick" in (r.get("notes") or "").lower()):
                y = by_name_v.get((r.get("slowmo") or "").strip())
                if y and y not in sticks:
                    sticks.append(y)
        for r in rows:
            if ((r.get("set") or "").strip().lower() in ("calibration", "yardstick")
                    or "yardstick" in (r.get("notes") or "").lower()):
                continue
            l = by_name_l.get((r.get("lidar") or "").strip())
            v = by_name_v.get((r.get("slowmo") or "").strip())
            if not (l or v):
                continue
            same = next((q for q in pairs if l and q.lidar == str(l)), None)
            if same is not None and v:      # another clip for the same LiDAR recording: a session
                same.slowmo_clips = (same.slowmo_clips or [same.slowmo]) + [str(v)]
                same.slowmo = None
                continue
            pairs.append(Pair(sdir.name, (r.get("swing") or "").strip() or (l or v).stem, l and str(l), v and str(v),
                              (r.get("hitter") or "").strip() or default_hitter, (r.get("set") or "").strip(), "log",
                              yardstick=_yardstick_for(v and str(v), sticks)))
        return pairs
    grouped = _group_by_time(lid, vids) if vids and lid else None
    if grouped is None and len(lid) == 1 and len(vids) > 1:
        grouped = ({lid[0]: list(vids)}, [], "session_order")
    if grouped is not None:
        groups, leftover, method = grouped
        for i, l in enumerate(lid):
            vs = groups.get(l, [])
            pairs.append(Pair(sdir.name, f"{i + 1:03d}", str(l), str(vs[0]) if len(vs) == 1 else None, default_hitter,
                              "", method if vs else "lidar_only", [], yardstick=_yardstick_for(vs and str(vs[0]), sticks),
                              slowmo_clips=[str(v) for v in vs] if len(vs) > 1 else []))
        for k, v in enumerate(leftover):
            pairs.append(Pair(sdir.name, f"{len(lid) + k + 1:03d}", None, str(v), default_hitter, "", "slowmo_only",
                              ["filmed while no LiDAR recording was running"], yardstick=_yardstick_for(str(v), sticks)))
        return pairs
    n = max(len(lid), len(vids))
    note = [] if len(lid) == len(vids) else [f"count mismatch: {len(lid)} LiDAR vs {len(vids)} slow-motion; pairs guessed by order"]
    for i in range(n):
        l = lid[i] if i < len(lid) else None
        v = vids[i] if i < len(vids) else None
        method = "order" if (l and v) else ("lidar_only" if l else "slowmo_only")
        pairs.append(Pair(sdir.name, f"{i + 1:03d}", l and str(l), v and str(v), default_hitter, "", method, list(note),
                          yardstick=_yardstick_for(v and str(v), sticks)))
    return pairs


def _group_by_time(lid: list[Path], vids: list[Path], slack_s: float = 10.0):
    """({lidar: [clips]}, leftover clips, "time") when every recording and clip has a wall-clock time
    and each clip falls inside one LiDAR recording's span (± slack); else None."""
    from .capture import load_any, recording_start_time, video_start_time
    spans = []
    for l in lid:
        t0 = recording_start_time(l)
        if t0 is None:
            return None
        try:
            cap = load_any(l)
            dur = float(cap.timestamps[-1] - cap.timestamps[0])
        except Exception:
            return None
        spans.append((l, t0, t0 + dur))
    groups: dict = {l: [] for l in lid}
    leftover = []
    for v in vids:
        tv = video_start_time(v)
        if tv is None:
            return None
        hit = [l for l, a, b in spans if a - slack_s <= tv <= b + slack_s]
        (groups[hit[0]] if hit else leftover).append(v)
    if not any(groups.values()):
        return None
    if all(len(vs) <= 1 for vs in groups.values()) and not leftover and len(vids) == len(lid):
        return None                      # plain one-clip-per-recording: the order pairing says the same
    return groups, leftover, "time"


def scan(root: str | Path, names: list[str] | None = None) -> list[Pair]:
    """Pairs for every session folder. Hitters: hitters.csv rows and ``names`` (profile names)."""
    root = Path(root)
    hitters = {**{n: {} for n in (names or [])}, **read_hitters(root)}
    default = next(iter(read_hitters(root)), None) or (names[0] if names else None)
    out = []
    drop = not (root / "sessions").is_dir()          # drop folder: one subfolder per hitter, files loose inside
    for sdir in sorted((root / "sessions").glob("*") if not drop else root.glob("*")):
        if sdir.is_dir() and not sdir.name.startswith((".", "_")):
            if drop:
                name = next((h for h in hitters if h.lower() == sdir.name.lower()), None)
                if name is None:
                    continue                          # not a hitter's folder
                out += pair_session(sdir, name)
                continue
            # the upload page writes the chosen hitter to hitter.txt; log.csv rows still win
            ht = sdir / "hitter.txt"
            session_hitter = ht.read_text(encoding="utf-8").strip() if ht.exists() else ""
            out += pair_session(sdir, session_hitter if session_hitter in hitters else default)
    return out


def load_state(root: Path) -> dict:
    p = Path(root) / ".swinglab_processed.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(root: Path, state: dict) -> None:
    (Path(root) / ".swinglab_processed.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def key(p: Pair) -> str:
    clips = p.slowmo or ",".join(Path(c).name for c in p.slowmo_clips) or None   # new clips -> run again
    return f"{p.session}/{p.swing}|{p.lidar}|{clips}"


def process(root: str | Path, *, analyze_recording=None, save=None, save_session=None,
            force: bool = False, log=print, userdata: str | Path | None = None) -> list[dict]:
    """Analyze every new pair. Every swing found in a recording is analyzed the same way (one or
    many per recording) and saved in its hitter's profile folder (profiles.py, user data); a
    recording with 2+ swings also gets a summary. The hitter's profile is created from
    hitters.csv if it doesn't exist yet. Defaults: pipeline.analyze_recording / save_result /
    save_session. ``save(rec, folder)`` gets the profile id as folder."""
    from . import pipeline, profiles
    root = Path(root)
    analyze_recording = analyze_recording or pipeline.analyze_recording
    save = save or (lambda r, folder: pipeline.save_result(r, folder, out_root=userdata))
    save_session = save_session or (lambda sm, folder: pipeline.save_session(sm, folder, out_root=userdata))
    csv_hitters = read_hitters(root)
    state = {} if force else load_state(root)
    done = []
    for p in scan(root, [pr["name"] for pr in profiles.list_profiles(userdata) if not pr.get("synthetic")]):
        k = key(p)
        if k in state:
            continue
        name = p.hitter or ""
        prof = profiles.by_name(name, userdata)
        if prof is None and name in csv_hitters:
            prof = profiles.create({**csv_hitters[name], "name": name}, userdata)
            log(f"[profile] created {prof['name']} from hitters.csv")
        if prof is None:
            log(f"[skip] {p.session}/{p.swing}: hitter '{name}' has no profile (add one in the viewer or hitters.csv)")
            continue
        h = {**csv_hitters.get(name, {}), **profiles.hitter(prof)}
        log(f"[run] {p.session}/{p.swing} ({p.method}) hitter={prof['name']} lidar={p.lidar and Path(p.lidar).name} "
            f"slowmo={(p.slowmo and Path(p.slowmo).name) or (p.slowmo_clips and f'{len(p.slowmo_clips)} clips')}"
            + (f" yardstick={Path(p.yardstick).name}" if p.yardstick else ""))
        extra = {k: v for k, v in (("yardstick", p.yardstick), ("tape", bat_tape(h)),
                                   ("slowmo_clips", p.slowmo_clips or None)) if v is not None}
        out = analyze_recording(p.lidar, p.slowmo, h, **extra)
        recs = out.get("recs") or []
        saved = []
        for r in recs:
            r["pair"] = asdict(p)
            r["profile"] = prof["id"]
            r["id"] = f"{p.session}__{r.get('id') or p.swing}"
            if "analysis" not in r:
                log(f"      !! {r['id']}: not analyzed: " + "; ".join(
                    f"{s_}: {v.get('error')}" for s_, v in (r.get("stages") or {}).items() if not v.get("ok")))
                continue
            saved.append(str(save(r, prof["id"])))
            log(f"      -> {r['id']}: {r.get('mode', '?')} swing, form {r['analysis']['scores'].get('form')}"
                f", contact {r['analysis']['scores'].get('contact')}, stages "
                f"{ {s_: v.get('ok') for s_, v in (r.get('stages') or {}).items()} }")
        if out.get("summary"):
            out["summary"]["recording"] = f"{p.session}__{Path(p.lidar or 'recording').stem}"
            save_session(out["summary"], prof["id"])
            for note in out["summary"].get("notes") or []:
                log(f"      !! {note}")
        state[k] = {"profile": prof["id"], "swings": len(saved), "results": saved}
        done.extend(r for r in recs if "analysis" in r or not saved)
        save_state(root, state)
    return done


def bat_tape(h: dict):
    """BatTape from optional hitters.csv columns bat_length_m, tip_band_m, hand_band_m (meters from the knob)."""
    from .ballflight import BatTape
    vals = {k: h.get(k) for k in ("bat_length_m", "tip_band_m", "hand_band_m")}
    vals = {k: float(v) for k, v in vals.items() if v not in (None, "")}
    return BatTape(**vals) if vals else None


def template(root: str | Path) -> None:
    """Create the folder layout and example CSVs."""
    root = Path(root)
    (root / "sessions" / "example_session" / "lidar").mkdir(parents=True, exist_ok=True)
    (root / "sessions" / "example_session" / "slowmo").mkdir(parents=True, exist_ok=True)
    h = root / "hitters.csv"
    if not h.exists():
        h.write_text("name,bats,height_m,mass_kg,age,level\nAdrian,R,1.78,75,20,college\nWilson,R,1.80,80,20,college\n", encoding="utf-8")
    lg = root / "sessions" / "example_session" / "log.csv"
    if not lg.exists():
        lg.write_text("swing,hitter,set,lidar,slowmo,contact,notes\n001,Adrian,normal,<stray folder name>,<video file name>,line drive,\n", encoding="utf-8")
