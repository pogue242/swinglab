"""Driveline OpenBiomechanics (OBP) hitting data: download, verify, load.

Source: https://github.com/drivelineresearch/openbiomechanics (data license
CC BY-NC-SA 4.0 + exclusion for pro-sports / financial-firm affiliates; code MIT).

What we use
-----------
* ``metadata.csv`` (in git): one row per swing: height, mass, age, level,
  handedness, bat size, bat speed (Blast sensor + mocap), exit velocity.
* ``poi/poi_metrics.csv`` (in git): Driveline's own per-swing metrics. Used to
  VALIDATE our features, never as model inputs (the phone can't compute them).
* ``poi/hittrax.csv`` (in git): launch angle, distance, result, point of impact.
* ``landmarks.csv`` (release asset ``hitting_landmarks.zip``): 360 Hz joint
  centers in meters (lab frame: +x toward pitcher, +y toward the right-handed
  batter's box, +z up) plus event times ``fp_10_time``, ``fp_100_time``
  (lead-foot force > 10 N / 100 N = foot plant) and ``contact_time``.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RELEASE_ASSETS = {
    "hitting": ["hitting_landmarks.zip", "hitting_joint_angles.zip", "hitting_joint_velos.zip"],
    "hitting_c3d": ["hitting_c3d.zip"],
    "pitching": ["pitching_landmarks.zip", "pitching_joint_velos.zip"],
}
REPO_FILES = [
    "baseball_hitting/data/metadata.csv",
    "baseball_hitting/data/poi/poi_metrics.csv",
    "baseball_hitting/data/poi/hittrax.csv",
    "scripts/release_checksums.sha256",
    "baseball_hitting/data/data_dictionary.csv",
]

# Landmark columns we keep (prefix -> canonical name). Driveline names are not
# in the data dictionary; meanings were checked from geometry (see
# tests/test_obp_geometry.py): l/r + {s,e,w,k,a} + "jc" = shoulder, elbow,
# wrist, knee, ankle joint centers; "lhjc"/"rhjc" are HAND centers (~7.6 cm
# from the wrist), NOT hips; the hips are "left_hip"/"right_hip".
LANDMARK_MAP = {
    "lsjc": "l_shoulder", "rsjc": "r_shoulder",
    "lejc": "l_elbow", "rejc": "r_elbow",
    "lwjc": "l_wrist", "rwjc": "r_wrist",
    "left_hip": "l_hip", "right_hip": "r_hip",
    "lhjc": "l_hand", "rhjc": "r_hand",
    "lkjc": "l_knee", "rkjc": "r_knee",
    "lajc": "l_ankle", "rajc": "r_ankle",
    "thorax_prox": "neck",          # top of thorax segment ~ base of neck
    "centerofmass": "com",
    "sweet_spot": "bat_sweet_spot",  # bat marker-derived; phone can't see it
    "blast_hand": "bat_handle",
}
EVENT_COLS = ["fp_10_time", "fp_100_time", "contact_time"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checksums() -> dict[str, str]:
    p = config.RAW / "release_checksums.sha256"
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                out[parts[1]] = parts[0]
    return out


def _fetch(url: str, out: Path) -> None:
    """Plain-Python download (works on Windows/macOS/Linux, no curl or gh needed)."""
    tmp = out.with_suffix(out.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.replace(out)


def download(groups=("hitting",), dest: Path | None = None) -> list[Path]:
    """Download repo CSVs + release assets, verifying checksums.

    No GitHub login is needed (release assets are public URLs).
    groups: "hitting" (~220 MB), "hitting_c3d" (430 MB), "pitching" (~70 MB).
    """
    dest = dest or config.RAW
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for rel in REPO_FILES:
        out = dest / Path(rel).name if "poi/" not in rel else dest / "poi" / Path(rel).name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            _fetch(config.OBP_REPO_RAW.format(path=rel), out)
        got.append(out)
    sums = _checksums()
    for g in groups:
        for name in RELEASE_ASSETS[g]:
            out = dest / "obp_release" / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                print("downloading", name)
                _fetch(config.OBP_RELEASE_URL.format(name=name), out)
            if name in sums and _sha256(out) != sums[name]:
                raise RuntimeError(f"checksum mismatch for {name}")
            got.append(out)
    return got


def load_metadata() -> pd.DataFrame:
    m = pd.read_csv(config.RAW / "metadata.csv")
    m["height_m"] = m["session_height_in"] * config.M_PER_IN
    m["mass_kg"] = m["session_mass_lbs"] * config.KG_PER_LB
    return m.set_index("session_swing")


def load_poi() -> pd.DataFrame:
    return pd.read_csv(config.RAW / "poi" / "poi_metrics.csv").set_index("session_swing")


def load_hittrax() -> pd.DataFrame:
    return pd.read_csv(config.RAW / "poi" / "hittrax.csv").set_index("session_swing")


def landmarks_parquet() -> Path:
    return config.PROCESSED / "obp_hitting_landmarks.parquet"


def build_landmarks_parquet(zip_path: Path | None = None) -> Path:
    """Extract the columns we need from landmarks.csv (220 MB) into a parquet."""
    zip_path = zip_path or config.RAW / "obp_release" / "hitting_landmarks.zip"
    out = landmarks_parquet()
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as f:
            header = pd.read_csv(f, nrows=0).columns
        keep = ["session_swing", "time"] + EVENT_COLS + [
            c for c in header if c.rsplit("_", 1)[0] in LANDMARK_MAP and c[-1] in "xyz"
        ]
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=keep, dtype={"session_swing": str})
    df.to_parquet(out, index=False)
    return out


class Swing:
    """One swing as arrays: ``joints[name] -> (T, 3)`` in meters, lab frame."""

    def __init__(self, swing_id: str, time: np.ndarray, joints: dict[str, np.ndarray], events: dict[str, float]):
        self.id = swing_id
        self.time = time
        self.joints = joints
        self.events = events

    @property
    def fps(self) -> float:
        return 1.0 / float(np.median(np.diff(self.time)))


def iter_swings(df: pd.DataFrame | None = None):
    df = df if df is not None else pd.read_parquet(landmarks_parquet())
    for sid, g in df.groupby("session_swing", sort=False):
        g = g.sort_values("time")
        joints = {}
        for prefix, name in LANDMARK_MAP.items():
            cols = [f"{prefix}_{a}" for a in "xyz"]
            if all(c in g.columns for c in cols):
                joints[name] = g[cols].to_numpy(dtype=float)
        ev = {k: float(g[k].iloc[0]) for k in EVENT_COLS if k in g.columns and pd.notna(g[k].iloc[0])}
        yield Swing(str(sid), g["time"].to_numpy(dtype=float), joints, ev)
