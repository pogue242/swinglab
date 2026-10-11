"""Canonical skeleton shared by the lab data and the phone.

Both sources are converted to the same representation before ANY feature is
computed, so the model sees identical inputs at training and inference time:

* joints renamed lead/rear (a right-handed hitter's lead side is the left)
* body frame defined from the hitter's own stance, not the lab or camera:
    origin  mid-hip at stance
    +z      up (lab z, or the floor normal for the phone)
    +x      horizontal direction rear ankle -> lead ankle at stance (toward the pitcher)
    +y      z × x, then mirrored for left-handed hitters, so every hitter is
            expressed as a right-handed hitter (plate on the -y side)
* resampled to the phone frame rate (config.PHONE_FPS)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config

CANONICAL = [
    "lead_shoulder", "rear_shoulder", "lead_elbow", "rear_elbow", "lead_wrist", "rear_wrist",
    "lead_hip", "rear_hip", "lead_knee", "rear_knee", "lead_ankle", "rear_ankle", "neck",
]
# Bones for drawing (pairs of canonical names)
BONES = [
    ("lead_shoulder", "rear_shoulder"), ("lead_hip", "rear_hip"),
    ("lead_shoulder", "lead_hip"), ("rear_shoulder", "rear_hip"),
    ("lead_shoulder", "lead_elbow"), ("lead_elbow", "lead_wrist"),
    ("rear_shoulder", "rear_elbow"), ("rear_elbow", "rear_wrist"),
    ("lead_hip", "lead_knee"), ("lead_knee", "lead_ankle"),
    ("rear_hip", "rear_knee"), ("rear_knee", "rear_ankle"),
]
# MediaPipe Pose landmark index -> side-specific name (for the phone path)
MEDIAPIPE = {
    11: "l_shoulder", 12: "r_shoulder", 13: "l_elbow", 14: "r_elbow", 15: "l_wrist", 16: "r_wrist",
    23: "l_hip", 24: "r_hip", 25: "l_knee", 26: "r_knee", 27: "l_ankle", 28: "r_ankle",
}


@dataclass
class Skeleton:
    """``J`` is (T, n_joints, 3) in meters in the canonical body frame."""

    time: np.ndarray
    J: np.ndarray
    names: list[str] = field(default_factory=lambda: list(CANONICAL))
    bats: str = "R"
    height_m: float | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)  # e.g. bat points (lab only)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.J[:, self.names.index(name), :]

    @property
    def fps(self) -> float:
        return 1.0 / float(np.median(np.diff(self.time)))

    @property
    def mid_hip(self) -> np.ndarray:
        return 0.5 * (self["lead_hip"] + self["rear_hip"])

    @property
    def mid_shoulder(self) -> np.ndarray:
        return 0.5 * (self["lead_shoulder"] + self["rear_shoulder"])

    @property
    def mid_wrist(self) -> np.ndarray:
        return 0.5 * (self["lead_wrist"] + self["rear_wrist"])


def to_lead_rear(joints: dict[str, np.ndarray], bats: str) -> dict[str, np.ndarray]:
    """Map l_/r_ joints to lead_/rear_ by handedness. Right-handed hitter: lead = left."""
    lead, rear = ("l", "r") if bats.upper() == "R" else ("r", "l")
    out = {}
    for name, arr in joints.items():
        if name.startswith(lead + "_"):
            out["lead_" + name[2:]] = arr
        elif name.startswith(rear + "_"):
            out["rear_" + name[2:]] = arr
        else:
            out[name] = arr
    if "neck" not in out and "lead_shoulder" in out:
        out["neck"] = 0.5 * (out["lead_shoulder"] + out["rear_shoulder"])
    return out


def stance_window(n: int, frac: float = 0.08) -> slice:
    return slice(0, max(3, int(n * frac)))


def body_frame(joints: dict[str, np.ndarray], bats: str, up: np.ndarray | None = None):
    """Return (origin, R) mapping input coords -> canonical: p' = R @ (p - origin)."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else up / np.linalg.norm(up)
    w = stance_window(len(joints["lead_ankle"]))
    d = np.nanmean(joints["lead_ankle"][w] - joints["rear_ankle"][w], axis=0)
    x = d - up * (d @ up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    if bats.upper() == "L":
        y = -y  # mirror lefties into a right-handed hitter
    R = np.vstack([x, y, up])
    origin = np.nanmean(0.5 * (joints["lead_hip"][w] + joints["rear_hip"][w]), axis=0)
    # Height is measured from the floor, estimated from the ankles at stance
    # (ankle joint centers sit ~ANKLE_HEIGHT_M above the floor). The lab floor
    # is z=0 already; ARKit/phone world origins are wherever the phone started.
    ankles = np.r_[joints["lead_ankle"][w], joints["rear_ankle"][w]]
    floor = float(np.nanmedian(ankles @ up)) - ANKLE_HEIGHT_M
    origin = origin - up * (origin @ up) + up * floor
    return origin, R


ANKLE_HEIGHT_M = 0.08


def resample(time: np.ndarray, arr: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear resample of (T, ...) array to a uniform grid at ``fps``."""
    t_new = np.arange(time[0], time[-1] + 1e-9, 1.0 / fps)
    flat = arr.reshape(len(time), -1)
    out = np.empty((len(t_new), flat.shape[1]))
    for k in range(flat.shape[1]):
        col = flat[:, k]
        ok = np.isfinite(col)
        out[:, k] = np.interp(t_new, time[ok], col[ok]) if ok.sum() > 1 else np.nan
    return t_new, out.reshape((len(t_new),) + arr.shape[1:])


def build(time: np.ndarray, joints: dict[str, np.ndarray], bats: str, *, fps: float | None = None,
          up: np.ndarray | None = None, height_m: float | None = None, noise_m: float = 0.0,
          rng: np.random.Generator | None = None, extras: tuple[str, ...] = ()) -> Skeleton:
    """Raw joints (side-named, any frame) -> canonical Skeleton at ``fps``.

    ``noise_m`` adds Gaussian jitter (meters) before resampling, to train on
    data that looks more like phone depth than clean mocap.
    """
    lr = to_lead_rear(joints, bats)
    origin, R = body_frame(lr, bats, up)
    J = np.stack([(lr[n] - origin) @ R.T for n in CANONICAL], axis=1)
    ex = {e: (lr[e] - origin) @ R.T for e in extras if e in lr}
    if noise_m > 0:
        rng = rng or np.random.default_rng(0)
        J = J + rng.normal(0, noise_m, J.shape)
    t = time - time[0]
    if fps:
        t_new, J = resample(t, J, fps)
        ex = {k: resample(t, v, fps)[1] for k, v in ex.items()}
        t = t_new
    return Skeleton(t, J, list(CANONICAL), bats.upper(), height_m, ex)


def from_obp(swing, bats: str, height_m: float, fps: float | None = config.PHONE_FPS, **kw) -> Skeleton:
    sk = build(swing.time, swing.joints, bats, fps=fps, height_m=height_m,
               extras=("bat_sweet_spot", "bat_handle"), **kw)
    return sk
