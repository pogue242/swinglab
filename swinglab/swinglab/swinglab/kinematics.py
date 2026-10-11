"""Angles, speeds and swing events computed from a canonical Skeleton.

Every function here works on any frame rate, so the same code runs on 360 Hz
mocap (for validation) and on phone-rate data (for training and inference).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .skeleton import Skeleton


def smooth(x: np.ndarray, fps: float, window_s: float = 0.1) -> np.ndarray:
    """Savitzky–Golay smoothing along time; window scales with frame rate."""
    n = len(x)
    w = int(round(window_s * fps)) | 1
    w = max(5, min(w, n - (1 - n % 2)))
    if n < 5 or w >= n:
        return x
    return savgol_filter(x, w, 2, axis=0)


def yaw_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Horizontal angle of segment a->b relative to +x (toward pitcher), degrees.

    For the rear->lead hip or shoulder line: ~0° closed at stance, increasing
    as the segment opens toward the pitcher (positive = open).
    """
    d = b - a
    return np.degrees(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b between b->a and b->c, degrees (180 = straight)."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), 1e-12)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def speed(p: np.ndarray, fps: float) -> np.ndarray:
    return np.linalg.norm(np.gradient(p, 1.0 / fps, axis=0), axis=1)


def series(sk: Skeleton) -> dict[str, np.ndarray]:
    """Per-frame signals used by events, features and the viewer."""
    fps = sk.fps
    s = {}
    s["pelvis_yaw"] = smooth(yaw_deg(sk["rear_hip"], sk["lead_hip"]), fps)
    s["torso_yaw"] = smooth(yaw_deg(sk["rear_shoulder"], sk["lead_shoulder"]), fps)
    s["separation"] = s["pelvis_yaw"] - s["torso_yaw"]          # hips more open than shoulders = +
    s["pelvis_vel"] = np.gradient(s["pelvis_yaw"], 1 / fps)     # deg/s
    s["torso_vel"] = np.gradient(s["torso_yaw"], 1 / fps)
    hands = smooth(sk.mid_wrist, fps)
    s["hand_speed"] = speed(hands, fps)                          # m/s
    s["lead_arm_vel"] = np.gradient(smooth(yaw_deg(sk.mid_shoulder, sk["lead_wrist"]), fps), 1 / fps)
    la = smooth(sk["lead_ankle"], fps)
    s["lead_ankle_speed"] = speed(la, fps)
    s["lead_ankle_z"] = la[:, 2]
    s["lead_knee_flex"] = 180 - joint_angle_deg(sk["lead_hip"], sk["lead_knee"], sk["lead_ankle"])
    s["rear_knee_flex"] = 180 - joint_angle_deg(sk["rear_hip"], sk["rear_knee"], sk["rear_ankle"])
    trunk = sk["neck"] - sk.mid_hip
    s["trunk_forward_tilt"] = np.degrees(np.arctan2(-trunk[:, 1], trunk[:, 2]))  # toward plate (-y) = +
    s["trunk_lateral_tilt"] = np.degrees(np.arctan2(-trunk[:, 0], trunk[:, 2]))  # away from pitcher (back) = +
    return s


def detect_events(sk: Skeleton, s: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    """Find stance, foot plant and contact from the skeleton alone (phone-feasible).

    * contact ~ peak hand speed (corrected by CONTACT_OFFSET_S, fitted on OBP)
    * foot plant = lead ankle settles after the stride, before contact
    * load = most closed torso (min torso yaw) before foot plant
    """
    s = s or series(sk)
    t, fps = sk.time, sk.fps
    i_hand = int(np.nanargmax(s["hand_speed"]))
    t_contact = t[i_hand] + CONTACT_OFFSET_S
    # foot plant: last frame the lead ankle is lifted above its stance height
    # (+FP_LIFT_M) before peak hand speed, then a fitted timing offset.
    lo = max(0, i_hand - int(1.2 * fps))
    z = s["lead_ankle_z"]
    z0 = np.nanmedian(z[: max(3, int(0.08 * len(z)))])
    up = np.where(z[lo:i_hand] > z0 + FP_LIFT_M)[0]
    if len(up):
        t_fp = t[min(lo + up[-1] + 1, len(t) - 1)] + FP_OFFSET_S
        fp_method = "ankle_lift"
    else:  # toe-tap / no-stride hitters: ankle settling after its fastest motion
        seg = s["lead_ankle_speed"][lo:i_hand]
        i_peak = lo + int(np.nanargmax(seg)) if len(seg) else lo
        if len(seg) and s["lead_ankle_speed"][i_peak] >= MIN_STRIDE_SPEED:
            after = np.where(s["lead_ankle_speed"][i_peak:i_hand] < max(0.25, 0.2 * s["lead_ankle_speed"][i_peak]))[0]
            t_fp = t[i_peak + int(after[0])] if len(after) else t[i_peak]
            fp_method = "ankle_settle"
        else:  # no stride at all: typical lab timing before contact
            t_fp = t_contact - TYPICAL_FP_TO_CONTACT_S
            fp_method = "typical_timing"
    t_fp = min(t_fp, t_contact - 1.0 / fps)
    i_fp = idx(sk, t_fp)
    i_load = int(np.nanargmin(s["torso_yaw"][: max(1, i_fp)])) if i_fp > 1 else 0
    return {"load": float(t[i_load]), "foot_plant": float(t_fp), "contact": float(t_contact),
            "peak_hand": float(t[i_hand]), "fp_method": fp_method}


# Fitted on the OBP hitting set by scripts/validate_events.py --fit (see docs/HANDOFF.md).
CONTACT_OFFSET_S = 0.065   # contact happens ~65 ms after peak hand speed
FP_LIFT_M = 0.03           # lead ankle "in the air" threshold above stance height
FP_OFFSET_S = -0.028
MIN_STRIDE_SPEED = 0.2          # m/s; below this the lead foot never really moved
TYPICAL_FP_TO_CONTACT_S = 0.19  # median foot plant -> contact in the OBP set       # ankle-lift rule lands slightly late vs. the 10 N force-plate time


def peak_offset(y: np.ndarray, i: int) -> float:
    """Sub-frame position of a peak at sample i: vertex of the parabola through
    samples i-1, i, i+1, in frames (-0.5..0.5). Turns 33 ms steps at 30 fps
    into continuous timing."""
    if i <= 0 or i >= len(y) - 1 or not np.all(np.isfinite(y[i - 1:i + 2])):
        return 0.0
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if den < 0 else 0.0


def idx(sk: Skeleton, t: float) -> int:
    return int(np.clip(np.searchsorted(sk.time, t), 0, len(sk.time) - 1))
