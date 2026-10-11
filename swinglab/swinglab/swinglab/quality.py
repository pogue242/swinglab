"""Recording quality checks that need no ground truth.

  bone_lengths(sk)                 a person's bones don't change length: thigh, shin, upper arm and
                                   forearm should vary < 10% across frames. Larger variation means
                                   depth sampling hit the background, another body part or the bat.
  framing(kps, size, n_frames)     how often all 13 joints were seen, and the feet and hands were inside
                                   the picture (MediaPipe guesses joints outside the frame).

Both return plain dicts that go into rec["quality"]; problems also become
plain-language warnings in rec["warnings"].
"""

from __future__ import annotations

import numpy as np

BONES = {
    "lead_thigh": ("lead_hip", "lead_knee"), "rear_thigh": ("rear_hip", "rear_knee"),
    "lead_shin": ("lead_knee", "lead_ankle"), "rear_shin": ("rear_knee", "rear_ankle"),
    "lead_upper_arm": ("lead_shoulder", "lead_elbow"), "rear_upper_arm": ("rear_shoulder", "rear_elbow"),
    "lead_forearm": ("lead_elbow", "lead_wrist"), "rear_forearm": ("rear_elbow", "rear_wrist"),
}
MAX_BONE_VARIATION = 0.10      # standard deviation ÷ median length
MIN_FRAMING = 0.90             # fraction of frames
EDGE_MARGIN = 0.01             # a joint within 1% of the image border counts as cut off


def bone_lengths(sk) -> dict:
    """Per bone: median length (m) and variation (sd ÷ median) over the frames where both ends exist."""
    out = {}
    for bone, (a, b) in BONES.items():
        L = np.linalg.norm(sk[a] - sk[b], axis=1)
        L = L[np.isfinite(L)]
        if len(L) < 3:
            continue
        med = float(np.median(L))
        out[bone] = {"median_m": round(med, 3), "variation": round(float(np.std(L) / med), 3) if med > 0 else None}
    scored = {k: v["variation"] for k, v in out.items() if v["variation"] is not None}
    worst = max(scored, key=scored.get) if scored else None
    return {"bones": out, "worst_bone": worst, "worst_variation": scored.get(worst),
            "ok": bool(scored) and max(scored.values()) < MAX_BONE_VARIATION}


def bone_warning(b: dict) -> str | None:
    if b.get("worst_bone") and not b["ok"]:
        return (f"{b['worst_bone'].replace('_', ' ')} length varies {b['worst_variation']:.0%} across frames "
                f"(should be < {MAX_BONE_VARIATION:.0%}): 3D joints are noisy; check depth coverage and framing")
    return None


def framing(kps: dict[int, dict[str, tuple[float, float, float]]], size: tuple[int, int], n_frames: int,
            *, min_vis: float = 0.5) -> dict:
    """Fractions of ALL frames: person found, all 13 joints visible (12 MediaPipe joints ≥ min_vis;
    the neck is derived from the shoulders), both ankles in frame, both wrists in frame."""
    w, h = size
    mx, my = EDGE_MARGIN * w, EDGE_MARGIN * h

    def inside(p):
        return mx <= p[0] <= w - mx and my <= p[1] <= h - my

    n = max(n_frames, 1)
    person = all_vis = feet = hands = 0
    for f in kps.values():
        person += 1
        all_vis += all(v[2] >= min_vis for v in f.values()) and len(f) >= 12
        feet += all(k in f and inside(f[k]) for k in ("l_ankle", "r_ankle"))
        hands += all(k in f and inside(f[k]) for k in ("l_wrist", "r_wrist"))
    r = lambda x: round(x / n, 3)  # noqa: E731
    out = {"person_found": r(person), "all_joints_visible": r(all_vis), "feet_in_frame": r(feet),
           "hands_in_frame": r(hands), "min_vis": min_vis}
    out["ok"] = all(out[k] >= MIN_FRAMING for k in ("person_found", "feet_in_frame", "hands_in_frame"))
    return out


def framing_warnings(fr: dict) -> list[str]:
    w = []
    if fr.get("person_found", 1) < MIN_FRAMING:
        w.append(f"hitter found in only {fr['person_found']:.0%} of frames")
    if fr.get("feet_in_frame", 1) < MIN_FRAMING:
        w.append(f"feet in the picture in only {fr['feet_in_frame']:.0%} of frames: step the tripod back or tilt down")
    if fr.get("hands_in_frame", 1) < MIN_FRAMING:
        w.append(f"hands in the picture in only {fr['hands_in_frame']:.0%} of frames: leave room above the load and at full extension")
    if fr.get("all_joints_visible", 1) < 0.5:
        w.append(f"all 13 joints clearly visible in only {fr['all_joints_visible']:.0%} of frames "
                 "(light, motion blur or the body hiding a joint)")
    return w


MAX_FALLBACK = 0.30   # warn when a joint's 3D position comes from bone length in more frames than this


def fallback_warnings(placed: dict[str, float], bats: str) -> list[str]:
    """``placed`` = {"l_wrist": 0.7, ...} from capture.lift_keypoints."""
    from .skeleton import to_lead_rear
    names = to_lead_rear({k: k for k in placed}, bats)   # {"lead_wrist": "l_wrist", ...}
    return [f"{lr.replace('_', ' ')} placed from bone length in {placed[side]:.0%} of frames (no usable depth there): "
            "its position is an estimate" for lr, side in names.items() if lr != "neck" and placed[side] > MAX_FALLBACK]
