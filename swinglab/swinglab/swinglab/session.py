"""Long recordings and sessions: one video of many swings -> one analysis per swing
-> trends across the session (batting-cage use case).

Swing finding works on the lifted 3D joints, so it is independent of camera
position: a swing is a burst of hand speed. Each detected swing is cut out with
enough stance before it (the body frame is defined from the first frames of a
window) and analyzed exactly like a single-swing recording.

  find_swings(t, joints)        -> list of SwingWindow (start, peak, end)
  analyze_long(folder, hitter)  -> (list of swing records, session summary)
  summarize(records)            -> trends, consistency, recurring issue, best swing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import features, kinematics as K

# Tuned on Driveline swings resampled to 30 fps: peak mid-wrist speed is 5.1–8.6 m/s
# (median 6.9); walking, waving or practice tees stay well below 3.5 m/s.
MIN_PEAK_HAND_SPEED = 3.5      # m/s
MIN_GAP_S = 2.0                # two swings closer than this are one swing
PRE_S, POST_S = 1.6, 0.8       # window around the peak hand speed
MIN_PRE_S = 0.6                # less recording than this before the peak: the stance and load are cut off
# Clip vs session is decided by CONTENT (number of swings found), not by duration.


@dataclass
class SwingWindow:
    index: int
    start: float
    peak: float
    end: float
    peak_hand_speed: float
    partial: bool = False          # the recording starts mid-swing (less than MIN_PRE_S before the peak)


def hand_speed(t: np.ndarray, joints: dict[str, np.ndarray]) -> np.ndarray:
    """Mid-wrist speed (m/s) in any fixed frame; NaN-tolerant."""
    w = 0.5 * (joints["l_wrist"] + joints["r_wrist"])
    ok = np.isfinite(w[:, 0])
    if ok.sum() < 5:
        return np.zeros(len(t))
    w = np.c_[[np.interp(t, t[ok], w[ok, a]) for a in range(3)]].T
    fps = 1.0 / float(np.median(np.diff(t)))
    return K.speed(K.smooth(w, fps), fps)


MIN_SHOULDER_TURN_DEG = 80.0   # lab swings turn the shoulders 133–260° around the hand-speed peak (677 swings); noise peaks < 55°


def shoulder_turn(t: np.ndarray, joints: dict[str, np.ndarray], i: int, before_s: float = 0.6, after_s: float = 0.4,
                  bats: str | None = None) -> float:
    """How far (deg) the shoulder line turns about the vertical around frame i, in any fixed frame
    (vertical = from the ankles to the shoulders). Shoulders are the best-measured joints, so this
    separates a real swing from noise in the hand positions. With ``bats`` ("R"/"L") only the turn
    toward the pitcher counts (the largest opening rotation in the window), so a bat waggle, the
    return to stance or a flip of the shoulder line doesn't pass for a swing; without it, the range."""
    pre = next((pr for pr in (("l_", "r_"), ("lead_", "rear_")) if f"{pr[0]}shoulder" in joints and f"{pr[0]}ankle" in joints), None)
    if pre is None:
        return 360.0                                   # can't tell: don't veto
    L, R = pre
    sh = joints[f"{L}shoulder"] - joints[f"{R}shoulder"]
    up = np.nanmean(0.5 * (joints[f"{L}shoulder"] + joints[f"{R}shoulder"]) - 0.5 * (joints[f"{L}ankle"] + joints[f"{R}ankle"]), 0)
    if not np.all(np.isfinite(up)):
        return 360.0                                   # can't tell: don't veto
    up = up / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    m = (t >= t[i] - before_s) & (t <= t[i] + after_s) & np.isfinite(sh[:, 0])
    if m.sum() < 3:
        return 0.0
    yaw = np.degrees(np.unwrap(np.arctan2(sh[m] @ b, sh[m] @ a)))   # counterclockwise seen from above = +
    if bats is None:
        return float(np.ptp(yaw))
    if L == "lead_":
        yaw = -yaw if bats.upper() == "L" else yaw    # lead/rear named: lead - rear opens like a righty's l - r
    elif bats.upper() == "L":
        yaw = -yaw                                     # a righty's l - r line opens counterclockwise, a lefty's clockwise
    return float(np.max(yaw - np.minimum.accumulate(yaw)))


def find_swings(t: np.ndarray, joints: dict[str, np.ndarray], *, min_peak: float = MIN_PEAK_HAND_SPEED,
                min_gap: float = MIN_GAP_S, min_turn: float = MIN_SHOULDER_TURN_DEG,
                bats: str | None = None) -> list[SwingWindow]:
    """Swings = hand-speed peaks above ``min_peak`` (m/s) where the shoulders also turn at least
    ``min_turn`` degrees (toward the pitcher, given ``bats``): hand positions rebuilt without depth
    can be noisy enough to make false speed peaks (night test, 2026-10-10), but the torso doesn't
    turn for noise. A swing with less than MIN_PRE_S of recording before its peak is ``partial``."""
    v = hand_speed(t, joints)
    cand = [i for i in range(1, len(v) - 1) if v[i] >= min_peak and v[i] >= v[i - 1] and v[i] >= v[i + 1]
            and shoulder_turn(t, joints, i, bats=bats) >= min_turn]
    cand.sort(key=lambda i: -v[i])           # strongest first, then suppress neighbors
    chosen: list[int] = []
    for i in cand:
        if all(abs(t[i] - t[j]) >= min_gap for j in chosen):
            chosen.append(i)
    out = []
    for k, i in enumerate(sorted(chosen)):
        out.append(SwingWindow(k + 1, float(max(t[0], t[i] - PRE_S)), float(t[i]),
                               float(min(t[-1], t[i] + POST_S)), round(float(v[i]), 2),
                               partial=bool(t[i] - t[0] < MIN_PRE_S)))
    return out


def cut(t: np.ndarray, joints: dict[str, np.ndarray], w: SwingWindow):
    m = (t >= w.start) & (t <= w.end)
    return t[m], {n: a[m] for n, a in joints.items()}


def analyze_long(folder: str | Path, hitter: dict, *, lifted=None, windows=None, **kw) -> tuple[list[dict], dict]:
    """``lifted`` = (t, joints, quality) from pipeline.lift_recording to avoid running pose twice."""
    from . import pipeline
    t, joints, q = lifted or pipeline.lift_recording(folder, height_m=float(hitter["height_m"]), **kw)
    wins = windows if windows is not None else find_swings(t, joints, bats=hitter.get("bats"))
    recs = []
    for w in wins:
        ts, js = cut(t, joints, w)
        try:
            r = pipeline.analyze_joints(ts, js, hitter, f"{Path(folder).name}-{w.index:02d}", q)
            r["window"] = asdict(w)
            recs.append(r)
        except Exception as e:  # keep the session going
            recs.append({"id": f"{Path(folder).name}-{w.index:02d}", "window": asdict(w), "error": repr(e)})
    q_summary = {k: v for k, v in q.items() if not k.startswith("_")}   # not the whole recording's 2D keypoints
    return recs, summarize([r for r in recs if "analysis" in r], source=str(folder), quality=q_summary)


KEY_MECHANICS = ["separation_fp", "pelvis_vel_peak", "torso_vel_peak", "lead_knee_extension",
                 "hands_forward_contact", "head_drift_fp_contact", "stride_length"]


def _slope_per_10(y: np.ndarray) -> float | None:
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return None
    x = np.arange(len(y))[ok]
    return round(float(np.polyfit(x, y[ok], 1)[0] * 10), 2)


def summarize(recs: list[dict], **meta) -> dict:
    """Trends and consistency over a session's swings (in order)."""
    if not recs:
        return {"n_swings": 0, **meta}
    bs = np.array([r["analysis"]["predicted"]["bat_speed"] for r in recs], float)
    ev = np.array([r["analysis"]["predicted"]["exit_velo"] for r in recs], float)
    ov = np.array([r["analysis"]["scores"]["overall"] for r in recs], float)
    meas = np.array([(r.get("measured") or {}).get("exit_velo_mph") or np.nan for r in recs], float)
    third = max(1, len(recs) // 3)
    focus = [((r["analysis"].get("focus") or {}).get("feature")) for r in recs]
    counts: dict[str, int] = {}
    for f in focus:
        if f:
            counts[f] = counts.get(f, 0) + 1
    recurring = max(counts, key=counts.get) if counts else None
    consistency = {}
    for k in KEY_MECHANICS:
        vals = np.array([r["features"].get(k, np.nan) for r in recs], float)
        if np.isfinite(vals).sum() >= 2:
            consistency[k] = {"label": features.FEATURES[k][1], "mean": round(float(np.nanmean(vals)), 2),
                              "sd": round(float(np.nanstd(vals)), 2), "unit": features.FEATURES[k][2]}
    best = int(np.nanargmax(ov))
    modes = [r.get("mode") or r["analysis"].get("mode") or "dry" for r in recs]
    form = np.array([r["analysis"]["scores"].get("form", r["analysis"]["scores"]["overall"]) for r in recs], float)
    cont = np.array([np.nan if r["analysis"]["scores"].get("contact") is None else r["analysis"]["scores"]["contact"]
                     for r in recs], float)
    bat_m = np.array([(r.get("measured") or {}).get("bat_speed_mph") or np.nan for r in recs], float)
    n_ball = sum(m in ("hit", "miss") for m in modes)
    stat = lambda a: None if not np.isfinite(a).any() else {"mean": round(float(np.nanmean(a)), 1),  # noqa: E731
                                                             "best": round(float(np.nanmax(a)), 1),
                                                             "worst": round(float(np.nanmin(a)), 1)}
    return {
        **meta, "n_swings": len(recs),
        # counted here, not by Gemini: Gemini only writes about these numbers
        "outcomes": {"hits": modes.count("hit"), "misses": modes.count("miss"), "dry_swings": modes.count("dry"),
                     "lab_swings": modes.count("lab"),
                     "hit_rate": None if not n_ball else round(modes.count("hit") / n_ball, 2)},
        "form_score": {**(stat(form) or {}), "first_third": round(float(form[:third].mean()), 1),
                       "last_third": round(float(form[-third:].mean()), 1), "trend_per_10_swings": _slope_per_10(form)},
        "contact_score": stat(cont),
        "measured_bat_speed": stat(bat_m),
        "per_swing": [{"id": r["id"], "t": (r.get("window") or {}).get("peak"), "outcome": md,
                       "bat_speed": round(float(b), 1), "exit_velo": round(float(e), 1), "overall": round(float(o), 1),
                       "form": round(float(fo), 1), "contact": None if not np.isfinite(co) else round(float(co), 1),
                       "focus": f, "measured_exit_velo": None if not np.isfinite(m) else float(m),
                       "measured_bat_speed": None if not np.isfinite(bm) else float(bm)}
                      for r, b, e, o, f, m, md, fo, co, bm in zip(recs, bs, ev, ov, focus, meas, modes, form, cont, bat_m)],
        "bat_speed": {"mean": round(float(bs.mean()), 1), "first_third": round(float(bs[:third].mean()), 1),
                      "last_third": round(float(bs[-third:].mean()), 1), "trend_per_10_swings": _slope_per_10(bs)},
        "overall_score": {"mean": round(float(ov.mean()), 1), "trend_per_10_swings": _slope_per_10(ov)},
        "measured_exit_velo": None if not np.isfinite(meas).any() else
            {"mean": round(float(np.nanmean(meas)), 1), "max": round(float(np.nanmax(meas)), 1)},
        "recurring_issue": recurring and {"feature": recurring, "label": features.FEATURES[recurring][1],
                                          "swings": counts[recurring]},
        "best_swing": recs[best]["id"], "consistency": consistency,
    }


# ---------------------------------------------------------------- slow-motion clips <-> swings

def match_clips(swing_t: list[float | None], clip_t: list[float | None] | None, *, tol_s: float = 1.5,
                max_offset_s: float | None = 15.0) -> dict:
    """Match per-swing slow-motion clips to the swings found in one long LiDAR recording.

    ``swing_t``: when each swing's ball left the tee (contact), on the LiDAR phone's clock;
    ``clip_t``: when each clip's ball left the tee, on the slow-motion phone's clock (None if unknown).
    Video creation times have 1 s resolution (MP4 headers and iPhone metadata store whole seconds),
    hence the 1.5 s tolerance; swings in a session are 10+ s apart.
    With times for at least two clips and swings, the clock offset is the one that lines up the
    most swings with clips within ``tol_s`` (ties: smaller error, then smaller offset; the phones'
    network clocks rarely differ by more than ``max_offset_s``), so a missing clip or an extra
    practice swing doesn't shift every pair. Otherwise clips are matched by order.
    Returns {"pairs": [(swing index, clip index)], "method", "offset_s", "unmatched_swings",
    "unmatched_clips", "notes"}."""
    ns, nc = len(swing_t), len(clip_t or [])
    S = [(i, t) for i, t in enumerate(swing_t) if t is not None]
    C = [(j, t) for j, t in enumerate(clip_t or []) if t is not None]
    best = None
    if len(S) >= 2 and len(C) >= 2 or (len(S) == 1 and len(C) == 1 and ns == nc == 1):
        for _, ts in S:
            for _, tc in C:
                d = tc - ts
                if max_offset_s is not None and abs(d) > max_offset_s:
                    continue
                used, pairs, err = set(), [], 0.0
                for i, a in S:
                    cand = [(abs(a + d - b), j) for j, b in C if j not in used and abs(a + d - b) <= tol_s]
                    if cand:
                        e, j = min(cand)
                        used.add(j)
                        pairs.append((i, j))
                        err += e
                key = (-len(pairs), err / max(len(pairs), 1), abs(d))
                if best is None or key < best[0]:
                    best = (key, pairs, d)
    if best and len(best[1]) >= max(1, min(len(S), len(C)) // 2):
        pairs, method, offset = sorted(best[1]), "time", round(best[2], 3)
    else:
        pairs, method, offset = [(k, k) for k in range(min(ns, nc))], "order", None
    us = [i for i in range(ns) if i not in {p[0] for p in pairs}]
    uc = [j for j in range(nc) if j not in {p[1] for p in pairs}]
    notes = []
    if ns != nc:
        notes.append(f"{ns} swings found but {nc} slow-motion clips" + (
            " (matched by when the ball left the tee)" if method == "time" else
            " — matched by order, so pairs after a missing or extra clip are probably wrong; "
            "name clips in log.csv or keep both phones' clocks set automatically"))
    if us:
        notes.append("no slow-motion clip for swing " + ", ".join(str(i + 1) for i in us))
    if uc:
        notes.append("slow-motion clip " + ", ".join(str(j + 1) for j in uc) + " matched no swing")
    return {"pairs": pairs, "method": method, "offset_s": offset, "unmatched_swings": us,
            "unmatched_clips": uc, "notes": notes}
