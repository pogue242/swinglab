"""Ball and bat speed from a high-frame-rate side camera (goal 2).

Pipeline for one tee swing filmed at ~240 fps from the side (camera axis
roughly perpendicular to the swing / ball-flight plane):

1. ``detect_ball``   find the ball in each frame (bright, round, small)
2. ``track``         keep the detections that form one smooth path
3. ``scale``         meters per pixel from a known size (ball diameter, bat
                     length or tape marks) — supplied by the caller
4. ``launch``        exit speed, launch angle from the first frames after
                     contact; contact frame = ball starts moving off the tee
5. ``trajectory``    ballistic projection (with optional drag) for drawing
                     the flight line on the video

Limits (documented in HANDOFF): a single side camera measures speed in the
image plane only, so motion toward/away from the camera is missed (speed
reads LOW). Wiffle/foam balls leave the bat far slower than baseballs, so the
exit speed is relative only; bat speed at contact is the comparable number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass
class Launch:
    contact_frame: int
    exit_speed_mps: float
    exit_speed_mph: float
    launch_angle_deg: float      # + = up
    direction: int               # +1 ball moves to image right, -1 left
    n_points: int
    fit_rmse_px: float
    contact_t: float | None = None   # seconds from the clip start, sub-frame (launch_from_flight)


BALL_RADIUS_M = 0.0737 / 2
MAX_BALLISTIC_RMSE_PX = 4.0   # a real ball's first frames off the tee fit a parabola this well


def detect_ball(frame_bgr: np.ndarray, min_r: float = 3, max_r: float = 40, bright: int = 200,
                expected_r: float | None = None):
    """Return (x, y, r) of the most ball-like bright blob, or None.

    Simple and dependency-light (OpenCV). Works for a white/yellow ball against
    a darker net/background; swap for a trained detector if the scene is busy.
    Bright tape on the bat is the main look-alike: bands seen side-on are rectangles
    (rejected by roundness), and among round blobs the one closest to ``expected_r``
    (the ball's radius in px when the scale is known) wins, else the largest.
    """
    if expected_r:
        min_r, max_r = 0.65 * expected_r, 1.5 * expected_r
    import cv2
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), bright, 255, cv2.THRESH_BINARY)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        r = 0.25 * (w + h)
        if not (min_r <= r <= max_r):
            continue
        roundness = area / (np.pi * r * r + 1e-9)          # ~1 for a filled circle
        aspect = min(w, h) / max(w, h)
        size = 1 - min(1.0, abs(r - expected_r) / expected_r) if expected_r else np.sqrt(r)
        score = roundness * aspect * size
        if score > best_score and 0.6 < roundness < 1.2:   # a filled circle is ~1.0; a tape band (rectangle) ~1.25
            best, best_score = (float(cents[i][0]), float(cents[i][1]), float(r)), score
    if best is not None:
        best = (best[0], best[1], _refine_radius(gray, best))
    return best


def _refine_radius(gray: np.ndarray, det: tuple[float, float, float]) -> float:
    """Radius from the area above the half-way level between local background and
    ball brightness. A fixed high threshold cuts off the soft edge and makes the
    ball look ~20% small, which inflates speeds when the ball is the scale."""
    x, y, r = det
    h = int(max(6, 3 * r))
    x0, x1, y0, y1 = max(0, int(x) - h), min(gray.shape[1], int(x) + h + 1), max(0, int(y) - h), min(gray.shape[0], int(y) + h + 1)
    win = gray[y0:y1, x0:x1].astype(float)
    border = np.r_[win[0], win[-1], win[:, 0], win[:, -1]]
    bg, peak = float(np.median(border)), float(np.percentile(win, 99))
    if peak - bg < 20:
        return r
    area = float(np.sum(win > 0.5 * (bg + peak)))
    return float(np.sqrt(area / np.pi))


def track(points: list[tuple[float, float] | None], max_jump_px: float = 80.0) -> np.ndarray:
    """(N, 2) array with NaN where missing; drops detections that jump too far."""
    out = np.full((len(points), 2), np.nan)
    last = None
    for i, p in enumerate(points):
        if p is None:
            continue
        if last is not None and np.hypot(p[0] - last[0], p[1] - last[1]) > max_jump_px * (i - last[2]):
            continue
        out[i] = p[:2]
        last = (p[0], p[1], i)
    return out


def contact_frame(xy: np.ndarray, still_px: float = 1.5) -> int | None:
    """First frame the ball leaves its resting position on the tee."""
    ok = np.where(np.isfinite(xy[:, 0]))[0]
    if len(ok) < 3:
        return None
    rest = np.nanmedian(xy[ok[: max(3, len(ok) // 5)]], axis=0)
    moved = np.where(np.hypot(*(xy[ok] - rest).T) > still_px)[0]
    return int(ok[moved[0]]) - 1 if len(moved) else None


def launch(xy: np.ndarray, fps: float, m_per_px: float, *, n_fit: int = 8, image_y_down: bool = True) -> Launch | None:
    """Exit speed and launch angle from the first ``n_fit`` tracked points after contact.

    Fits x(t) linear and y(t) quadratic with known gravity (in pixels), so a
    few noisy points give a stable velocity.
    """
    c = contact_frame(xy)
    if c is None:
        return None
    idx = [i for i in range(c + 1, len(xy)) if np.isfinite(xy[i, 0])][:n_fit]
    if len(idx) < 3:
        return None
    t = (np.array(idx) - c) / fps
    x, y = xy[idx, 0] * m_per_px, xy[idx, 1] * m_per_px
    if image_y_down:
        y = -y
    vx, x0 = np.polyfit(t, x, 1)
    # y = y0 + vy t - g/2 t^2  ->  fit y + g/2 t^2 linearly
    vy, y0 = np.polyfit(t, y + 0.5 * G * t**2, 1)
    pred = np.c_[x0 + vx * t, y0 + vy * t - 0.5 * G * t**2]
    rmse = float(np.sqrt(np.mean(np.sum((pred - np.c_[x, y]) ** 2, axis=1))) / m_per_px)
    speed = float(np.hypot(vx, vy))
    return Launch(c, speed, speed * 2.2369363, float(np.degrees(np.arctan2(vy, abs(vx)))),
                  int(np.sign(vx) or 1), len(idx), rmse)


def trajectory(l: Launch, start_xy_px: tuple[float, float], m_per_px: float, *, t_max: float = 3.0,
               ground_y_px: float | None = None, drag_k: float = 0.0, dt: float = 1 / 240,
               image_y_down: bool = True) -> np.ndarray:
    """Projected flight path in pixels for drawing. ``drag_k`` = 0.5·ρ·Cd·A/m (1/m); 0 = vacuum."""
    th = np.radians(l.launch_angle_deg)
    v = np.array([l.direction * l.exit_speed_mps * np.cos(th), l.exit_speed_mps * np.sin(th)])
    p = np.zeros(2)
    pts = [p.copy()]
    for _ in range(int(t_max / dt)):
        a = np.array([0.0, -G]) - drag_k * np.linalg.norm(v) * v
        v = v + a * dt
        p = p + v * dt
        pts.append(p.copy())
        if ground_y_px is not None:
            ypx = start_xy_px[1] - p[1] / m_per_px if image_y_down else start_xy_px[1] + p[1] / m_per_px
            if (image_y_down and ypx > ground_y_px) or (not image_y_down and ypx < ground_y_px):
                break
    pts = np.array(pts)
    sign = -1 if image_y_down else 1
    return np.c_[start_xy_px[0] + pts[:, 0] / m_per_px, start_xy_px[1] + sign * pts[:, 1] / m_per_px]


def tip_speed(tip_xy: np.ndarray, fps: float, m_per_px: float, around: int, half_window: int = 3) -> float:
    """Bat-tip (or any point) speed in mph, averaged over frames around ``around`` (e.g. contact)."""
    lo, hi = max(1, around - half_window), min(len(tip_xy) - 1, around + half_window)
    d = np.hypot(*np.diff(tip_xy[lo - 1:hi + 1], axis=0).T) * m_per_px * fps
    return float(np.nanmedian(d) * 2.2369363)


def scale_from_ball(radius_px: float, ball_diameter_m: float = 0.0737) -> float:
    """Meters per pixel from the ball's apparent radius (baseball ≈ 7.37 cm)."""
    return ball_diameter_m / (2 * radius_px)


# ---------------------------------------------------------------- ball: rest on the tee -> flight
#
# Frame-by-frame "best round blob" fails on real clips: the bat covers the ball at contact, a fast
# ball is a motion-blur streak (a capsule as WIDE as the ball, but longer), and the tape on the bat
# is also small and bright. So: find where the ball rests (the round blob that stays put longest),
# then the blobs that leave that spot as wide as the ball and moving like a projectile.

def ball_candidates(frame_bgr: np.ndarray, bright: int = 200, min_area: int = 8):
    """Bright blobs: [(x, y, width_px, length_px, roundness)]; width and length from second moments.
    A disc of radius r has width = length = 2r; a motion-blurred ball is as wide as the ball and longer."""
    import cv2
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    n, lab, stats, cents = cv2.connectedComponentsWithStats((gray > bright).astype(np.uint8))
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > 40000:
            continue
        ys, xs = np.nonzero(lab[stats[i, 1]:stats[i, 1] + stats[i, 3], stats[i, 0]:stats[i, 0] + stats[i, 2]] == i)
        ev = np.linalg.eigvalsh(np.cov(np.c_[xs, ys].T) + 1e-6 * np.eye(2))
        width, length = 4 * np.sqrt(ev[0]), 4 * np.sqrt(ev[1])
        r = 0.25 * (stats[i, 2] + stats[i, 3])
        roundness = float(area / (np.pi * r * r))
        if 0.6 < roundness < 1.2 and length < 1.4 * width:   # round: size at half brightness (a fixed
            width = length = 2 * _refine_radius(gray, (float(cents[i][0]), float(cents[i][1]), r))  # cut reads ~20% small)
        out.append((float(cents[i][0]), float(cents[i][1]), float(width), float(length), roundness))
    return out


def track_ball(cands: list[list[tuple]], *, expected_r: float | None = None, still_px: float = 2.0,
               max_flight: int = 40, max_step_px: float | None = None) -> dict | None:
    """Ball at rest on the tee, then in flight. ``cands`` = ball_candidates per frame.
    Returns {"rest": (x, y, r), "last_rest_frame", "flight": {frame: (x, y)}, "xy": (N,2)} or None.

    Flight candidates: as wide as 0.5–1.5 ball diameters (a blurred ball's bright core is narrower
    than the ball; tape bands are much thinner), not a static object (anything at the same spot in
    a quarter of the frames is scenery), within ``max_step_px`` per frame of the previous point
    (default: a 120 mph ball at 240 fps when the scale is known)."""
    n = len(cands)
    round_ = [[c for c in fc if 0.6 < c[4] < 1.2 and c[3] < 1.4 * c[2]
               and (expected_r is None or 0.65 * expected_r <= c[2] / 2 <= 1.5 * expected_r)] for fc in cands]
    pts = [(i, c) for i, fc in enumerate(round_) for c in fc]
    if not pts:
        return None
    best, best_n = None, 0                            # the rest position: seen in the most frames
    for _, c in pts[:: max(1, len(pts) // 400)]:
        frames_here = {i for i, d in pts if np.hypot(d[0] - c[0], d[1] - c[1]) <= still_px}
        if len(frames_here) > best_n:
            best, best_n = c, len(frames_here)
    near = [(i, d) for i, d in pts if np.hypot(d[0] - best[0], d[1] - best[1]) <= still_px]
    rx, ry = float(np.median([d[0] for _, d in near])), float(np.median([d[1] for _, d in near]))
    rr = float(np.median([d[2] for _, d in near])) / 2
    last_rest = max(i for i, _ in near)
    xy = np.full((n, 2), np.nan)
    for i, _ in near:
        xy[i] = (rx, ry)
    spots: dict[tuple[int, int], set] = {}           # scenery: bright things that never move
    for i, fc in enumerate(cands):
        for c in fc:
            spots.setdefault((round(c[0] / still_px), round(c[1] / still_px)), set()).add(i)
    static = {k for k, v in spots.items() if len(v) > 0.25 * n}
    is_static = lambda c: any((round(c[0] / still_px) + a, round(c[1] / still_px) + b) in static  # noqa: E731
                              for a in (-1, 0, 1) for b in (-1, 0, 1))
    D = 2 * rr
    step = max_step_px or (54.0 / 240 / (BALL_RADIUS_M / rr))   # 120 mph at 240 fps, scale from the ball's size
    flight, prev = {}, [(last_rest, np.array([rx, ry]))]
    for i in range(last_rest + 1, min(n, last_rest + 1 + max_flight)):
        ok = [np.array(c[:2]) for c in cands[i] if 0.5 * D <= c[2] <= 1.5 * D and np.hypot(c[0] - rx, c[1] - ry) > rr
              and not is_static(c)]
        if not ok:
            continue
        if len(prev) >= 2:                            # constant-velocity prediction from the last two points
            (i1, p1), (i2, p2) = prev[-1], prev[-2]
            pred = p1 + (p1 - p2) * (i - i1) / (i1 - i2)
            gate = 0.5 * np.linalg.norm(p1 - p2) / (i1 - i2) * (i - i1) + 3 * rr
        else:                                         # first point off the tee: nearest to the rest spot
            pred, gate = np.array([rx, ry]), step * (i - last_rest) + D
        p = min(ok, key=lambda q: np.linalg.norm(q - pred))
        if np.linalg.norm(p - pred) <= gate:
            flight[i] = p
            prev.append((i, p))
            xy[i] = p
    return {"rest": (rx, ry, rr), "last_rest_frame": int(last_rest), "flight": flight, "xy": xy}


def link_tracks(cands: list[list[tuple]], *, expected_r: float | None = None, max_gap: int = 4,
                min_len: int = 4, still_px: float = 2.0) -> list[dict]:
    """Moving bright round things linked across frames (constant-velocity prediction), scenery that
    never moves removed. Each track: {"pts": {frame: np.array([x, y])}, "w": [blob widths]}."""
    n = len(cands)
    spots: dict[tuple[int, int], set] = {}
    for i, fc in enumerate(cands):
        for c in fc:
            spots.setdefault((round(c[0] / still_px), round(c[1] / still_px)), set()).add(i)
    static = {k for k, v in spots.items() if len(v) > 0.25 * n}

    def usable(c):
        if any((round(c[0] / still_px) + a, round(c[1] / still_px) + b) in static for a in (-1, 0, 1) for b in (-1, 0, 1)):
            return False
        if expected_r:
            return 0.5 * 2 * expected_r <= c[2] <= 1.6 * 2 * expected_r and c[3] <= 8 * c[2]
        return c[2] >= 3 and c[3] <= 8 * c[2]

    tracks: list[dict] = []
    for i, fc in enumerate(cands):
        pts = [(np.array(c[:2]), c[2], c[4]) for c in fc if usable(c)]
        taken = set()
        for tr in sorted(tracks, key=lambda t: -len(t["pts"])):
            last = max(tr["pts"])
            if i - last > max_gap or not pts:
                continue
            p1 = tr["pts"][last]
            prev = sorted(tr["pts"])[-2] if len(tr["pts"]) >= 2 else None
            v = (p1 - tr["pts"][prev]) / (last - prev) if prev is not None else np.zeros(2)
            pred = p1 + v * (i - last)
            w = float(np.median(tr["w"]))
            gate = 3 * w + 0.6 * float(np.linalg.norm(v)) * (i - last) + (6 * w if prev is None else 0)
            best = min(((float(np.linalg.norm(p - pred)), k) for k, (p, _, _) in enumerate(pts) if k not in taken), default=None)
            if best and best[0] <= gate:
                taken.add(best[1])
                tr["pts"][i] = pts[best[1]][0]
                tr["w"].append(pts[best[1]][1])
                tr["rd"].append(pts[best[1]][2])
        for k, (p, w, rd) in enumerate(pts):
            if k not in taken:
                tracks.append({"pts": {i: p}, "w": [w], "rd": [rd]})
    return [t for t in tracks if len(t["pts"]) >= min_len]


def _speed_change(tr: dict) -> tuple[int, float] | None:
    """Frame where a track's motion breaks the way a bat hit breaks it: much faster afterwards,
    in a new direction. (A ball's free flight changes velocity smoothly.)"""
    fr = sorted(tr["pts"])
    if len(fr) < 5:
        return None
    best = None
    for k in range(2, len(fr) - 2):
        a, b, c = fr[k - 2], fr[k], fr[k + 2]
        v_before = (tr["pts"][b] - tr["pts"][a]) / (b - a)
        v_after = (tr["pts"][c] - tr["pts"][b]) / (c - b)
        s0, s1 = float(np.linalg.norm(v_before)), float(np.linalg.norm(v_after))
        if s1 < 1.8 * s0 + 1.0:
            continue
        cos = float(v_before @ v_after) / (s0 * s1 + 1e-9)
        if s0 > 1.0 and cos > np.cos(np.radians(25)):
            continue
        if best is None or s1 / (s0 + 1.0) > best[1]:
            best = (b, s1 / (s0 + 1.0))
    return best


def ball_like(pts: dict, widths: list, rounds: list, *, min_travel: float = 3.0) -> bool:
    """Does this stretch of a track look and move like a ball? Round, steady in size (a lit window
    partly covered by the hitter changes size and shape), and it really travels (``min_travel``
    ball widths), unlike scenery that only jitters with the camera. Found on the night test: a
    lit window behind the hitter passed as a ball before this check."""
    if len(pts) < 3:
        return False
    w = float(np.median(widths))
    P = np.array(list(pts.values()))
    travel = float(np.linalg.norm(P.max(0) - P.min(0)))
    steady = float(np.std(widths)) / max(w, 1e-6) < 0.35
    rd = float(np.median(rounds)) if rounds else 1.0
    return travel >= min_travel * w and steady and 0.55 < rd < 1.35


def falls(pts: dict, widths: list, fps: float | None) -> bool:
    """A ball in the air falls: fit y(t) with a parabola; its downward acceleration, in m/s² with
    the scale from the ball's own width (7.4 cm), must be roughly g (3–20 m/s², allowing for
    perspective and noise). Scenery drifting with a moving phone does not (night test: −0.25 m/s²).
    Without fps (unknown frame rate) this check passes."""
    if not fps or len(pts) < 6:
        return True
    fr = np.array(sorted(pts), float)
    y = np.array([pts[int(f)][1] for f in fr])
    a = np.polyfit(fr, y, 2)[0]                                   # px / frame²; image y points down
    acc = 2 * a * fps * fps * (2 * BALL_RADIUS_M / max(float(np.median(widths)), 1e-6))
    return 3.0 <= acc <= 20.0


def ball_events(cands: list[list[tuple]], *, expected_r: float | None = None, fps: float | None = None) -> dict:
    """What the ball did, for a tee or a toss ("we throw it up and hit it, sometimes missing"):
      hit   a ball resting on a tee leaves it, or a moving ball's path breaks (much faster, new
            direction) or ends and a much faster ball starts from the same spot within a few frames
      miss  a ball is followed for a while but its path never breaks (a swing and miss)
      none  no ball (a dry swing)
    Returns {"kind", "how" (tee | toss), "rest" (x, y, r: the ball where it was hit), "last_rest_frame"
    (the contact frame), "flight" {frame: xy} after contact, "xy" (N, 2) all ball points, "reason"}.
    ``rest``/``last_rest_frame``/``flight`` are what launch_from_flight needs."""
    n = len(cands)
    tb = track_ball(cands, expected_r=expected_r)
    if tb and len(tb["flight"]) >= 3:
        rx, ry, _ = tb["rest"]
        d = [float(np.hypot(p[0] - rx, p[1] - ry)) for _, p in sorted(tb["flight"].items())]
        rest_frames = int(np.isfinite(tb["xy"][: tb["last_rest_frame"] + 1, 0]).sum())
        fr = sorted(tb["flight"])[:4]
        steps = [float(np.linalg.norm(tb["flight"][b] - tb["flight"][a])) / (b - a) for a, b in zip(fr, fr[1:])]
        fast = bool(steps) and float(np.median(steps)) >= 0.4 * 2 * tb["rest"][2]   # a struck ball leaves fast; a bat starts slowly
        if all(b > a for a, b in zip(d, d[1:])) and rest_frames >= 5 and fast:
            return {**tb, "kind": "hit", "how": "tee", "reason": "the ball left the tee"}
    tracks = link_tracks(cands, expected_r=expected_r)
    xy = np.full((n, 2), np.nan)
    if not tracks:
        return {"kind": "none", "how": None, "rest": None, "last_rest_frame": None, "flight": {}, "xy": xy,
                "reason": "no ball seen"}
    tracks.sort(key=lambda t: min(t["pts"]))
    speed = lambda t, end: (lambda f: float(np.linalg.norm(t["pts"][f[-1 if end else 1]] - t["pts"][f[-2 if end else 0]]))  # noqa: E731
                            / max(1, abs(f[-1 if end else 1] - f[-2 if end else 0])))(sorted(t["pts"]))
    hit = None
    def parts(t, c):
        fr = sorted(t["pts"])
        k = fr.index(c) + 1 if c in t["pts"] else sum(f <= c for f in fr)
        return ({f: t["pts"][f] for f in fr[:k]}, t["w"][:k], t["rd"][:k]), ({f: t["pts"][f] for f in fr[k:]}, t["w"][k:], t["rd"][k:])

    def departs(after: dict, start: np.ndarray, w: float) -> bool:
        """After a hit the ball keeps getting farther from where it was hit, at least 4 ball widths."""
        d = [float(np.linalg.norm(p - start)) for _, p in sorted(after.items())]
        return len(d) >= 3 and d[-1] >= 4 * w and sum(b >= a for a, b in zip(d, d[1:])) >= 0.8 * (len(d) - 1)

    for t in sorted(tracks, key=lambda t: -len(t["pts"])):          # 1. a break inside one track
        br = _speed_change(t)
        if br:
            (bp, bw, brd), (ap, aw, ard) = parts(t, br[0])
            w = float(np.median(t["w"]))
            if ball_like(bp, bw, brd) and falls(bp, bw, fps) and departs(ap, bp[max(bp)], w):
                hit = (t, bp, ap, br[0])
                break
    if hit is None:                                                  # 2. the ball is lost at the bat, a fast one leaves
        for a in tracks:
            ea = max(a["pts"])
            for b in tracks:
                sb = min(b["pts"])
                if b is a or not 0 < sb - ea <= 10 or len(b["pts"]) < 3:
                    continue
                w = float(np.median(a["w"]))
                near = np.linalg.norm(b["pts"][sb] - a["pts"][ea]) <= 4 * w + 3 * speed(b, False) * (sb - ea)
                if (near and speed(b, False) > 1.8 * speed(a, True) + 1.0 and ball_like(a["pts"], a["w"], a["rd"])
                        and falls(a["pts"], a["w"], fps) and departs(b["pts"], a["pts"][ea], w)):
                    hit = (b, a["pts"], b["pts"], ea)
                    break
            if hit:
                break
    if hit:
        t, before, after, c = hit
        for f, p in {**before, **after}.items():
            xy[f] = p
        p0 = before[max(before)] if before else after[min(after)]
        r = float(np.median(t["w"])) / 2
        return {"kind": "hit", "how": "toss", "rest": (float(p0[0]), float(p0[1]), r), "last_rest_frame": int(c),
                "flight": {f: p for f, p in after.items()}, "xy": xy,
                "reason": "the ball's path broke: it left much faster in a new direction"}
    balls = [t for t in tracks if ball_like(t["pts"], t["w"], t["rd"], min_travel=4.0) and falls(t["pts"], t["w"], fps)]
    if not balls:
        return {"kind": "none", "how": None, "rest": None, "last_rest_frame": None, "flight": {}, "xy": xy,
                "reason": "no ball seen (bright things in view did not move like a ball)"}
    main = max(balls, key=lambda t: len(t["pts"]))
    for f, p in main["pts"].items():
        xy[f] = p
    if len(main["pts"]) >= 8:
        return {"kind": "miss", "how": "toss", "rest": None, "last_rest_frame": None, "flight": {}, "xy": xy,
                "radius": float(np.median(main["w"])) / 2,
                "reason": f"a ball was followed for {len(main['pts'])} frames but its path never broke (swing and miss)"}
    return {"kind": "none", "how": None, "rest": None, "last_rest_frame": None, "flight": {}, "xy": xy,
            "reason": "nothing followed long enough to be a ball"}


def ball_hit_in_frames(frames, *, scale: float = 0.5, fps: float | None = None) -> dict:
    """Hit, miss or no ball in a video (ball_events), e.g. the LiDAR phone's own color video when
    there is no slow-motion clip. Returns {"kind", "hit", "how", "reason", "frames_with_ball",
    "xy" (N, 2) ball points in full-size pixels, NaN where none)}."""
    import cv2
    cands = []
    for fr in frames:
        small = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else fr
        cands.append(ball_candidates(small))
    ev = ball_events(cands, fps=fps)
    return {"kind": ev["kind"], "hit": ev["kind"] == "hit", "how": ev["how"], "reason": ev["reason"],
            "frames_with_ball": int(np.isfinite(ev["xy"][:, 0]).sum()), "xy": ev["xy"] / scale}


def launch_from_flight(tb: dict, fps: float, m_per_px: float, *, n_fit: int = 8, max_skip: int = 3,
                       image_y_down: bool = True) -> Launch | None:
    """Exit speed and launch angle from ``n_fit`` flight points; the contact frame is where the fitted
    path left the rest spot (extrapolated back, so a bat covering the ball doesn't matter): the last
    frame at or before contact, as in ``contact_frame``.

    Right after contact the bat follows through next to the ball and their blurs merge, which
    shifts the first few ball positions by several pixels (synthetic test: up to 6 px for 3 frames).
    So up to ``max_skip`` of the earliest points may be skipped: the first start with a clean fit
    (rmse <= 1.5 px) wins, else the best one; at least 5 points are always used."""
    order = sorted(tb["flight"])
    fits = []
    for k in range(0, max_skip + 1):
        fr = order[k:k + n_fit]
        if len(fr) < 5 and not (k == 0 and len(fr) >= 3):
            break
        t = np.array(fr, float) / fps
        P = np.array([tb["flight"][i] for i in fr]) * m_per_px
        x, y = P[:, 0], (-P[:, 1] if image_y_down else P[:, 1])
        vx, x0 = np.polyfit(t, x, 1)
        t_c = (tb["rest"][0] * m_per_px - x0) / vx if abs(vx) > 1e-9 else t[0]   # when the ball left the tee
        tau = t - t_c                                 # gravity acts from contact: time since contact, not clip time
        vy, y0 = np.polyfit(tau, y + 0.5 * G * tau**2, 1)
        rmse = float(np.sqrt(np.mean((x0 + vx * t - x) ** 2 + (y0 + vy * tau - 0.5 * G * tau**2 - y) ** 2)) / m_per_px)
        fits.append((rmse, k, fr, vx, x0, vy, t_c))
        if rmse <= 1.5:
            break
    if not fits:
        return None
    rmse, k, fr, vx, x0, vy, t_c = fits[-1] if fits[-1][0] <= 1.5 else min(fits)
    c = int(np.clip(np.round(t_c * fps), tb["last_rest_frame"], order[0] - 1))   # the frame nearest the contact instant
    speed = float(np.hypot(vx, vy))
    return Launch(c, speed, speed * 2.2369363, float(np.degrees(np.arctan2(vy, abs(vx)))),
                  int(np.sign(vx) or 1), len(fr), rmse, float(t_c))


def ball_positions(l: Launch, rest_xy: tuple[float, float], m_per_px: float, fps: float, n: int) -> dict[int, np.ndarray]:
    """Where the fitted flight puts the ball in every frame after contact (px), also where it was
    hidden by the bat or blurred: used to keep the ball out of the bat's tape-band tracking."""
    th = np.radians(l.launch_angle_deg)
    out = {}
    for i in range(l.contact_frame + 1, n):
        tau = i / fps - (l.contact_t if l.contact_t is not None else l.contact_frame / fps)
        if tau <= 0:
            continue
        x = rest_xy[0] + l.direction * l.exit_speed_mps * np.cos(th) * tau / m_per_px
        y = rest_xy[1] - (l.exit_speed_mps * np.sin(th) * tau - 0.5 * G * tau * tau) / m_per_px
        out[i] = np.array([x, y])
    return out


# ---------------------------------------------------------------- bat: two bright tape bands
#
# The recording plan puts one ~5 cm band of bright tape at the bat's end and one just above the
# hands. At 240 fps the bat is a blur, but the camera is still: the median frame is the background,
# and the tape is the only thing much BRIGHTER than it (a dark bat is darker; the ball resting on the
# tee is part of the background). A blurred band is a streak whose centroid is the band's position
# at mid-exposure, so centroid-to-centroid motion gives the speed even when the streak is long.
# Velocity varies linearly along a rigid bat, so the two bands give the speed at any point of it,
# e.g. the sweet spot, which is where bat sensors (and Driveline's bat speed) measure.

MPS_TO_MPH = 2.2369363


@dataclass
class BatTape:
    """Where the tape is, measured from the knob along the bat (meters). Measure your bat."""
    bat_length_m: float = 0.84           # 33 in
    tip_band_m: float = 0.815            # center of the end band (5 cm band at the end)
    hand_band_m: float = 0.22            # center of the band just above the hands
    sweet_spot_from_end_m: float = 0.15  # 6 in from the end, where bat sensors measure

    @property
    def separation_m(self) -> float:
        return self.tip_band_m - self.hand_band_m


def _gray(img: np.ndarray) -> np.ndarray:
    import cv2
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img).astype(np.float32)


def background(frames: list[np.ndarray], n: int = 60) -> np.ndarray:
    """Median of up to ``n`` frames spread over the clip (grayscale float)."""
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    return np.median(np.stack([_gray(frames[i]) for i in idx]), axis=0).astype(np.float32)


def detect_bright(frame_bgr: np.ndarray, bg: np.ndarray, *, thresh: float = 25.0, min_area: int = 12):
    """Blobs much brighter than the background: [(x, y, area)], centroid weighted by the brightness gain."""
    import cv2
    gain = cv2.GaussianBlur(_gray(frame_bgr) - bg, (3, 3), 0)
    mask = cv2.morphologyEx((gain > thresh).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.nonzero(lab == i)
        w = gain[ys, xs]
        out.append((float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum()), int(len(xs))))
    return out


def track_bands(frames, *, m_per_px: float | None = None, tape: BatTape = BatTape(),
                tol: float = 0.25, max_miss: int = 4, bg: np.ndarray | None = None,
                exclude: dict[int, tuple[float, float, float]] | None = None) -> dict:
    """Follow the two tape bands. Returns {"tip": (N,2), "hand": (N,2), "found_fraction"}: pixel
    tracks, NaN where not found.

    From the side camera the bat points nearly AT the camera around contact, so the bands look
    much closer together then than when the bat is side-on (94 px instead of 238 in the synthetic
    test). Spacing therefore only has an upper bound (the full tape spacing + ``tol``). A track
    starts on a pair whose spacing matches the tape (bat side-on); after that each band's position
    is predicted from its last two positions (constant velocity) and the pair closest to the
    predictions is taken, which also keeps the flying ball from being mistaken for a band. The band
    that travels farther over the clip is the end band (the bat turns around the hands).
    ``frames``: a list, or any iterable (a video stream) when ``bg`` (see ``background``) is given.
    ``exclude`` = {frame: (x, y, radius)}: ignore bright blobs there (the ball in flight, which right
    after contact travels next to the bat and is as bright as the tape)."""
    if bg is None:
        bg = background(frames)
    sep_px = tape.separation_m / m_per_px if m_per_px else None
    A, B = [], []

    def predict(T, i):
        known = [r for r in range(i - 1, max(-1, i - 1 - max_miss), -1) if np.isfinite(T[r][0])]
        if not known:
            return None
        if len(known) == 1:
            return T[known[0]]
        r1, r2 = known[0], known[1]
        return T[r1] + (T[r1] - T[r2]) * (i - r1) / (r1 - r2)

    for i, f in enumerate(frames):
        blobs = detect_bright(f, bg)
        if exclude and i in exclude:
            ex, ey, er = exclude[i]
            blobs = [b for b in blobs if np.hypot(b[0] - ex, b[1] - ey) > er]
        pa, pb = predict(A, i), predict(B, i)
        best, best_cost = None, np.inf
        for j in range(len(blobs)):
            for k in range(j + 1, len(blobs)):
                p, q = np.array(blobs[j][:2]), np.array(blobs[k][:2])
                d = float(np.linalg.norm(p - q))
                if d < 6 or (sep_px is not None and d > (1 + tol) * sep_px):
                    continue
                if pa is not None and pb is not None:     # tracking: closest to where the bands should be
                    e1 = np.linalg.norm(p - pa) + np.linalg.norm(q - pb)
                    e2 = np.linalg.norm(q - pa) + np.linalg.norm(p - pb)
                    cost, pair = (e1, (p, q)) if e1 <= e2 else (e2, (q, p))
                    limit = 0.5 * (sep_px or d) + 0.5 * np.linalg.norm(pa - pb)
                    if cost > 2 * limit:
                        continue
                elif sep_px is not None:                    # starting: bat side-on, spacing matches the tape
                    if abs(d - sep_px) > tol * sep_px:
                        continue
                    cost, pair = abs(d - sep_px) / sep_px, (p, q)
                else:
                    cost, pair = -min(blobs[j][2], blobs[k][2]), (p, q)
                if cost < best_cost:
                    best, best_cost = pair, cost
        A.append(best[0] if best is not None else np.full(2, np.nan))
        B.append(best[1] if best is not None else np.full(2, np.nan))
    A, B = np.array(A).reshape(-1, 2), np.array(B).reshape(-1, 2)
    path = lambda T: float(np.nansum(np.linalg.norm(np.diff(T, axis=0), axis=1)))  # noqa: E731
    tip, hand = (A, B) if path(A) >= path(B) else (B, A)
    return {"tip": tip, "hand": hand, "found_fraction": round(float(np.isfinite(tip[:, 0]).mean()), 3)}


def _velocity(xy: np.ndarray, i: int, fps: float) -> np.ndarray | None:
    """Velocity at frame i in px/s: central difference, else one-sided."""
    ok = lambda j: 0 <= j < len(xy) and np.isfinite(xy[j, 0])  # noqa: E731
    if ok(i - 1) and ok(i + 1):
        return (xy[i + 1] - xy[i - 1]) * fps / 2
    if ok(i) and ok(i + 1):
        return (xy[i + 1] - xy[i]) * fps
    if ok(i - 1) and ok(i):
        return (xy[i] - xy[i - 1]) * fps
    return None


def bat_at(track: dict, frame: int, fps: float, m_per_px: float, tape: BatTape = BatTape(),
           direction: int | None = None) -> dict | None:
    """Bat speed and angles at ``frame`` (e.g. contact) in the image plane, from the two band tracks.

    bat_speed_mph     speed of the sweet spot (``tape.sweet_spot_from_end_m`` from the end)
    tip_speed_mph     speed of the end band
    bat_angle_deg     bat (hands -> end) above horizontal; + = end above the hands
    attack_angle_deg  direction the sweet spot moves; + = upward
    ``direction``: +1 if the ball goes to image right, -1 left; default: the sweet spot's motion."""
    tip, hand = track["tip"], track["hand"]
    if not (0 <= frame < len(tip)) or not (np.isfinite(tip[frame, 0]) and np.isfinite(hand[frame, 0])):
        return None
    vt, vh = _velocity(tip, frame, fps), _velocity(hand, frame, fps)
    if vt is None or vh is None:
        return None
    a = (tape.bat_length_m - tape.sweet_spot_from_end_m - tape.hand_band_m) / tape.separation_m
    v_sweet = vh + a * (vt - vh)                    # rigid bat: velocity linear along its length
    mph = lambda v: float(np.linalg.norm(v) * m_per_px * MPS_TO_MPH)  # noqa: E731
    fwd = direction or (int(np.sign(v_sweet[0])) or 1)
    d = tip[frame] - hand[frame]
    return {"frame": int(frame), "bat_speed_mph": round(mph(v_sweet), 1), "tip_speed_mph": round(mph(vt), 1),
            "hand_band_speed_mph": round(mph(vh), 1),
            "bat_angle_deg": round(float(np.degrees(np.arctan2(-d[1], abs(d[0])))), 1),
            "attack_angle_deg": round(float(np.degrees(np.arctan2(-v_sweet[1], fwd * v_sweet[0]))), 1),
            "band_spacing_px": round(float(np.linalg.norm(d)), 1)}


def fastest_frame(track: dict, fps: float) -> int | None:
    """Frame of peak end-band speed (the contact proxy for a dry swing)."""
    v = [_velocity(track["tip"], i, fps) for i in range(len(track["tip"]))]
    s = [float(np.linalg.norm(x)) if x is not None else -1.0 for x in v]
    return int(np.argmax(s)) if max(s, default=-1.0) > 0 else None


def scale_from_yardstick(frames, length_m: float = 0.9144, *, min_angle_deg: float = 60.0) -> tuple[float | None, dict]:
    """Meters per pixel from a clip (list of frames) or one image of a yardstick held upright at the tee,
    in the ball's plane. The stick is the longest near-vertical straight edge in the median frame.
    Returns (m_per_px, info); (None, {"reason": ...}) if no upright edge is found."""
    import cv2
    img = (np.median(np.stack(frames[:: max(1, len(frames) // 30)]), axis=0).astype(np.uint8)
           if isinstance(frames, list) else frames)
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 40, 120), 1, np.pi / 720, threshold=80,
                            minLineLength=int(0.15 * img.shape[0]), maxLineGap=8)
    if lines is None:
        return None, {"reason": "no straight edge found"}
    best = None
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):   # (N,1,4) in OpenCV 4, (N,4) in 5
        L = float(np.hypot(x2 - x1, y2 - y1))
        ang = float(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
        if ang >= min_angle_deg and (best is None or L > best[0]):
            best = (L, ang, [int(x1), int(y1), int(x2), int(y2)])
    if best is None:
        return None, {"reason": "no upright straight edge (hold the stick vertical)"}
    # Hough segments stop a few px short of soft (compressed, blurred) ends: measure the stick's
    # length from the contrast profile along it instead, at half height of its plateau.
    L = _extent_along(gray, np.array(best[2][:2], float), np.array(best[2][2:], float)) or best[0]
    return length_m / L, {"length_px": round(L, 1), "hough_length_px": round(best[0], 1),
                          "angle_deg": round(best[1], 1), "segment": best[2]}


def _extent_along(gray: np.ndarray, p1: np.ndarray, p2: np.ndarray, half_band: int = 6) -> float | None:
    """Length (px) of the object whose edge runs p1 -> p2: mean contrast vs the background in a
    band across that edge, sampled from 30% before p1 to 30% past p2; ends at half the plateau."""
    import cv2
    L = float(np.linalg.norm(p2 - p1))
    d = (p2 - p1) / L
    n = np.array([-d[1], d[0]])
    ts = np.arange(-0.3 * L, 1.3 * L, 0.5)
    off = np.arange(-half_band, half_band + 1)
    X = (p1[0] + ts[:, None] * d[0] + off[None, :] * n[0]).astype(np.float32)
    Y = (p1[1] + ts[:, None] * d[1] + off[None, :] * n[1]).astype(np.float32)
    vals = cv2.remap(gray.astype(np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    outside = (ts < -0.15 * L) | (ts > 1.15 * L)
    prof = np.abs(vals - np.median(vals[outside], axis=0)).mean(axis=1)
    plateau = float(np.median(prof[(ts > 0.2 * L) & (ts < 0.8 * L)]))
    if plateau < 5:
        return None
    lvl, mid = 0.5 * plateau, int(np.argmin(np.abs(ts - L / 2)))

    def crossing(step):
        i = mid
        while 0 < i < len(ts) - 1 and prof[i] >= lvl:
            i += step
        j = i - step                                  # last index above the level
        return ts[j] + (ts[i] - ts[j]) * (prof[j] - lvl) / max(prof[j] - prof[i], 1e-9)
    return float(crossing(1) - crossing(-1))


# ---------------------------------------------------------------- object detector: is a ball in view at all?
#
# The blob tracker above looks for bright round things that move like a ball. On the LiDAR phone's
# 60 fps video it also finds them on dry swings (white shoes, a bright bat end, sky through the arms:
# TestSet2, 9 of 10 dry swings labeled hit or miss). A COCO detector that knows "sports ball" is the
# independent check: a swing is only a hit or a miss when the detector also sees a ball in the frames
# before contact, near where the tracker says the ball was. YOLOX-S from the OpenCV model zoo
# (Apache-2.0, runs on cv2.dnn: no PyTorch), downloaded by scripts/get_ball_model.py.

YOLO_MODEL_FILE = "object_detection_yolox_2022nov.onnx"
YOLO_MODEL_URL = "https://huggingface.co/opencv/object_detection_yolox/resolve/main/" + YOLO_MODEL_FILE
COCO_SPORTS_BALL, COCO_PERSON, COCO_BAT = 32, 0, 34
YOLO_SIZE = 640
MIN_BALL_SCORE = 0.25          # detector confidence for "sports ball"

_YOLO = None


def _yolo():
    global _YOLO
    if _YOLO is None:
        import cv2

        from . import config
        path = config.MODELS / YOLO_MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python scripts/get_ball_model.py")
        _YOLO = cv2.dnn.readNet(str(path))
        grids, strides = [], []
        for s in (8, 16, 32):
            n = YOLO_SIZE // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        _YOLO_GRID[:] = [np.concatenate(grids), np.concatenate(strides)]
    return _YOLO


_YOLO_GRID: list = [None, None]


def yolo_detect(frame_bgr: np.ndarray, classes: tuple[int, ...] = (COCO_SPORTS_BALL,),
                min_score: float = MIN_BALL_SCORE) -> list[tuple[int, float, float, float, float, float]]:
    """Objects of ``classes`` in one frame: [(class, score, x, y, w, h)] in pixels (box center and size)."""
    import cv2
    net = _yolo()
    h, w = frame_bgr.shape[:2]
    r = min(YOLO_SIZE / h, YOLO_SIZE / w)
    pad = np.full((YOLO_SIZE, YOLO_SIZE, 3), 114, np.uint8)
    pad[: int(h * r), : int(w * r)] = cv2.resize(frame_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    net.setInput(cv2.dnn.blobFromImage(pad))          # BGR, 0–255, as YOLOX was trained
    out = net.forward()[0]
    grid, stride = _YOLO_GRID
    xy = (out[:, :2] + grid) * stride / r
    wh = np.exp(out[:, 2:4]) * stride / r
    dets = []
    for c in classes:
        sc = out[:, 4] * out[:, 5 + c]
        keep = np.where(sc >= min_score)[0]
        if not len(keep):
            continue
        boxes = [[float(xy[i, 0] - wh[i, 0] / 2), float(xy[i, 1] - wh[i, 1] / 2), float(wh[i, 0]), float(wh[i, 1])] for i in keep]
        for j in np.array(cv2.dnn.NMSBoxes(boxes, sc[keep].astype(float).tolist(), min_score, 0.45)).reshape(-1):
            i = keep[j]
            dets.append((c, float(sc[i]), float(xy[i, 0]), float(xy[i, 1]), float(wh[i, 0]), float(wh[i, 1])))
    return dets


def confirm_ball(frames, tracked_xy: np.ndarray | None = None, *, every: int = 3, near_px: float = 120.0,
                 min_frames: int = 2, detect=None) -> dict:
    """Does the object detector see a ball in these frames (the ones before contact)?
    Every ``every``-th frame is checked. Confirmed when a "sports ball" is found in at least
    ``min_frames`` frames and, if the blob tracker's ball points ``tracked_xy`` (full-size pixels,
    one row per frame, NaN = none) are given, at least one detection lies within ``near_px`` of a
    tracked point within 3 frames. ``detect(frame) -> [(class, score, x, y, w, h)]`` replaces YOLOX
    (tests). Returns {"confirmed", "frames_checked", "frames_with_ball", "near_track", "best_score"}."""
    detect = detect or yolo_detect
    checked, hits, near, best = 0, 0, False, 0.0
    for i, fr in enumerate(frames):
        if i % every:
            continue
        checked += 1
        balls = [d for d in detect(fr) if d[0] == COCO_SPORTS_BALL]
        if not balls:
            continue
        hits += 1
        best = max(best, max(d[1] for d in balls))
        if tracked_xy is not None and not near:
            lo, hi = max(0, i - 3), min(len(tracked_xy), i + 4)
            pts = tracked_xy[lo:hi]
            pts = pts[np.isfinite(pts[:, 0])] if len(pts) else pts
            near = any(np.hypot(*(pts - [d[2], d[3]]).T).min() <= near_px for d in balls if len(pts))
    confirmed = hits >= min_frames and (tracked_xy is None or near)
    return {"confirmed": bool(confirmed), "frames_checked": checked, "frames_with_ball": hits,
            "near_track": bool(near), "best_score": round(best, 2)}


# ---------------------------------------------------------------- hit / miss from the object detector (toss hitting)
#
# TestSet4 (240 fps, self-tossed balls, phone held to the side): the blob tracker followed white clutter, while
# YOLOX found the real ball in many frames. The ball vanishes at the bat and reappears farther away, so a hit is
# not one continuous path. From the detector's own tracks instead:
#   hit   a FAST track (>= MIN_FLIGHT_PX_PER_FRAME) moving away from the hitter, starting near contact; or the toss
#         is seen coming down by the hitter, vanishes at bat height and is never seen falling past the waist (hit
#         out of the camera's view: toward it, or straight away; TestSet4, where the team reported misses that were hits)
#   miss  the toss is seen falling on past the hitter's waist (it went by the bat)
#   none  no toss seen near the hitter
# Exit speed: the flight's image-plane speed x the scale from the ball's own box size (0.0737 m) - rough, and low
# when the ball flies toward or away from the camera.

MIN_FLIGHT_PX_PER_FRAME = 8.0


def link_detections(pts: list[tuple[int, float, float, float]], *, max_gap: int = 4, max_step: float = 70.0,
                    min_len: int = 4) -> list[list[tuple]]:
    """[(frame, x, y, size)] -> tracks (lists of the same tuples), nearest continuation within max_step px/frame."""
    trs: list[list[tuple]] = []
    for p in sorted(pts):
        i, x, y = p[0], p[1], p[2]
        best, bd = None, None
        for t in trs:
            j, xj, yj = t[-1][0], t[-1][1], t[-1][2]
            d = float(np.hypot(x - xj, y - yj))
            if 0 < i - j <= max_gap and d <= max_step * (i - j) and (bd is None or d < bd):
                best, bd = t, d
        if best is not None:
            best.append(p)
        else:
            trs.append([p])
    return [t for t in trs if len(t) >= min_len]


def detector_ball_outcome(frames, *, fps: float, contact_frame: int | None = None, detect=None,
                          before_s: float = 0.25, after_s: float = 0.35) -> dict:
    """Hit / miss / none for a toss (or tee) swing from YOLOX ball and person detections on every frame.
    ``contact_frame``: when the bat reaches the ball (default: the middle of the clip). ``detect(frame)`` replaces
    YOLOX (tests). Returns {"kind", "reason", "frames", "frames_with_ball", "flight" | None, "exit_speed_mph",
    "launch_angle_deg", "tracks"}."""
    detect = detect or (lambda fr: yolo_detect(fr, (COCO_SPORTS_BALL, COCO_PERSON), min_score=0.12))
    pts, people, n = [], [], 0
    for i, fr in enumerate(frames):
        n += 1
        for c, sc, x, y, w, h in detect(fr):
            if c == COCO_SPORTS_BALL:
                pts.append((i, x, y, min(w, h)))
            elif c == COCO_PERSON and sc >= 0.3:
                people.append((x, y, w, h))
    c0 = n // 2 if contact_frame is None else int(contact_frame)
    lo, hi = c0 - int(before_s * fps), c0 + int(after_s * fps)
    px, py, pw, ph = (float(v) for v in np.median(np.array(people), axis=0)) if people else (None, None, None, None)
    trs = link_detections(pts)
    sizes = [p[3] for p in pts if p[3] > 2]
    m_per_px = 2 * BALL_RADIUS_M / float(np.median(sizes)) if sizes else None
    flights, tosses = [], []
    for t in trs:
        (i0, x0, y0, _), (i1, x1, y1, _) = t[0], t[-1]
        if np.hypot(x1 - x0, y1 - y0) < 15:                   # a ball lying still (ground, bucket)
            continue
        speed = float(np.hypot(x1 - x0, y1 - y0)) / max(1, i1 - i0)
        away = px is None or (x1 - x0) * (x0 - px) > 0 or (y1 - y0) < -2 * abs(x1 - x0)
        if speed >= MIN_FLIGHT_PX_PER_FRAME and away and lo <= i0 <= hi:
            flights.append((t, speed))
        elif y1 > y0 and (px is None or abs(float(np.median([p[1] for p in t])) - px) <= 0.9 * (pw or 400)):
            tosses.append(t)                                   # coming down by the hitter
    out = {"frames": n, "frames_with_ball": len({p[0] for p in pts}), "flight": None, "exit_speed_mph": None,
           "launch_angle_deg": None, "tracks": len(trs), "scale_m_per_px": m_per_px}
    if flights:
        t, speed = max(flights, key=lambda f: len(f[0]) * f[1])
        (i0, x0, y0, _), (i1, x1, y1, _) = t[0], t[-1]
        out["flight"] = {"frames": [int(i0), int(i1)], "from_px": [round(x0), round(y0)], "to_px": [round(x1), round(y1)],
                         "px_per_frame": round(speed, 1)}
        if m_per_px:
            out["exit_speed_mph"] = round(speed * fps * m_per_px * 2.2369363, 1)
        out["launch_angle_deg"] = round(float(np.degrees(np.arctan2(-(y1 - y0), abs(x1 - x0)))), 1)
        return {**out, "kind": "hit", "reason": f"the ball detector saw the ball fly away from the hitter after contact "
                                                f"({len(t)} frames)"}
    waist = None if py is None else py + 0.05 * ph            # image y grows downward
    if tosses and waist is not None and any(max(p[2] for p in t) > waist for t in tosses):
        return {**out, "kind": "miss", "reason": "the ball detector saw the toss fall on past the hitter's waist: it went "
                                                 "by the bat (swing and miss)"}
    if tosses:
        return {**out, "kind": "hit", "how": "left_view", "reason": "the toss came down by the hitter and vanished at bat "
                "height, never falling past him: hit out of the camera's view (direction and speed not seen)"}
    return {**out, "kind": "none", "reason": "the ball detector saw no toss near the hitter"}
