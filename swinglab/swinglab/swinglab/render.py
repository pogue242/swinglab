"""Annotated swing video (OpenCV + Pillow), like the inspirations' render.py: the swing with the
skeleton drawn on it, a side panel with live numbers, scores at the end, and the ball's flight
from the slow-motion camera when there is one.

  render_swing(rec, out)       phone recording (rec["video"] from the pipeline: frames + 2D keypoints)
                               or a lab swing drawn on a blank background (3D joints projected)
  draw_keypoints_2d(img, kp)   MediaPipe 2D keypoints colored by visibility (sanity overlay)

On the picture: bones, the hip line (blue) and shoulder line (red) labeled with the hip–shoulder
separation MEASURED IN 3D (a 2D angle would be distorted by the camera angle), the focus joint.
Panel: phase, pelvis and torso rotation speed, separation, hand speed, a kinematic-sequence chart
with a moving cursor. End card: Power / Transfer / Contact / Overall, predicted bat speed,
measured exit speed, the coach headline. Then a slow-motion replay with the ball's track and
projected flight if rec["slowmo"] has one.

Video codec: H.264 MP4 with x264 (CRF 23) through the ffmpeg binary that ``imageio-ffmpeg`` ships
(plays in every browser and phone, ~1 MB per second of 720p). Fallbacks through OpenCV: H.264 via
Windows Media Foundation (plays everywhere but ~20 MB/s: no bitrate control), VP9 WebM, then
MPEG-4 Part 2 (``mp4v``: desktop players only, not Chrome).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# side-named MediaPipe joints (skeleton.MEDIAPIPE values) for the 2D overlay
BONES_2D = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
            ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
            ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]
GOOD, WEAK = (80, 220, 80), (0, 165, 255)          # BGR: green = visibility >= threshold, orange = below
# BGR colors matching the viewer
INK, PELVIS, TORSO, HANDS, SEAM, MUTED = (39, 32, 27), (179, 109, 47), (45, 54, 194), (31, 109, 138), (45, 54, 194), (112, 102, 93)
PANEL_BG, PANEL_INK = (28, 23, 19), (236, 233, 230)
# RGB colors for the (Pillow-drawn) panel, the viewer's dark-theme palette
P_PELVIS, P_TORSO, P_HANDS, P_SEAM = (106, 166, 236), (255, 107, 94), (217, 183, 90), (255, 107, 94)
PANEL_W, OUT_H = 380, 720
END_CARD_S, SLOWMO_PLAYBACK_FPS = 3.0, 30.0


def draw_keypoints_2d(img: np.ndarray, kp: dict[str, tuple[float, float, float]], *, scale: float = 1.0,
                      min_vis: float = 0.5, thickness: int = 3) -> np.ndarray:
    """Draw ``{name: (u, v, visibility)}`` on ``img`` (in place). ``scale`` maps keypoint pixels to img pixels.
    Weak joints (visibility < min_vis, which get no depth sample) are drawn orange."""
    import cv2
    P = {n: (int(round(u * scale)), int(round(v * scale)), vis) for n, (u, v, vis) in kp.items() if np.isfinite(u)}
    for a, b in BONES_2D:
        if a in P and b in P:
            col = GOOD if min(P[a][2], P[b][2]) >= min_vis else WEAK
            cv2.line(img, P[a][:2], P[b][:2], col, thickness, cv2.LINE_AA)
    for n, (x, y, vis) in P.items():
        cv2.circle(img, (x, y), thickness + 2, GOOD if vis >= min_vis else WEAK, -1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- video writer

class _FfmpegWriter:
    """BGR frames -> H.264 MP4 through imageio-ffmpeg's bundled ffmpeg."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import imageio_ffmpeg
        self._gen = imageio_ffmpeg.write_frames(str(path), size, fps=float(fps), codec="libx264", pix_fmt_in="rgb24",
                                                pix_fmt_out="yuv420p", macro_block_size=2,
                                                output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"])
        self._gen.send(None)

    def write(self, bgr: np.ndarray) -> None:
        self._gen.send(np.ascontiguousarray(bgr[:, :, ::-1]))

    def release(self) -> None:
        self._gen.close()


def open_writer(out: Path, fps: float, size: tuple[int, int]):
    """(writer, path, codec) for ``out`` + extension. x264 MP4 if imageio-ffmpeg is installed, else
    OpenCV: H.264 MP4 (Media Foundation), VP9 WebM, mp4v."""
    import cv2
    try:
        path = Path(f"{out}.mp4")
        return _FfmpegWriter(path, fps, size), path, "h264"
    except Exception:
        pass
    tries = []
    if hasattr(cv2, "CAP_MSMF"):
        tries.append((Path(f"{out}.mp4"), cv2.CAP_MSMF, "H264", "h264-msmf"))
    tries += [(Path(f"{out}.webm"), cv2.CAP_FFMPEG, "VP90", "vp9"),
              (Path(f"{out}.mp4"), cv2.CAP_FFMPEG, "mp4v", "mp4v")]
    for path, api, fourcc, name in tries:
        try:
            vw = cv2.VideoWriter(str(path), api, cv2.VideoWriter_fourcc(*fourcc), float(fps), size)
        except Exception:
            continue
        if vw.isOpened():
            return vw, path, name
    raise RuntimeError("no video codec available in this OpenCV build")


# ---------------------------------------------------------------- text (Pillow: real fonts, °, →)

_FONTS: dict = {}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    key = (size, bold)
    if key not in _FONTS:
        names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold else
                 ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
        for n in names:
            try:
                _FONTS[key] = ImageFont.truetype(n, size)
                break
            except OSError:
                continue
        else:
            _FONTS[key] = ImageFont.load_default(size=size)
    return _FONTS[key]


def _n(x: float, nd: int = 0) -> str:
    """Rounded number without a "-0"."""
    v = round(float(x), nd)
    return f"{(v if v != 0 else 0.0):.{nd}f}"


def _rgb(bgr):
    return (bgr[2], bgr[1], bgr[0])


# ---------------------------------------------------------------- swing timeline helpers

def phase_at(ev: dict, t: float, fps: float) -> str:
    if t >= ev["contact"] - 0.5 / fps:
        return "Finish" if t > ev["contact"] + 0.15 else "Contact"
    if t >= ev["foot_plant"]:
        return "Foot plant → contact"
    if t >= ev["load"]:
        return "Load → stride"
    return "Stance"


def series_at(rec: dict, t: float) -> dict:
    tt = np.asarray(rec["time"], float)
    return {k: float(np.interp(t, tt, np.asarray(v, float))) for k, v in rec["series"].items()}


# ---------------------------------------------------------------- 2D positions per frame

def project_lab(J: np.ndarray, size: tuple[int, int], azimuth_deg: float = -30.0, margin: float = 0.08):
    """Canonical 3D joints (T, n, 3; x toward pitcher, z up) -> pixels (T, n, 2) on a blank canvas,
    seen from the plate side like camera A (facing the chest, turned toward the catcher)."""
    a = np.radians(azimuth_deg)
    right = np.array([np.cos(a), np.sin(a), 0.0])
    u, v = J @ right, -J[..., 2]
    ok = np.isfinite(u)
    lo_u, hi_u, lo_v, hi_v = np.nanmin(u[ok]), np.nanmax(u[ok]), np.nanmin(v[ok]), max(np.nanmax(v[ok]), 0.0)
    w, h = size
    s = min((1 - 2 * margin) * w / max(hi_u - lo_u, 1e-6), (1 - 2 * margin) * h / max(hi_v - lo_v, 1e-6))
    cu, cv = 0.5 * (lo_u + hi_u), 0.5 * (lo_v + hi_v)
    return np.stack([w / 2 + (u - cu) * s, h / 2 + (v - cv) * s], -1), (h / 2 + (0 - cv) * s)


def canonical_kp(rec: dict, P2: np.ndarray, i: int) -> dict[str, tuple[float, float, float]]:
    """Canonical names -> side names (l_/r_) so one drawing path handles phone and lab swings."""
    lead, rear = ("l", "r") if rec.get("bats", "R").upper() == "R" else ("r", "l")
    out = {}
    for k, n in enumerate(rec["joints"]):
        if n == "neck":
            continue
        side, part = n.split("_", 1)
        out[f"{lead if side == 'lead' else rear}_{part}"] = (float(P2[i, k, 0]), float(P2[i, k, 1]), 1.0)
    return out


# ---------------------------------------------------------------- drawing

def _draw_body(img, kp: dict, scale: float, focus: str | None, bats: str, thick: int):
    """Bones, extended hip and shoulder lines, focus ring. Returns where to put the separation label."""
    import cv2
    P = {n: np.array([u * scale, v * scale]) for n, (u, v, _) in kp.items() if np.isfinite(u)}
    for a, b in BONES_2D:
        if a in P and b in P and {a, b} not in ({"l_hip", "r_hip"}, {"l_shoulder", "r_shoulder"}):
            cv2.line(img, tuple(P[a].astype(int)), tuple(P[b].astype(int)), (245, 245, 245), thick + 2, cv2.LINE_AA)
            cv2.line(img, tuple(P[a].astype(int)), tuple(P[b].astype(int)), INK, thick, cv2.LINE_AA)
    label_at = None
    for (a, b), col in ((("l_hip", "r_hip"), PELVIS), (("l_shoulder", "r_shoulder"), TORSO)):
        if a in P and b in P:
            m, d = 0.5 * (P[a] + P[b]), P[b] - P[a]
            p0, p1 = m - 0.9 * d, m + 0.9 * d          # lines extended past the joints, like a coach's markup
            cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), col, thick + 2, cv2.LINE_AA)
            label_at = m if label_at is None else 0.5 * (label_at + m)
    for p in P.values():
        cv2.circle(img, tuple(p.astype(int)), thick + 2, (245, 245, 245), -1, cv2.LINE_AA)
    if focus:
        lead, rear = ("l", "r") if bats.upper() == "R" else ("r", "l")
        side, part = focus.split("_", 1) if focus != "neck" else ("lead", "shoulder")
        fn = f"{lead if side == 'lead' else rear}_{part}"
        if fn in P:
            cv2.circle(img, tuple(P[fn].astype(int)), 6 * thick, SEAM, max(2, thick), cv2.LINE_AA)
    return label_at


def _label(img, xy, text: str, *, size: int = 22, fg=(255, 255, 255), bg=(20, 20, 20)):
    from PIL import Image, ImageDraw
    f = _font(size, True)
    x0, y0, x1, y1 = f.getbbox(text)
    pad = 6
    w, h = x1 - x0 + 2 * pad, y1 - y0 + 2 * pad
    x, y = int(np.clip(xy[0] + 14, 0, img.shape[1] - w)), int(np.clip(xy[1] - h / 2, 0, img.shape[0] - h))
    patch = Image.new("RGB", (w, h), _rgb(bg))
    ImageDraw.Draw(patch).text((pad - x0, pad - y0), text, font=f, fill=_rgb(fg))
    img[y:y + h, x:x + w] = np.asarray(patch)[:, :, ::-1]


def _panel(rec: dict, t: float, *, title: str, end: bool = False, slow: dict | None = None) -> np.ndarray:
    """Side panel (OUT_H x PANEL_W, BGR)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (PANEL_W, OUT_H), _rgb(PANEL_BG))
    d = ImageDraw.Draw(im)
    ink, muted = _rgb(PANEL_INK), (150, 160, 170)
    d.text((22, 18), "SWING", font=_font(30, True), fill=ink)
    d.text((22 + _font(30, True).getlength("SWING"), 18), "LAB", font=_font(30, True), fill=P_SEAM)
    d.text((22, 56), title, font=_font(15), fill=muted)
    ev, fps = rec["events"], float(rec.get("fps", 30))
    if not end:
        d.text((22, 92), phase_at(ev, t, fps).upper(), font=_font(26, True), fill=P_SEAM)
        d.text((22, 124), f"{t - ev['contact']:+.3f} s from contact", font=_font(15), fill=muted)
        s = series_at(rec, t)
        rows = [("Pelvis rotation", _n(s["pelvis_vel"]), "°/s", P_PELVIS),
                ("Torso rotation", _n(s["torso_vel"]), "°/s", P_TORSO),
                ("Hip–shoulder separation", _n(s["separation"]), "°", ink),
                ("Hand speed", _n(s["hand_speed"], 1), "m/s", P_HANDS)]
        y = 162
        for lab, val, unit, col in rows:
            d.text((22, y), lab, font=_font(15), fill=muted)
            d.text((22, y + 18), val, font=_font(34, True), fill=col)
            d.text((22 + _font(34, True).getlength(val) + 6, y + 34), unit, font=_font(15), fill=muted)
            y += 70
        _chart(d, rec, t, (22, 460, PANEL_W - 22, 640))
        d.text((22, 652), "speeds measured in 3D · separation = hips ahead of shoulders", font=_font(12), fill=muted)
    else:
        an = rec.get("analysis") or {}
        sc = an.get("scores") or {}
        d.text((22, 92), "RESULT", font=_font(26, True), fill=P_SEAM)
        d.text((22, 126), f"{sc.get('overall', 0):.0f}", font=_font(64, True), fill=ink)
        d.text((22 + _font(64, True).getlength(f"{sc.get('overall', 0):.0f}") + 10, 160),
               "overall\n" + ("form + ball contact" if sc.get("contact") is not None else "form (dry swing)"),
               font=_font(14), fill=muted)
        y = 214
        rows = [("form", "Form", True), ("power", "Power", False), ("transfer", "Sequencing", False),
                ("posture", "Posture", False), ("contact", "Contact", True)]
        for k, lab, main in rows:
            if sc.get(k) is None:
                continue
            v = float(sc[k])
            col = (92, 196, 137) if v >= 60 else (240, 166, 58) if v >= 35 else (255, 122, 112)
            d.text((22 if main else 36, y), lab, font=_font(18 if main else 15, main), fill=ink if main else muted)
            d.rounded_rectangle((140, y + 4, 300, y + 16), 6, fill=(43, 51, 60))
            d.rounded_rectangle((140, y + 4, 140 + int(160 * v / 100), y + 16), 6, fill=col)
            d.text((310, y), f"{v:.0f}", font=_font(18 if main else 15, True), fill=ink)
            y += 30
        pred = (an.get("predicted") or {}).get("bat_speed")
        if pred is not None:
            d.text((22, y + 8), "Bat speed (predicted from body)", font=_font(15), fill=muted)
            d.text((22, y + 26), f"{pred:.1f} mph", font=_font(28, True), fill=ink)
            y += 70
        ms = (rec.get("measured") or {})
        if ms.get("bat_speed_mph") is not None:
            d.text((22, y), f"Bat speed measured: {ms['bat_speed_mph']:.1f} mph", font=_font(16), fill=ink)
            y += 26
        if slow and slow.get("exit_speed_mph") is not None:
            d.text((22, y), f"Exit speed off the tee: {slow['exit_speed_mph']:.1f} mph", font=_font(16), fill=ink)
            d.text((22, y + 22), f"Launch angle: {slow['launch_angle_deg']:.0f}°", font=_font(16), fill=ink)
            y += 52
        head = (rec.get("report") or {}).get("headline", "")
        if head:
            d.text((22, y + 14), "Coach", font=_font(15), fill=muted)
            _wrap(d, head, (22, y + 34), PANEL_W - 44, _font(20, True), ink)
    return np.asarray(im)[:, :, ::-1].copy()


def _wrap(d, text, xy, width, font, fill):
    words, line, y = text.split(), "", xy[1]
    for w in words:
        if font.getlength((line + " " + w).strip()) > width:
            d.text((xy[0], y), line, font=font, fill=fill)
            line, y = w, y + font.size + 6
        else:
            line = (line + " " + w).strip()
    if line:
        d.text((xy[0], y), line, font=font, fill=fill)


def _chart(d, rec, t, box):
    """Kinematic sequence: pelvis and torso rotation speed (deg/s) and hand speed (scaled), cursor at t."""
    x0, y0, x1, y1 = box
    tt = np.asarray(rec["time"], float)
    ev = rec["events"]
    a, b = max(tt[0], ev["load"] - 0.1), min(tt[-1], ev["contact"] + 0.15)
    if b <= a:
        a, b = tt[0], tt[-1]
    m = (tt >= a) & (tt <= b)
    pv, tv, hs = (np.asarray(rec["series"][k], float)[m] for k in ("pelvis_vel", "torso_vel", "hand_speed"))
    top = max(250.0, np.nanmax(np.r_[pv, tv]) if len(pv) else 250.0)
    hmax = max(1.0, np.nanmax(hs) if len(hs) else 1.0)
    X = lambda v: x0 + (v - a) / (b - a) * (x1 - x0)  # noqa: E731
    Y = lambda v, mx: y1 - max(0.0, v) / mx * (y1 - y0)  # noqa: E731
    d.rectangle(box, outline=(55, 63, 72))
    d.text((x0, y0 - 20), "Kinematic sequence", font=_font(15, True), fill=(200, 205, 210))
    for k in ("foot_plant", "contact"):
        if a <= ev[k] <= b:
            d.line((X(ev[k]), y0, X(ev[k]), y1), fill=(90, 98, 108), width=1)
    for arr, mx, col in ((pv, top, P_PELVIS), (tv, top, P_TORSO), (hs, hmax, P_HANDS)):
        pts = [(X(x), Y(v, mx)) for x, v in zip(tt[m], arr) if np.isfinite(v)]
        if len(pts) > 1:
            d.line(pts, fill=col, width=2)
    if a <= t <= b:
        d.line((X(t), y0, X(t), y1), fill=(236, 233, 230), width=2)


# ---------------------------------------------------------------- main entry

def render_swing(rec: dict, out: str | Path, *, video: str | Path | None = None, slowmo_video: str | Path | None = None,
                 title: str | None = None, max_height: int = OUT_H, end_card_s: float = END_CARD_S) -> dict:
    """Write the annotated video. Returns sync metadata for the viewer:
    {"path", "codec", "fps", "t_start", "t_end", "main_frames", "frame_times", "end_card_s", "slowmo_s"}.
    Video second v (v < main duration) shows swing time t_start + v (rec time, seconds)."""
    import cv2
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vid = rec.get("video") or {}
    video = video or vid.get("path")
    slowmo_video = slowmo_video or rec.get("slowmo_video")
    title = title or (f"{rec.get('hitter') or 'Hitter'} · {rec.get('id', '')}" if video else f"Lab swing {rec.get('id', '')}")
    focus = (rec.get("report") or {}).get("focus_joint") if (rec.get("analysis") or {}).get("focus") else None
    focus_phase = (rec.get("report") or {}).get("focus_phase", "contact")
    tt = np.asarray(rec["time"], float)
    sep = np.asarray(rec["series"]["separation"], float)

    if video and Path(str(video).replace("%d", "0")).exists():   # a file, or the first frame of a %d.jpg sequence
        from .capture import iter_frames
        fps = float(vid.get("fps") or 30.0)
        f0, f1 = vid.get("frames", [0, 10 ** 9])
        t_off = float(vid.get("t_offset", 0.0))
        ts_all = vid.get("timestamps")
        kp2d = {int(k): v for k, v in (vid.get("kp2d") or {}).items()}

        def frames():
            for i, img in enumerate(iter_frames(video)):
                if i < f0:
                    continue
                if i > f1:
                    break
                ti = (ts_all[i - f0] if ts_all else i / fps) - t_off   # timestamps cover frames f0..f1
                yield img, ti, {n: tuple(p) for n, p in kp2d.get(i, {}).items()}
        src_size = tuple(vid.get("size") or (0, 0))
    else:   # lab swing (or no video file): blank background, 3D joints projected
        fps = float(rec.get("fps", 30))
        J = np.asarray(rec["J"], float)
        canvas = (int(OUT_H * 0.8), OUT_H)
        P2, floor_y = project_lab(J, canvas)

        def frames():
            for i, ti in enumerate(tt):
                img = np.full((canvas[1], canvas[0], 3), (241, 244, 243), np.uint8)
                cv2.line(img, (0, int(floor_y)), (canvas[0], int(floor_y)), (200, 205, 200), 2)
                yield img, float(ti), canonical_kp(rec, P2, i)
        src_size = canvas

    w0, h0 = src_size
    s = max_height / h0 if h0 else 1.0
    vw_w = int(round(w0 * s)) // 2 * 2
    size = (vw_w + PANEL_W, max_height)
    writer, path, codec = open_writer(out, fps, size)
    thick = max(2, int(round(max_height / 240)))
    times, n = [], 0
    try:
        for img, ti, kp in frames():
            img = cv2.resize(img, (vw_w, max_height))
            sep_now = float(np.interp(ti, tt, sep))
            near_focus = abs(ti - rec["events"].get(focus_phase, -9)) < 0.25   # ring the focus joint around its phase
            at = _draw_body(img, kp, s, focus if near_focus else None, rec.get("bats", "R"), thick) if kp else None
            if at is not None:
                _label(img, at, f"separation {_n(sep_now)}°", size=max(16, max_height // 32))
            frame = np.hstack([img, _panel(rec, ti, title=title)])
            writer.write(frame)
            times.append(round(ti, 4))
            n += 1
        last = img if n else np.zeros((max_height, vw_w, 3), np.uint8)
        slow = rec.get("slowmo") or None
        end = np.hstack([(last * 0.35).astype(np.uint8), _panel(rec, times[-1] if times else 0, title=title, end=True, slow=slow)])
        for _ in range(int(round(end_card_s * fps))):
            writer.write(end)
        slow_s = 0.0
        if slow and slowmo_video and slow.get("contact_frame") is not None and Path(slowmo_video).exists():
            slow_s = _slowmo_segment(writer, slowmo_video, slow, size, fps, rec, title)
    finally:
        writer.release()
    return {"path": str(path), "codec": codec, "fps": fps, "t_start": times[0] if times else 0.0,
            "t_end": times[-1] if times else 0.0, "main_frames": n, "frame_times": times,
            "end_card_s": end_card_s, "slowmo_s": round(slow_s, 2), "size": list(size)}


def _slowmo_segment(writer, video, slow: dict, size, fps: float, rec: dict, title: str) -> float:
    """Slow-motion replay around contact with the ball's track and projected flight. Returns seconds written."""
    import cv2
    from .capture import iter_frames
    c = int(slow["contact_frame"])
    lo, hi = max(0, c - 24), c + 72
    traj = np.asarray(slow.get("trajectory_px") or [], float)
    track = {int(k): v for k, v in (slow.get("track_px") or {}).items()}
    vw_w = size[0] - PANEL_W
    n = 0
    for i, img in enumerate(iter_frames(video)):
        if i < lo:
            continue
        if i > hi:
            break
        h, w = img.shape[:2]
        s = min(vw_w / w, size[1] / h)
        canvas = np.zeros((size[1], vw_w, 3), np.uint8)
        small = cv2.resize(img, (int(w * s), int(h * s)))
        oy, ox = (size[1] - small.shape[0]) // 2, (vw_w - small.shape[1]) // 2
        canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
        P = lambda p: (int(ox + p[0] * s), int(oy + p[1] * s))  # noqa: E731
        if i > c and len(traj) > 1:
            cv2.polylines(canvas, [np.array([P(p) for p in traj], np.int32)], False, (94, 107, 255), 3, cv2.LINE_AA)
        for k in sorted(track):
            if k <= i and track[k] is not None:
                cv2.circle(canvas, P(track[k]), 4, (255, 255, 255), -1, cv2.LINE_AA)
        bt = (slow.get("bat_track_px") or {}).get(str(i))
        if bt:                                       # the bat between its two tape bands
            cv2.line(canvas, P(bt[1]), P(bt[0]), (90, 183, 217), 3, cv2.LINE_AA)
            cv2.circle(canvas, P(bt[0]), 6, (90, 183, 217), -1, cv2.LINE_AA)
        if i >= c and slow.get("exit_speed_mph") is not None:
            _label(canvas, (20, 40), f"exit {slow['exit_speed_mph']:.1f} mph · launch {slow['launch_angle_deg']:.0f}°", size=24)
        if i >= c and slow.get("bat"):
            _label(canvas, (20, 84), f"bat {slow['bat']['bat_speed_mph']:.1f} mph · attack {slow['bat']['attack_angle_deg']:.0f}°", size=22)
        _label(canvas, (20, size[1] - 40), f"slow motion · {slow.get('fps', 240):.0f} fps", size=16)
        writer.write(np.hstack([canvas, _panel(rec, rec["events"]["contact"], title=title, end=True, slow=slow)]))
        n += 1
    return n / fps
