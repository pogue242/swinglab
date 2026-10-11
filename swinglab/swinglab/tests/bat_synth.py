"""Synthetic 240 fps slow-motion tee swing filmed like camera B, rendered with real motion blur.

3D scene, perspective camera: the camera stands ``cam_dist`` meters to the side of the tee, level
with it, looking straight across it (x = the ball's direction, to the image right; z up). A taped
bat (two white 5 cm bands: end, above the hands) swings as a rigid lever around the hitter's body:
at contact its sweet spot meets the ball on the tee while the bat points TOWARD the camera (across
the plate) and ``vba_deg`` downward, and the barrel moves toward the field ``attack_deg`` upward.
So, as on the field, the bat is foreshortened near contact and the two bands sit at different
distances from the camera. Each frame averages ``sub`` renders over the exposure, so fast parts
smear into streaks as in a real clip. Ground truth comes from the same motion.
"""
from __future__ import annotations

import numpy as np

from swinglab.ballflight import BatTape

MPH = 2.2369363


def _rot(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def render_swing(*, fps=240.0, n=100, omega=40.0, knob_radius=0.45, vba_deg=25.0, attack_deg=8.0,
                 cam_dist=4.5, m_per_px=0.0025, size=(1280, 720), tape: BatTape = BatTape(),
                 exposure_s=1 / 480, sub=16, noise=4.0, ball=False, exit_speed=35.0, launch_deg=12.0,
                 seed=0, contact_frame=55, before_deg=150.0, after_deg=70.0, tee=True, toss=False, miss=False):
    """Returns (frames BGR uint8, truth). ``omega`` rad/s: the lever's angular speed, so the sweet spot
    moves at omega * (knob_radius + distance knob->sweet spot).
    ``toss``: instead of resting on a tee, the ball is thrown up from below-left 0.6 s before contact
    and falls into the hitting zone; ``miss``: it keeps falling (no hit)."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = size
    f = cam_dist / m_per_px                                   # focal length in px; m_per_px holds at the tee
    tee_z = 0.0                                               # the tee/ball is the world origin
    cam = np.array([0.0, -cam_dist, tee_z])

    def px(p):                                                # world -> pixel (camera looks along +y)
        d = p - cam
        return np.array([W / 2 + f * d[0] / d[1], H / 2 - f * d[2] / d[1]]), d[1]

    vba, att = np.radians(vba_deg), np.radians(attack_deg)
    u_c = np.array([0.0, -np.cos(vba), -np.sin(vba)])         # bat direction at contact: toward the camera, down
    w = np.array([0.0, -np.sin(vba), np.cos(vba)])            # perpendicular to u_c, pointing up
    v_dir = np.cos(att) * np.array([1.0, 0, 0]) + np.sin(att) * w
    axis = np.cross(u_c, v_dir)                               # d(u)/dt = omega * axis x u
    s_sweet = tape.bat_length_m - tape.sweet_spot_from_end_m
    center = -(knob_radius + s_sweet) * u_c                   # sweet spot at the tee at contact
    t_c = contact_frame / fps
    a0, a1 = -np.radians(before_deg), np.radians(after_deg)

    def angle(t):
        return float(np.clip(omega * (t - t_c), a0, a1))

    def point(s, t):
        return center + (knob_radius + s) * (_rot(axis, angle(t)) @ u_c)

    yy, xx = np.mgrid[0:H, 0:W]
    bg = (95 + 30 * xx / W + 10 * np.sin(yy / 37.0)).astype(np.float32)
    bg = np.repeat(bg[:, :, None], 3, 2)
    cv2.rectangle(bg, (60, H - 80), (180, H - 60), (230, 230, 230), -1)       # a static bright object
    if tee and not toss:
        p0, _ = px(np.array([0, 0, -0.04]))
        p1, _ = px(np.array([0, 0, -0.75]))
        cv2.line(bg, tuple(p0.astype(int)), tuple(p1.astype(int)), (30, 30, 30), 6)

    def bat_quads(t):
        """(depth, polygon px, color) pieces, drawn far to near."""
        u = _rot(axis, angle(t)) @ u_c
        side = np.cross(u, np.array([0.0, 1.0, 0.0]))
        side = side / (np.linalg.norm(side) + 1e-9) if np.linalg.norm(side) > 1e-6 else np.array([0, 0, 1.0])
        pieces = []
        edges = np.linspace(0, tape.bat_length_m, 25)
        width = lambda s: 0.032 + (0.066 - 0.032) * np.clip((s - 0.2) / 0.4, 0, 1)  # noqa: E731
        bands = [(tape.tip_band_m - 0.025, tape.tip_band_m + 0.025), (tape.hand_band_m - 0.025, tape.hand_band_m + 0.025)]
        cuts = sorted(set(np.r_[edges, [b for ab in bands for b in ab]]))
        for s0, s1 in zip(cuts[:-1], cuts[1:]):
            sm = 0.5 * (s0 + s1)
            col = (235, 235, 235) if any(a <= sm <= b for a, b in bands) else (35, 38, 45)
            P = [point(s, t) for s in (s0, s1)]
            quad, depth = [], 0.0
            for p, s, sg in ((P[0], s0, 1), (P[1], s1, 1), (P[1], s1, -1), (P[0], s0, -1)):
                q, d = px(p + sg * width(s) / 2 * side)
                quad.append(q)
                depth += d / 4
            pieces.append((depth, np.array(quad, np.int32), col))
        return sorted(pieces, key=lambda x: -x[0])

    def draw(t):
        img = bg.copy()
        if ball:
            TOSS_S = 0.6
            v0 = np.array([0.25 / TOSS_S, 0.0, (0.2 + 4.905 * TOSS_S ** 2) / TOSS_S])   # from (-0.25, 0, -0.2) to the contact spot
            if toss and (t <= t_c or miss):
                tau = t - (t_c - TOSS_S)
                if tau < 0:
                    return _draw_bat(img, t)
                b = np.array([-0.25, 0.0, -0.2]) + v0 * tau + np.array([0, 0, -4.905 * tau * tau])
            elif t <= t_c:
                b = np.zeros(3)
            else:
                dt = t - t_c
                b = exit_speed * dt * np.array([np.cos(np.radians(launch_deg)), 0, np.sin(np.radians(launch_deg))]) \
                    + np.array([0, 0, -4.905 * dt * dt])
            q, d = px(b)
            cv2.circle(img, tuple(np.round(q).astype(int)), max(1, int(round(0.0368 * f / d))), (240, 240, 240), -1, cv2.LINE_AA)
        return _draw_bat(img, t)

    def _draw_bat(img, t):
        for _, quad, col in bat_quads(t):
            cv2.fillPoly(img, [quad], col, cv2.LINE_AA)
        return img

    frames = []
    for i in range(n):
        t = i / fps
        acc = np.zeros((H, W, 3), np.float32)
        for k in range(sub):
            acc += draw(t + (k + 0.5) / sub * exposure_s - exposure_s / 2)
        acc = acc / sub + rng.normal(0, noise, acc.shape)
        frames.append(np.clip(acc, 0, 255).astype(np.uint8))

    def vel(s, t, h=1e-5):
        return (point(s, t + h) - point(s, t - h)) / (2 * h)

    v_sweet = vel(s_sweet, t_c)
    tip_img, _ = px(point(tape.tip_band_m, t_c))
    hand_img, _ = px(point(tape.hand_band_m, t_c))
    d = tip_img - hand_img
    truth = {"contact_frame": contact_frame, "fps": fps, "m_per_px": m_per_px,
             "bat_speed_mph": float(np.linalg.norm(v_sweet)) * MPH,
             "tip_speed_mph": float(np.linalg.norm(vel(tape.tip_band_m, t_c))) * MPH,
             "attack_angle_deg": float(np.degrees(np.arctan2(v_sweet[2], v_sweet[0]))),
             "bat_angle_image_deg": float(np.degrees(np.arctan2(-d[1], abs(d[0])))),
             "band_spacing_px_at_contact": float(np.linalg.norm(d)),
             "band_spacing_px_full": tape.separation_m / m_per_px}
    return frames, truth


def write(frames, path, fps=240.0):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for fr in frames:
        vw.write(fr)
    vw.release()
    return path


def yardstick_frames(m_per_px=0.0025, size=(1280, 720), tilt_deg=3.0, seed=0, length_m=0.9144, n=20):
    """A yellow yardstick held (nearly) upright at the tee, among other straight edges: a horizontal
    fence rail and the short dark tee post. Returns (frames, true length in px)."""
    import cv2
    rng = np.random.default_rng(seed)
    W, H = size
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.repeat((90 + 40 * xx / W + 8 * np.sin(yy / 23.0))[:, :, None], 3, 2).astype(np.uint8)
    cv2.line(img, (0, 470), (W, 470), (60, 60, 60), 3)
    cv2.line(img, (900, 700), (900, 560), (30, 30, 30), 6)
    L = length_m / m_per_px
    a = np.radians(tilt_deg)
    c = np.array([640.0, 360.0])
    d = np.array([np.sin(a), -np.cos(a)]) * L / 2
    nrm = np.array([np.cos(a), np.sin(a)]) * 0.035 / m_per_px / 2
    cv2.fillPoly(img, [np.array([c - d - nrm, c - d + nrm, c + d + nrm, c + d - nrm], np.int32)], (40, 200, 230), cv2.LINE_AA)
    return [np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8) for _ in range(n)], L
