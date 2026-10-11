"""Analysis stages for real recordings, as functions (used by the inbox processor,
the upload server and the scripts).

  analyze_lidar(folder, hitter)   Stray Scanner folder -> skeleton, features, scores, report
  analyze_slowmo(video)           240 fps side video -> contact frame, exit speed, launch angle
  analyze_swing(lidar, slowmo, hitter) both, merged; slo-mo numbers go to Gemini as "measured"

Every stage records what it did in rec["stages"] so the UI and logs show
exactly which parts ran and which failed.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import capture, config, features, kinematics as K, model, quality as Q, scoring, skeleton

r3 = lambda x: np.round(np.asarray(x, float), 3).tolist()  # noqa: E731

_MODELS = None


def _models():
    global _MODELS
    if _MODELS is None:
        import joblib
        _MODELS = (model.load(), joblib.load(config.MODELS / "scorers.joblib"))
    return _MODELS


# Driveline training range (metadata of the 677 swings). Predictions outside it are extrapolation.
TRAINING_RANGE = {"height_m": (1.60, 1.96), "mass_kg": (63.5, 118.0), "athlete_age": (17, 31)}
DEFAULT_MASS_KG, DEFAULT_AGE = 80.0, 20.0
MIN_JOINT_COVERAGE = 0.5   # each joint needs a 3D position in at least half the frames of a swing


def covariates(h: dict) -> dict:
    return {"height_m": float(h["height_m"]), "mass_kg": float(h.get("mass_kg") or DEFAULT_MASS_KG),
            "athlete_age": float(h.get("age") or DEFAULT_AGE), "level": model.LEVELS.get(h.get("level", "college"), 1)}


def hitter_warnings(h: dict) -> list[str]:
    """Plain-language notes when the hitter's profile is missing or outside the lab data."""
    w = []
    if not h.get("mass_kg"):
        w.append(f"weight not given; assumed {DEFAULT_MASS_KG:.0f} kg")
    if not h.get("age"):
        w.append(f"age not given; assumed {DEFAULT_AGE:.0f}")
    c = covariates(h)
    names = {"height_m": "height", "mass_kg": "weight", "athlete_age": "age"}
    for k, (lo, hi) in TRAINING_RANGE.items():
        if not lo <= c[k] <= hi:
            w.append(f"{names[k]} {c[k]:g} is outside the lab hitters' range ({lo:g}–{hi:g}); "
                     "speed predictions are less reliable, mechanics comparisons still apply")
    return w


def lift_recording(folder: str | Path, *, pose_model: str | Path | None = None, keypoints_fn=None,
                   min_vis: float = 0.5, height_m: float | None = None, bone_fallback: bool = True,
                   pose_fps: float | None = None, use_mediapipe_3d: bool = True):
    """Stray Scanner folder or Record3D export -> (timestamps, cleaned world-frame joints, quality).
    Pose runs once. Keypoints below ``min_vis`` MediaPipe visibility get no depth sample; with
    ``bone_fallback`` a limb joint without usable depth is placed by bone length (capture.lift_keypoints).
    With MediaPipe (no ``keypoints_fn``) and ``use_mediapipe_3d``, the joints are MediaPipe's 3D pose
    anchored by the LiDAR (capture.lift_world); quality["lift"] says which was used."""
    cap = capture.load_any(folder)
    pose_model = pose_model or config.MODELS / "pose_landmarker_heavy.task"
    # pose on every n-th frame: the model works at config.PHONE_FPS (30) and MediaPipe is the slowest
    # stage (~12 frames/s on a laptop CPU), so a 60 fps Record3D clip needs only every second frame
    every = 1 if keypoints_fn else max(1, int(round(cap.fps / (pose_fps or config.PHONE_FPS))))
    world: dict = {}
    kps, size = (keypoints_fn or (lambda v: capture.mediapipe_keypoints(v, pose_model, every=every, fps=cap.fps,
                                                                        world=world)))(cap.video)
    lift_info: dict = {}
    joints = capture.lift_keypoints(cap, kps, size, min_vis=min_vis, height_m=height_m,
                                    bone_fallback=bone_fallback, info=lift_info)
    valid = float(np.mean([np.isfinite(v[:, 0]).mean() for v in joints.values()]))
    joints, replaced = capture.clean_joints(joints, cap.timestamps)
    # body shape from MediaPipe 3D, position from the LiDAR (lift_world); depth-only joints if it can't anchor
    hybrid = capture.lift_world(cap, world, joints, info=lift_info) if world and use_mediapipe_3d else None
    if hybrid is not None:
        joints = hybrid
        # bone-length placement only shaped the LiDAR anchor here, not the joints: keep it out of the warnings
        lift_info["depth_placed_by_bone_length"] = lift_info.pop("placed_by_bone_length", {})
    else:
        lift_info["lift"] = "lidar_depth"
    quality = {"source_app": cap.kind, "pose_frames": len(kps), "frames": cap.n, "capture_fps": round(cap.fps, 1),
               "duration_s": round(float(cap.timestamps[-1] - cap.timestamps[0]), 2),
               "valid_joint_fraction": round(valid, 3),
               "depth_outliers_replaced": round(float(np.mean(list(replaced.values()))), 3),
               "pose_every_n_frames": every,
               "framing": Q.framing(kps, size, len(range(0, cap.n, every)), min_vis=min_vis), **lift_info,
               # for the annotated video (render.py); analyze_joints keeps only its swing's frames
               "_video": {"path": str(cap.video), "fps": round(cap.fps, 3), "size": list(size),
                          "timestamps": np.round(cap.timestamps, 4).tolist(), "kp2d": kps}}
    return cap.timestamps, joints, quality


def analyze_joints(t: np.ndarray, joints: dict, hitter: dict, rec_id: str, quality: dict | None = None) -> dict:
    """World-frame joints for ONE swing -> full record (skeleton, events, features, scores)."""
    # a joint seen in too few frames would be one straight interpolated line (or NaN, which crashes smoothing)
    lr = skeleton.to_lead_rear(joints, hitter["bats"])
    cover = {n: float(np.isfinite(lr[n][:, 0]).mean()) for n in skeleton.CANONICAL if n in lr}
    missing = [n for n in skeleton.CANONICAL if n not in lr] + [n for n, c in cover.items() if c < MIN_JOINT_COVERAGE]
    if missing:
        raise ValueError("not enough 3D data for " + ", ".join(f"{n.replace('_', ' ')} ({cover.get(n, 0):.0%} of frames)"
                                                               for n in missing)
                         + f"; need {MIN_JOINT_COVERAGE:.0%}. Usually no depth on that body part (dark clothing, "
                         "fast motion, out of range) or it was hidden; check the pose overlay and check_recording.py")
    # resample to the SAME rate the model was trained at (config.PHONE_FPS)
    sk = skeleton.build(t, joints, hitter["bats"], up=capture.WORLD_UP_ARKIT,
                        height_m=float(hitter["height_m"]), fps=config.PHONE_FPS)
    s = K.series(sk)
    ev = K.detect_events(sk, s)
    f = features.compute(sk, ev, s)
    models, scorers = _models()
    an = scoring.analyze(f, models, scorers, covariates(hitter))
    quality = dict(quality or {})
    video = _video_for(quality.pop("_video", None), t)
    quality["bones"] = Q.bone_lengths(sk)
    warnings = hitter_warnings(hitter) + Q.framing_warnings(quality.get("framing") or {})
    if Q.bone_warning(quality["bones"]):
        warnings.append(Q.bone_warning(quality["bones"]))
    warnings += Q.fallback_warnings(quality.get("placed_by_bone_length") or {}, hitter["bats"])
    return {"id": rec_id, "hitter": hitter.get("name", ""), "bats": hitter["bats"],
            "height_m": float(hitter["height_m"]), "level": hitter.get("level", "college"), "fps": round(sk.fps, 2),
            "time": r3(sk.time), "joints": sk.names, "J": r3(np.nan_to_num(sk.J)), "bat": None,
            "events": {k: v for k, v in ev.items() if isinstance(v, float)},
            "series": {k: r3(np.nan_to_num(s[k])) for k in ("pelvis_vel", "torso_vel", "hand_speed", "separation", "lead_knee_flex")},
            "features": {k: round(float(v), 3) for k, v in f.items()}, "analysis": an,
            "measured": {"bat_speed_mph": None, "exit_velo_mph": None},
            "warnings": warnings, "quality": quality, "video": video}


def _video_for(v: dict | None, t: np.ndarray) -> dict | None:
    """The recording's video info cut to this swing: frames whose time lies in [t[0], t[-1]].
    ``t_offset`` = recording time of the swing's time 0 (skeleton time starts at 0)."""
    if not v:
        return None
    ts = np.asarray(v["timestamps"])
    idx = np.where((ts >= t[0] - 1e-6) & (ts <= t[-1] + 1e-6))[0]
    if not len(idx):
        return None
    f0, f1 = int(idx[0]), int(idx[-1])
    r = lambda x: [round(float(a), 1) for a in x[:2]] + [round(float(x[2]), 2)]  # noqa: E731
    return {"path": v["path"], "fps": v["fps"], "size": v["size"], "frames": [f0, f1], "t_offset": float(t[0]),
            "timestamps": ts[f0:f1 + 1].tolist(), "kp2d": {str(i): {n: r(p) for n, p in v["kp2d"][i].items()}
                                                 for i in range(f0, f1 + 1) if i in v["kp2d"]}}


def analyze_lidar(folder: str | Path, hitter: dict, **lift_kw) -> dict:
    """One recording = one swing. ``hitter``: {name, bats, height_m, mass_kg?, age?, level?}.
    ``lift_kw`` go to lift_recording, e.g. ``keypoints_fn(video) -> (kps, size)`` to replace
    MediaPipe (tests, or a different pose model), ``bone_fallback=False``."""
    t, joints, q = lift_recording(folder, height_m=float(hitter["height_m"]), **lift_kw)
    return analyze_joints(t, joints, hitter, Path(folder).name, q)


def video_frames(video: str | Path):
    import cv2
    cap = cv2.VideoCapture(str(video))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def analyze_slowmo(video: str | Path, *, fps: float = 240.0, m_per_px: float | None = None,
                   max_frames: int = 2400, yardstick: str | Path | None = None, tape=None,
                   track_bat: bool = True, bat_width_px: int = 960, stick_length_m: float = 0.9144) -> dict:
    """240 fps side video -> ball off the tee (contact frame, exit speed, launch angle) and the bat
    from its two tape bands (bat speed at the sweet spot, end-band speed, bat and attack angle).
    All in the image plane: motion toward or away from the camera is not seen.

    Scale (m/px at the tee): ``m_per_px``, else the ``yardstick`` clip (scale_from_yardstick), else
    the ball's size on the tee. Contact = the ball leaving the tee; on a dry swing, the frame of
    peak bat speed. ``tape`` = ballflight.BatTape (where the tape is on the bat).
    Two passes over the file, so long clips never sit in memory."""
    import cv2
    from . import ballflight as B
    tape = tape or B.BatTape()
    from itertools import islice
    scale_source, yinfo = ("given" if m_per_px else None), None
    if m_per_px is None and yardstick:   # every 8th frame of the first second: the stick is still
        m_per_px, yinfo = B.scale_from_yardstick(list(islice(video_frames(yardstick), 0, 240, 8)), stick_length_m)
        scale_source = "yardstick" if m_per_px else None
    dets, bg_samples, n_total = [], [], None
    vc = cv2.VideoCapture(str(video))
    n_total = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    vc.release()
    step = max(1, min(n_total, max_frames) // 60)
    shrink = None
    exp_r = B.BALL_RADIUS_M / m_per_px if m_per_px else None   # ball size in px when the scale is known
    for i, fr in enumerate(video_frames(video)):
        if i >= max_frames:
            break
        dets.append(B.ball_candidates(fr))
        if shrink is None:
            shrink = min(1.0, bat_width_px / fr.shape[1])
        if track_bat and i % step == 0:
            bg_samples.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None, fx=shrink, fy=shrink))
    ev = B.ball_events(dets, expected_r=exp_r, fps=fps)          # tee hit, toss hit, miss, or no ball
    tb = ev if ev["kind"] == "hit" else None
    xy = ev["xy"]
    c = None
    if m_per_px is None and tb and tb["rest"][2] > 0:
        m_per_px = B.scale_from_ball(tb["rest"][2])
        scale_source = "ball size" if m_per_px else None
    l = B.launch_from_flight(tb, fps, m_per_px) if (tb and m_per_px) else None
    if l:
        c = l.contact_frame
    out = {"frames": len(dets), "ball_found_fraction": round(float(np.isfinite(xy[:, 0]).mean()), 3) if dets else 0,
           "contact_frame": c, "m_per_px": m_per_px, "scale_source": scale_source, "yardstick": yinfo, "exit_speed_mph": None,
           "launch_angle_deg": None, "bat_speed_mph": None, "bat": None, "fps": fps,
           "ball_outcome": ev["kind"], "ball_how": ev["how"], "ball_reason": ev["reason"],
           # ball positions around contact, for the annotated replay (render.py)
           "track_px": {str(i): [round(float(xy[i, 0]), 1), round(float(xy[i, 1]), 1)]
                        for i in range(max(0, (c or 0) - 20), min(len(xy), (c or 0) + 80)) if np.isfinite(xy[i, 0])}}
    direction = None
    if l and l.fit_rmse_px > B.MAX_BALLISTIC_RMSE_PX:   # not a ball in flight (e.g. a tape band on the bat)
        note = f"tracked object did not fly like a ball (fit error {l.fit_rmse_px:.0f} px); no ball found"
        out.update(contact_frame=None, ball_outcome="none", ball_note=note, ball_reason=note)
        c, l = None, None
    elif tb is None:
        out["ball_note"] = ev["reason"]
    if l:
        direction = l.direction
        out.update(contact_frame=c, exit_speed_mph=round(l.exit_speed_mph, 1), launch_angle_deg=round(l.launch_angle_deg, 1),
                   fit_rmse_px=round(l.fit_rmse_px, 2), contact_time_s=round(c / fps, 4), ball_rest_px=r3(tb["rest"]),
                   trajectory_px=r3(B.trajectory(l, tb["rest"][:2], m_per_px, t_max=1.0)[::8]))
    if track_bat and m_per_px and bg_samples:
        bg = np.median(np.stack(bg_samples), axis=0).astype(np.float32)
        small = (cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None, fx=shrink, fy=shrink)
                 for i, fr in enumerate(video_frames(video)) if i < max_frames)
        # the ball is as bright as the tape and travels next to the bat (a toss before contact, the hit
        # after it): keep every tracked ball position, and the fitted flight where it was hidden, out
        # of the band tracking
        r_ball = (tb["rest"][2] if tb else ev.get("radius")) or (exp_r or 12.0)
        r_mask = 1.5 * 2 * r_ball * shrink
        ball_mask = {i: (float(xy[i, 0]) * shrink, float(xy[i, 1]) * shrink, r_mask)
                     for i in range(len(xy)) if np.isfinite(xy[i, 0])}
        if l:
            ball_mask.update({i: (p[0] * shrink, p[1] * shrink, r_mask)
                              for i, p in B.ball_positions(l, tb["rest"][:2], m_per_px, fps, len(dets)).items()})
        trk = B.track_bands(small, m_per_px=m_per_px / shrink, tape=tape, bg=bg, exclude=ball_mask)
        at = c if c is not None else B.fastest_frame(trk, fps)
        bat = B.bat_at(trk, at, fps, m_per_px / shrink, tape, direction=direction) if at is not None else None
        if bat:
            bat.update(contact_from="ball leaving the tee" if c is not None else "peak bat speed (no ball found)",
                       bands_found_fraction=trk["found_fraction"])
            out["bat"] = bat
            out["bat_speed_mph"] = bat["bat_speed_mph"]
        lo, hi = max(0, (at or 0) - 30), min(len(trk["tip"]), (at or 0) + 20)
        out["bat_track_px"] = {str(i): [r3(trk["tip"][i] / shrink), r3(trk["hand"][i] / shrink)]
                               for i in range(lo, hi) if np.isfinite(trk["tip"][i, 0])}
    return out


def analyze_swing(lidar: str | Path | None, slowmo: str | Path | None, hitter: dict, *,
                  use_gemini: bool = True, slowmo_fps: float = 240.0, m_per_px: float | None = None,
                  lifted=None, window=None, yardstick: str | Path | None = None, tape=None,
                  stick_length_m: float = 0.9144, **kw) -> dict:
    """Run every available stage for ONE swing; never raises. rec["stages"] says what happened.
    ``lifted``/``window``: reuse an already-lifted recording and analyze only that swing's window."""
    from . import session
    stages, rec = {}, None
    if lidar:
        try:
            if lifted is None:
                rec = analyze_lidar(lidar, hitter, **kw)
            else:
                t, joints, q = lifted
                if window is not None:
                    t, joints = session.cut(t, joints, window)
                rec = analyze_joints(t, joints, hitter, Path(lidar).name, q)
                if window is None:
                    rec["warnings"].append("no clear swing found (hand speed never reached a swing's); analyzed the whole clip")
            stages["lidar"] = {"ok": True, **rec["quality"]}
        except Exception as e:
            stages["lidar"] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
    slow = None
    if slowmo:
        try:
            slow = analyze_slowmo(slowmo, fps=slowmo_fps, m_per_px=m_per_px, yardstick=yardstick, tape=tape,
                                  stick_length_m=stick_length_m)
            stages["slowmo"] = {"ok": slow.get("exit_speed_mph") is not None or slow.get("bat_speed_mph") is not None,
                                **{k: slow[k] for k in ("frames", "ball_found_fraction", "contact_frame", "scale_source", "bat_speed_mph")}}
        except Exception as e:
            stages["slowmo"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if rec is None:
        return {"id": Path(lidar or slowmo).name, "hitter": hitter.get("name", ""), "stages": stages, "slowmo": slow}
    rec["stages"] = stages
    _finish(rec, hitter, slow, slowmo, use_gemini)
    return rec


def attach_slowmo(rec: dict, slow: dict | None, video: str | Path | None) -> dict | None:
    """Put a slow-motion result on a swing record (rec["slowmo"], rec["measured"], the clip for the
    replay) and return the "measured" block for the coach payload (None if nothing was measured)."""
    rec["slowmo"], rec["slowmo_video"] = slow, (str(video) if video and slow else None)
    if not slow:
        return None
    measured = None
    if slow.get("exit_speed_mph") is not None:
        measured = {"exit_speed_mph_tee": slow["exit_speed_mph"], "launch_angle_deg": slow["launch_angle_deg"],
                    "note": "measured off a tee; lower than lab exit velocity, which had a ~65 mph incoming pitch"}
        rec["measured"]["exit_velo_mph"] = slow["exit_speed_mph"]
    if slow.get("bat"):
        bat = slow["bat"]
        measured = {**(measured or {}), "bat_speed_mph_sweet_spot": bat["bat_speed_mph"],
                    "attack_angle_deg": bat["attack_angle_deg"],
                    "bat_note": "slow-motion camera, image plane; compare with predicted bat speed"}
        rec["measured"]["bat_speed_mph"] = bat["bat_speed_mph"]
    return measured


# Gemini reports for every swing of a recording (team decision 2026-10-10). The free tier allows 20 calls
# per model per day; set GEMINI_MAX_SWINGS to cap it (the rest get template reports).
GEMINI_MAX_SWINGS = int(__import__("os").environ.get("GEMINI_MAX_SWINGS", "0")) or None


def read_frames(video: str | Path, f0: int, f1: int):
    """BGR frames f0..f1 of a video or a %d.jpg sequence (sequences are read directly by index)."""
    import cv2
    if "%d" in str(video):
        for i in range(f0, f1 + 1):
            img = cv2.imread(str(video).replace("%d", str(i)))
            if img is not None:
                yield img
        return
    for i, fr in enumerate(capture.iter_frames(video)):
        if i > f1:
            break
        if i >= f0:
            yield fr


def decide_mode(rec: dict, *, detect=None) -> None:
    """Hit, miss or dry swing, decided by looking for a ball, never by the user:
    the slow-motion clip if there is one (ball seen leaving the tee), else the LiDAR phone's own
    color video around the swing. Dry is the default: on the LiDAR video the blob tracker's hit or
    miss only stands when the object detector (YOLOX "sports ball", ballflight.confirm_ball) also
    sees a ball near the tracked one before contact. ``detect`` replaces YOLOX (tests).
    Sets rec["mode"] ("hit" | "miss" | "dry"), rec["mode_evidence"] and rec["ball_check"]."""
    from . import ballflight as B
    slow = rec.get("slowmo")
    kind, ev = "none", "no video to look for a ball in: treated as a dry swing"
    if slow:
        kind = slow.get("ball_outcome") or ("hit" if slow.get("exit_speed_mph") is not None else "none")
        ev = f"slow-motion clip: {slow.get('ball_reason') or slow.get('ball_note') or kind}"
    else:
        v = rec.get("video") or {}
        if v.get("path") and v.get("frames"):
            try:
                r = B.ball_hit_in_frames(read_frames(v["path"], *v["frames"]), fps=v.get("fps"))
                kind, ev = r["kind"], f"LiDAR phone video: {r['reason']}"
                if kind in ("hit", "miss"):
                    f0, f1 = v["frames"]
                    fc = _contact_frame(rec, v)
                    chk = B.confirm_ball(read_frames(v["path"], f0, fc), r["xy"][: fc - f0 + 1], detect=detect)
                    rec["ball_check"] = {"tracker": kind, **chk}
                    if not chk["confirmed"]:
                        kind = "none"
                        ev = (f"LiDAR phone video: the blob tracker reported a {r['kind']}, but the ball detector "
                              f"found no ball near it in {chk['frames_checked']} frames before contact: dry swing")
            except FileNotFoundError as e:
                kind, ev = "none", f"ball detector model missing ({e}): treated as a dry swing"
            except Exception as e:
                kind, ev = "none", f"could not check the video for a ball ({type(e).__name__}): treated as a dry swing"
    rec["mode"] = {"hit": "hit", "miss": "miss"}.get(kind, "dry")
    rec["mode_evidence"] = ev
    if "analysis" in rec:
        rec["analysis"]["mode"] = rec["mode"]


def _contact_frame(rec: dict, v: dict) -> int:
    """Recording frame index of this swing's contact (+0.1 s: a toss can still be in the air)."""
    f0, f1 = v["frames"]
    tc = v.get("t_offset", 0.0) + (rec.get("events") or {}).get("contact", 0.0) + 0.1
    ts = np.asarray(v.get("timestamps") or [])
    if not len(ts):
        return f1
    return int(np.clip(f0 + np.searchsorted(ts, tc), f0, f1))


def contact_for(rec: dict) -> dict | None:
    """Ball-contact score: 0 for a miss; for a hit it needs exit speed (the slow-motion camera for now)."""
    if rec.get("mode") == "miss":
        return {"score": 0.0, "smash_factor": None, "launch_deg": None, "missed": True,
                "rule": "swing and miss: contact 0"}
    if rec.get("mode") != "hit":
        return None
    slow = rec.get("slowmo") or {}
    bat = (rec.get("measured") or {}).get("bat_speed_mph")
    src = "measured"
    if bat is None:
        bat, src = (rec.get("analysis") or {}).get("predicted", {}).get("bat_speed"), "predicted from the body"
    c = scoring.contact_score(slow.get("exit_speed_mph"), slow.get("launch_angle_deg"), bat)
    if c:
        c["bat_speed_source"] = src
    return c


def _finish(rec: dict, hitter: dict, slow: dict | None, slow_video, use_gemini: bool, *,
            template_reason: str = "disabled") -> None:
    """Last stage of every analyzed swing: attach the slow-motion result, decide hit vs dry swing,
    score ball contact (hits only), then the coach report."""
    measured = attach_slowmo(rec, slow, slow_video)
    if "analysis" not in rec:
        return
    decide_mode(rec)
    contact = contact_for(rec)
    scoring.with_contact(rec["analysis"], contact)
    if rec["mode"] == "hit" and contact is None:
        rec.setdefault("warnings", []).append("ball hit detected, but no exit speed was measured, so no contact score "
                                              "(needs the slow-motion camera)")
    _coach(rec, hitter, measured, use_gemini, template_reason=template_reason)


def slowmo_contact_time(slow: dict | None, video: str | Path) -> float | None:
    """Wall-clock time the ball left the tee in a slow-motion clip (clip start from its metadata +
    contact frame; on a dry swing, the bat's peak), or None."""
    if not slow:
        return None
    start = capture.video_start_time(video)
    if start is None:
        return None
    if slow.get("contact_frame") is not None:
        return start + slow["contact_frame"] / slow["fps"]
    if slow.get("bat"):
        return start + slow["bat"]["frame"] / slow["fps"]
    return None


def swing_contact_times(lidar: str | Path, recs: list[dict], t_first: float) -> list[float | None]:
    """Wall-clock contact time of each swing of a LiDAR recording (None when the recording's start
    time is unknown, e.g. Stray Scanner)."""
    start = capture.recording_start_time(lidar)
    out = []
    for r in recs:
        if start is None or "events" not in r:
            out.append(None)
            continue
        t0 = (r.get("video") or {}).get("t_offset", (r.get("window") or {}).get("start", t_first))
        out.append(start + (t0 - t_first) + r["events"]["contact"])
    return out


def match_slowmo(lidar, recs: list[dict], clips: list, t_first: float, **slow_kw) -> tuple[dict, dict]:
    """Analyze each clip and match clips to swings (session.match_clips).
    Returns ({swing index: (slow-motion result, clip path)}, matching info for the summary)."""
    from . import session
    results = []
    for c in clips:
        try:
            results.append(analyze_slowmo(c, **slow_kw))
        except Exception as e:
            results.append({"error": f"{type(e).__name__}: {e}"})
    ct = [slowmo_contact_time(r if "error" not in r else None, c) for r, c in zip(results, clips)]
    m = session.match_clips(swing_contact_times(lidar, recs, t_first), ct)
    assigned = {}
    for i, j in m["pairs"]:
        if "error" not in results[j]:
            assigned[i] = (results[j], clips[j])
            recs[i].setdefault("stages", {})["slowmo"] = {"ok": True, "clip": Path(clips[j]).name, "matched_by": m["method"]}
    info = {k: m[k] for k in ("method", "offset_s", "notes")}
    info["pairs"] = [[recs[i].get("id"), Path(clips[j]).name] for i, j in m["pairs"]]
    info["clips"] = [Path(c).name for c in clips]
    return assigned, info


def analyze_recording(lidar: str | Path | None, slowmo: str | Path | None, hitter: dict, *,
                      use_gemini: bool = True, slowmo_fps: float = 240.0, m_per_px: float | None = None,
                      yardstick: str | Path | None = None, tape=None, slowmo_clips: list | None = None,
                      stick_length_m: float = 0.9144, gemini_max_swings: int | None = GEMINI_MAX_SWINGS, **kw) -> dict:
    """Entry point for any recording: lift it ONCE, find every swing in it and analyze each one the
    same way, whatever the recording's length (no clip-vs-session distinction).
    Returns {"recs": [one record per swing], "summary": trends when there are 2+ swings, else None,
    "n_swings"}. ``slowmo``/``slowmo_clips``: slow-motion clip(s), matched to the swings by when the
    ball left the tee if both clocks are known, else by order. Each swing is a hit, a miss or a dry
    swing, decided by looking for a ball (decide_mode). Gemini writes every swing's report and the
    recording summary (outcome counts and stats are computed first and handed to it);
    ``gemini_max_swings`` caps the per-swing calls (free-tier quota)."""
    from . import session
    from .coach import cues, gemini
    clips = list(slowmo_clips or ([slowmo] if slowmo else []))
    slow_kw = dict(fps=slowmo_fps, m_per_px=m_per_px, yardstick=yardstick, tape=tape, stick_length_m=stick_length_m)
    if not lidar:
        rec = analyze_swing(None, clips[0] if clips else None, hitter, use_gemini=use_gemini, slowmo_fps=slowmo_fps,
                            m_per_px=m_per_px, yardstick=yardstick, tape=tape, stick_length_m=stick_length_m)
        return {"recs": [rec], "summary": None, "n_swings": 0}
    try:
        lifted = lift_recording(lidar, height_m=float(hitter["height_m"]), **kw)
    except Exception as e:
        return {"recs": [{"id": Path(lidar).stem, "hitter": hitter.get("name", ""),
                          "stages": {"lidar": {"ok": False, "error": f"{type(e).__name__}: {e}"}}}],
                "summary": None, "n_swings": 0}
    t, joints, q = lifted
    wins = session.find_swings(t, joints, bats=hitter["bats"])
    recs = []
    for k, w in enumerate(wins or [None]):
        rid = Path(lidar).stem if len(wins) <= 1 else f"{Path(lidar).stem}-{k + 1:02d}"
        try:
            ts, js = session.cut(t, joints, w) if w is not None else (t, joints)
            r = analyze_joints(ts, js, hitter, rid, q)
            if w is None:
                r["warnings"].append("no complete swing found (hand speed and shoulder turn never reached a swing's; "
                                     "the recording may start after the swing); analyzed the whole clip, numbers unreliable")
            else:
                r["window"] = asdict(w)
                if w.partial:
                    r["warnings"].append("the recording starts mid-swing: the stance and load were not filmed, so "
                                         "stance, stride and timing numbers for this swing are unreliable")
            r["stages"] = {"lidar": {"ok": True, **r["quality"]}}
        except Exception as e:
            r = {"id": rid, "hitter": hitter.get("name", ""), "window": asdict(w) if w else None,
                 "stages": {"lidar": {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}}}
        r["recording"] = Path(lidar).name
        r["recording_t0"] = float(t[0])          # window times minus this = seconds into the recording's video
        recs.append(r)
    good = [r for r in recs if "analysis" in r]
    assigned, matching = {}, None
    if clips and good:
        by_index, matching = match_slowmo(lidar, good, clips, float(t[0]), **slow_kw)
        assigned = {id(good[i]): v for i, v in by_index.items()}
        if len(good) == 1:
            good[0].setdefault("warnings", []).extend(matching["notes"])
    for k, r in enumerate(good):
        slow, video = assigned.get(id(r), (None, None))
        _finish(r, hitter, slow, video, use_gemini and (gemini_max_swings is None or k < gemini_max_swings),
                template_reason="disabled" if not use_gemini else f"GEMINI_MAX_SWINGS={gemini_max_swings}")
    summary = None
    if len(good) >= 2:
        summary = session.summarize(good, source=str(lidar), quality={k: v for k, v in q.items() if not k.startswith("_")})
        if matching:
            summary["slowmo_matching"] = matching
            summary["notes"] = list(matching["notes"])
        srep, sinfo = (gemini.session_report(summary) if use_gemini
                       else (cues.session_report(summary), {"source": "template"}))
        summary.update(hitter=hitter.get("name", ""), report=srep.model_dump(), report_source=sinfo,
                       warnings=hitter_warnings(hitter), recording=Path(lidar).name)
        v = q.get("_video") or {}
        if v.get("path"):
            summary["recording_video"] = {"source": v["path"], "fps": v.get("fps"), "t0": float(t[0]),
                                          "swings": [{"id": r["id"], **{k: round(r["window"][k] - float(t[0]), 3)
                                                                        for k in ("start", "peak", "end")},
                                                      "partial": r["window"].get("partial", False)}
                                                     for r in good if r.get("window")]}
    return {"recs": recs, "summary": summary, "n_swings": len(wins)}


def _coach(rec: dict, hitter: dict, measured: dict | None, use_gemini: bool, *, template_reason: str = "disabled") -> None:
    """Payload + report (Gemini or template) for an analyzed swing record."""
    from .coach import cues, gemini, payload
    pl = payload.build(rec["analysis"], rec["features"], {"name": hitter.get("name", ""), "bats": hitter["bats"],
                                                          "height_m": float(hitter["height_m"]),
                                                          "level": hitter.get("level", "college")}, measured=measured)
    rep, info = gemini.report(pl) if use_gemini else (cues.report(pl), {"source": "template", "reason": template_reason})
    rec.update(payload=pl, report=rep.model_dump(), report_source=info)
    rec.setdefault("stages", {})["coach"] = {"ok": True, **{k: v for k, v in info.items() if k in ("source", "reason", "seconds", "model")}}


def analyze_session(lidar: str | Path, hitter: dict, *, use_gemini: bool = True, **kw) -> tuple[list[dict], dict]:
    """Kept for older callers: analyze_recording, returned as (records, summary)."""
    out = analyze_recording(lidar, None, hitter, use_gemini=use_gemini, **kw)
    return out["recs"], out["summary"] or {"n_swings": len(out["recs"])}


def save_session(summary: dict, session: str, out_root: Path | None = None, *, render_video: bool = True) -> Path:
    """Summary of a recording with 2+ swings: <out_root>/swings/<folder>/_summary_<recording>.json,
    plus the whole recording as a plain video, _recording_<recording>.mp4 (render_recording), whose
    path goes into summary["recording_video"]["path"] (relative to out_root)."""
    out_root = Path(out_root or config.USERDATA)
    d = out_root / "swings" / session
    d.mkdir(parents=True, exist_ok=True)
    stem = Path(str(summary.get("recording") or "recording")).stem
    rv = summary.get("recording_video")
    if render_video and rv and rv.get("source"):
        try:
            path = render_recording(rv["source"], d / f"_recording_{stem}", fps=rv.get("fps") or 60.0)
            rv["path"] = path.relative_to(out_root).as_posix()
        except Exception as e:
            rv["error"] = f"{type(e).__name__}: {e}"[:300]
    p = d / f"_summary_{stem}.json"
    p.write_text(json.dumps(summary, separators=(",", ":")), encoding="utf-8")
    rebuild_index(out_root)
    return p


def render_recording(source: str | Path, out_base: Path, *, fps: float = 60.0, out_fps: float = 30.0,
                     height: int = 960) -> Path:
    """The whole recording's color video as a plain H.264 MP4 (``out_base`` + .mp4), every
    fps/out_fps-th frame, scaled to ``height`` px: what the viewer's whole-recording view plays.
    Video second s = recording time t0 + s (summary["recording_video"]["t0"])."""
    import cv2
    from .render import open_writer
    step = max(1, int(round(fps / out_fps)))
    writer = path = None
    try:
        for i, fr in enumerate(capture.iter_frames(source)):
            if i % step:
                continue
            h, w = fr.shape[:2]
            size = (int(round(w * height / h / 2)) * 2, height)
            if writer is None:
                writer, path, _ = open_writer(out_base, fps / step, size)
            writer.write(cv2.resize(fr, size, interpolation=cv2.INTER_AREA))
    finally:
        if writer is not None:
            writer.release()
    if path is None:
        raise ValueError(f"no frames in {source}")
    return path


def render_overlay(rec: dict, out_base: Path, out_root: Path) -> None:
    """Annotated video next to the record (render.py). Never raises; adds rec["overlay"] (sync data for
    the viewer, path relative to out_root) and rec["stages"]["render"]."""
    import time
    from . import render
    if "J" not in rec or "series" not in rec:
        return
    t0 = time.time()
    try:
        meta = render.render_swing(rec, out_base)
        rec["overlay"] = {**meta, "path": Path(meta["path"]).relative_to(out_root).as_posix()}
        rec.setdefault("stages", {})["render"] = {"ok": True, "codec": meta["codec"], "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        rec.setdefault("stages", {})["render"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}


def save_result(rec: dict, session: str, out_root: Path | None = None, *, render_video: bool = True) -> Path:
    """Write <out_root>/swings/<folder>/<id>.json (+ the annotated video <id>.mp4) and refresh the index.
    ``out_root`` defaults to the user-data folder (config.USERDATA); ``session`` is the folder,
    normally the hitter's profile id."""
    import time
    out_root = Path(out_root or config.USERDATA)
    d = out_root / "swings" / session
    d.mkdir(parents=True, exist_ok=True)
    rec.setdefault("analyzed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    rec.setdefault("recorded_at", recorded_at(rec))
    if render_video:
        render_overlay(rec, d / str(rec["id"]), out_root)
    p = d / f"{rec['id']}.json"
    p.write_text(json.dumps(rec, separators=(",", ":")), encoding="utf-8")
    rebuild_index(out_root)
    return p


def recorded_at(rec: dict) -> str | None:
    """When the swing was recorded: the recording's start time (Record3D names) + the swing's offset,
    else the recording file's modification time."""
    import time
    src = rec.get("recording") or (rec.get("pair") or {}).get("lidar")
    start = None
    path = None
    for cand in (src, (rec.get("video") or {}).get("path")):
        if cand:
            start = start or capture.recording_start_time(cand)
            path = path or (Path(str(cand).split("%d")[0]).parent if "%d" in str(cand) else Path(cand))
    if start is not None:
        start += float((rec.get("window") or {}).get("start") or 0.0)
    elif path is not None and path.exists():
        start = path.stat().st_mtime
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)) if start else None


def rebuild_index(out_root: Path | None = None) -> Path:
    out_root = Path(out_root or config.USERDATA)
    items, sessions = [], []
    for p in sorted((out_root / "swings").glob("*/_summary_*.json")):
        try:
            sm = json.loads(p.read_text(encoding="utf-8"))
            sessions.append({"path": str(p.relative_to(out_root)).replace("\\", "/"), "session": p.parent.name,
                             "n_swings": sm.get("n_swings"), "headline": (sm.get("report") or {}).get("headline")})
        except Exception:
            pass
    for p in sorted((out_root / "swings").glob("*/*.json")):
        if p.name.startswith("_"):
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        an = r.get("analysis") or {}
        items.append({"path": str(p.relative_to(out_root)).replace("\\", "/"), "session": p.parent.name, "id": r.get("id"),
                      "hitter": r.get("hitter", ""), "overall": (an.get("scores") or {}).get("overall"),
                      "headline": (r.get("report") or {}).get("headline"),
                      "stages": {k: v.get("ok") for k, v in (r.get("stages") or {}).items()}})
    idx = out_root / "index.json"
    idx.write_text(json.dumps({"swings": items, "sessions": sessions}, indent=1), encoding="utf-8")
    return idx
