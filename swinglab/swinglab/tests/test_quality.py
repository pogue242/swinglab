"""Recording quality checks (item 4): bone lengths and framing, on synthetic recordings
with injected errors. A rigid synthetic hitter (every bone a fixed length) is rendered
into a Stray Scanner folder; errors are injected into the 2D keypoints."""
import numpy as np

from swinglab import pipeline, quality as Q
from conftest import needs_models
from test_kinematics import synthetic_swing
from test_pipeline import H, W, render_stray

HITTER = {"name": "Test", "bats": "R", "height_m": 1.8, "mass_kg": 80, "age": 20}


def rigid_swing(fps=60, T=2.0):
    """synthetic_swing with knees and ankles turning with the hips, so thigh and shin keep their length
    (the original keeps knees fixed while hips rotate: its thigh changes 25% by construction)."""
    t, J = synthetic_swing(fps=fps, T=T, smooth=True)
    yaw = np.arctan2(J["l_hip"][:, 1], J["l_hip"][:, 0])

    def ring(r, z, a0):
        return np.c_[r * np.cos(a0 + yaw), r * np.sin(a0 + yaw), np.full(len(t), z)]
    J.update(l_knee=ring(0.15, 0.5, 0.0), r_knee=ring(0.15, 0.5, np.pi),
             l_ankle=ring(0.15, 0.08, 0.0), r_ankle=ring(0.15, 0.08, np.pi))
    return t, J


def _rendered(tmp_path, name="rec"):
    t, J = rigid_swing()
    kps = render_stray(tmp_path / name, t, J)
    return tmp_path / name, kps


@needs_models
def test_clean_rigid_hitter_passes_bone_check(tmp_path):
    root, kps = _rendered(tmp_path)
    rec = pipeline.analyze_lidar(root, HITTER, keypoints_fn=lambda v: (kps, (W, H)))
    b = rec["quality"]["bones"]
    assert b["ok"] and b["worst_variation"] < 0.03, b
    assert set(b["bones"]) == set(Q.BONES) and abs(b["bones"]["lead_thigh"]["median_m"] - 0.4) < 0.01
    assert not any("length varies" in w for w in rec["warnings"])


@needs_models
def test_injected_depth_error_is_caught_and_named(tmp_path):
    """The lead knee keypoint slips 60 px off the leg for 0.5 s (a pose error): its depth
    comes from the background, the thigh and shin stretch, and the check names the bone."""
    root, kps = _rendered(tmp_path)
    for i in range(40, 70):
        u, v, vis = kps[i]["l_knee"]
        kps[i]["l_knee"] = (u + 60, v, vis)
    rec = pipeline.analyze_lidar(root, HITTER, keypoints_fn=lambda v: (kps, (W, H)), bone_fallback=False)
    b = rec["quality"]["bones"]
    assert not b["ok"] and b["worst_bone"] in ("lead_thigh", "lead_shin") and b["worst_variation"] > 0.10, b
    assert any(w.startswith(b["worst_bone"].replace("_", " ")) for w in rec["warnings"]), rec["warnings"]
    # with the bone-length fallback (default) the knee is put back at thigh length from the hip
    rec2 = pipeline.analyze_lidar(root, HITTER, keypoints_fn=lambda v: (kps, (W, H)))
    assert rec2["quality"]["bones"]["bones"]["lead_thigh"]["variation"] < 0.05
    assert 0.2 < rec2["quality"]["placed_by_bone_length"]["l_knee"] < 0.3   # 30 of 120 frames


def test_bone_check_on_a_skeleton_directly():
    from swinglab import skeleton
    t, J = rigid_swing(fps=30)
    sk = skeleton.build(t, J, "R", fps=30)
    assert Q.bone_lengths(sk)["ok"]
    J["l_wrist"] = J["l_wrist"] + np.random.default_rng(0).normal(0, 0.04, J["l_wrist"].shape)  # 4 cm jitter
    b = Q.bone_lengths(skeleton.build(t, J, "R", fps=30))
    assert not b["ok"] and b["worst_bone"] == "lead_forearm"   # right-handed hitter: left arm is the lead arm


def _kps(n=100, w=W, h=H):
    base = {"l_shoulder": (900, 500), "r_shoulder": (1000, 500), "l_elbow": (880, 650), "r_elbow": (1020, 650),
            "l_wrist": (870, 780), "r_wrist": (1030, 780), "l_hip": (920, 800), "r_hip": (980, 800),
            "l_knee": (910, 1000), "r_knee": (990, 1000), "l_ankle": (905, 1250), "r_ankle": (995, 1250)}
    return {i: {k: (u, v, 0.95) for k, (u, v) in base.items()} for i in range(n)}


def test_framing_clean():
    fr = Q.framing(_kps(), (W, H), 100)
    assert fr == {"person_found": 1.0, "all_joints_visible": 1.0, "feet_in_frame": 1.0, "hands_in_frame": 1.0,
                  "min_vis": 0.5, "ok": True}
    assert Q.framing_warnings(fr) == []


def test_framing_flags_cut_off_feet_hidden_wrists_and_lost_person():
    k = _kps()
    for i in range(30):                           # feet below the bottom edge in 30 frames
        k[i]["l_ankle"] = (905, H + 40, 0.3)
    for i in range(30, 90):                        # lead wrist blurred in 60 frames
        u, v, _ = k[i]["l_wrist"]
        k[i]["l_wrist"] = (u, v, 0.2)
    for i in range(90, 100):                       # nobody found in the last 10 frames
        del k[i]
    fr = Q.framing(k, (W, H), 100)
    assert fr["person_found"] == 0.9 and fr["feet_in_frame"] == 0.6 and fr["hands_in_frame"] == 0.9
    assert fr["all_joints_visible"] == 0.0 and not fr["ok"]
    ws = " | ".join(Q.framing_warnings(fr))
    assert "feet in the picture in only 60%" in ws and "all 13 joints clearly visible in only 0%" in ws


@needs_models
def test_framing_reaches_the_record_and_warnings(tmp_path):
    root, kps = _rendered(tmp_path)
    for i in range(0, len(kps), 2):                # hands raised out of the top of the frame in half the frames
        for j in ("l_wrist", "r_wrist"):
            u, v, vis = kps[i][j]
            kps[i][j] = (u, -30.0, vis)
    rec = pipeline.analyze_lidar(root, HITTER, keypoints_fn=lambda v: (kps, (W, H)))
    fr = rec["quality"]["framing"]
    assert fr["feet_in_frame"] == 1.0 and abs(fr["hands_in_frame"] - 0.5) < 0.02 and not fr["ok"]
    assert any("hands in the picture" in w for w in rec["warnings"])


@needs_models
def test_mostly_missing_joint_gives_a_plain_error_not_a_crash(tmp_path):
    """Real night recording: the lead wrist had depth in 1% of frames and smoothing crashed with
    'array must not contain infs or NaNs'. Now the stage fails with a readable reason."""
    root, kps = _rendered(tmp_path)
    for i in range(len(kps)):
        if i % 10:
            del kps[i]["l_wrist"]              # no 2D point at all in 90% of frames: nothing to rebuild from
    rec = pipeline.analyze_swing(root, None, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    err = rec["stages"]["lidar"]["error"]
    assert not rec["stages"]["lidar"]["ok"] and "lead wrist" in err and "of frames" in err, err


def _remove_depth(root, kps, joints, frames, H_rgb=H, W_rgb=W):
    """Simulate the night recording: no depth (confidence 0) under these joints in these frames."""
    import cv2
    for i in frames:
        cp = root / "confidence" / f"{i:06d}.png"
        c = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
        for n in joints:
            u, v, _ = kps[i][n]
            x, y = int(round(u * 256 / W_rgb)), int(round(v * 192 / H_rgb))
            c[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = 0
        cv2.imwrite(str(cp), c)


@needs_models
def test_bone_fallback_recovers_arms_without_depth(tmp_path):
    """No depth under both elbows and wrists for 1.25 s through the swing (as in the night test): with the
    fallback the arms are rebuilt from 2D + bone length and the features match ground truth;
    without it the wrists are missing and the stage refuses."""
    from swinglab import config, features, skeleton
    t, J = rigid_swing()
    root = tmp_path / "rec"
    kps = render_stray(root, t, J)
    gone = list(range(40, 115))   # the whole swing (0.67-1.9 s) has no arm depth
    _remove_depth(root, kps, ["l_elbow", "l_wrist", "r_elbow", "r_wrist"], gone)
    rec = pipeline.analyze_swing(root, None, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    q = rec["quality"]
    assert q["placed_by_bone_length"]["l_wrist"] > 0.5 and q["bone_lengths_m"]["l_elbow-l_wrist"][1] == "depth"
    from swinglab import kinematics as K
    sk = skeleton.build(t, J, "R", height_m=1.8, fps=config.PHONE_FPS)
    truth, ev = features.compute(sk), K.detect_events(sk)
    # events within one frame: a few cm of wrist error can move the hand-speed peak by one 33 ms frame,
    # and this synthetic hip turns ~19 deg per frame, so angles sampled AT an event are compared separately
    for e in ("load", "foot_plant", "contact", "peak_hand"):
        assert abs(rec["events"][e] - ev[e]) <= 1.01 / config.PHONE_FPS, (e, rec["events"][e], ev[e])
    at_event = {"pelvis_yaw_fp", "torso_yaw_fp", "separation_fp", "pelvis_rotation_range", "lead_knee_flex_fp"}
    for k in truth:
        if k in at_event:
            continue
        atol = 20.0 if k.endswith("_ms") else 1.5
        assert np.isclose(rec["features"][k], truth[k], rtol=0.05, atol=atol, equal_nan=True), (k, rec["features"][k], truth[k])
    # arms rebuilt from 2D + bone length only, through the whole swing: peak hand speed within 10% (measured 6%)
    assert abs(rec["features"]["hand_speed_peak"] - truth["hand_speed_peak"]) / truth["hand_speed_peak"] < 0.10
    assert any("placed from bone length" in w for w in rec["warnings"])
    off = pipeline.analyze_swing(root, None, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)), bone_fallback=False)
    assert not off["stages"]["lidar"]["ok"] and "wrist" in off["stages"]["lidar"]["error"]
