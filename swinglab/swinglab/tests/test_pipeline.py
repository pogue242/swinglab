"""Phone path end to end on synthetic recordings (everything except MediaPipe itself).

A synthetic hitter is placed 3.5 m in front of an ARKit-style camera, rendered
into per-frame depth maps (patches at each joint) and pixel keypoints, written
as a Stray Scanner folder, and run through pipeline.analyze_lidar. The features
must match those computed directly from the original 3D joints.
"""
import csv
import json

import numpy as np
import pytest

from swinglab import config, features, inbox, pipeline, skeleton
from test_kinematics import synthetic_swing
from conftest import needs_models

FX, CX, CY, W, H = 1400.0, 960.0, 720.0, 1920, 1440


def render_stray(root, t, J_zup, bg_m=8.0):
    import cv2
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    names = sorted(J_zup)
    # z-up body coords -> ARKit world (y up), hitter 3.5 m in front of the camera, floor 1 m below it
    Wd = {n: np.c_[v[:, 0], v[:, 2], -v[:, 1]] + np.array([0.0, -1.0, -3.5]) for n, v in J_zup.items()}
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    kps = {}
    sx, sy = 256 / W, 192 / H
    for i, ti in enumerate(t):
        rows.append(f"{ti:.6f}, {i}, 0, 0, 0, 0, 0, 0, 1, {FX}, {FX}, {CX}, {CY}")
        depth = np.full((192, 256), int(bg_m * 1000), np.uint16)
        pts = []
        for n in names:
            X, Y, Z = Wd[n][i] * np.array([1, -1, -1])   # world -> OpenCV-style camera coords
            pts.append((Z, n, FX * X / Z + CX, FX * Y / Z + CY))
        kps[i] = {}
        for Z, n, u, v in sorted(pts, reverse=True):   # far first, near overwrites
            x, y = int(round(u * sx)), int(round(v * sy))
            depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = int(round(Z * 1000))
            kps[i][n] = (u, v, 0.99)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), depth)
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), np.full((192, 256), 2, np.uint8))
    (root / "odometry.csv").write_text("\n".join(rows), encoding="utf-8")
    return kps


@needs_models
def test_lidar_pipeline_matches_ground_truth(tmp_path):
    t, J = synthetic_swing(fps=60, T=2.0, smooth=True)
    root = tmp_path / "rec1"
    kps = render_stray(root, t, J)
    hitter = {"name": "Test", "bats": "R", "height_m": 1.8}
    rec = pipeline.analyze_lidar(root, hitter, keypoints_fn=lambda v: (kps, (W, H)))
    truth = features.compute(skeleton.build(t, J, "R", height_m=1.8, fps=config.PHONE_FPS))
    assert rec["quality"]["valid_joint_fraction"] > 0.99
    # every feature must match ground truth (1 mm depth rounding + occlusion cleanup)
    # timing features: allow ±20 ms (wrists partly hidden for a few frames ≈ half a frame of noise)
    for k in truth:
        atol = 20.0 if k.endswith("_ms") else 1.0
        assert np.isclose(rec["features"][k], truth[k], rtol=0.03, atol=atol, equal_nan=True), (k, rec["features"][k], truth[k])
    assert 0 <= rec["analysis"]["scores"]["overall"] <= 100


def write_ball_video(path, v=30.0, ang=15.0, fps=240, rest=30, n=80, m_per_px=0.004):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))
    r_px = 0.0737 / 2 / m_per_px
    for i in range(n):
        img = np.full((540, 960, 3), 40, np.uint8)
        tt = max(0, i - rest) / fps
        x = 200 + v * np.cos(np.radians(ang)) * tt / m_per_px
        y = 400 - (v * np.sin(np.radians(ang)) * tt - 4.905 * tt * tt) / m_per_px
        if x < 950:
            cv2.circle(img, (int(round(x)), int(round(y))), int(round(r_px)), (240, 240, 240), -1)
        vw.write(img)
    vw.release()


def test_slowmo_exit_speed_from_video(tmp_path):
    p = tmp_path / "shot.mp4"
    write_ball_video(p)
    out = pipeline.analyze_slowmo(p, fps=240)
    assert out["contact_frame"] is not None and abs(out["contact_frame"] - 29) <= 2
    mph = 30.0 * 2.2369363
    assert abs(out["exit_speed_mph"] - mph) / mph < 0.08, out["exit_speed_mph"]
    assert abs(out["launch_angle_deg"] - 15) < 4


def test_inbox_pairs_by_order_and_by_log(tmp_path):
    root = tmp_path / "SwingLab"
    inbox.template(root)
    s = root / "sessions" / "s1"
    for i in range(2):
        d = s / "lidar" / f"rec{i}"
        d.mkdir(parents=True)
        (d / "odometry.csv").write_text("timestamp, frame\n", encoding="utf-8")
        (s / "slowmo").mkdir(exist_ok=True)
        (s / "slowmo" / f"IMG_{i}.MOV").write_bytes(b"x")
    pairs = [p for p in inbox.scan(root) if p.session == "s1"]
    assert [(p.method, p.lidar.endswith("rec0"), p.slowmo.endswith("IMG_0.MOV")) for p in pairs][0] == ("order", True, True)
    with open(s / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["swing", "hitter", "set", "lidar", "slowmo", "contact", "notes"])
        w.writerow(["7", "Wilson", "flaw_arms_only", "rec1", "IMG_0.MOV", "", ""])
    p = [p for p in inbox.scan(root) if p.session == "s1"]
    assert len(p) == 1 and p[0].method == "log" and p[0].hitter == "Wilson" and p[0].lidar.endswith("rec1") and p[0].slowmo.endswith("IMG_0.MOV")


def test_inbox_process_runs_once_and_skips_done(tmp_path):
    root = tmp_path / "SwingLab"
    inbox.template(root)
    s = root / "sessions" / "s1" / "lidar" / "recA"
    s.mkdir(parents=True)
    (s / "odometry.csv").write_text("timestamp, frame\n", encoding="utf-8")
    calls = []
    fake = lambda l, v, h: (calls.append((l, v, h["name"])) or {"recs": [{"stages": {"lidar": {"ok": True}}, "analysis": {"scores": {}}}], "summary": None})
    saved = []
    sv = lambda rec, session: (saved.append(rec) or tmp_path / "out.json")
    inbox.process(root, analyze_recording=fake, save=sv, log=lambda *a: None)
    inbox.process(root, analyze_recording=fake, save=sv, log=lambda *a: None)
    assert len(calls) == 1 and calls[0][2] == "Adrian" and saved[0]["pair"]["method"] == "lidar_only"


@needs_models
def test_long_clip_with_one_swing_is_single_and_short_clip_works(tmp_path):
    """Mode comes from content: a 12 s recording holding ONE swing is a single swing."""
    hitter = {"bats": "R", "height_m": 1.8, "mass_kg": 80, "age": 20}
    t, J = synthetic_swing(fps=30, T=12.0, smooth=True, hip_rate=1000)   # realistic hand speed
    kps = render_stray(tmp_path / "one", t, J)
    out = pipeline.analyze_recording(tmp_path / "one", None, hitter, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    assert len(out["recs"]) == 1 and out["summary"] is None and "analysis" in out["recs"][0]
    assert out["recs"][0]["window"]["index"] == 1
    # a motion too slow to be a swing: still analyzed as one clip, with a warning
    t2, J2 = synthetic_swing(fps=30, T=4.0, smooth=True, hip_rate=500)
    kps2 = render_stray(tmp_path / "slow", t2, J2)
    out2 = pipeline.analyze_recording(tmp_path / "slow", None, hitter, use_gemini=False, keypoints_fn=lambda v: (kps2, (W, H)))
    assert len(out2["recs"]) == 1 and any("no complete swing" in w for w in out2["recs"][0]["warnings"])


def test_save_result_builds_index(tmp_path):
    rec = {"id": "001", "hitter": "Adrian", "analysis": {"scores": {"overall": 61.0}},
           "report": {"headline": "x"}, "stages": {"lidar": {"ok": True}, "slowmo": {"ok": False}}}
    pipeline.save_result(rec, "s1", out_root=tmp_path)
    idx = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert idx["swings"][0]["path"] == "swings/s1/001.json" and idx["swings"][0]["stages"] == {"lidar": True, "slowmo": False}


def stray_to_r3d(stray, out, fps=60.0, zipped=True):
    """Convert a synthetic Stray Scanner folder into a Record3D export (.r3d zip or folder)."""
    import json, zipfile
    import cv2, lzfse
    from swinglab import capture
    cap = capture.load_stray(stray)
    K = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1]], float)
    meta = {"w": W, "h": H, "K": K.T.flatten().tolist(), "fps": fps,
            "poses": [list(q) + list(p) for q, p in zip(cap.cam_quat, cap.cam_pos)]}
    files = {"metadata": json.dumps(meta).encode()}
    for i in range(cap.n):
        d = cv2.imread(str(stray / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        files[f"rgbd/{i}.depth"] = lzfse.compress(d.tobytes())
        files[f"rgbd/{i}.conf"] = lzfse.compress(np.full(d.shape, 2, np.uint8).tobytes())
        files[f"rgbd/{i}.jpg"] = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    if zipped:
        with zipfile.ZipFile(out, "w") as z:
            for k, v in files.items():
                z.writestr(k, v)
    else:
        for k, v in files.items():
            (out / k).parent.mkdir(parents=True, exist_ok=True)
            (out / k).write_bytes(v)
    return out


@needs_models
def test_record3d_export_gives_same_features_as_stray(tmp_path):
    pytest.importorskip("lzfse")
    from swinglab import capture, inbox
    t, J = synthetic_swing(fps=60, T=2.0, smooth=True)
    stray = tmp_path / "rec1"
    kps = render_stray(stray, t, J)
    r3d = stray_to_r3d(stray, tmp_path / "swing.r3d")
    cap = capture.load_any(r3d)
    assert cap.kind == "record3d" and cap.n == len(t) and abs(cap.fps - 60) < 0.5
    assert np.allclose(cap.depth(0)[np.isfinite(capture.load_stray(stray).depth(0))],
                       capture.load_stray(stray).depth(0)[np.isfinite(capture.load_stray(stray).depth(0))], atol=1e-6)
    hitter = {"name": "Test", "bats": "R", "height_m": 1.8}
    a = pipeline.analyze_lidar(stray, hitter, keypoints_fn=lambda v: (kps, (W, H)))
    b = pipeline.analyze_lidar(r3d, hitter, keypoints_fn=lambda v: (kps, (W, H)))
    assert b["quality"]["source_app"] == "record3d"
    for k, v in a["features"].items():
        assert np.isclose(b["features"][k], v, rtol=1e-3, atol=0.05, equal_nan=True), k
    # the inbox finds .r3d files and extracted Record3D folders (not our own extraction folder twice)
    lid = tmp_path / "inbox"
    lid.mkdir()
    import shutil
    shutil.copy(r3d, lid / "a.r3d")
    capture.load_any(lid / "a.r3d")              # creates a_r3d/ next to it
    folder = lid / "b"
    folder.mkdir()
    stray_to_r3d(stray, folder, zipped=False)
    found = sorted(p.name for p in inbox.find_lidar(lid))
    assert found == ["a.r3d", "b"], found


def test_saved_json_has_no_bare_nan(tmp_path):
    """UI request 2026-10-11: bare NaN made browsers refuse whole swing files."""
    rec = {"id": "n1", "analysis": {"scores": {"overall": float("nan")}}, "quality": {"r_pelvis": np.nan,
           "a": [1.0, np.inf, np.float32(2.5)]}}
    p = pipeline.save_result(rec, "s", out_root=tmp_path, render_video=False)
    back = json.loads(p.read_text(encoding="utf-8"))
    assert "NaN" not in p.read_text(encoding="utf-8") and back["quality"] == {"r_pelvis": None, "a": [1.0, None, 2.5]}
