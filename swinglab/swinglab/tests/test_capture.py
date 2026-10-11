"""Synthetic Stray Scanner folder: load it and back-project a known point."""
import numpy as np

from swinglab import capture


def make_stray(tmp_path, n=3, depth_m=2.0):
    import cv2
    (tmp_path / "depth").mkdir()
    (tmp_path / "confidence").mkdir()
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    for i in range(n):
        rows.append(f"{i/60:.6f}, {i}, 0, 0, 0, 0, 0, 0, 1, 1400, 1400, 960, 720")
        cv2.imwrite(str(tmp_path / "depth" / f"{i:06d}.png"), np.full((192, 256), int(depth_m * 1000), np.uint16))
        cv2.imwrite(str(tmp_path / "confidence" / f"{i:06d}.png"), np.full((192, 256), 2, np.uint8))
    (tmp_path / "odometry.csv").write_text("\n".join(rows), encoding="utf-8")
    return tmp_path


def test_load_and_depth(tmp_path):
    cap = capture.load_stray(make_stray(tmp_path))
    assert cap.n == 3 and abs(cap.fps - 60) < 1e-3
    assert np.allclose(cap.depth(0), 2.0)


def test_backproject_center_and_offset(tmp_path):
    cap = capture.load_stray(make_stray(tmp_path))
    K = cap.K_rgb[0]
    P = capture.backproject(np.array([[960.0, 720.0], [960 + 700.0, 720.0]]), np.array([2.0, 2.0]), K)
    assert np.allclose(P[0], [0, 0, 2])
    assert np.allclose(P[1], [1.0, 0, 2])  # 700 px * 2 m / 1400 px


def test_lift_keypoints_world_frame(tmp_path):
    cap = capture.load_stray(make_stray(tmp_path))
    kps = {i: {"l_hip": (960.0, 720.0, 0.9)} for i in range(cap.n)}
    J = capture.lift_keypoints(cap, kps, rgb_size=(1920, 1440))
    # identity pose, ARKit camera looks down -Z: a point 2 m ahead is at z = -2 in world
    assert np.allclose(J["l_hip"][0], [0, 0, -2.0], atol=1e-6)


def test_check_recording_on_synthetic(tmp_path):
    import cv2, importlib.util, pathlib
    root = make_stray(tmp_path, n=90)
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60, (320, 240))
    for _ in range(90):
        vw.write(np.full((240, 320, 3), 120, np.uint8))
    vw.release()
    spec = importlib.util.spec_from_file_location("chk", pathlib.Path(__file__).parents[1] / "scripts" / "check_recording.py")
    chk = importlib.util.module_from_spec(spec); spec.loader.exec_module(chk)
    info = chk.check_stray(root)
    assert info["depth_fps"] == 60 and info["depth_frames"] == 90 and info["valid_depth_fraction_mid_frame"] == 1.0
    assert info["video"]["ok"] and info["set_SWINGLAB_PHONE_FPS_to"] == 60
    slow = chk.check_video(root / "rgb.mp4", expect_fps=240)
    assert "warning" in slow  # 60 fps file flagged when 240 expected


def test_record3d_portrait_metadata_depth_size_and_per_frame_K(tmp_path):
    """Real iPhone 17 Pro exports are portrait (color 1440x1920, depth dw=192 x dh=256) with
    per-frame [fx, fy, cx, cy]; the loader must use both."""
    import json, zipfile
    import cv2
    lzfse = __import__("pytest").importorskip("lzfse")
    n, dw, dh = 3, 192, 256
    meta = {"w": 1440, "h": 1920, "dw": dw, "dh": dh, "fps": 60, "K": [1337.0, 0, 0, 0, 1337.0, 0, 720.7, 956.8, 1],
            "poses": [[0, 0, 0, 1, 0, 0, 0]] * n, "frameTimestamps": [i / 60 for i in range(n)],
            "perFrameIntrinsicCoeffs": [[1337.0 + i, 1337.0 + i, 720.7, 956.8] for i in range(n)]}
    depth = np.tile(np.linspace(1.0, 3.0, dh, dtype=np.float32)[:, None], (1, dw))   # rows get farther
    with zipfile.ZipFile(tmp_path / "p.r3d", "w") as z:
        z.writestr("metadata", json.dumps(meta))
        for i in range(n):
            z.writestr(f"rgbd/{i}.depth", lzfse.compress(depth.tobytes()))
            z.writestr(f"rgbd/{i}.jpg", cv2.imencode(".jpg", np.zeros((1920, 1440, 3), np.uint8))[1].tobytes())
    cap = capture.load_any(tmp_path / "p.r3d")
    d = cap.depth(0)
    assert d.shape == (256, 192) and np.isclose(d[0, 0], 1.0) and np.isclose(d[-1, 0], 3.0)
    assert cap.K_rgb[2, 0, 0] == 1339.0 and cap.K_rgb[0, 0, 2] == 720.7
    # the bottom of the portrait image samples the bottom rows of the depth map
    assert capture.sample_depth(d, np.array([[720.0, 1915.0]]), cap.rgb_size)[0] > 2.9


from conftest import needs_real_r3d, REAL_R3D  # noqa: E402


@needs_real_r3d
def test_real_record3d_orientation_and_K():
    """First real exports: depth and color share orientation, K fits the color image,
    and the poses put the floor horizontal in ARKit world (normal along +Y)."""
    for path in REAL_R3D:
        cap = capture.load_any(path)
        w, h = cap.rgb_size
        d = cap.depth(cap.n // 2)
        assert (d.shape[1] >= d.shape[0]) == (w >= h), "depth and color orientation differ"
        assert abs(cap.K_rgb[0, 0, 2] - w / 2) < 0.02 * w and abs(cap.K_rgb[0, 1, 2] - h / 2) < 0.02 * h
        assert 40 < cap.fps < 70
        # floor = bottom band of the depth map, lifted to world with the loader's convention
        vv, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        sel = (vv > d.shape[0] - 40) & np.isfinite(d)
        uv = np.c_[(uu[sel] + 0.5) * w / d.shape[1], (vv[sel] + 0.5) * h / d.shape[0]]
        P = capture.backproject(uv, d[sel], cap.K_rgb[cap.n // 2])
        Wp = (capture.quat_to_R(cap.cam_quat[cap.n // 2]) @ (np.diag([1, -1, -1]) @ P.T)).T
        normal = np.linalg.svd(Wp - Wp.mean(0))[2][-1]
        assert abs(normal[1]) > 0.95, (path.name, normal)
