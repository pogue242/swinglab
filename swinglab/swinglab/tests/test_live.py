"""Live capture (swinglab/live.py) without a phone: the buffer, the intrinsics fix and the saved recording."""
import numpy as np

from swinglab import capture, live


def _frames(buf, n=40, fps=30.0):
    for i in range(n):
        rgb = np.full((192, 144, 3), 40 + i, np.uint8)
        depth = np.full((64, 48), 2.0 + 0.01 * i, np.float32)
        K = np.array([[160.0, 0, 72.0], [0, 160.0, 96.0], [0, 0, 1]])
        buf.add(i / fps, rgb, depth, None, K, [0.1 * i, 1.0, -2.0, 0, 0, 0, 1])


def test_buffer_keeps_only_the_last_seconds():
    buf = live.LiveBuffer(seconds=0.5)
    _frames(buf, n=40)
    assert len(buf.frames) <= 17 and abs(buf.latest_t() - 39 / 30) < 1e-9
    assert all(0.8 <= f.t <= 1.0 for f in buf.window(0.8, 1.0))


def test_intrinsics_are_scaled_to_the_color_frame():
    K = np.array([[200.0, 0, 128.0], [0, 200.0, 96.0], [0, 0, 1]])        # centered on a 256-wide depth image
    Kc = live.rgb_intrinsics(K, rgb_w=1440, depth_w=256)
    assert abs(Kc[0, 2] - 720.0) < 1e-6 and abs(Kc[0, 0] - 200.0 * 1440 / 256) < 1e-6
    K2 = np.array([[1300.0, 0, 720.0], [0, 1300.0, 960.0], [0, 0, 1]])   # already the color frame's
    assert np.allclose(live.rgb_intrinsics(K2, 1440, 256), K2)


def test_saved_clip_loads_as_a_recording(tmp_path):
    buf = live.LiveBuffer(seconds=10)
    _frames(buf, n=40)
    out = live.save_stray(list(buf.frames), tmp_path / "live 1")
    assert capture.is_recording(out) and not (tmp_path / "live 1.part").exists()
    cap = capture.load_any(out)
    assert cap.n == 40 and abs(cap.fps - 30.0) < 0.5
    assert abs(float(np.nanmedian(cap.depth(10))) - 2.10) < 0.01                       # depth survives in mm
    assert np.allclose(cap.cam_pos[5], [0.5, 1.0, -2.0]) and abs(cap.K_rgb[0][0, 0] - 160.0) < 1e-6
