"""Live capture (swinglab/live.py) without a phone: the intrinsics fix, the .r3d session writer, the motion score
and a whole LiveSession (fake stream) that ends by itself once the hitter stands still."""
import time

import numpy as np
import pytest

from swinglab import capture, live

pytest.importorskip("lzfse")


def _frame(i: int, fps: float = 30.0, moving: bool = False):
    rgb = np.full((192, 144, 3), 40, np.uint8)
    if moving:                                    # a bright block that jumps around: lots of changed pixels
        y = (i * 37) % 150
        rgb[y:y + 40, 20:120] = 250
    depth = np.full((64, 48), 2.0 + 0.01 * (i % 50), np.float32)
    conf = np.full((64, 48), 2, np.uint8)
    K = np.array([[160.0, 0, 72.0], [0, 160.0, 96.0], [0, 0, 1]])
    return i / fps, rgb, depth, conf, K, [0.1 * i, 1.0, -2.0, 0, 0, 0, 1]


def test_intrinsics_are_scaled_to_the_color_frame():
    K = np.array([[200.0, 0, 128.0], [0, 200.0, 96.0], [0, 0, 1]])        # centered on a 256-wide depth image
    Kc = live.rgb_intrinsics(K, rgb_w=1440, depth_w=256)
    assert abs(Kc[0, 2] - 720.0) < 1e-6 and abs(Kc[0, 0] - 200.0 * 1440 / 256) < 1e-6
    K2 = np.array([[1300.0, 0, 720.0], [0, 1300.0, 960.0], [0, 0, 1]])   # already the color frame's
    assert np.allclose(live.rgb_intrinsics(K2, 1440, 256), K2)


def test_session_writer_makes_an_r3d_that_loads(tmp_path):
    final = tmp_path / "drop" / "2026-10-11--10-15-02.r3d"
    w = live.R3dWriter(tmp_path / "staging" / "s1", final, queue_frames=64)   # no drops while the thread imports cv2
    for i in range(40):
        assert w.add(*_frame(i))
    info = w.finish()
    assert info["frames"] == 40 and info["dropped"] == 0 and final.exists()
    assert not (tmp_path / "staging" / "s1").exists() and not final.with_name(final.name + ".part").exists()
    assert capture.is_recording(final) and capture.recording_start_time(final) is not None
    assert live.is_live_recording(final) and not live.is_live_recording(tmp_path / "missing.r3d")
    cap = capture.load_any(final)
    assert cap.n == 40 and abs(cap.fps - 30.0) < 0.5
    assert abs(float(np.nanmedian(cap.depth(10))) - 2.10) < 1e-4                      # float32 meters survive
    assert np.allclose(cap.cam_pos[5], [0.5, 1.0, -2.0]) and abs(cap.K_rgb[0][0, 0] - 160.0) < 1e-6


def test_session_writer_refuses_too_short_a_session(tmp_path):
    w = live.R3dWriter(tmp_path / "staging" / "s2", tmp_path / "drop" / "x.r3d")
    for i in range(3):
        w.add(*_frame(i))
    with pytest.raises(ValueError):
        w.finish()
    assert not (tmp_path / "staging" / "s2").exists() and not (tmp_path / "drop" / "x.r3d").exists()


def test_motion_score_is_zero_when_still_and_high_when_moving():
    a = live.small_gray(_frame(0)[1])
    assert live.motion_score(None, a) == 0.0 and live.motion_score(a, a) == 0.0
    assert live.motion_score(live.small_gray(_frame(1, moving=True)[1]), live.small_gray(_frame(2, moving=True)[1])) > 0.05


def test_stream_handles_the_newest_frame_and_never_falls_behind(monkeypatch):
    """A slow handler (50 ms a frame) on a 60 fps phone: frames in between are skipped, so what is handled stays
    within a few frames of what the phone just sent (it used to queue up: 10+ s behind on a real phone)."""
    import sys
    import threading
    from types import ModuleType, SimpleNamespace

    class FakeRecord3DStream:
        @staticmethod
        def get_connected_devices():
            return ["iphone"]

        def __init__(self):
            self.i, self.is_on = 0, True
            self.on_new_frame = self.on_stream_stopped = None

        def connect(self, dev):
            def run():
                while self.is_on:
                    self.i += 1
                    self.on_new_frame()
                    time.sleep(1 / 60)
            threading.Thread(target=run, daemon=True).start()

        def get_rgb_frame(self):
            return np.full((192, 144, 3), 40, np.uint8)

        def get_depth_frame(self):
            return np.full((64, 48), 2.0, np.float32)

        def get_confidence_frame(self):
            return np.full((64, 48), 2, np.uint8)

        def get_intrinsic_mat(self):
            return SimpleNamespace(fx=160.0, fy=160.0, tx=72.0, ty=96.0)

        def get_camera_pose(self):
            return SimpleNamespace(qx=0.0, qy=0.0, qz=0.0, qw=1.0, tx=0.0, ty=0.0, tz=0.0)

    fake = ModuleType("record3d")
    fake.Record3DStream = FakeRecord3DStream
    monkeypatch.setitem(sys.modules, "record3d", fake)
    lags = []

    class SlowSink:
        def add(self, t, rgb, depth, conf, K, pose):
            sent_before = s.i
            time.sleep(0.05)
            lags.append(s.i - sent_before)     # frames the phone sent while this one was handled

    s = live.stream(SlowSink())
    try:
        time.sleep(1.5)
        sent, handled = s.i, len(lags)
    finally:
        s.is_on = False
        s.on_stream_stopped()
    assert sent >= 60                          # the phone was never held up by the slow handler
    assert 10 <= handled < sent / 2            # frames in between were skipped, not queued
    assert max(lags) <= 6                      # each handled frame was the newest (about 0.1 s at most)


def _fake_connect(sink, on_stopped=None, still_after_s: float = 1.0, fps: float = 30.0):
    """A stand-in for live.stream: a thread feeds frames (moving for still_after_s, then still) in real time."""
    stop = {"done": False}

    def run():
        i, t0 = 0, time.time()
        while not stop["done"]:
            t = time.time()
            _, rgb, depth, conf, K, pose = _frame(i, moving=t - t0 < still_after_s)
            sink.add(t, rgb, depth, conf, K, pose)
            i += 1
            time.sleep(1 / fps)

    import threading
    threading.Thread(target=run, daemon=True).start()
    return stop


def test_live_session_ends_by_itself_after_stillness(tmp_path):
    saved = []
    streams = []

    def connect(sink, on_stopped=None):
        streams.append(_fake_connect(sink, on_stopped))
        return streams[-1]

    s = live.LiveSession(tmp_path / "staging", idle_s=1.0, max_s=20.0, connect=connect, on_saved=saved.append,
                         log=lambda *a: None)
    try:
        s.connect()
        time.sleep(0.3)
        assert s.preview_jpeg() is not None and s.status()["connected"]
        st = s.start("Angela", tmp_path / "drop" / "Angela")
        assert st["state"] == "recording"
        with pytest.raises(RuntimeError):
            s.start("Angela", tmp_path / "drop" / "Angela")       # one session at a time
        deadline = time.time() + 10
        while s.state != "saved" and time.time() < deadline:
            time.sleep(0.1)
        assert s.state == "saved", s.status()
        info = saved[0]
        assert info["reason"].startswith("still") and info["hitter"] == "Angela" and info["frames"] >= 20
        cap = capture.load_any(info["path"])
        assert cap.n == info["frames"]
    finally:
        for st_ in streams:
            st_["done"] = True


def test_live_session_stop_saves_right_away(tmp_path):
    saved = []
    streams = []

    def connect(sink, on_stopped=None):
        streams.append(_fake_connect(sink, on_stopped, still_after_s=100))
        return streams[-1]

    s = live.LiveSession(tmp_path / "staging", idle_s=5.0, tail_s=1.0, connect=connect, on_saved=saved.append,
                         log=lambda *a: None)
    try:
        s.start("Angela", tmp_path / "drop")
        time.sleep(1.0)
        st = s.stop()
        assert st["state"] == "recording" and st["finishing"]      # keeps recording: the video arrives late
        frames_at_stop = st["frames"] or 0
        deadline = time.time() + 10
        while s.state != "saved" and time.time() < deadline:
            time.sleep(0.1)
        assert s.state == "saved" and saved[0]["reason"] == "stopped by user"
        assert saved[0]["frames"] >= frames_at_stop + 20 and saved[0]["seconds"] >= 1.8
    finally:
        for st_ in streams:
            st_["done"] = True
