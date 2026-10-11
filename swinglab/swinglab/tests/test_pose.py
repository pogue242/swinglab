"""MediaPipe input paths (item 3): video files and Record3D-style JPG sequences.

Synthetic tests check frame reading (no person needed). Real-data tests run only
when a Record3D recording sits in data/captures and the pose model is downloaded.
"""
import numpy as np
import pytest

from swinglab import capture
from conftest import POSE_MODEL, REAL_R3D, needs_pose_model, needs_real_r3d


def _write_sequence(folder, n=12, size=(64, 48)):
    import cv2
    folder.mkdir(parents=True)
    for i in range(n):
        img = np.zeros((size[1], size[0], 3), np.uint8)
        img[:, :, 1] = i * 20          # frame number encoded in green
        cv2.imwrite(str(folder / f"{i}.jpg"), img)
    return folder / "%d.jpg"


def test_jpg_sequence_is_read_in_order_from_a_path_with_spaces(tmp_path):
    pat = _write_sequence(tmp_path / "hack knights" / "rgbd")
    frames = list(capture.iter_frames(pat))
    assert len(frames) == 12 and frames[0].shape == (48, 64, 3)
    greens = [int(round(f[:, :, 1].mean())) for f in frames]
    assert all(abs(g - 20 * i) <= 2 for i, g in enumerate(greens)), greens


def test_record3d_video_is_its_jpg_sequence(tmp_path):
    pytest.importorskip("lzfse")
    from test_pipeline import render_stray, stray_to_r3d
    from test_kinematics import synthetic_swing
    t, J = synthetic_swing(fps=60, T=0.3)
    render_stray(tmp_path / "s", t, J)
    cap = capture.load_any(stray_to_r3d(tmp_path / "s", tmp_path / "x y.r3d"))
    assert str(cap.video).endswith("%d.jpg") and len(list(capture.iter_frames(cap.video))) == cap.n


def test_mediapipe_timestamps_use_the_given_fps(monkeypatch, tmp_path):
    """OpenCV says 25 fps for any JPG sequence; the true rate must reach MediaPipe."""
    mp = pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import vision
    pat = _write_sequence(tmp_path / "rgbd", n=4)
    stamps = []

    class Fake:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def detect_for_video(self, img, ts):
            stamps.append(ts)
            return type("R", (), {"pose_landmarks": []})()
    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", staticmethod(lambda o: Fake()))
    capture.mediapipe_keypoints(pat, "unused.task", fps=60)
    assert stamps == [0, 16, 33, 50]


@needs_real_r3d
@needs_pose_model
def test_real_record3d_pose_and_jpg_roundtrip(tmp_path):
    """Real phone frames: a person is found in nearly every frame, and running pose on
    an MP4 of the frames or on JPGs cut from that MP4 gives the same keypoints."""
    import cv2
    cap = capture.load_any(REAL_R3D[-1])
    frames = [f for _, f in zip(range(40), capture.iter_frames(cap.video))]
    mp4 = tmp_path / "phone video.mp4"
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), cap.fps, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    (tmp_path / "cut frames").mkdir()
    for i, f in enumerate(capture.iter_frames(mp4)):
        cv2.imwrite(str(tmp_path / "cut frames" / f"{i}.jpg"), f)
    a, sa = capture.mediapipe_keypoints(mp4, POSE_MODEL, fps=cap.fps)
    b, sb = capture.mediapipe_keypoints(tmp_path / "cut frames" / "%d.jpg", POSE_MODEL, fps=cap.fps)
    assert sa == sb == cap.rgb_size and len(a) >= 36 and len(b) >= 36
    common = sorted(set(a) & set(b))
    d = [np.hypot(a[k][n][0] - b[k][n][0], a[k][n][1] - b[k][n][1]) for k in common for n in ("l_hip", "r_hip", "l_shoulder", "r_shoulder")]
    assert np.median(d) < 5, np.median(d)
