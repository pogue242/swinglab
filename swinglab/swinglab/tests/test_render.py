"""Annotated swing video (item 5): lab swing on a blank background, synthetic phone recording
with an RGB video, and the slow-motion replay with the ball's flight."""
import json

import numpy as np
import pytest

from swinglab import config, pipeline, render
from conftest import needs_models
from test_pipeline import H, W, render_stray, write_ball_video
from test_quality import HITTER, rigid_swing

cv2 = pytest.importorskip("cv2")


def _read(path):
    vc = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = vc.read()
        if not ok:
            return frames
        frames.append(f)


def _near(img, xy, bgr, r=6, tol=60):
    x, y = int(round(xy[0])), int(round(xy[1]))
    patch = img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3).astype(int)
    return bool((np.abs(patch - np.array(bgr)).max(1) < tol).any())


def test_lab_swing_on_blank_background(tmp_path):
    app = config.WEB_DATA / "app.json"
    if not app.exists():
        pytest.skip("no web/data/app.json (run scripts/export_demo.py)")
    rec = json.loads(app.read_text(encoding="utf-8"))["reference"]     # the fastest lab swing
    meta = render.render_swing(rec, tmp_path / "lab")
    frames = _read(meta["path"])
    assert meta["main_frames"] == len(rec["time"]) and meta["frame_times"] == pytest.approx(rec["time"], abs=1e-3)
    assert len(frames) == meta["main_frames"] + round(render.END_CARD_S * meta["fps"])
    w = meta["size"][0] - render.PANEL_W
    first = frames[0]
    assert first.shape[:2] == (render.OUT_H, meta["size"][0])
    assert (first[:, :w].mean(2) < 80).sum() > 500           # dark skeleton on the light canvas
    assert first[:, w + 5:].mean() < 60                       # dark side panel
    assert frames[-1][:, :w].mean() < first[:, :w].mean()     # end card dims the picture


@needs_models
def test_phone_recording_overlay_lines_up_with_the_body(tmp_path):
    t, J = rigid_swing()
    root = tmp_path / "rec"
    kps = render_stray(root, t, J)
    vw = cv2.VideoWriter(str(root / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60, (W, H))
    for i in range(len(t)):                                  # a plain gray "camera" picture
        vw.write(np.full((H, W, 3), 128, np.uint8))
    vw.release()
    rec = pipeline.analyze_swing(root, None, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    assert rec["video"]["frames"] == [0, len(t) - 1] and len(rec["video"]["kp2d"]) == len(t)
    p = pipeline.save_result(rec, "s1", out_root=tmp_path / "web")
    saved = json.loads(p.read_text(encoding="utf-8"))
    ov = saved["overlay"]
    assert saved["stages"]["render"]["ok"] and ov["path"] == f"swings/s1/{rec['id']}.{'webm' if ov['codec'] == 'vp9' else 'mp4'}"
    frames = _read(tmp_path / "web" / ov["path"])
    assert ov["main_frames"] == len(t) and len(frames) == len(t) + round(render.END_CARD_S * ov["fps"])
    assert ov["frame_times"][0] == pytest.approx(0, abs=1e-6) and np.all(np.diff(ov["frame_times"]) > 0)
    s = render.OUT_H / H
    k = int(np.argmin(np.abs(np.array(ov["frame_times"]) - rec["events"]["contact"])))
    hip_mid = 0.5 * (np.array(kps[k]["l_hip"][:2]) + np.array(kps[k]["r_hip"][:2])) * s
    sh_mid = 0.5 * (np.array(kps[k]["l_shoulder"][:2]) + np.array(kps[k]["r_shoulder"][:2])) * s
    assert _near(frames[k], hip_mid, render.PELVIS) and _near(frames[k], sh_mid, render.TORSO)
    assert not _near(frames[k], (20, 20), render.PELVIS)     # and not everywhere


@needs_models
def test_slowmo_replay_draws_the_ball_flight(tmp_path):
    t, J = rigid_swing()
    root = tmp_path / "rec"
    kps = render_stray(root, t, J)
    shot = tmp_path / "shot.mp4"
    write_ball_video(shot)
    rec = pipeline.analyze_swing(root, shot, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    assert rec["slowmo"]["exit_speed_mph"] and rec["slowmo"]["track_px"]
    meta = render.render_swing(rec, tmp_path / "out")        # no rgb.mp4: blank background + slow-mo replay
    frames = _read(meta["path"])
    assert meta["slowmo_s"] > 0
    n_slow = len(frames) - meta["main_frames"] - round(render.END_CARD_S * meta["fps"])
    assert n_slow == round(meta["slowmo_s"] * meta["fps"]) and n_slow > 30
    last = frames[-1][:, :meta["size"][0] - render.PANEL_W]
    red = (last[:, :, 2] > 200) & (last[:, :, 1] < 140) & (last[:, :, 0] < 140)
    assert red.sum() > 100                                    # the projected flight line
