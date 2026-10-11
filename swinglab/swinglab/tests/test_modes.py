"""Hit vs dry swing, decided by looking for a ball; Form and Contact scores kept separate."""
import numpy as np
import pytest

from swinglab import ballflight as B, pipeline, scoring
from conftest import needs_models
from test_pipeline import H, W, render_stray, write_ball_video
from test_quality import HITTER, rigid_swing


def test_contact_score_rules():
    best = scoring.contact_score(84.0, 18.0, 70.0)            # smash 1.2, line drive
    assert best["score"] == 100.0 and best["smash_factor"] == 1.2
    weak = scoring.contact_score(56.0, 40.0, 70.0)            # smash 0.8, high pop-up
    assert weak["score"] == pytest.approx(100 * (0.7 * (0.8 - 0.6) / 0.6 + 0.3 * (45 - 40) / 20), abs=0.1)   # 30.8
    grounder = scoring.contact_score(84.0, -10.0, 70.0)       # square but straight into the ground
    assert grounder["score"] == 70.0
    assert scoring.contact_score(None, 10.0, 70.0) is None and scoring.contact_score(80.0, 10.0, None) is None


def test_overall_is_form_for_a_dry_swing_and_the_mean_for_a_hit():
    a = {"scores": {"form": 60.0, "contact": None, "overall": 60.0}}
    assert scoring.with_contact(a, None)["scores"]["overall"] == 60.0
    assert scoring.with_contact(a, {"score": 80.0})["scores"]["overall"] == 70.0


def test_ball_hit_detected_in_video_frames():
    from bat_synth import render_swing
    hit, _ = render_swing(size=(960, 540), m_per_px=0.0033, sub=8, n=80, omega=27, ball=True)
    dry, _ = render_swing(size=(960, 540), m_per_px=0.0033, sub=8, n=80, omega=27, ball=False, tee=False)
    r = B.ball_hit_in_frames(hit, scale=1.0)
    assert r["hit"] and r["kind"] == "hit" and r["frames_with_ball"] >= 3
    assert not B.ball_hit_in_frames(dry, scale=1.0)["hit"]


def test_a_still_light_is_not_a_hit():
    import cv2
    frames = []
    for i in range(40):
        img = np.full((360, 640, 3), 40, np.uint8)
        cv2.circle(img, (500, 80), 9, (250, 250, 250), -1, cv2.LINE_AA)       # a street light, never moves
        frames.append(img)
    r = B.ball_hit_in_frames(frames, scale=1.0)
    assert not r["hit"] and r["kind"] == "none"


@needs_models
def test_dry_swing_gets_form_only(tmp_path):
    t, J = rigid_swing()
    kps = render_stray(tmp_path / "rec", t, J)
    out = pipeline.analyze_recording(tmp_path / "rec", None, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    rec = out["recs"][0]
    sc = rec["analysis"]["scores"]
    assert rec["mode"] == "dry" and sc["contact"] is None and sc["overall"] == sc["form"]
    assert set(sc) >= {"power", "transfer", "posture", "form"}
    assert rec["payload"]["swing_type"].startswith("dry swing") and "contact" not in rec["payload"]["scores_0_100"]


@needs_models
def test_hit_gets_a_contact_score(tmp_path):
    t, J = rigid_swing()
    kps = render_stray(tmp_path / "rec", t, J)
    shot = tmp_path / "shot.mp4"
    write_ball_video(shot)
    rec = pipeline.analyze_swing(tmp_path / "rec", shot, HITTER, use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    sc = rec["analysis"]["scores"]
    assert rec["mode"] == "hit" and "slow-motion" in rec["mode_evidence"]
    assert sc["contact"] is not None and sc["overall"] == pytest.approx((sc["form"] + sc["contact"]) / 2, abs=0.06)
    assert rec["analysis"]["contact_detail"]["bat_speed_source"] == "predicted from the body"   # no tape in this clip


def test_a_miss_scores_zero_contact():
    rec = {"mode": "miss", "analysis": {"scores": {"form": 64.0, "contact": None, "overall": 64.0}}}
    c = pipeline.contact_for(rec)
    assert c["score"] == 0.0 and c["missed"]
    assert scoring.with_contact(rec["analysis"], c)["scores"]["overall"] == 32.0


def test_scenery_is_not_a_ball():
    """Night test false alarms: a lit window that jitters (the phone moved) and is partly covered by
    the hitter, and a small light drifting slowly with the camera, must not count as a hit or a miss."""
    import cv2
    rng = np.random.default_rng(1)
    frames = []
    for i in range(140):
        img = np.full((480, 360, 3), 25, np.uint8)
        jx, jy = rng.normal(0, 1.2, 2)
        cv2.rectangle(img, (int(240 + jx), int(160 + jy)), (int(262 + jx), int(178 + jy)), (245, 245, 245), -1)
        if 100 <= i <= 125:                                    # the bat passes in front of the window
            cv2.rectangle(img, (240, 150), (252 + (i - 100) % 8, 190), (40, 40, 45), -1)
        cv2.circle(img, (int(60 + 0.5 * i), int(300 + 0.25 * i)), 3, (250, 250, 250), -1)   # slow drift
        frames.append(img)
    r = B.ball_hit_in_frames(frames, scale=1.0, fps=60)
    assert r["kind"] == "none", r


def _fake_detector(ball_at: dict[int, tuple[float, float]]):
    """detect(frame) for confirm_ball: frames are ints here; a ball at ball_at[frame]."""
    def detect(fr):
        return [(B.COCO_SPORTS_BALL, 0.6, *ball_at[fr], 20.0, 20.0)] if fr in ball_at else []
    return detect


def test_confirm_ball_needs_detections_near_the_tracked_ball():
    frames = list(range(30))
    xy = np.full((30, 2), np.nan)
    xy[:, :] = [500.0, 300.0]                                  # tracker: ball resting at (500, 300)
    on_tee = _fake_detector({i: (505.0, 302.0) for i in range(0, 30, 3)})
    assert B.confirm_ball(frames, xy, detect=on_tee)["confirmed"]
    elsewhere = _fake_detector({i: (100.0, 900.0) for i in range(0, 30, 3)})   # a ball on the ground, far away
    r = B.confirm_ball(frames, xy, detect=elsewhere)
    assert not r["confirmed"] and r["frames_with_ball"] == 10 and not r["near_track"]
    once = _fake_detector({3: (505.0, 302.0)})                  # a single detection is not enough
    assert not B.confirm_ball(frames, xy, detect=once)["confirmed"]
    assert not B.confirm_ball(frames, xy, detect=_fake_detector({}))["confirmed"]


def test_tracker_hit_without_a_detected_ball_is_a_dry_swing(monkeypatch):
    """TestSet2: dry swings were labeled hit/miss from bright blobs; the detector must veto them."""
    xy = np.full((40, 2), np.nan)
    xy[5:20] = [400.0, 500.0]
    monkeypatch.setattr(B, "ball_hit_in_frames", lambda frames, fps=None: (list(frames), {
        "kind": "hit", "hit": True, "how": "toss", "reason": "the ball's path broke", "frames_with_ball": 15, "xy": xy})[1])
    monkeypatch.setattr(pipeline, "read_frames", lambda path, f0, f1: iter(range(f0, f1 + 1)))
    rec = {"video": {"path": "x.mp4", "frames": [0, 39], "fps": 60.0, "t_offset": 0.0,
                     "timestamps": [i / 60 for i in range(40)]},
           "events": {"contact": 0.4}, "analysis": {"scores": {}}}
    pipeline.decide_mode(rec, detect=_fake_detector({}))
    assert rec["mode"] == "dry" and "found no ball" in rec["mode_evidence"] and rec["ball_check"]["tracker"] == "hit"
    pipeline.decide_mode(rec, detect=_fake_detector({i: (402.0, 498.0) for i in range(0, 40)}))
    assert rec["mode"] == "hit" and rec["ball_check"]["confirmed"]
