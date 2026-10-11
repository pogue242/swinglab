"""Bat tracking from two tape bands in 240 fps slow motion (item 6), on synthetic clips rendered in
3D with perspective and motion blur (tests/bat_synth.py). Target: bat speed within 5%."""
import numpy as np
import pytest

from swinglab import ballflight as B, pipeline
from bat_synth import render_swing, write, yardstick_frames

cv2 = pytest.importorskip("cv2")
FAST = dict(size=(960, 540), m_per_px=0.0033, sub=8, n=90)   # smaller frames keep the suite quick


@pytest.mark.parametrize("kw", [dict(omega=27), dict(omega=18), dict(omega=32, vba_deg=35, attack_deg=15),
                                dict(omega=27, cam_dist=6.0, m_per_px=0.0044)])
def test_bat_speed_at_contact_within_5_percent(kw):
    frames, tr = render_swing(**{**FAST, **kw})
    trk = B.track_bands(frames, m_per_px=tr["m_per_px"])
    r = B.bat_at(trk, tr["contact_frame"], tr["fps"], tr["m_per_px"], direction=1)
    assert r is not None
    assert abs(r["bat_speed_mph"] / tr["bat_speed_mph"] - 1) < 0.05, (r, tr["bat_speed_mph"])
    assert abs(r["tip_speed_mph"] / tr["tip_speed_mph"] - 1) < 0.05
    assert abs(r["attack_angle_deg"] - tr["attack_angle_deg"]) < 2.0
    assert abs(r["bat_angle_deg"] - tr["bat_angle_image_deg"]) < 5.0


def test_bands_tracked_through_the_foreshortened_contact_zone():
    """Near contact the bat points at the camera: the bands look less than half as far apart as
    when the bat is side-on. The tracker must not drop them there."""
    frames, tr = render_swing(**FAST, omega=27)
    assert tr["band_spacing_px_at_contact"] < 0.5 * tr["band_spacing_px_full"]
    trk = B.track_bands(frames, m_per_px=tr["m_per_px"])
    c = tr["contact_frame"]
    assert np.isfinite(trk["tip"][c - 8:c + 9, 0]).all() and np.isfinite(trk["hand"][c - 8:c + 9, 0]).all()


@pytest.mark.parametrize("mpp,tilt", [(0.0025, 0), (0.0025, 8), (0.0035, 3), (0.0018, -5)])
def test_yardstick_scale_within_2_percent(mpp, tilt):
    frames, _ = yardstick_frames(mpp, tilt_deg=tilt)
    s, info = B.scale_from_yardstick(frames)
    assert s is not None and abs(s / mpp - 1) < 0.02, info


def test_yardstick_missing_is_reported():
    s, info = B.scale_from_yardstick([np.full((540, 960, 3), 120, np.uint8)] * 3)
    assert s is None and "reason" in info


def test_ball_detector_ignores_a_tape_band():
    img = np.full((540, 960, 3), 90, np.uint8)
    cv2.rectangle(img, (300, 200), (320, 226), (235, 235, 235), -1)          # 5 cm band seen side-on
    assert B.detect_ball(img) is None
    cv2.circle(img, (600, 300), 12, (240, 240, 240), -1, cv2.LINE_AA)
    x, y, r = B.detect_ball(img)
    assert abs(x - 600) < 2 and abs(y - 300) < 2


@pytest.mark.parametrize("exit_mps,launch", [(35.0, 12.0), (40.0, -5.0)])
def test_slowmo_file_with_yardstick_gives_bat_and_ball(tmp_path, exit_mps, launch):
    """Whole path: a slow-motion file (taped bat hits the ball off the tee) + a yardstick clip.
    The low line drive (-5 deg) flies right next to the bat: the ball must not be taken for tape."""
    frames, tr = render_swing(**FAST, omega=27, ball=True, exit_speed=exit_mps, launch_deg=launch)
    video = write(frames, tmp_path / "swing.mp4")
    yframes, _ = yardstick_frames(tr["m_per_px"], size=FAST["size"], tilt_deg=2)
    yclip = write(yframes, tmp_path / "yardstick.mp4")
    out = pipeline.analyze_slowmo(video, fps=240, yardstick=yclip)
    assert out["scale_source"] == "yardstick" and abs(out["m_per_px"] / tr["m_per_px"] - 1) < 0.02
    assert out["contact_frame"] == tr["contact_frame"], out["contact_frame"]
    assert out["bat"]["contact_from"] == "ball leaving the tee"
    assert abs(out["bat_speed_mph"] / tr["bat_speed_mph"] - 1) < 0.05, (out["bat"], tr["bat_speed_mph"])
    assert abs(out["bat"]["attack_angle_deg"] - tr["attack_angle_deg"]) < 2.5
    assert abs(out["exit_speed_mph"] / (exit_mps * 2.2369363) - 1) < 0.03 and abs(out["launch_angle_deg"] - launch) < 1.5
    assert len(out["bat_track_px"]) > 20


def test_dry_swing_uses_peak_bat_speed(tmp_path):
    frames, tr = render_swing(**FAST, omega=27, ball=False, tee=False)
    out = pipeline.analyze_slowmo(write(frames, tmp_path / "dry.mp4"), fps=240, m_per_px=tr["m_per_px"])
    assert out["contact_frame"] is None and out["bat"]["contact_from"].startswith("peak bat speed")
    assert abs(out["bat_speed_mph"] / tr["bat_speed_mph"] - 1) < 0.05   # constant-speed arc: same speed at its peak


def test_inbox_finds_the_yardstick_and_tape_positions(tmp_path):
    import csv
    import os
    import time
    from swinglab import inbox
    root = tmp_path / "SwingLab"
    inbox.template(root)
    (root / "hitters.csv").write_text("name,bats,height_m,mass_kg,age,level,bat_length_m,tip_band_m,hand_band_m\n"
                                      "Adrian,R,1.78,75,20,college,0.81,0.785,0.2\n", encoding="utf-8")
    s = root / "sessions" / "s1"
    (s / "slowmo").mkdir(parents=True)
    t0 = time.time() - 100
    for k, name in enumerate(["yardstick_setup1.MOV", "IMG_1.MOV", "IMG_2.MOV"]):
        f = s / "slowmo" / name
        f.write_bytes(b"x")
        os.utime(f, (t0 + k, t0 + k))
    calls = []
    fake = lambda l, v, h, **kw: (calls.append((v, kw)) or {"recs": [{"stages": {}, "analysis": {"scores": {}}}], "summary": None})  # noqa: E731
    inbox.process(root, analyze_recording=fake, save=lambda r, sess: tmp_path / "x.json", log=lambda *a: None)
    mine = [c for c in calls if "s1" in c[0]]
    assert [p.split("\\")[-1].split("/")[-1] for p, _ in mine] == ["IMG_1.MOV", "IMG_2.MOV"]   # not the yardstick
    assert all(kw["yardstick"].endswith("yardstick_setup1.MOV") for _, kw in mine)
    assert mine[0][1]["tape"].bat_length_m == 0.81 and mine[0][1]["tape"].hand_band_m == 0.2
    # a log.csv calibration row names the yardstick even without "yardstick" in the file name
    (s / "slowmo" / "yardstick_setup1.MOV").rename(s / "slowmo" / "IMG_0.MOV")
    with open(s / "log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["swing", "hitter", "set", "lidar", "slowmo", "contact", "notes"])
        w.writerow(["001", "Adrian", "calibration", "", "IMG_0.MOV", "", "yardstick"])
        w.writerow(["002", "Adrian", "normal", "", "IMG_2.MOV", "", ""])
    pairs = [p for p in inbox.scan(root) if p.session == "s1"]
    assert [p.swing for p in pairs] == ["002"] and pairs[0].yardstick.endswith("IMG_0.MOV")


@pytest.mark.parametrize("miss", [False, True])
def test_tossed_ball_hit_or_missed(tmp_path, miss):
    """The team's real clips: the ball is thrown up and hit (sometimes missed), no tee."""
    frames, tr = render_swing(**FAST, omega=27, ball=True, toss=True, miss=miss, exit_speed=35.0, launch_deg=12.0)
    out = pipeline.analyze_slowmo(write(frames, tmp_path / "toss.mp4"), fps=240, m_per_px=tr["m_per_px"])
    assert out["ball_outcome"] == ("miss" if miss else "hit") and out["ball_how"] == "toss", out["ball_reason"]
    assert abs(out["bat_speed_mph"] / tr["bat_speed_mph"] - 1) < 0.05            # the tossed ball is not taken for tape
    if miss:
        assert out["contact_frame"] is None and out["exit_speed_mph"] is None
    else:
        assert out["contact_frame"] == tr["contact_frame"]
        assert abs(out["exit_speed_mph"] / (35 * 2.2369363) - 1) < 0.03 and abs(out["launch_angle_deg"] - 12) < 1.5


def test_bat_at_rest_is_not_a_ball_on_a_tee(tmp_path):
    """Tape bands sit still during the stance like a ball on a tee; when the bat starts it moves slowly,
    so it must not count as a hit."""
    frames, tr = render_swing(**FAST, omega=27, ball=False, tee=False)
    assert B.ball_events([B.ball_candidates(f) for f in frames], expected_r=B.BALL_RADIUS_M / tr["m_per_px"])["kind"] == "none"
