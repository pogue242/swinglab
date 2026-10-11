"""Long sessions with one slow-motion clip per swing (item 7): match N clips to the N swings of
one long LiDAR recording, by when the ball left the tee when both clocks are known, else by order;
flag count mismatches. Synthetic data through inbox.process."""
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pytest

from swinglab import capture, inbox, pipeline, session
from conftest import needs_models

# ---------------------------------------------------------------- the matcher


def test_order_when_no_times():
    m = session.match_clips([None, None, None], [None, None, None])
    assert m["method"] == "order" and m["pairs"] == [(0, 0), (1, 1), (2, 2)] and m["notes"] == []


def test_order_mismatch_is_flagged():
    m = session.match_clips([None] * 4, [None] * 3)
    assert m["pairs"] == [(0, 0), (1, 1), (2, 2)] and m["unmatched_swings"] == [3]
    assert "4 swings found but 3 slow-motion clips" in m["notes"][0] and "probably wrong" in m["notes"][0]


def test_time_matching_with_clock_offset_and_a_missing_clip():
    swings = [10.0, 41.3, 69.8, 103.1, 131.0]           # irregular, like a real hitter
    clips = [t + 3.2 + 0.4 * (k % 2) for k, t in enumerate(swings) if k != 2]   # clocks 3.2 s apart (+1 s rounding), clip 3 missing
    m = session.match_clips(swings, clips)
    assert m["method"] == "time" and abs(m["offset_s"] - 3.2) < 0.5
    assert m["pairs"] == [(0, 0), (1, 1), (3, 2), (4, 3)] and m["unmatched_swings"] == [2]
    assert any("no slow-motion clip for swing 3" in n for n in m["notes"])


def test_time_matching_with_an_extra_clip():
    swings = [5.0, 30.0, 61.0]
    clips = [4.8, 18.0, 30.3, 60.6]                      # a clip with no swing (filmed between swings)
    m = session.match_clips(swings, clips)
    assert m["pairs"] == [(0, 0), (1, 2), (2, 3)] and m["unmatched_clips"] == [1]


def test_regular_spacing_prefers_the_small_clock_offset():
    swings = [0.0, 30.0, 60.0, 90.0]
    clips = [30.4, 60.4, 90.4]                           # first clip missing; a 30 s shift would also fit 3
    m = session.match_clips(swings, clips)
    assert m["pairs"] == [(1, 0), (2, 1), (3, 2)] and abs(m["offset_s"] - 0.4) < 1e-6


# ---------------------------------------------------------------- wall-clock times


def _with_creation_time(src, dst, when: float):
    import imageio_ffmpeg
    iso = datetime.fromtimestamp(when, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src), "-c", "copy",
                    "-metadata", f"creation_time={iso}", str(dst)], check=True)
    os.remove(src)
    return dst


def test_video_and_record3d_start_times(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    cv2 = pytest.importorskip("cv2")
    vw = cv2.VideoWriter(str(tmp_path / "a.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(5):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    assert capture.video_start_time(tmp_path / "a.mp4") is None          # OpenCV writes no creation time
    when = 1791640805.25
    v = _with_creation_time(tmp_path / "a.mp4", tmp_path / "IMG_1.mp4", when)
    assert abs(capture.video_start_time(v) - int(when)) < 1e-6   # MP4 headers keep whole seconds
    t = capture.recording_start_time("2026-10-10--10-00-00.r3d")
    assert datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") == "2026-10-10 10:00:00"   # local time
    assert capture.recording_start_time("8f3a2c1d") is None              # Stray Scanner folder


# ---------------------------------------------------------------- through the shared folder


def _session_folder(tmp_path, *, record3d: bool, clips_for, clock_offset=2.0, swing_gap=4.0):
    """Long LiDAR recording with 3 swings + one slow-motion clip per swing in ``clips_for``."""
    from test_pipeline import render_stray, stray_to_r3d
    from test_session import long_recording
    from bat_synth import render_swing, write
    root = tmp_path / "SwingLab"
    inbox.template(root)
    sdir = root / "sessions" / "cage"
    (sdir / "lidar").mkdir(parents=True)
    (sdir / "slowmo").mkdir(parents=True)
    t, J = long_recording(n_swings=3, gap=swing_gap, fps=30)
    stray = tmp_path / "stray_long"
    kps = render_stray(stray, t, J)
    if record3d:
        lid = stray_to_r3d(stray, sdir / "lidar" / "2026-10-10--10-00-00.r3d", fps=30.0)
    else:
        import shutil
        lid = sdir / "lidar" / "a1b2c3d4"
        shutil.copytree(stray, lid)
    start = capture.recording_start_time("2026-10-10--10-00-00.r3d")
    truths = {}
    for k in clips_for:
        frames, tr = render_swing(size=(960, 540), m_per_px=0.0033, sub=8, n=80, omega=20 + 4 * k, ball=True,
                                  exit_speed=30.0 + 3 * k, launch_deg=10.0, seed=k)
        contact_lidar = k * swing_gap + 1.9 + 0.065       # when the swing's ball leaves the tee (LiDAR clock)
        name = f"IMG_{100 + k}.mp4"
        src = write(frames, sdir / "slowmo" / f"tmp_{k}.mp4")
        if record3d:   # clip starts so its contact lands at the swing's contact, on a clock 2 s ahead
            _with_creation_time(src, sdir / "slowmo" / name, start + contact_lidar - tr["contact_frame"] / 240 + clock_offset)
        else:
            os.replace(src, sdir / "slowmo" / name)
        os.utime(sdir / "slowmo" / name, (1e9 + k, 1e9 + k))
        truths[k] = tr
    return root, lid, kps, truths


def _process(root, kps, tmp_path):
    from test_pipeline import W, H
    out = tmp_path / "web"
    logs = []
    recs = inbox.process(
        root, analyze_recording=lambda l, v, h, **kw: pipeline.analyze_recording(
            l, v, h, use_gemini=False, keypoints_fn=lambda k: (kps, (W, H)), **kw),
        save=lambda r, s: pipeline.save_result(r, s, out_root=out, render_video=False),
        save_session=lambda sm, s: pipeline.save_session(sm, s, out_root=out), log=logs.append)
    sess = json.loads(next((out / "swings").glob("*/_summary_*.json")).read_text(encoding="utf-8"))
    return recs, sess, logs


@needs_models
def test_session_clips_matched_by_time_with_one_missing(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    pytest.importorskip("lzfse")
    root, lid, kps, truths = _session_folder(tmp_path, record3d=True, clips_for=[0, 2])   # no clip for swing 2
    pairs = [p for p in inbox.scan(root) if p.session == "cage"]
    assert len(pairs) == 1 and pairs[0].method == "time" and len(pairs[0].slowmo_clips) == 2
    recs, sess, logs = _process(root, kps, tmp_path)
    m = sess["slowmo_matching"]
    assert sess["n_swings"] == 3 and m["method"] == "time" and abs(m["offset_s"] - 2.0) < 1.0   # 1 s metadata resolution
    assert [p[1] for p in m["pairs"]] == ["IMG_100.mp4", "IMG_102.mp4"]
    by = {r["id"]: r for r in recs}
    ids = sorted(by)
    assert by[ids[0]]["slowmo_video"].endswith("IMG_100.mp4") and by[ids[2]]["slowmo_video"].endswith("IMG_102.mp4")
    assert by[ids[1]].get("slowmo") is None
    for k, rid in ((0, ids[0]), (2, ids[2])):
        assert abs(by[rid]["measured"]["bat_speed_mph"] / truths[k]["bat_speed_mph"] - 1) < 0.05
        assert by[rid]["payload"]["measured"]["bat_speed_mph_sweet_spot"] == by[rid]["measured"]["bat_speed_mph"]
    assert any("no slow-motion clip for swing 2" in n for n in sess["notes"])
    assert any("!! " in line and "swings found but 2 slow-motion clips" in line for line in logs)
    assert sess["measured_exit_velo"] is not None                  # the summary includes the measured speeds


@needs_models
def test_session_clips_matched_by_order_without_clocks(tmp_path):
    root, lid, kps, truths = _session_folder(tmp_path, record3d=False, clips_for=[0, 1, 2])
    pairs = [p for p in inbox.scan(root) if p.session == "cage"]
    assert len(pairs) == 1 and pairs[0].method == "session_order" and len(pairs[0].slowmo_clips) == 3
    recs, sess, _ = _process(root, kps, tmp_path)
    m = sess["slowmo_matching"]
    assert m["method"] == "order" and m["notes"] == []
    for k, r in enumerate(sorted(recs, key=lambda r: r["id"])):
        assert r["slowmo_video"].endswith(f"IMG_{100 + k}.mp4")
        assert abs(r["measured"]["bat_speed_mph"] / truths[k]["bat_speed_mph"] - 1) < 0.05
