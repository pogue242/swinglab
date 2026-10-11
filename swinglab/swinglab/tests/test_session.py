"""Batting-cage mode: one long recording with several swings."""
import json

import numpy as np

from swinglab import inbox, pipeline, session
from swinglab.coach import cues
from test_kinematics import body_from_yaw
from test_pipeline import render_stray, W, H
from conftest import needs_models


def long_recording(n_swings=5, gap=4.0, fps=30.0, turns=(160, 150, 140, 130, 120)):
    """Swings every `gap` s: fast smoothstep turn (one clear hand-speed peak), then a slow
    return to stance that stays below the swing threshold. Later swings turn less (a 'tiring' hitter)."""
    T = n_swings * gap + 1.0
    t = np.arange(0, T, 1 / fps)
    yaw = np.zeros_like(t)
    for k in range(n_swings):
        t0 = k * gap + 1.8
        up = np.clip((t - t0) / 0.2, 0, 1)
        down = np.clip((t - (t0 + 0.9)) / 1.8, 0, 1)
        s = lambda u: 3 * u**2 - 2 * u**3  # noqa: E731
        yaw += np.radians(turns[k]) * (s(up) - s(down))
    return t, body_from_yaw(t, yaw)


def test_find_swings_counts_and_times():
    t, J = long_recording()
    wins = session.find_swings(t, J)
    assert len(wins) == 5
    assert np.allclose([w.peak for w in wins], [k * 4.0 + 1.9 for k in range(5)], atol=0.1)
    assert all(w.start < w.peak < w.end for w in wins)


def test_no_swings_when_still():
    t = np.arange(0, 10, 1 / 30)
    J = body_from_yaw(t, np.zeros_like(t))
    assert session.find_swings(t, J) == []


@needs_models
def test_analyze_long_and_summary(tmp_path):
    t, J = long_recording(fps=60)
    kps = render_stray(tmp_path / "cage", t, J)
    hitter = {"name": "A", "bats": "R", "height_m": 1.8, "mass_kg": 80, "age": 20}
    recs, summ = session.analyze_long(tmp_path / "cage", hitter, keypoints_fn=lambda v: (kps, (W, H)))
    assert len(recs) == 5 and all("analysis" in r for r in recs)
    assert summ["n_swings"] == 5 and len(summ["per_swing"]) == 5
    # smaller turns later in the session -> pelvis rotation range trends down
    rng = [r["features"]["pelvis_rotation_range"] for r in recs]
    assert rng[0] > rng[-1]
    rep = cues.session_report(summ)
    assert rep.best_swing_id in {r["id"] for r in recs}


@needs_models
def test_inbox_splits_long_recording(tmp_path):
    root = tmp_path / "SwingLab"
    inbox.template(root)
    t, J = long_recording(n_swings=3, fps=30)
    kps = render_stray(root / "sessions" / "cage1" / "lidar" / "longrec", t, J)
    out = tmp_path / "web"
    recs = inbox.process(
        root,
        analyze_recording=lambda l, v, h: pipeline.analyze_recording(l, v, h, use_gemini=False, keypoints_fn=lambda k: (kps, (W, H))),
        save=lambda r, s: pipeline.save_result(r, s, out_root=out),
        save_session=lambda sm, s: pipeline.save_session(sm, s, out_root=out), log=lambda *a: None)
    assert len(recs) == 3
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert len(idx["swings"]) == 3 and idx["sessions"][0]["n_swings"] == 3


@needs_models
def test_short_recording_with_three_swings_is_a_session(tmp_path):
    """Mode comes from content: 3 quick swings in 10 s is a session, not a clip."""
    t, J = long_recording(n_swings=3, gap=3.0, fps=30)
    kps = render_stray(tmp_path / "quick", t, J)
    out = pipeline.analyze_recording(tmp_path / "quick", None, {"bats": "R", "height_m": 1.8, "mass_kg": 80, "age": 20},
                                     use_gemini=False, keypoints_fn=lambda v: (kps, (W, H)))
    assert t[-1] < 12 and out["summary"] is not None and len(out["recs"]) == 3


def test_hitter_warnings_outside_training_range():
    w = pipeline.hitter_warnings({"bats": "R", "height_m": 1.50, "mass_kg": 50, "age": 14})
    assert sum("outside the lab" in x for x in w) == 3
    assert any("weight not given" in x for x in pipeline.hitter_warnings({"bats": "R", "height_m": 1.8, "age": 20}))
    assert pipeline.hitter_warnings({"bats": "R", "height_m": 1.8, "mass_kg": 80, "age": 20}) == []


from conftest import needs_obp


@needs_obp
def test_one_swing_per_lab_clip():
    import pandas as pd
    from swinglab import obp, skeleton
    df = pd.read_parquet(obp.landmarks_parquet())
    n = 0
    for i, sw in enumerate(obp.iter_swings(df)):
        if i >= 150:
            break
        names = ["l_wrist", "r_wrist", "l_shoulder", "r_shoulder", "l_ankle", "r_ankle"]   # shoulders: the turn check
        t, W2 = skeleton.resample(sw.time, np.stack([sw.joints[k] for k in names], 1), 30)
        n += len(session.find_swings(t - t[0], {k: W2[:, j] for j, k in enumerate(names)})) == 1
    assert n == 150



def test_summary_counts_hits_misses_and_dry_swings():
    def rec(i, mode, form, contact):
        return {"id": str(i), "mode": mode, "features": {}, "measured": {},
                "analysis": {"predicted": {"bat_speed": 60.0 + i, "exit_velo": 80.0}, "focus": None,
                             "scores": {"overall": form if contact is None else (form + contact) / 2, "form": form, "contact": contact}}}
    recs = [rec(0, "hit", 60, 80), rec(1, "miss", 50, 0), rec(2, "dry", 70, None), rec(3, "hit", 66, 90)]
    s = session.summarize(recs)
    assert s["outcomes"] == {"hits": 2, "misses": 1, "dry_swings": 1, "lab_swings": 0, "hit_rate": 0.67}
    assert s["form_score"]["mean"] == 61.5 and s["contact_score"]["mean"] == 56.7
    assert [p["outcome"] for p in s["per_swing"]] == ["hit", "miss", "dry", "hit"]
    from swinglab.coach import cues
    assert "2 hits, 1 misses, 1 dry swings" in cues.session_report(s).headline
