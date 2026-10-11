"""Checks on the real Driveline data (skipped if data isn't downloaded)."""
import json

import numpy as np
import pandas as pd

from swinglab import config, kinematics as K, obp, skeleton
from conftest import needs_models, needs_obp


@needs_obp
def test_landmark_meaning_by_geometry():
    """Hips are left_hip/right_hip (not lhjc); lhjc/rhjc are hands near the wrists."""
    df = pd.read_parquet(obp.landmarks_parquet()).head(20000)
    P = lambda p: df[[f"{p}_{a}" for a in "xyz"]].to_numpy()
    d = lambda a, b: np.nanmedian(np.linalg.norm(P(a) - P(b), axis=1))
    assert 0.30 < d("left_hip", "lkjc") < 0.50      # thigh
    assert d("rhjc", "rwjc") < 0.12                  # hand next to wrist
    assert 0.2 < d("left_hip", "right_hip") < 0.4    # pelvis width


@needs_obp
def test_events_match_force_plate_and_contact():
    meta, df = obp.load_metadata(), pd.read_parquet(obp.landmarks_parquet())
    fp_err, c_err = [], []
    for i, sw in enumerate(obp.iter_swings(df)):
        if i >= 60:
            break
        m = meta.loc[sw.id]
        sk = skeleton.from_obp(sw, m.hitter_side, m.height_m, fps=30)
        ev = K.detect_events(sk)
        t0 = sw.time[0]
        if "fp_10_time" in sw.events:
            fp_err.append(abs(ev["foot_plant"] - (sw.events["fp_10_time"] - t0)))
        c_err.append(abs(ev["contact"] - (sw.events["contact_time"] - t0)))
    assert np.mean(np.array(c_err) < 0.05) > 0.95
    assert np.mean(np.array(fp_err) < 0.05) > 0.6


@needs_obp
def test_pelvis_speed_matches_driveline():
    f = pd.read_csv(config.PROCESSED / "features_30fps.csv", index_col=0)
    poi = obp.load_poi()
    j = f[["pelvis_vel_peak"]].join(poi[["pelvis_angular_velocity_swing_max_x"]]).dropna()
    assert j.corr().iloc[0, 1] > 0.85
    assert abs(j.median().iloc[0] - j.median().iloc[1]) < 60


@needs_models
def test_models_beat_baseline():
    r = json.loads((config.MODELS / "metrics.json").read_text(encoding="utf-8"))
    bs = r["cv"]["bat_speed"]
    assert bs["both"]["ridge"]["r2"] > 0.2
    assert bs["both"]["ridge"]["mae"] < bs["both"]["baseline_mean_mae"]
    assert bs["both"]["ridge"]["r2"] > bs["covariates"]["ridge"]["r2"]  # mechanics add information
