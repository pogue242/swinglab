"""Build the training table: one row per OBP swing, features at phone frame rate.

Run: python scripts/build_dataset.py [--fps 30] [--noise 0.0]
Writes data/processed/features_<fps>fps.csv
"""
import argparse
import time

import numpy as np
import pandas as pd

from swinglab import config, features, kinematics as K, obp, skeleton

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=float, default=config.PHONE_FPS)
ap.add_argument("--noise", type=float, default=0.0, help="joint jitter in meters before resampling")
a = ap.parse_args()

if not obp.landmarks_parquet().exists():
    obp.build_landmarks_parquet()
meta = obp.load_metadata()
df = pd.read_parquet(obp.landmarks_parquet())
rng = np.random.default_rng(0)
rows, t0 = [], time.time()
for sw in obp.iter_swings(df):
    m = meta.loc[sw.id]
    try:
        sk = skeleton.from_obp(sw, m.hitter_side, m.height_m, fps=a.fps, noise_m=a.noise, rng=rng)
        s = K.series(sk)
        ev = K.detect_events(sk, s)
        f = features.compute(sk, ev, s)
        f.update(session_swing=sw.id, fp_method=ev["fp_method"])
        rows.append(f)
    except Exception as e:  # keep going; report failures
        rows.append(dict(session_swing=sw.id, error=repr(e)))
out = pd.DataFrame(rows).set_index("session_swing")
path = config.PROCESSED / f"features_{int(a.fps)}fps{'_noise' if a.noise else ''}.csv"
out.to_csv(path)
n_err = out["error"].notna().sum() if "error" in out else 0
print(f"{len(out)} swings, {n_err} failed, {time.time()-t0:.0f}s -> {path}")
