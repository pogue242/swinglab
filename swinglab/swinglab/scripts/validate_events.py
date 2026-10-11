"""Validate skeleton-only event detection against Driveline's force-plate foot plant and contact times.

Run: python scripts/validate_events.py [--n 200]
"""
import argparse, json
import numpy as np, pandas as pd
from swinglab import obp, skeleton, kinematics as K, config

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=10**6); a = ap.parse_args()
meta = obp.load_metadata(); df = pd.read_parquet(obp.landmarks_parquet())
rows = []
for i, sw in enumerate(obp.iter_swings(df)):
    if i >= a.n: break
    m = meta.loc[sw.id]; t0 = sw.time[0]
    if "contact_time" not in sw.events: continue
    for fps in (360, 60, config.PHONE_FPS):
        sk = skeleton.from_obp(sw, m.hitter_side, m.height_m, fps=fps)
        ev = K.detect_events(sk)
        fp_true = sw.events.get("fp_10_time", np.nan) - t0
        rows.append(dict(id=sw.id, fps=fps, fp_err_ms=1000*(ev["foot_plant"]-fp_true),
                         contact_err_ms=1000*(ev["contact"]-(sw.events["contact_time"]-t0)),
                         peak_minus_contact_ms=1000*(ev["peak_hand"]-(sw.events["contact_time"]-t0))))
r = pd.DataFrame(rows)
print("median offsets to add if re-fitting (ms):", r.groupby("fps")[["fp_err_ms","contact_err_ms"]].median().round(1).to_dict())
summ = r.groupby("fps")[["fp_err_ms", "contact_err_ms", "peak_minus_contact_ms"]].agg(
    ["median", lambda x: np.nanmedian(np.abs(x)), lambda x: np.nanmean(np.abs(x) < 50)]).round(1)
summ.columns = [f"{c}_{s}" for c, s in [(c, {"median":"median","<lambda_0>":"MAE_med","<lambda_1>":"within50ms"}[s]) for c, s in summ.columns]]
print(summ.T.to_string())
config.PROCESSED.mkdir(parents=True, exist_ok=True)
r.to_csv(config.PROCESSED / "event_validation.csv", index=False)
