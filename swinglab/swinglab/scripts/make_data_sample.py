"""Build an Excel workbook (and CSVs) to look at the Driveline data and our features.

Run: python scripts/make_data_sample.py   ->  data/sample/driveline_sample.xlsx
Needs the hitting data (python scripts/setup_data.py) and `pip install openpyxl`.
"""
import numpy as np
import pandas as pd

from swinglab import config, kinematics as K, model, obp, skeleton

SWING = "291_6"  # the strongest swing in the demo viewer
out_dir = config.DATA / "sample"
out_dir.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(config.RAW / "metadata.csv")
poi = pd.read_csv(config.RAW / "poi" / "poi_metrics.csv")
htx = pd.read_csv(config.RAW / "poi" / "hittrax.csv")
dd = pd.read_csv(config.RAW / "data_dictionary.csv") if (config.RAW / "data_dictionary.csv").exists() else None
feats = pd.read_csv(config.PROCESSED / f"features_{int(config.PHONE_FPS)}fps.csv")
feats = feats.merge(meta[["session_swing", "hitter_side", "highest_playing_level", "session_height_in",
                          "bat_speed_mph_max_x", "exit_velo_mph_x"]], on="session_swing", how="left")
lm = pd.read_parquet(obp.landmarks_parquet())
one = lm[lm.session_swing == SWING].reset_index(drop=True)

m = obp.load_metadata().loc[SWING]
sw = next(obp.iter_swings(one))
sk = skeleton.from_obp(sw, m.hitter_side, m.height_m, fps=config.PHONE_FPS)
s = K.series(sk)
canon = pd.DataFrame({"time_s": sk.time})
for j, name in enumerate(sk.names):
    for a, ax in enumerate("xyz"):
        canon[f"{name}_{ax}"] = sk.J[:, j, a].round(4)
for k in ("pelvis_yaw", "torso_yaw", "separation", "pelvis_vel", "torso_vel", "hand_speed", "lead_knee_flex"):
    canon[k] = np.round(s[k], 2)

readme = pd.DataFrame({"tab": [
    "metadata", "poi_metrics", "hittrax", "our_features_30fps", f"swing_{SWING}_360hz", f"swing_{SWING}_30fps_ours",
    "data_dictionary"], "what it is": [
    "One row per swing (677). Hitter height/weight/age/level/handedness, bat size, bat speed (mocap max and Blast sensor), exit velocity. Join key: session_swing.",
    "Driveline's own computed metrics per swing (angles at events, speeds). We use these only to check our numbers.",
    "Ball-flight data for 604 swings: launch angle (la), distance (dist), result (res), point of impact on the ball (poi_x/y/z).",
    "Our 23 phone-reproducible mechanics per swing, computed after resampling to phone frame rate, plus the outcomes. This is the model's training table.",
    f"Raw 360 Hz joint centers (meters) for one swing. Lab axes: +x toward pitcher, +y toward the right-handed batter's box, +z up. lhjc/rhjc = HANDS; left_hip/right_hip = hips. Events: fp_10_time (foot plant), contact_time.",
    "The same swing after our pipeline: lead/rear joints in the hitter's body frame at phone frame rate, plus rotation angles and speeds.",
    "Driveline's column descriptions (hitting)."]})
xlsx = out_dir / "driveline_sample.xlsx"
with pd.ExcelWriter(xlsx) as w:
    readme.to_excel(w, sheet_name="README", index=False)
    meta.to_excel(w, sheet_name="metadata", index=False)
    poi.to_excel(w, sheet_name="poi_metrics", index=False)
    htx.to_excel(w, sheet_name="hittrax", index=False)
    feats.to_excel(w, sheet_name="our_features_30fps", index=False)
    one.to_excel(w, sheet_name=f"swing_{SWING}_360hz", index=False)
    canon.to_excel(w, sheet_name=f"swing_{SWING}_30fps_ours", index=False)
    if dd is not None:
        dd.to_excel(w, sheet_name="data_dictionary", index=False)
one.to_csv(out_dir / f"swing_{SWING}_360hz.csv", index=False)
canon.to_csv(out_dir / f"swing_{SWING}_30fps_ours.csv", index=False)
print(xlsx, round(xlsx.stat().st_size / 1e6, 2), "MB")
