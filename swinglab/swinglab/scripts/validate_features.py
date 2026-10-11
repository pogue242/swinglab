"""Compare our phone-reproducible features with Driveline's own POI metrics.

Correlation shows we measure the same thing; median offsets come from different
angle conventions (Driveline uses K-Vest conventions), which is fine as long as
training and inference both use ours.
Run: python scripts/validate_features.py
"""
import pandas as pd
from swinglab import obp, config
PAIRS = [("separation_fp", "x_factor_fp_z"), ("pelvis_yaw_fp", "pelvis_angle_fp_z"),
         ("torso_yaw_fp", "torso_angle_fp_z"), ("pelvis_vel_peak", "pelvis_angular_velocity_swing_max_x"),
         ("lead_knee_flex_fp", "lead_knee_stride_max_x")]
poi = obp.load_poi()
rows = []
for fps in (360, 30):
    f = pd.read_csv(config.PROCESSED / f"features_{fps}fps.csv", index_col=0)
    for a, b in PAIRS:
        j = f[[a]].join(poi[[b]], how="inner").dropna()
        rows.append(dict(fps=fps, ours=a, driveline=b, n=len(j), r=round(j.corr().iloc[0, 1], 2),
                         ours_median=round(j[a].median(), 1), driveline_median=round(j[b].median(), 1)))
print(pd.DataFrame(rows).to_string(index=False))
