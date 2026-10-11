"""Cross-validate and train the bat-speed and exit-velocity models.

Run: python scripts/train.py [--fps 30]
Prints athlete-grouped CV results for three input sets so you can see whether
mechanics add information beyond body size:
  covariates  height, mass, age, level only
  mechanics   our skeleton features only
  both        mechanics + covariates  (the model we ship)
Writes data/models/swing_models.joblib and metrics.json.
"""
import argparse
import json

from sklearn.tree import export_text

from swinglab import config, model

ap = argparse.ArgumentParser()
ap.add_argument("--fps", type=int, default=int(config.PHONE_FPS))
a = ap.parse_args()

tab = model.load_table(a.fps)
report = {"fps": a.fps, "n_swings": len(tab), "n_athletes": int(tab["user"].nunique()), "cv": {}}
for key, target in model.TARGETS.items():
    report["cv"][key] = {
        "covariates": model.cross_validate(tab, target, model.COVARIATES),
        "mechanics": model.cross_validate(tab, target, model.MECHANICS),
        "both": model.cross_validate(tab, target, model.MECHANICS + model.COVARIATES),
    }
print(json.dumps(report, indent=1))

models = {key: model.fit(tab, key) for key in model.TARGETS}
for key, m in models.items():
    print(f"\nExplanation tree for {key} (size-adjusted):")
    print(export_text(m.tree, feature_names=m.tree_inputs, decimals=1))
from swinglab import scoring
import joblib
scorers = scoring.fit_scorers(tab)
model.save(models, report)
joblib.dump(scorers, config.MODELS / "scorers.joblib")
# sanity: score every training swing, scores should spread ~0-100 with median ~50
import numpy as np, pandas as pd
cov = ["height_m", "mass_kg", "athlete_age", "level"]
S = pd.DataFrame([scoring.analyze(r[model.MECHANICS].to_dict(), models, scorers, r[cov].to_dict())["scores"] for _, r in tab.iterrows()])
print("score distribution on training swings:"); print(S.describe().round(1).loc[["mean","25%","50%","75%"]])
print("saved", config.MODELS / "swing_models.joblib")
