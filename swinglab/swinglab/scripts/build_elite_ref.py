"""Build data/models/elite_ref.json (the elite standard for elite.match) from the lab swings, and check it.

Run after scripts/build_dataset.py:  python scripts/build_elite_ref.py
Prints the validation: elite-match headline for elite vs other lab swings, how it follows size-adjusted bat
speed (rank correlation, per swing and per hitter), and how much one hitter's swings vary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swinglab import config, elite, model  # noqa: E402


def lab_table() -> pd.DataFrame:
    f = pd.read_csv(config.PROCESSED / f"features_{int(config.PHONE_FPS)}fps.csv", index_col=0)
    extra = [c for c in elite.factor_inputs() if c in f and c not in model.MECHANICS]
    return model.load_table().join(f[extra])


def main() -> dict:
    tab = lab_table()
    ref = elite.build_reference(tab)
    (config.MODELS / elite.ELITE_FILE).write_text(json.dumps(ref, indent=1), encoding="utf-8")
    print("saved", config.MODELS / elite.ELITE_FILE, f"({ref['n_elite_swings']} elite swings, {ref['n_elite_hitters']} hitters)")
    # validation on the lab itself
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    target = model.TARGETS["bat_speed"]
    d = tab.dropna(subset=[target]).copy()
    d["adj"] = d[target] - make_pipeline(SimpleImputer(), RidgeCV()).fit(d[model.COVARIATES], d[target]).predict(d[model.COVARIATES])
    d["elite"] = d["adj"] >= d["adj"].quantile(1 - elite.ELITE_FRACTION)
    d["match"] = [elite.match(r.to_dict(), height_m=r.height_m, mass_kg=r.mass_kg, ref=ref)["match"] for _, r in d.iterrows()]
    g = d.groupby("user")[["match", "adj"]].mean()
    val = {"elite_median": float(d.loc[d.elite, "match"].median()), "rest_median": float(d.loc[~d.elite, "match"].median()),
           "rho_swing": round(float(spearmanr(d["match"], d["adj"])[0]), 2), "rho_hitter": round(float(spearmanr(g["match"], g["adj"])[0]), 2),
           "within_hitter_sd": round(float(d.groupby("user")["match"].std().median()), 1)}
    print("validation:", val)
    return val


if __name__ == "__main__":
    main()
