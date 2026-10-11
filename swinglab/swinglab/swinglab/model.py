"""Train and apply the swing models.

Targets (all from OBP metadata, one model each):
  bat_speed  -> bat_speed_mph_max_x (mocap bat speed)      -> Power
  exit_velo  -> exit_velo_mph_x (ball off a ~65 mph machine) -> Contact quality

Inputs: our skeleton features (phone-reproducible) + body covariates
(height, mass, age, playing level). Covariates are included so the model's
mechanics effects are "for a hitter of your size and level", not "be bigger".

Validation: GroupKFold by athlete (``user``) so no athlete appears in both
train and test folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config, features, obp

TARGETS = {"bat_speed": "bat_speed_mph_max_x", "exit_velo": "exit_velo_mph_x"}
LEVELS = {"high_school": 0, "college": 1, "independent": 2, "milb": 3}
COVARIATES = ["height_m", "mass_kg", "athlete_age", "level"]
MECHANICS = list(features.FEATURES)
# Not used to EXPLAIN (they restate the outcome or can't be coached directly),
# but still used to PREDICT.
NOT_ACTIONABLE = {"hand_speed_peak", "seq_order_ok"}
ACTIONABLE = [f for f in MECHANICS if f not in NOT_ACTIONABLE]


def load_table(fps: int = int(config.PHONE_FPS), noise: bool = False) -> pd.DataFrame:
    f = pd.read_csv(config.PROCESSED / f"features_{fps}fps{'_noise' if noise else ''}.csv", index_col=0)
    m = obp.load_metadata()
    m["level"] = m["highest_playing_level"].map(LEVELS)
    cols = COVARIATES + ["user", "hitter_side"] + list(TARGETS.values())
    return f[[c for c in MECHANICS if c in f]].join(m[cols], how="inner")


def _models():
    return {
        "ridge": make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "gbm": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20),
    }


def cross_validate(tab: pd.DataFrame, target: str, inputs: list[str], n_splits: int = 5) -> dict:
    d = tab.dropna(subset=[target])
    X, y, g = d[inputs], d[target], d["user"]
    out = {}
    for name, mdl in _models().items():
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits).split(X, y, g):
            mdl.fit(X.iloc[tr], y.iloc[tr])
            pred[te] = mdl.predict(X.iloc[te])
        out[name] = {"r2": round(r2_score(y, pred), 3), "mae": round(mean_absolute_error(y, pred), 2), "n": int(len(y))}
    out["baseline_mean_mae"] = round(float(np.mean(np.abs(y - y.mean()))), 2)
    return out


@dataclass
class SwingModel:
    target: str
    inputs: list[str]
    gbm: object
    tree: DecisionTreeRegressor
    tree_inputs: list[str]
    reference: pd.DataFrame  # training features, for percentiles

    def predict(self, row: dict) -> float:
        x = pd.DataFrame([{k: row.get(k, np.nan) for k in self.inputs}])
        return float(self.gbm.predict(x)[0])

    def tree_path(self, row: dict) -> list[dict]:
        """Splits this swing passed through in the explanation tree."""
        x = pd.DataFrame([{k: row.get(k, np.nan) for k in self.tree_inputs}]).fillna(self.reference[self.tree_inputs].median())
        t = self.tree.tree_
        node, path = 0, []
        while t.children_left[node] != -1:
            fi, thr = t.feature[node], t.threshold[node]
            name = self.tree_inputs[fi]
            val = float(x.iloc[0, fi])
            go_left = val <= thr
            left, right = t.children_left[node], t.children_right[node]
            path.append({"feature": name, "value": round(val, 3), "threshold": round(float(thr), 3),
                         "went": "below" if go_left else "above",
                         "mean_if_below": round(float(t.value[left][0][0]), 1),
                         "mean_if_above": round(float(t.value[right][0][0]), 1),
                         "is_mechanic": name in MECHANICS})
            node = left if go_left else right
        return path


def fit(tab: pd.DataFrame, key: str, tree_depth: int = 3) -> SwingModel:
    target = TARGETS[key]
    d = tab.dropna(subset=[target])
    inputs = MECHANICS + COVARIATES
    gbm = _models()["ridge"].fit(d[inputs], d[target])  # ridge matched or beat GBM in CV
    # Explanation tree: mechanics only, on the target adjusted for body covariates,
    # so splits describe mechanics, not size.
    cov = make_pipeline(SimpleImputer(), RidgeCV()).fit(d[COVARIATES], d[target])
    resid = d[target] - cov.predict(d[COVARIATES]) + d[target].mean()
    Xt = d[ACTIONABLE].fillna(d[ACTIONABLE].median())
    tree = DecisionTreeRegressor(max_depth=tree_depth, min_samples_leaf=30, random_state=0).fit(Xt, resid)
    return SwingModel(target, inputs, gbm, tree, ACTIONABLE, d[MECHANICS + COVARIATES + [target]].copy())


def save(models: dict[str, SwingModel], meta: dict):
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, config.MODELS / "swing_models.joblib")
    (config.MODELS / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load() -> dict[str, SwingModel]:
    return joblib.load(config.MODELS / "swing_models.joblib")
