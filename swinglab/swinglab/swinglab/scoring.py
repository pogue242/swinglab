"""Turn features + models into 0–100 scores and the coaching focus.

FORM (how you move; every swing, with or without a ball) = mean of three scores, each the
percentile, among the 677 Driveline swings, of a size-adjusted outcome predicted from ONE group
of features:
  Power       power features     -> bat speed
  Transfer    transfer features  -> bat speed       (shown as "Sequencing")
  Posture     posture features   -> exit velocity   (body position at contact; was "Contact")
CONTACT (how you hit the ball; only when a ball is seen leaving the tee) = contact_score(),
from the slow-motion camera's exit speed, launch angle and bat speed.
OVERALL = Form for a dry swing; the mean of Form and Contact for a hit.
"Size-adjusted" = the part of the outcome not explained by height, mass, age
and level, so a small high-schooler with great mechanics can score high.

The coaching focus is the actionable feature in the weakest group that costs
the most (standardized ridge coefficient × how far this swing is from the
typical Driveline value), reported in mph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features, model

MIN_FOCUS_MPH = 0.3   # ignore "issues" worth less than this
STRONG_SWING = 85.0   # overall score above which we lead with strengths

GROUP_TARGET = {"power": "bat_speed", "transfer": "bat_speed", "posture": "exit_velo"}


@dataclass
class GroupScorer:
    group: str
    target_key: str
    feats: list[str]
    pipe: object
    train_pred: np.ndarray
    medians: pd.Series
    scales: pd.Series

    def score(self, row: dict) -> tuple[float, float]:
        x = pd.DataFrame([{k: row.get(k, np.nan) for k in self.feats}])
        p = float(self.pipe.predict(x)[0])
        pct = float((self.train_pred < p).mean() * 100)
        return round(pct, 1), p

    def contributions(self, row: dict) -> list[dict]:
        """mph effect of each feature vs. the typical Driveline hitter (negative = costing you)."""
        ridge = self.pipe[-1]
        coef = ridge.coef_ / self.pipe[1].scale_  # per original unit
        out = []
        for f, c in zip(self.feats, coef):
            v = row.get(f, np.nan)
            if f in model.NOT_ACTIONABLE or not np.isfinite(v):
                continue
            out.append({"feature": f, "value": round(float(v), 3), "typical": round(float(self.medians[f]), 3),
                        "effect_mph": round(float(c * (v - self.medians[f])), 2),
                        "label": features.FEATURES[f][1], "unit": features.FEATURES[f][2]})
        return sorted(out, key=lambda d: d["effect_mph"])


def fit_scorers(tab: pd.DataFrame) -> dict[str, GroupScorer]:
    scorers = {}
    for g, key in GROUP_TARGET.items():
        target = model.TARGETS[key]
        d = tab.dropna(subset=[target])
        cov = make_pipeline(SimpleImputer(), RidgeCV()).fit(d[model.COVARIATES], d[target])
        y = d[target] - cov.predict(d[model.COVARIATES])
        feats = [f for f, (grp, *_) in features.FEATURES.items() if grp == g]
        pipe = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(d[feats], y)
        scorers[g] = GroupScorer(g, key, feats, pipe, pipe.predict(d[feats]), d[feats].median(), d[feats].std())
    return scorers


def analyze(row: dict, models: dict, scorers: dict, covariates: dict) -> dict:
    """Full result for one swing: predictions, scores, focus, tree paths."""
    full = {**row, **covariates}
    res = {"predicted": {k: round(m.predict(full), 1) for k, m in models.items()}, "scores": {}}
    for g, sc in scorers.items():
        res["scores"][g] = sc.score(row)[0]
    res["scores"]["form"] = round(float(np.mean([res["scores"][g] for g in GROUP_TARGET])), 1)
    res["scores"]["contact"] = None          # set by with_contact() when a ball was hit
    res["scores"]["overall"] = res["scores"]["form"]
    weakest = min(GROUP_TARGET, key=lambda g: res["scores"][g])
    contrib = scorers[weakest].contributions(row)
    res["weakest_group"] = weakest
    # Only coach a fix when it costs something meaningful; strong swings get praise.
    res["focus"] = contrib[0] if contrib and contrib[0]["effect_mph"] <= -MIN_FOCUS_MPH and res["scores"]["overall"] < STRONG_SWING else None
    res["contributions"] = {g: scorers[g].contributions(row) for g in GROUP_TARGET}
    res["tree_path"] = {k: m.tree_path(row) for k, m in models.items()}
    return res


# ---------------------------------------------------------------- ball contact (hit mode only)
#
# Provisional, rule-based until we have our own tee data (Driveline's exit speeds came off a ~65 mph
# pitch, so they are not a fair reference for tee hits):
#   smash factor = exit speed / bat speed. Off a tee a square hit gives ~1.2 (ball-bat collision
#   physics with typical bat mass); glancing hits lose speed. 70% of the score.
#   launch angle: line drives (8–25°) score full marks, fading to 0 at -10° and 45°. 30%.
SMASH_FULL, SMASH_ZERO = 1.2, 0.6
LAUNCH_BEST, LAUNCH_ZERO = (8.0, 25.0), (-10.0, 45.0)


def contact_score(exit_mph: float | None, launch_deg: float | None, bat_mph: float | None) -> dict | None:
    """0–100 ball-contact score with its parts, or None when exit speed or bat speed is missing."""
    if exit_mph is None or bat_mph is None or bat_mph <= 0:
        return None
    smash = exit_mph / bat_mph
    s_part = float(np.clip((smash - SMASH_ZERO) / (SMASH_FULL - SMASH_ZERO), 0, 1))
    a_part = 0.0
    if launch_deg is not None:
        lo, hi = LAUNCH_BEST
        z0, z1 = LAUNCH_ZERO
        a_part = 1.0 if lo <= launch_deg <= hi else float(np.clip(
            (launch_deg - z0) / (lo - z0) if launch_deg < lo else (z1 - launch_deg) / (z1 - hi), 0, 1))
    return {"score": round(100 * (0.7 * s_part + 0.3 * a_part), 1), "smash_factor": round(smash, 2),
            "launch_deg": launch_deg, "rule": "provisional: 70% smash factor (1.2 = square), 30% launch angle (8–25° best)"}


def with_contact(analysis: dict, contact: dict | None) -> dict:
    """Put the ball-contact score into an analysis and recompute Overall."""
    sc = analysis["scores"]
    sc["contact"] = contact["score"] if contact else None
    sc["overall"] = round((sc["form"] + sc["contact"]) / 2, 1) if contact else sc["form"]
    analysis["contact_detail"] = contact
    return analysis
