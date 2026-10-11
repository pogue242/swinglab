"""Turn features into 0–100 scores and the coaching focus.

FORM (how you move; every swing, with or without a ball) = mean of three group scores:
  Power       stance, stride, hip–shoulder separation, hips open at plant, hand speed
  Transfer    foot plant -> contact time, shoulders closed at plant, hips -> chest -> hands order
              (shown as "Sequencing")
  Posture     front knee, upper-body lean, head and hip drift, hands out front
Each group score is a weighted mean of per-measurement GRADES (grade_feature): where the swing's
value sits in the spread of the 677 Driveline lab swings at phone frame rate
(data/models/lab_ranges.json, written by scripts/train.py from data/processed/features_30fps.csv):
  "higher is better"  grade = the lab percentile of the value
  "lower is better"   grade = 100 - the lab percentile
  "range"             100 inside the middle half of the lab (p25–p75), falling linearly to 0 at
                      p2.5 / p97.5
  yes/no              100 or 0
Weights: 0.5–1.5 per measurement, by how strongly the lab model ties it to bat speed (Power,
Sequencing) or exit speed (Posture): 0.5 + |standardized ridge coefficient| / largest in the group.
Measurements the phone can't measure reliably yet (NOT_GRADED) are shown but not graded.
CONTACT (how you hit the ball; only when a ball is seen leaving the tee) = contact_score().
OVERALL = Form for a dry swing; the mean of Form and Contact for a hit.

Why not the old score (until 2026-10-11): it was the lab percentile of a bat speed predicted by a
ridge model from one group's features. From mechanics alone that model explains ~16% of bat speed
(cross-validated R² 0.16), so its predictions bunch together and small measurement noise moved a
score from 0 to 100 (TestSet2: Power 0 on every Wilson swing, Sequencing 100 on noisy timings).
It is still computed, as analysis["scores_lab_model"], for comparison.

The coaching focus is the graded measurement in the weakest group that falls furthest below the
lab spread (weight × (100 - grade)), when its grade is under FOCUS_MAX_GRADE; its effect_mph (the
lab model's estimate of what it costs, size-adjusted) is kept for the coach text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features, model

STRONG_SWING = 85.0      # overall score above which we lead with strengths
FOCUS_MAX_GRADE = 50.0   # only coach a measurement graded below this

GROUP_TARGET = {"power": "bat_speed", "transfer": "bat_speed", "posture": "exit_velo"}
QUANTILES = (2.5, 10, 25, 50, 75, 90, 97.5)

# Measured on TestSet2 (2026-10-11; phone in front of the hitter's chest, MediaPipe 3D + LiDAR)
# against the lab ranges, these come out off on every swing of both hitters, so grading them would
# grade the camera, not the hitter. Shown, not graded, until a side view or a calibration fixes them.
NOT_GRADED = {
    "pelvis_rotation_range": "hip turn toward / away from the camera reads 2–3× too small",
    "pelvis_vel_peak": "hip rotation toward / away from the camera reads 2–3× too slow",
    "torso_vel_peak": "chest rotation toward / away from the camera reads about 2× too slow",
    "gain_torso_over_pelvis": "a ratio of the two rotation speeds above, so it inherits their error",
    "seq_pelvis_to_torso_ms": "the lab gap is ~35 ms, about one frame at 30 fps, and the pose lags the turn",
    "seq_torso_to_hands_ms": "the lab gap is ~15 ms, under one frame at 30 fps",
    "trunk_forward_tilt_contact": "leaning toward the plate is leaning toward the camera, where depth is weakest",
}


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


def lab_ranges(tab: pd.DataFrame) -> dict[str, dict]:
    """Lab spread of every feature: {feature: {"p2.5", "p10", ..., "p97.5", "better"(, "share_yes")}}."""
    out = {}
    for f in features.FEATURES:
        col = tab[f].dropna()
        r = {f"p{q:g}": round(float(np.percentile(col, q)), 4) for q in QUANTILES}
        r["better"] = features.FEATURE_HELP[f][1]
        if features.FEATURES[f][2] == "bool":
            r["share_yes"] = round(float(col.mean()), 3)
        out[f] = r
    return out


_RANGES: dict | None = None


def load_ranges() -> dict:
    global _RANGES
    if _RANGES is None:
        from . import config
        _RANGES = json.loads((config.MODELS / "lab_ranges.json").read_text(encoding="utf-8"))
    return _RANGES


def grade_feature(f: str, v: float | None, rng: dict) -> float | None:
    """0–100: where value ``v`` of feature ``f`` sits in the lab spread ``rng`` (see the module doc)."""
    if v is None or not np.isfinite(v):
        return None
    if features.FEATURES[f][2] == "bool":
        return 100.0 if v >= 0.5 else 0.0
    if rng["better"] in ("higher", "lower"):
        pct = float(np.interp(v, [rng[f"p{q:g}"] for q in QUANTILES], QUANTILES, left=0.0, right=100.0))
        return round(pct if rng["better"] == "higher" else 100.0 - pct, 1)
    lo, p25, p75, hi = rng["p2.5"], rng["p25"], rng["p75"], rng["p97.5"]
    if p25 <= v <= p75:
        return 100.0
    g = (v - lo) / (p25 - lo) if v < p25 else (hi - v) / (hi - p75)
    return round(100.0 * float(np.clip(g, 0.0, 1.0)), 1)


def feature_weights(scorers: dict) -> dict[str, float]:
    """0.5–1.5 per feature: 0.5 + |standardized ridge coefficient| / the largest in its group."""
    w = {}
    for sc in scorers.values():
        c = np.abs(sc.pipe[-1].coef_)
        top = float(c.max()) or 1.0
        w.update({f: round(0.5 + float(ci) / top, 3) for f, ci in zip(sc.feats, c)})
    return w


def grade_swing(row: dict, scorers: dict, ranges: dict) -> dict:
    """Per-feature grades and the three group scores: {"grades": {feature: {grade, weight, value,
    better, lab_p25, lab_median, lab_p75, label, unit}}, "scores": {group: 0–100 or None},
    "not_graded": [{feature, label, reason}]}."""
    w = feature_weights(scorers)
    grades, groups, skipped = {}, {g: [] for g in GROUP_TARGET}, []
    for f, (grp, label, unit) in features.FEATURES.items():
        if f in NOT_GRADED:
            skipped.append({"feature": f, "label": label, "reason": NOT_GRADED[f]})
            continue
        v = row.get(f)
        v = np.nan if v is None else float(v)
        g = grade_feature(f, v, ranges[f])
        if g is None:
            continue
        rng = ranges[f]
        grades[f] = {"grade": g, "weight": w.get(f, 1.0), "value": round(v, 3), "better": rng["better"],
                     "lab_p25": rng["p25"], "lab_median": rng["p50"], "lab_p75": rng["p75"], "label": label, "unit": unit}
        groups[grp].append((g, w.get(f, 1.0)))
    scores = {g: (round(sum(a * b for a, b in v) / sum(b for _, b in v), 1) if v else None) for g, v in groups.items()}
    return {"grades": grades, "scores": scores, "not_graded": skipped}


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


def analyze(row: dict, models: dict, scorers: dict, covariates: dict, ranges: dict | None = None) -> dict:
    """Full result for one swing: predictions, graded scores, focus, tree paths."""
    full = {**row, **covariates}
    gs = grade_swing(row, scorers, ranges or load_ranges())
    res = {"predicted": {k: round(m.predict(full), 1) for k, m in models.items()}, "grading": "lab_range_v2",
           "scores": {g: (v if v is not None else 0.0) for g, v in gs["scores"].items()}}
    res["scores"]["form"] = round(float(np.mean([res["scores"][g] for g in GROUP_TARGET])), 1)
    res["scores"]["contact"] = None          # set by with_contact() when a ball was hit
    res["scores"]["overall"] = res["scores"]["form"]
    res["scores_lab_model"] = {g: sc.score(row)[0] for g, sc in scorers.items()}
    res["grades"], res["not_graded"] = gs["grades"], gs["not_graded"]
    weakest = min(GROUP_TARGET, key=lambda g: res["scores"][g])
    res["weakest_group"] = weakest
    gr = gs["grades"]
    res["contributions"] = {g: [{**c, "grade": gr[c["feature"]]["grade"], "lab_p25": gr[c["feature"]]["lab_p25"],
                                 "lab_p75": gr[c["feature"]]["lab_p75"]}
                                for c in scorers[g].contributions(row) if c["feature"] in gr] for g in GROUP_TARGET}
    # Coach the measurement furthest below the lab spread in the weakest group; strong swings get praise.
    low = [c for c in res["contributions"][weakest] if c["grade"] < FOCUS_MAX_GRADE]
    res["focus"] = (max(low, key=lambda c: gr[c["feature"]]["weight"] * (100 - c["grade"]))
                    if low and res["scores"]["overall"] < STRONG_SWING else None)
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
