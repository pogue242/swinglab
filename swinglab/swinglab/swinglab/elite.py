"""Elite match: how close a swing is to elite hitters, factor by factor (scoring agreed with the team 2026-10-11).

Standard: ELITE = the top third of the 677 Driveline lab swings by size-adjusted bat speed (bat speed minus what
height, weight, age and level predict), so "elite" means hitters who get the most bat speed out of their size.

Each FACTOR is a few measurements; each measurement gets a percent of elite by its rule:
  ratio      value / elite median, capped at 100 (speeds and turn amounts, where 0 means "none");
             hand speed against elite hitters of similar height and weight
  higher     100 at or above the elite's 25th percentile, sliding to 0 at the lab's 2.5th percentile
  lower      the mirror: 100 at or below the elite's 75th percentile, 0 at the lab's 97.5th percentile
  range      100 inside the elite's 10th–90th percentile, 0 at the lab's 2.5th / 97.5th percentile
  heel_turn  coaching rule (the lab has no foot markers): back foot turned >= 30° just after contact = 100,
             10° = 0, linear between
A factor's percent = the mean of its measurements' percents. The headline ELITE MATCH = the mean over the
factors that could be judged; "at elite level" = factors at 90% or more. Verdicts: >= 90 elite, >= 70 close,
else work on. One factor, Explosiveness, is rated by the AI coach from the video (with_ai) and labeled so.

Kinetic chain behind the factors: ground -> legs -> pelvis -> torso -> arms -> bat, each part speeding up and then
braking to pass the speed on. The back side drives (Leg drive), the front side brakes (Front-leg block), the hips
turn before the shoulders (Stretch & stay closed, Hip turn, Shoulder turn), the head stays still (Step-through),
and the result is hand speed and where the hands meet the ball (Contact point).

Validation (scripts/build_elite_ref.py prints it): lab elite swings score a median ~95%, the rest ~90%; the
headline follows size-adjusted bat speed (rank correlation ~0.4 per hitter); one hitter's swings vary ±3 points.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, features

ELITE_FILE = "elite_ref.json"
ELITE_FRACTION = 1 / 3
ELITE_PCT, CLOSE_PCT = 90.0, 70.0
HEEL_FULL, HEEL_ZERO = 30.0, 10.0
FRONT_CAMERA = "reads lower than true from a camera in front of the hitter"

FACTORS = [
    {"key": "hand_speed", "label": "Hand speed",
     "help": "How fast your hands move: the result of everything else, and the closest thing to bat speed we can "
             "measure from one camera.",
     "parts": [("hand_speed_peak", "ratio")]},
    {"key": "shoulder_turn", "label": "Shoulder turn",
     "help": "How far your shoulders rotate from the coil to the finish.",
     "parts": [("torso_turn_total", "ratio")], "note": "shoulder turn " + FRONT_CAMERA + " (about 80%)"},
    {"key": "hip_turn", "label": "Hip turn",
     "help": "How far your hips rotate from the coil to the finish: the hips start the turn.",
     "parts": [("pelvis_turn_total", "ratio")], "note": "hip turn " + FRONT_CAMERA + " (about 60%)"},
    {"key": "leg_drive", "label": "Leg drive",
     "help": "The back side drives: your legs push your body toward the pitcher during the stride, and the back heel "
             "turns up and out as the back hip drives through.",
     "parts": [("drive_speed", "ratio"), ("rear_foot_turn", "heel_turn")]},
    {"key": "front_leg_block", "label": "Front-leg block",
     "help": "The front side brakes: the front leg lands and firms up (straightens) into contact, stopping the "
             "forward move so it turns into rotation.",
     "parts": [("lead_knee_extension", "higher"), ("lead_knee_flex_contact", "lower")]},
    {"key": "contact_point", "label": "Contact point",
     "help": "Your hands are out in front of your body when the bat reaches the ball.",
     "parts": [("hands_forward_contact", "higher")]},
    {"key": "stretch", "label": "Stretch & stay closed",
     "help": "Your hips turn ahead of your shoulders (the stretch across your core) while your shoulders stay closed "
             "until the front foot lands.",
     "parts": [("separation_max", "higher"), ("torso_yaw_fp", "lower")]},
    {"key": "step_through", "label": "Step-through",
     "help": "The stride and plant: a controlled step, the front foot landing at the right time, then turning on it "
             "without sliding your hips or head toward the pitcher.",
     "parts": [("stride_length", "range"), ("fp_to_contact_ms", "range"), ("hips_drift_fp_contact", "range"),
               ("head_drift_fp_contact", "range")]},
]
AI_FACTOR = {"key": "explosiveness", "label": "Explosiveness (AI coach's eye)",
             "help": "An overall rating of how athletic and explosive the swing looks, by the AI coach from the video "
                     "alone (before it sees any numbers). Rated, not measured.",
             "note": "rated by the AI from the video, not measured"}
QUANTS_ELITE = (10, 25, 50, 75, 90)
QUANTS_ALL = (2.5, 5, 50, 95, 97.5)


def factor_inputs() -> list[str]:
    return [f for fc in FACTORS for f, _ in fc["parts"]]


def _label(f: str) -> tuple[str, str]:
    if f in features.FEATURES:
        return features.FEATURES[f][1], features.FEATURES[f][2]
    return features.FACTOR_FEATURES[f]


# ---------------------------------------------------------------- the elite reference (built from the lab)

def build_reference(tab: pd.DataFrame) -> dict:
    """``tab``: lab swings with factor_inputs(), model.COVARIATES (height_m, mass_kg, athlete_age, level), the bat
    speed target and "user". Returns the JSON-able reference: quantiles per measurement for the elite and for all
    swings, and the elite swings' (height, mass, hand speed) for size matching."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from . import model
    target = model.TARGETS["bat_speed"]
    d = tab.dropna(subset=[target]).copy()
    cov = make_pipeline(SimpleImputer(), RidgeCV()).fit(d[model.COVARIATES], d[target])
    d["adj"] = d[target] - cov.predict(d[model.COVARIATES])
    elite = d["adj"] >= d["adj"].quantile(1 - ELITE_FRACTION)
    ref = {"definition": "top third of the Driveline lab swings by size-adjusted bat speed",
           "n_swings": int(len(d)), "n_elite_swings": int(elite.sum()), "n_elite_hitters": int(d[elite]["user"].nunique()),
           "measurements": {}}
    for f in factor_inputs():
        if f not in d or d[f].notna().sum() < 20:
            continue
        e, a = d.loc[elite, f].dropna(), d[f].dropna()
        ref["measurements"][f] = {"elite": {f"p{q:g}": round(float(np.percentile(e, q)), 4) for q in QUANTS_ELITE},
                                  "all": {f"p{q:g}": round(float(np.percentile(a, q)), 4) for q in QUANTS_ALL}}
    ref["size_matched"] = {"hand_speed_peak": d.loc[elite, ["height_m", "mass_kg", "hand_speed_peak"]].round(3).values.tolist()}
    return ref


_REF: dict | None = None


def load_reference() -> dict:
    global _REF
    if _REF is None:
        _REF = json.loads((config.MODELS / ELITE_FILE).read_text(encoding="utf-8"))
    return _REF


def size_matched_median(ref: dict, f: str, height_m: float | None, mass_kg: float | None, min_n: int = 30) -> float:
    """Elite median of ``f`` among elite swings of similar height and weight (widening until >= min_n swings)."""
    rows = np.array(ref.get("size_matched", {}).get(f) or [])
    if not len(rows) or height_m is None:
        return ref["measurements"][f]["elite"]["p50"]
    mass_kg = mass_kg if mass_kg is not None else float(np.median(rows[:, 1]))
    for dh, dm in ((0.05, 10), (0.08, 15), (0.12, 25)):
        m = (np.abs(rows[:, 0] - height_m) <= dh) & (np.abs(rows[:, 1] - mass_kg) <= dm)
        if m.sum() >= min_n:
            return float(np.median(rows[m, 2]))
    return ref["measurements"][f]["elite"]["p50"]


# ---------------------------------------------------------------- percents

def part_percent(f: str, v: float, rule: str, ref: dict, height_m=None, mass_kg=None) -> tuple[float | None, dict]:
    """Percent of elite for one measurement, and what it was compared with."""
    if v is None or not np.isfinite(v):
        return None, {}
    if rule == "heel_turn":
        return float(np.clip((v - HEEL_ZERO) / (HEEL_FULL - HEEL_ZERO), 0, 1) * 100), {
            "target": f">= {HEEL_FULL:g}°", "rule": "coaching rule (the lab has no foot markers)"}
    m = ref["measurements"].get(f)
    if m is None:
        return None, {}
    e, a = m["elite"], m["all"]
    if rule == "ratio":
        med = size_matched_median(ref, f, height_m, mass_kg) if f in ref.get("size_matched", {}) else e["p50"]
        return float(np.clip(v / med, 0, 1) * 100) if med > 0 else None, {"elite_median": round(med, 3)}
    if rule == "higher":
        lo, top = a["p2.5"], e["p25"]
        return float(np.clip((v - lo) / (top - lo), 0, 1) * 100), {"elite_from": top, "elite_median": e["p50"]}
    if rule == "lower":
        hi, top = a["p97.5"], e["p75"]
        return float(np.clip((hi - v) / (hi - top), 0, 1) * 100), {"elite_up_to": top, "elite_median": e["p50"]}
    lo_b, hi_b, lo, hi = e["p10"], e["p90"], a["p2.5"], a["p97.5"]
    if lo_b <= v <= hi_b:
        p = 100.0
    else:
        p = (v - lo) / (lo_b - lo) if v < lo_b else (hi - v) / (hi - hi_b)
        p = float(np.clip(p, 0, 1) * 100)
    return p, {"elite_band": [lo_b, hi_b], "elite_median": e["p50"]}


def verdict(p: float | None) -> str:
    if p is None:
        return "not judged"
    return "elite" if p >= ELITE_PCT else "close" if p >= CLOSE_PCT else "work on"


def match(feats: dict, *, height_m: float | None = None, mass_kg: float | None = None, ref: dict | None = None,
          unreliable: dict | None = None) -> dict:
    """The review for one swing: {"match", "at_elite", "n_judged", "factors": [...], "standard"}.
    ``unreliable``: {feature: reason} not judged for this swing (e.g. stance-based ones on a partial swing)."""
    ref = ref or load_reference()
    unreliable = unreliable or {}
    out = []
    for fc in FACTORS:
        parts, ps = [], []
        for f, rule in fc["parts"]:
            label, unit = _label(f)
            v = feats.get(f)
            v = None if v is None or not np.isfinite(float(v)) else float(v)
            if f in unreliable:
                parts.append({"feature": f, "label": label, "unit": unit, "value": v, "percent": None,
                              "rule": rule, "skipped": unreliable[f]})
                continue
            p, cmp_ = part_percent(f, v, rule, ref, height_m, mass_kg)
            parts.append({"feature": f, "label": label, "unit": unit, "value": None if v is None else round(v, 3),
                          "percent": None if p is None else round(p), "rule": rule, **cmp_})
            if p is not None:
                ps.append(p)
        pct = round(float(np.mean(ps))) if ps else None
        out.append({"key": fc["key"], "label": fc["label"], "help": fc["help"], "percent": pct, "verdict": verdict(pct),
                    "parts": parts, **({"note": fc["note"]} if fc.get("note") else {})})
    return _summarize({"factors": out, "standard": ref.get("definition"),
                       "elite_reference": {k: ref.get(k) for k in ("n_elite_swings", "n_elite_hitters", "n_swings")}})


def _summarize(el: dict) -> dict:
    judged = [f["percent"] for f in el["factors"] if f["percent"] is not None]
    el["match"] = round(float(np.mean(judged))) if judged else None
    el["at_elite"] = sum(p >= ELITE_PCT for p in judged)
    el["n_judged"] = len(judged)
    return el


def with_ai(el: dict, rating: float | None, notes: str = "", source: dict | None = None) -> dict:
    """Add (or replace) the AI-rated Explosiveness factor and recompute the headline."""
    el["factors"] = [f for f in el["factors"] if f["key"] != AI_FACTOR["key"]]
    pct = None if rating is None else round(float(np.clip(rating, 0, 100)))
    el["factors"].append({**AI_FACTOR, "percent": pct, "verdict": verdict(pct), "parts": [], "observations": notes,
                          "rated_by": (source or {}).get("model")})
    return _summarize(el)


def weakest(el: dict, *, include_ai: bool = False) -> tuple[dict, dict] | None:
    """(factor, part) to coach: the lowest judged factor below elite level and its lowest part."""
    cands = [f for f in el["factors"] if f["percent"] is not None and f["percent"] < ELITE_PCT
             and (include_ai or f["key"] != AI_FACTOR["key"]) and f["parts"]]
    if not cands:
        return None
    fc = min(cands, key=lambda f: f["percent"])
    parts = [p for p in fc["parts"] if p.get("percent") is not None]
    return (fc, min(parts, key=lambda p: p["percent"])) if parts else None
