"""Pack one analyzed swing into the compact JSON Gemini receives."""

from __future__ import annotations

import numpy as np

from .. import features
from . import exercises, plain

FOCUS_ANCHOR = {
    # feature -> (joint to highlight, phase to jump to)
    "separation_fp": ("lead_shoulder", "foot_plant"), "separation_max": ("lead_shoulder", "foot_plant"),
    "torso_yaw_fp": ("lead_shoulder", "foot_plant"), "pelvis_yaw_fp": ("lead_hip", "foot_plant"),
    "pelvis_rotation_range": ("lead_hip", "contact"), "pelvis_vel_peak": ("lead_hip", "foot_plant"),
    "torso_vel_peak": ("lead_shoulder", "contact"), "stride_length": ("lead_ankle", "foot_plant"),
    "stance_width": ("lead_ankle", "stance"), "seq_pelvis_to_torso_ms": ("lead_hip", "foot_plant"),
    "seq_torso_to_hands_ms": ("lead_wrist", "contact"), "gain_torso_over_pelvis": ("lead_shoulder", "contact"),
    "fp_to_contact_ms": ("lead_ankle", "foot_plant"), "lead_knee_flex_fp": ("lead_knee", "foot_plant"),
    "lead_knee_flex_contact": ("lead_knee", "contact"), "lead_knee_extension": ("lead_knee", "contact"),
    "trunk_forward_tilt_contact": ("neck", "contact"), "trunk_lateral_tilt_contact": ("neck", "contact"),
    "head_drift_fp_contact": ("neck", "contact"), "hips_drift_fp_contact": ("lead_hip", "contact"),
    "hands_forward_contact": ("lead_wrist", "contact"),
    "torso_turn_total": ("lead_shoulder", "contact"), "pelvis_turn_total": ("lead_hip", "contact"),
    "drive_speed": ("rear_ankle", "foot_plant"), "rear_foot_turn": ("rear_ankle", "contact"),
}


def elite_block(el: dict | None) -> dict | None:
    """The elite-match review, compact, for Gemini (elite.match output)."""
    if not el:
        return None
    keep = ("value", "unit", "percent", "elite_median", "elite_band", "elite_from", "elite_up_to", "target", "skipped")
    return {"match_percent": el["match"], "at_elite": el["at_elite"], "factors_judged": el["n_judged"],
            "standard": el.get("standard"),
            "factors": [{"factor": f["label"], "percent_of_elite": f["percent"], "verdict": f["verdict"],
                         "weight": f.get("weight", 1.0),
                         **({"note": f["note"]} if f.get("note") else {}),
                         "measurements": [{"name": p["label"], **{k: p[k] for k in keep if p.get(k) is not None}}
                                          for p in f["parts"]]} for f in el["factors"]]}


def build(analysis: dict, feats: dict, hitter: dict, measured: dict | None = None, vitals: dict | None = None) -> dict:
    """analysis = scoring.analyze(...) output; feats = features.compute(...)."""
    def m(name):
        grp, label, unit = features.FEATURES[name] if name in features.FEATURES else ("factor", *features.FACTOR_FEATURES[name])
        return {"label": label, "value": round(float(feats[name]), 3), "unit": unit, "group": grp}

    focus = analysis.get("focus")
    mode = analysis.get("mode", "dry")
    payload = {
        "hitter": hitter,
        "swing_type": {"hit": "hit: the ball was struck (off a tee or a toss)",
                       "miss": "swing and miss: a ball was in the air but not hit; rate form, say it was a miss, no exit speed",
                       "lab": "lab swing at a pitched ball (Driveline motion capture)"}.get(
            mode, "dry swing (no ball): rate form only; do not mention ball contact, exit speed or launch"),
        "elite_match": elite_block(analysis.get("elite")),
        "plain_language": {"faults": plain.faults(analysis.get("elite")), "strengths": plain.strengths(analysis.get("elite"))},
        "scores_0_100": {k: v for k, v in analysis["scores"].items() if v is not None},
        "score_meaning": {"form": "the elite match: mean of the factors' percent of elite (see elite_match)",
                          "power / transfer / posture": "older lab-range group grades, secondary",
                          "contact": "how squarely the ball was hit (hits only)",
                          "overall": "form for a dry swing; mean of form and contact for a hit"},
        "predicted": {"bat_speed_mph": analysis["predicted"].get("bat_speed"),
                      "exit_velo_mph_vs_65mph_pitch": analysis["predicted"].get("exit_velo")},
        "weakest_area": analysis["weakest_group"],
        "focus": focus and {**focus, "anchor_joint": FOCUS_ANCHOR.get(focus["feature"], ("neck", "contact"))[0],
                            "anchor_phase": FOCUS_ANCHOR.get(focus["feature"], ("neck", "contact"))[1]},
        "top_costs": {g: [c for c in cs if c["effect_mph"] < 0][:2] for g, cs in analysis["contributions"].items()},
        "top_strengths": {g: [c for c in cs[::-1] if c["effect_mph"] > 0][:1] for g, cs in analysis["contributions"].items()},
        "measurements": {k: m(k) for k in (*features.FEATURES, *features.FACTOR_FEATURES)
                         if k in feats and feats[k] is not None and np.isfinite(float(feats[k]))},
        "tree_path_bat_speed": analysis["tree_path"].get("bat_speed", []),
        "notes": ["the elite match compares the swing with elite hitters: the top third of 677 Driveline lab "
                  "swings (98 hitters, mostly college) by bat speed for their size",
                  "effects are size-adjusted associations, not guarantees"],
        "not_graded": [n["label"] for n in analysis.get("not_graded", [])],
    }
    areas = exercises.weak_areas(analysis["contributions"])
    payload["training_areas"] = [exercises.AREAS[a] for a in areas]
    payload["exercise_options"] = exercises.options_for(areas)
    payload["training_disclaimer"] = exercises.DISCLAIMER
    if measured:
        payload["measured"] = measured  # e.g. bat speed from the slow-motion camera
    if analysis.get("contact_detail"):
        payload["contact_detail"] = analysis["contact_detail"]
    if vitals:
        payload["vitals"] = vitals      # Presage, between swings
    return payload
