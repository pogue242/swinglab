"""Plain-language reading of the elite-match factors: what the body did, in words a beginner understands.

The coach report opens with a short paragraph built from these phrases (no numbers, no jargon), e.g. "your back
foot stayed flat instead of turning", "your front leg stayed soft when it landed". Gemini gets them in the payload
(`plain_language`) and rewrites them naturally; the template report uses them directly. The stats go in the
detailed part of the report.
"""

from __future__ import annotations

# measurement -> (fault when too LOW, fault when too HIGH); None = not a fault in that direction
FAULTS = {
    "hand_speed_peak": ("your hands didn't whip through the hitting zone very fast", None),
    "torso_turn_total": ("your shoulders didn't turn all the way through the swing", None),
    "pelvis_turn_total": ("your hips didn't open up all the way toward the pitcher, so they couldn't pass much power up", None),
    "drive_speed": ("your legs didn't push your body forward into the swing", None),
    "rear_foot_turn": ("your back foot stayed flat instead of turning up onto the toe, which keeps the back hip from driving through", None),
    "lead_knee_extension": ("your front leg didn't firm up after it landed, so your body kept drifting forward instead of stopping and turning", None),
    "lead_knee_flex_contact": (None, "your front knee stayed bent and soft at contact instead of bracing"),
    "hands_forward_contact": ("your hands were too close to your body when the bat reached the ball", None),
    "separation_max": ("your hips and shoulders turned together, instead of the hips going first and stretching your core", None),
    "torso_yaw_fp": (None, "your shoulders started opening before your front foot was down"),
    "stride_length": ("you barely stepped toward the pitcher, so there was little forward move to turn into power",
                      "your step toward the pitcher was too long, which makes it hard to stay balanced"),
    "fp_to_contact_ms": ("your front foot landed late, rushing the swing",
                         "your front foot landed early and you waited on it, losing the energy from the step"),
    "hips_drift_fp_contact": ("your hips leaned back away from the pitcher", "your hips slid toward the pitcher instead of turning"),
    "head_drift_fp_contact": ("your head pulled back during the swing", "your head moved forward during the swing"),
}
STRENGTHS = {
    "hand_speed": "fast hands", "shoulder_turn": "a full shoulder turn", "hip_turn": "a strong hip turn",
    "leg_drive": "good drive from your legs", "front_leg_block": "a firm front leg", "contact_point": "your hands out in front at contact",
    "stretch": "hips leading your shoulders", "step_through": "a balanced, well-timed step",
}
WEAK_PCT, STRONG_PCT = 70, 90


def _direction(part: dict) -> int:
    """-1 = below the elite range, +1 = above it, 0 = can't tell."""
    v = part.get("value")
    if v is None:
        return 0
    if part.get("elite_band"):
        lo, hi = part["elite_band"]
        return -1 if v < lo else 1 if v > hi else 0
    if part.get("elite_up_to") is not None:
        return 1 if v > part["elite_up_to"] else 0
    if part.get("elite_from") is not None:
        return -1 if v < part["elite_from"] else 0
    return -1          # ratio / heel rule: only "too little" is a fault


def faults(elite: dict | None, max_n: int = 3) -> list[dict]:
    """The weakest factors (under WEAK_PCT), worst first by weight x shortfall, each with its plain fault phrase
    from its weakest measurement(s): [{"factor", "phrase"}]."""
    if not elite:
        return []
    out = []
    for f in elite.get("factors", []):
        if f.get("percent") is None or f["percent"] >= WEAK_PCT or not f.get("parts"):
            continue
        phrases = []
        for p in sorted((p for p in f["parts"] if p.get("percent") is not None), key=lambda p: p["percent"]):
            if p["percent"] >= WEAK_PCT:
                continue
            low, high = FAULTS.get(p["feature"], (None, None))
            d = _direction(p)
            ph = high if d > 0 else low if d < 0 else (low or high)
            if ph and ph not in phrases:
                phrases.append(ph)
        if phrases:
            out.append({"factor": f["label"], "phrase": phrases[0], "also": phrases[1:2],
                        "_rank": f.get("weight", 1.0) * (100 - f["percent"])})
    out.sort(key=lambda x: -x["_rank"])
    return [{k: v for k, v in x.items() if k != "_rank"} for x in out[:max_n]]


def strengths(elite: dict | None, max_n: int = 2) -> list[str]:
    if not elite:
        return []
    good = sorted((f for f in elite.get("factors", []) if f.get("percent") is not None and f["percent"] >= STRONG_PCT
                   and f["key"] in STRENGTHS), key=lambda f: -f.get("weight", 1.0))
    if not good:   # nothing at elite level: the best one or two, if decent
        good = sorted((f for f in elite.get("factors", []) if f.get("percent") is not None and f["key"] in STRENGTHS
                       and f["percent"] >= WEAK_PCT), key=lambda f: -f["percent"])
    return [STRENGTHS[f["key"]] for f in good[:max_n]]


def summary(elite: dict | None) -> str:
    """The template's opening paragraph (Gemini writes its own from the same phrases)."""
    fs, ss = faults(elite), strengths(elite)
    parts = []
    if ss:
        parts.append(f"What's working: {' and '.join(ss)}.")
    if fs:
        parts.append(f"The biggest thing holding you back: {fs[0]['phrase']}.")
        if len(fs) > 1:
            parts.append(f"Also, {fs[1]['phrase']}.")
        parts.append("Fix the first one and the rest of the swing usually gets easier.")
    else:
        parts.append("Nothing big is holding you back; keep repeating this swing.")
    return " ".join(parts)
