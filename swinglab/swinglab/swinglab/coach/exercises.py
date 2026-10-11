"""Curated training library: exercises baseball strength & conditioning programs
commonly use, mapped to the swing areas they support.

Gemini is given only the exercises that match this hitter's weak areas and must
choose from them (ids are validated), so it cannot invent unsafe or
non-existent movements. It personalizes the "why" and the order; prescriptions
come from here.

This is general training information, not medical advice. Youth athletes
should train with a coach; stop any exercise that causes pain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Training "areas": what a group of mechanics needs physically.
AREAS = {
    "hip_power": "Hip rotation speed and power",
    "hip_mobility": "Hip mobility (turning into the front hip)",
    "trunk_rotation": "Upper-body rotation speed",
    "thoracic_mobility": "Upper-back rotation mobility (hip-shoulder separation)",
    "lead_leg_block": "Front-leg strength to brake and firm up",
    "sequencing": "Hips-first timing (kinematic sequence)",
    "stability": "Balance and head/trunk stability",
    "lower_body_strength": "General lower-body strength",
    "footwork": "Footwork: the step, the plant and the back-foot turn",
}

# Which areas each measured mechanic draws on (features.FEATURES keys).
FEATURE_AREAS = {
    "pelvis_vel_peak": ["hip_power", "lower_body_strength"],
    "pelvis_rotation_range": ["hip_mobility", "hip_power"],
    "pelvis_yaw_fp": ["sequencing", "hip_mobility"],
    "separation_fp": ["thoracic_mobility", "sequencing"],
    "separation_max": ["thoracic_mobility", "sequencing"],
    "torso_yaw_fp": ["sequencing", "thoracic_mobility"],
    "torso_vel_peak": ["trunk_rotation", "thoracic_mobility"],
    "gain_torso_over_pelvis": ["trunk_rotation", "sequencing"],
    "seq_pelvis_to_torso_ms": ["sequencing"],
    "seq_torso_to_hands_ms": ["sequencing", "trunk_rotation"],
    "fp_to_contact_ms": ["footwork", "sequencing"],
    "stride_length": ["footwork", "stability"],
    "stance_width": ["stability"],
    "lead_knee_extension": ["lead_leg_block", "footwork"],
    "lead_knee_flex_contact": ["lead_leg_block"],
    "lead_knee_flex_fp": ["lead_leg_block", "stability"],
    "trunk_forward_tilt_contact": ["stability", "thoracic_mobility"],
    "trunk_lateral_tilt_contact": ["stability", "thoracic_mobility"],
    "head_drift_fp_contact": ["stability"],
    "hips_drift_fp_contact": ["stability", "lead_leg_block"],
    "hands_forward_contact": ["sequencing", "trunk_rotation"],
    # elite-match factor measurements (features.FACTOR_FEATURES)
    "torso_turn_total": ["trunk_rotation", "thoracic_mobility"],
    "pelvis_turn_total": ["hip_mobility", "hip_power"],
    "drive_speed": ["hip_power", "lower_body_strength", "footwork"],
    "rear_foot_turn": ["footwork", "hip_power"],
}


@dataclass(frozen=True)
class Exercise:
    id: str
    name: str
    areas: tuple[str, ...]
    kind: str            # power | strength | mobility | drill
    prescription: str    # default sets/reps
    how: str             # one-line description
    equipment: str


LIBRARY = [
    Exercise("mb_scoop_toss", "Med-ball rotational scoop toss", ("hip_power", "trunk_rotation", "sequencing"), "power",
             "3 × 5 each side, full rest, max intent", "Side-on to a wall, load the back hip, turn hips first and throw the ball into the wall at hip height.", "4–6 lb med ball, wall"),
    Exercise("mb_shotput", "Med-ball rotational shot-put throw", ("trunk_rotation", "hip_power"), "power",
             "3 × 5 each side", "Ball at the back shoulder, rotate and push it forward like a shot put.", "4–6 lb med ball"),
    Exercise("mb_step_behind", "Step-behind rotational throw", ("hip_power", "sequencing"), "power",
             "3 × 4 each side", "Step behind with the back foot, land, then throw: trains hips leading the upper body.", "4–6 lb med ball, wall"),
    Exercise("lateral_bound", "Lateral bounds with stick", ("hip_power", "lead_leg_block", "stability"), "power",
             "3 × 5 each side", "Jump sideways off one leg and land on the other, hold the landing for 2 seconds.", "none"),
    Exercise("landmine_rotation", "Landmine rotation", ("trunk_rotation", "hip_power"), "strength",
             "3 × 8 each side", "Arc a barbell end from hip to hip, turning through the hips and trunk.", "barbell, landmine or corner"),
    Exercise("cable_chop", "Cable or band woodchop (high to low)", ("trunk_rotation", "stability"), "strength",
             "3 × 10 each side", "Pull diagonally across the body while the hips stay square and stable.", "cable or band"),
    Exercise("pallof_press", "Half-kneeling Pallof press", ("stability", "sequencing"), "strength",
             "3 × 10 each side, 2 s hold", "Press a band straight out and resist it pulling you into rotation.", "band or cable"),
    Exercise("open_book", "Open books (side-lying thoracic rotation)", ("thoracic_mobility",), "mobility",
             "2 × 8 each side, slow", "Lie on your side, knees bent, rotate the top arm open and follow it with your eyes.", "none"),
    Exercise("quadruped_tspine", "Quadruped thoracic rotations", ("thoracic_mobility",), "mobility",
             "2 × 8 each side", "On hands and knees, hand behind head, rotate the elbow toward the ceiling.", "none"),
    Exercise("ninety_ninety", "90/90 hip switches", ("hip_mobility",), "mobility",
             "2 × 8 switches", "Sit with both knees bent at 90°, rotate both knees to the other side without using hands.", "none"),
    Exercise("hip_ir_stretch", "Lead-hip internal rotation stretch", ("hip_mobility",), "mobility",
             "2 × 30 s each side", "Lying on your back, drop one knee inward toward the floor while the foot stays wide.", "none"),
    Exercise("worlds_greatest", "World's greatest stretch", ("hip_mobility", "thoracic_mobility"), "mobility",
             "1 × 5 each side", "Deep lunge, elbow to instep, then rotate the arm to the sky.", "none"),
    Exercise("couch_stretch", "Couch stretch (hip flexors)", ("hip_mobility",), "mobility",
             "2 × 45 s each side", "Back knee on the floor near a wall, shin up the wall, squeeze the glute.", "wall, pad"),
    Exercise("sl_rdl", "Single-leg Romanian deadlift", ("lead_leg_block", "stability", "lower_body_strength"), "strength",
             "3 × 6 each leg", "Hinge on one leg with a flat back, reach the other leg behind you.", "dumbbell or none"),
    Exercise("bulgarian_split_squat", "Bulgarian split squat", ("lower_body_strength", "lead_leg_block"), "strength",
             "3 × 6–8 each leg", "Back foot on a bench, lower straight down on the front leg.", "bench, dumbbells"),
    Exercise("trap_bar_deadlift", "Trap-bar deadlift", ("lower_body_strength", "hip_power"), "strength",
             "3 × 5, leave 2 reps in reserve", "Push the floor away with both legs, chest up.", "trap bar"),
    Exercise("lateral_lunge", "Lateral lunge", ("lower_body_strength", "hip_mobility", "lead_leg_block"), "strength",
             "3 × 6 each side", "Step wide and sit into one hip, push back to the start.", "none or dumbbell"),
    Exercise("depth_drop_stick", "Depth drop to stick", ("lead_leg_block", "stability"), "power",
             "3 × 4", "Step off a low box and land in an athletic stance, freeze instantly.", "low box (30–45 cm)"),
    Exercise("sl_balance_reach", "Single-leg balance reach", ("stability",), "strength",
             "2 × 6 each leg", "Stand on one leg and reach the other foot forward, side and back without touching down.", "none"),
    Exercise("hip_lead_turns", "Hip-lead turns with bat behind back", ("sequencing", "thoracic_mobility"), "drill",
             "2 × 10", "Bat across the shoulders, turn the hips first while the chest stays closed, then let the shoulders follow.", "bat"),
    Exercise("step_back_drill", "Step-back drill", ("sequencing", "hip_power"), "drill",
             "2 × 8 swings", "Step the front foot back to the back foot, then stride and swing: builds momentum into the turn.", "bat, tee"),
    Exercise("overload_underload", "Overload/underload bat swings", ("trunk_rotation", "hip_power"), "drill",
             "3 sets of 5 swings per bat (heavy, light, game), only when fresh", "Alternate a slightly heavier and lighter bat with your game bat at full intent. Stop if your mechanics break down.", "heavier and lighter bats"),
    Exercise("stride_and_stick", "Stride and stick", ("footwork", "lead_leg_block", "stability"), "drill",
             "2 × 8", "Take your normal step, land on the inside of the front foot with the knee firm, and freeze for 2 seconds "
             "before swinging: teaches a firm, on-time plant.", "bat"),
    Exercise("back_heel_pivot", "Back-heel pivot (\"squash the bug\") swings", ("footwork", "hip_power"), "drill",
             "2 × 10 slow, then 1 × 5 full", "Swing slowly and finish up on the ball of your back foot with the heel pointing "
             "to the sky and the back knee driving toward the front knee.", "bat"),
]
BY_ID = {e.id: e for e in LIBRARY}

DISCLAIMER = ("General training information, not medical advice. Warm up first, use light loads to learn each "
              "movement, train with a coach if you're under 18, and stop if anything hurts.")


def video_url(e: Exercise) -> str:
    """A YouTube search for the exercise (so the link always works; individual videos are not vetted)."""
    from urllib.parse import quote_plus
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"{e.name} baseball hitting exercise")


def weak_areas_elite(elite: dict | None, top_n: int = 3) -> list[str]:
    """Training areas behind the weakest elite-match factors (weight x shortfall, worst first)."""
    if not elite:
        return []
    costs: dict[str, float] = {}
    for f in elite.get("factors", []):
        if f.get("percent") is None or f["percent"] >= 90 or not f.get("parts"):
            continue
        for p in f["parts"]:
            if p.get("percent") is None:
                continue
            for k, area in enumerate(FEATURE_AREAS.get(p["feature"], [])):
                costs[area] = costs.get(area, 0.0) + f.get("weight", 1.0) * (100 - p["percent"]) / (k + 1)
    return [a for a, _ in sorted(costs.items(), key=lambda kv: -kv[1])][:top_n]


def weak_areas(contributions: dict, top_n: int = 3) -> list[str]:
    """Areas behind the costliest mechanics across all three scores (most costly first)."""
    costs: dict[str, float] = {}
    for items in contributions.values():
        for c in items:
            if c["effect_mph"] >= 0:
                continue
            for area in FEATURE_AREAS.get(c["feature"], []):
                costs[area] = costs.get(area, 0.0) + c["effect_mph"]
    return [a for a, _ in sorted(costs.items(), key=lambda kv: kv[1])][:top_n]


def options_for(areas: list[str], max_items: int = 8) -> list[dict]:
    """Exercises covering the given areas, best matches first, mixed by kind."""
    scored = []
    for e in LIBRARY:
        hit = [a for a in areas if a in e.areas]
        if hit:
            scored.append((-len(hit), min(areas.index(a) for a in hit), e))
    scored.sort(key=lambda s: (s[0], s[1]))
    out = []
    for _, _, e in scored[:max_items]:
        d = asdict(e)
        d["areas"] = [AREAS[a] for a in e.areas]
        d["video_url"] = video_url(e)
        out.append(d)
    return out
