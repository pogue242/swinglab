"""Template report used when Gemini is unavailable (no key, offline, rate-limited).

Guarantees the demo never breaks. Same schema as the Gemini report.
"""

from __future__ import annotations

from . import exercises
from .schemas import Drill, SessionReport, SwingReport, TrainingItem

TEMPLATES = {
    "separation_fp": ("Your hips and shoulders turn together",
                      "Keep your front shoulder closed until your front foot lands",
                      Drill(name="Step-and-hold", steps=["Stride and land with your shoulders still pointing at the plate",
                                                         "Hold one second, feel the stretch across your stomach",
                                                         "Then turn and swing"]), "overlay"),
    "separation_max": ("Not much stretch between hips and shoulders",
                       "Let your hips start the turn while your shoulders wait",
                       Drill(name="Hip-lead turns", steps=["Bat across your shoulders", "Turn hips only, keep chest still", "Then let shoulders follow"]), "overlay"),
    "pelvis_vel_peak": ("Your hips turn slowly", "Fire your back hip toward the pitcher",
                        Drill(name="Med-ball hip toss", steps=["Side-on to a wall", "Throw a light ball by turning your hips first"]), "sequencing"),
    "torso_vel_peak": ("Your upper body rotation is slow", "Turn your chest through the ball, not your arms",
                       Drill(name="Rotational throws", steps=["Hold a light ball at your back hip", "Rotate hard and throw it forward"]), "sequencing"),
    "gain_torso_over_pelvis": ("Energy isn't building from hips to chest", "Hips first, then chest, then hands",
                               Drill(name="Separation pauses", steps=["Swing in slow motion", "Pause after the hips turn", "Then finish with chest and hands"]), "sequencing"),
    "seq_pelvis_to_torso_ms": ("Your hips and chest fire at the same time", "Hips first, then chest",
                               Drill(name="Pause swings", steps=["Load and stride", "Turn hips, pause a beat", "Swing"]), "sequencing"),
    "torso_yaw_fp": ("Your shoulders open too early", "Show the pitcher your back shoulder until your foot lands",
                     Drill(name="Closed-shoulder stride", steps=["Stride with your front shoulder pointing at the pitcher", "Land, then turn"]), "overlay"),
    "pelvis_yaw_fp": ("Your hips open early", "Land with your front hip still closed",
                      Drill(name="Toe-in stride", steps=["Land with your front toe slightly closed", "Then rotate"]), "overlay"),
    "hands_forward_contact": ("Your contact point is too deep", "Meet the ball out in front of your front hip",
                              Drill(name="Forward tee", steps=["Move the tee a ball-length toward the pitcher", "Hit line drives without letting the ball get deep"]), "overlay"),
    "stride_length": ("Your stride is short", "Take a controlled step toward the pitcher",
                      Drill(name="Stride line", steps=["Put tape where your front foot should land", "Hit that spot every swing"]), "overlay"),
    "lead_knee_extension": ("Your front leg isn't firming up", "Straighten your front leg as you turn",
                            Drill(name="Front-leg block", steps=["Stride and land", "Push the ground away with your front leg as you swing"]), "overlay"),
    "lead_knee_flex_contact": ("Your front knee stays bent at contact", "Brace your front leg at contact",
                               Drill(name="Front-leg block", steps=["Swing at 50% speed", "Feel the front leg straighten at contact"]), "overlay"),
    "head_drift_fp_contact": ("Your head drifts forward", "Keep your head still and let the ball come to you",
                              Drill(name="Head-still tee work", steps=["Partner holds a hand near your head", "Swing without touching it"]), "overlay"),
    "hips_drift_fp_contact": ("Your hips slide forward", "Turn your hips, don't slide them",
                              Drill(name="Back-knee-to-ball", steps=["Turn so your back knee points at the ball", "Keep your weight centered"]), "overlay"),
}
DEFAULT = ("Solid swing overall", "Repeat this swing and stay smooth",
           Drill(name="Tee repetitions", steps=["Ten swings at 80% effort", "Focus on a balanced finish"]), "scores")


def report(payload: dict) -> SwingReport:
    f = payload.get("focus") or {}
    name = f.get("feature")
    headline, cue, drill, diagram = TEMPLATES.get(name, DEFAULT)
    if name and name not in TEMPLATES:
        headline = f"Work on: {f.get('label', name).lower()}"
    s = payload["scores_0_100"]
    what = (f"Your form scores: power {s['power']:.0f}, sequencing {s['transfer']:.0f}, posture {s['posture']:.0f} "
            f"(percentiles vs. Driveline lab swings).")
    if f:
        what += f" {f['label']}: {f['value']:g} {f['unit']} vs. about {f['typical']:g} for a typical lab hitter."
    why = (f"In the lab data, hitters your size with values like this tended to swing about "
           f"{abs(f['effect_mph']):.1f} mph {'slower' if f['effect_mph'] < 0 else 'faster'}." if f else
           "No single area stands out as holding you back.")
    plan, kinds = [], set()
    for o in payload.get("exercise_options", []):  # one of each kind first (power, strength, mobility, drill)
        if o["kind"] not in kinds and len(plan) < 4:
            kinds.add(o["kind"])
            plan.append(TrainingItem(exercise_id=o["id"], prescription=o["prescription"],
                                     why=f"Builds {o['areas'][0].lower()}."))
    areas = payload.get("training_areas", [])
    return SwingReport(headline=headline, what_happened=what, why_it_matters=why, cue=cue, drill=drill,
                       focus_joint=f.get("anchor_joint", "neck"), focus_phase=f.get("anchor_phase", "contact"),
                       diagram=diagram, caption=f"{f.get('label', 'Your scores')} — you vs. typical lab hitter",
                       training_focus=("Targets: " + ", ".join(a.lower() for a in areas) + ".") if areas else "",
                       training_plan=plan)


def session_report(summary: dict) -> SessionReport:
    n = summary.get("n_swings", 0)
    bs, ov = summary.get("bat_speed", {}), summary.get("overall_score", {})
    d = (bs.get("last_third", 0) or 0) - (bs.get("first_third", 0) or 0)
    rec = summary.get("recurring_issue") or {}
    cons = summary.get("consistency", {})
    most_var = max(cons.items(), key=lambda kv: kv[1]["sd"] / (abs(kv[1]["mean"]) + 1e-6))[1] if cons else None
    return SessionReport(
        headline=(f"{n} swings: {(summary.get('outcomes') or {}).get('hits', 0)} hits, "
                  f"{(summary.get('outcomes') or {}).get('misses', 0)} misses, {(summary.get('outcomes') or {}).get('dry_swings', 0)} dry swings; "
                  f"average predicted bat speed {bs.get('mean', 0):.1f} mph."),
        trend=(f"Predicted bat speed went from {bs.get('first_third', 0):.1f} mph (first third) to "
               f"{bs.get('last_third', 0):.1f} mph (last third); overall score averaged {ov.get('mean', 0):.0f}."),
        consistency=(f"Least consistent: {most_var['label'].lower()} (average {most_var['mean']:g}, spread ±{most_var['sd']:g} {most_var['unit']})."
                     if most_var else "Not enough swings to judge consistency."),
        recurring_issue=(f"{rec['label']} came up on {rec['swings']} of {n} swings." if rec else "No single issue repeated."),
        best_swing_id=str(summary.get("best_swing", "")),
        next_session_focus=(f"Work on {rec['label'].lower()}." if rec else "Keep repeating your best swing."),
        fatigue_note=("Bat speed dropped late in the session; take more rest between swings." if d <= -1.0 else ""))
