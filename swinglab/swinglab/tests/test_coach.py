"""Coach: payload -> report, with Gemini mocked (no API key needed)."""
import json
import numpy as np
from types import SimpleNamespace

import pytest

from swinglab.coach import cues, gemini, payload
from swinglab.coach.schemas import SwingReport
from conftest import needs_models

FAKE_ANALYSIS = {
    "predicted": {"bat_speed": 66.2, "exit_velo": 88.0},
    "scores": {"power": 41.0, "transfer": 22.5, "posture": 63.0, "form": 42.2, "contact": None, "overall": 42.2},
    "weakest_group": "transfer",
    "focus": {"feature": "seq_pelvis_to_torso_ms", "value": 0.0, "typical": 33.3, "effect_mph": -0.9,
              "label": "Pelvis peak → torso peak", "unit": "ms"},
    "contributions": {"power": [], "transfer": [], "contact": []},
    "tree_path": {"bat_speed": []},
}
FAKE_FEATS = {"separation_fp": 12.0, "seq_pelvis_to_torso_ms": 0.0}


def make_payload():
    return payload.build(FAKE_ANALYSIS, FAKE_FEATS, {"bats": "R", "height_m": 1.8})


def test_payload_is_json_and_anchored():
    p = make_payload()
    json.dumps(p)
    assert p["focus"]["anchor_joint"] == "lead_hip" and p["focus"]["anchor_phase"] == "foot_plant"


def test_template_report_validates():
    r = cues.report(make_payload())
    assert isinstance(r, SwingReport) and r.diagram == "sequencing"


def test_gemini_path_with_fake_client():
    good = cues.report(make_payload()).model_copy(update={"headline": "from gemini"})
    resp = SimpleNamespace(parsed=good, text=good.model_dump_json())
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))
    r, info = gemini.report(make_payload(), client=client)
    assert info["source"] == "gemini" and r.headline == "from gemini"


def test_gemini_failure_falls_back():
    def boom(**kw):
        raise RuntimeError("quota")
    client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    r, info = gemini.report(make_payload(), client=client)
    assert info["source"] == "template" and "quota" in info["reason"]


def test_no_key_uses_template(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r, info = gemini.report(make_payload())
    assert info["source"] == "template"


@needs_models
def test_end_to_end_on_a_driveline_swing():
    import pandas as pd
    from swinglab import config, model, scoring
    import joblib
    models = model.load()
    scorers = joblib.load(config.MODELS / "scorers.joblib")
    tab = model.load_table()
    row = tab.iloc[0]
    res = scoring.analyze(row[model.MECHANICS].to_dict(), models, scorers, row[model.COVARIATES].to_dict())
    p = payload.build(res, row[model.MECHANICS].to_dict(), {"bats": row.hitter_side})
    rep = cues.report(p)
    assert 0 <= res["scores"]["overall"] <= 100 and rep.cue


def test_training_plan_from_weak_areas():
    from swinglab.coach import exercises
    contrib = {"power": [{"feature": "pelvis_vel_peak", "effect_mph": -1.2}],
               "transfer": [{"feature": "separation_fp", "effect_mph": -0.5}], "contact": []}
    areas = exercises.weak_areas(contrib)
    assert areas[0] in ("hip_power", "lower_body_strength") and "thoracic_mobility" in areas
    opts = exercises.options_for(areas)
    assert opts and all(o["id"] in exercises.BY_ID for o in opts)
    a = dict(FAKE_ANALYSIS, contributions=contrib)
    p = payload.build(a, FAKE_FEATS, {"bats": "R"})
    r = cues.report(p)
    assert 2 <= len(r.training_plan) <= 4 and all(t.exercise_id in {o["id"] for o in p["exercise_options"]} for t in r.training_plan)


def test_gemini_invented_exercise_is_dropped():
    from swinglab.coach.schemas import TrainingItem
    contrib = {"power": [{"feature": "pelvis_vel_peak", "effect_mph": -1.2}], "transfer": [], "contact": []}
    p = payload.build(dict(FAKE_ANALYSIS, contributions=contrib), FAKE_FEATS, {"bats": "R"})
    ok_id = p["exercise_options"][0]["id"]
    bad = cues.report(p).model_copy(update={"training_plan": [
        TrainingItem(exercise_id="made_up_lift", why="x", prescription="1x1"),
        TrainingItem(exercise_id=ok_id, why="y", prescription="3x5")]})
    resp = SimpleNamespace(parsed=bad, text=bad.model_dump_json())
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))
    r, info = gemini.report(p, client=client)
    ids = [t.exercise_id for t in r.training_plan]
    assert "made_up_lift" not in ids and ok_id in ids and info["dropped_exercises"] == ["made_up_lift"]


# ---- no invented numbers (item 2) -------------------------------------------------

def test_number_check_flags_invented_and_accepts_rounding():
    from swinglab.coach import numbers
    p = make_payload()   # separation_fp 12.0, focus value 0.0, typical 33.3, effect -0.9, bat speed 66.2
    ok = cues.report(p).model_copy(update={
        "what_happened": "Your hips and chest peaked together (0 ms apart vs. about 33 ms); predicted bat speed 66 mph.",
        "why_it_matters": "Hitters like this swung about 0.9 mph slower; 0.03 s is a long gap."})
    assert numbers.unsupported(ok, p) == []
    bad = ok.model_copy(update={"what_happened": "Your bat speed was 81.7 mph and you lost 4 mph."})
    found = {b["number"] for b in numbers.unsupported(bad, p)}
    assert found == {"81.7", "4"}, found


def test_number_check_unit_changes_and_instructions():
    from swinglab.coach import numbers
    p = {"x": {"value": 293.0, "unit": "ms"}, "r": 0.15, "pelvis": 709.4}
    rep = {"what_happened": "Plant to contact took 0.29 s; hands 15% of your height out front; about 710 deg/s.",
           "drill": {"name": "Tee", "steps": ["10 swings at 80% effort"]}, "prescription": "3 × 5"}
    assert numbers.unsupported(rep, p) == []
    assert [b["number"] for b in numbers.unsupported({"cue": "aim for 25% more"}, p)] == ["25"]


def test_gemini_report_flags_unsupported_numbers():
    invented = cues.report(make_payload()).model_copy(update={"headline": "You swing 91 mph"})
    resp = SimpleNamespace(parsed=invented, text=invented.model_dump_json())
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp))
    r, info = gemini.report(make_payload(), client=client)
    assert info["source"] == "gemini" and [b["number"] for b in info["unsupported_numbers"]] == ["91"]


@needs_models
def test_template_reports_use_only_payload_numbers():
    """The checker must not cry wolf: template text is built only from payload numbers,
    so every lab swing's template report has to come out clean."""
    import joblib
    from swinglab import config, model, scoring
    from swinglab.coach import numbers
    models, scorers = model.load(), joblib.load(config.MODELS / "scorers.joblib")
    tab = model.load_table()
    flagged = []
    for sid, row in tab.iloc[::7].iterrows():   # ~100 swings across the table
        res = scoring.analyze(row[model.MECHANICS].to_dict(), models, scorers, row[model.COVARIATES].to_dict())
        p = payload.build(res, row[model.MECHANICS].to_dict(), {"bats": row.hitter_side})
        flagged += [(sid, b["number"], b["context"]) for b in numbers.unsupported(cues.report(p), p)]
    assert not flagged, flagged[:5]


def test_bad_key_falls_back_to_template(monkeypatch):
    """Real SDK client with an invalid key: the API (or the network) refuses, nothing raises,
    the template report is used and the UI is told why."""
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-swinglab-test")
    monkeypatch.setattr(gemini, "TIMEOUT_S", 15.0)
    r, info = gemini.report(make_payload())
    assert info["source"] == "template" and info["reason"] and isinstance(r, SwingReport)
    srep, sinfo = gemini.session_report({"n_swings": 0})
    assert sinfo["source"] == "template"
    assert "unavailable" in gemini.chat(make_payload(), [], "Why is my score low?")


def test_busy_model_retries_on_fallback_model():
    good = cues.report(make_payload()).model_copy(update={"headline": "from fallback"})
    asked = []

    class Busy(Exception):
        code = 503

    def gen(model, **kw):
        asked.append(model)
        if model == "pinned":
            raise Busy("high demand")
        return SimpleNamespace(parsed=good, text=good.model_dump_json(), model_version="fallback-001")
    r, info = gemini.report(make_payload(), client=SimpleNamespace(models=SimpleNamespace(generate_content=gen)), model="pinned")
    assert asked == ["pinned", gemini.FALLBACK_MODELS[0]] and info["source"] == "gemini" and info["served_by"] == "fallback-001"


def test_quota_exhausted_walks_the_whole_chain_then_template():
    asked = []

    class Quota(Exception):
        code = 429

    def gen(model, **kw):
        asked.append(model)
        raise Quota("20 per day")
    r, info = gemini.report(make_payload(), client=SimpleNamespace(models=SimpleNamespace(generate_content=gen)), model="pinned")
    assert asked == ["pinned", *gemini.FALLBACK_MODELS] and info["source"] == "template" and "Quota" in info["reason"]


def test_non_retryable_error_does_not_burn_other_models():
    asked = []

    class Bad(Exception):
        code = 400

    def gen(model, **kw):
        asked.append(model)
        raise Bad("schema problem")
    r, info = gemini.report(make_payload(), client=SimpleNamespace(models=SimpleNamespace(generate_content=gen)), model="pinned")
    assert asked == ["pinned"] and info["source"] == "template"


@pytest.mark.live
def test_live_gemini_report_has_no_invented_numbers():
    """Real API call (pytest -m live): schema-valid report, and every number traceable to the payload."""
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("no GEMINI_API_KEY")
    r, info = gemini.report(make_payload())
    assert info["source"] == "gemini", info
    assert not info.get("unsupported_numbers"), info["unsupported_numbers"]
    assert isinstance(r, SwingReport) and r.headline and r.cue


def test_gemini_html_entities_are_unescaped():
    rep = cues.report(make_payload())
    plan = [t.model_copy(update={"prescription": "3 &#215; 5 each side"}) for t in rep.training_plan] or []
    sent = rep.model_copy(update={"cue": "Hips first &amp; chest second", "training_plan": plan})
    resp = SimpleNamespace(parsed=sent, text=sent.model_dump_json())
    r, info = gemini.report(make_payload(), client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: resp)))
    assert r.cue == "Hips first & chest second" and all("×" in t.prescription for t in r.training_plan)
    assert not info.get("unsupported_numbers")


def test_number_check_reads_payload_keys():
    from swinglab.coach import numbers
    assert numbers.unsupported({"trend": "down 1.2 mph per 10 swings"}, {"trend_per_10_swings": -1.2}) == []


def test_grades_follow_the_lab_spread():
    from swinglab import scoring
    rng = {"p2.5": 0.0, "p10": 10.0, "p25": 25.0, "p50": 50.0, "p75": 75.0, "p90": 90.0, "p97.5": 100.0}
    assert scoring.grade_feature("hand_speed_peak", 50.0, {**rng, "better": "higher"}) == 50.0
    assert scoring.grade_feature("hand_speed_peak", 120.0, {**rng, "better": "higher"}) == 100.0
    assert scoring.grade_feature("lead_knee_flex_contact", 90.0, {**rng, "better": "lower"}) == 10.0
    r = {**rng, "better": "range"}
    assert scoring.grade_feature("stance_width", 60.0, r) == 100.0               # inside the middle half
    assert scoring.grade_feature("stance_width", 12.5, r) == 50.0                # halfway to p2.5
    assert scoring.grade_feature("stance_width", 110.0, r) == 0.0
    assert scoring.grade_feature("seq_order_ok", 1.0, r) == 100.0 and scoring.grade_feature("stance_width", np.nan, r) is None


@needs_models
def test_unreliable_phone_measurements_are_not_graded():
    """TestSet2: rotation speeds read 2-3x low from the front camera; they must not sink Power."""
    import joblib
    from swinglab import config, model, scoring
    models, scorers = model.load(), joblib.load(config.MODELS / "scorers.joblib")
    row = model.load_table().iloc[0]
    feats = row[model.MECHANICS].to_dict()
    a = scoring.analyze(feats, models, scorers, row[model.COVARIATES].to_dict())
    b = scoring.analyze({**feats, "pelvis_vel_peak": 150.0, "torso_vel_peak": 300.0}, models, scorers,
                        row[model.COVARIATES].to_dict())
    assert a["scores"] == b["scores"] and "pelvis_vel_peak" not in a["grades"]
    assert {n["feature"] for n in a["not_graded"]} == set(scoring.NOT_GRADED)
    slow = scoring.analyze({**feats, "hand_speed_peak": 3.0}, models, scorers, row[model.COVARIATES].to_dict())
    assert slow["scores"]["power"] < a["scores"]["power"] and slow["grades"]["hand_speed_peak"]["grade"] == 0.0
    assert "scores_lab_model" in a and a["grading"] == "elite_match_v1"


@needs_models
def test_key_features_are_graded_and_the_heaviest():
    import joblib
    from swinglab import config, features, scoring
    w = scoring.feature_weights(joblib.load(config.MODELS / "scorers.joblib"))
    graded = sorted((f for f in w if f not in scoring.NOT_GRADED), key=lambda f: -w[f])
    assert not set(features.KEY_FEATURES) & set(scoring.NOT_GRADED)
    assert set(features.KEY_FEATURES) == set(graded[:6])


@needs_models
def test_partial_swing_is_not_coached_on_its_stance():
    """UI report: a partial swing (no stance filmed) got 'Work on: stance width'."""
    import joblib
    from swinglab import config, model, scoring
    models, scorers = model.load(), joblib.load(config.MODELS / "scorers.joblib")
    row = model.load_table().iloc[0]
    feats = {**row[model.MECHANICS].to_dict(), "stance_width": 0.05}          # absurd stance
    unrel = dict.fromkeys(scoring.NEEDS_STANCE, scoring.PARTIAL_REASON)
    a = scoring.analyze(feats, models, scorers, row[model.COVARIATES].to_dict(), unreliable=unrel)
    assert not set(scoring.NEEDS_STANCE) & set(a["grades"]) and (a["focus"] or {}).get("feature") not in scoring.NEEDS_STANCE
    assert any(n["reason"] == scoring.PARTIAL_REASON for n in a["not_graded"])
    assert a["scores"]["transfer"] is None and a["scores"]["form"] == a["elite"]["match"]
    skipped = [p for fc in a["elite"]["factors"] for p in fc["parts"] if p.get("skipped")]
    assert {p["feature"] for p in skipped} == {f for fc in a["elite"]["factors"] for f in [p["feature"] for p in fc["parts"]]} & set(scoring.NEEDS_STANCE)
    assert all(c["feature"] not in scoring.NEEDS_STANCE for cs in a["contributions"].values() for c in cs)


# ---------------------------------------------------------------- elite match (2026-10-11)

REF = {"definition": "test", "measurements": {
    "hand_speed_peak": {"elite": {"p10": 6.5, "p25": 6.8, "p50": 7.0, "p75": 7.3, "p90": 7.6},
                        "all": {"p2.5": 5.5, "p5": 5.8, "p50": 6.8, "p95": 7.6, "p97.5": 7.8}},
    "lead_knee_flex_contact": {"elite": {"p10": 5.0, "p25": 8.0, "p50": 13.0, "p75": 18.0, "p90": 22.0},
                               "all": {"p2.5": 2.0, "p5": 4.0, "p50": 15.0, "p95": 30.0, "p97.5": 34.0}},
    "stride_length": {"elite": {"p10": 0.02, "p25": 0.06, "p50": 0.11, "p75": 0.16, "p90": 0.2},
                      "all": {"p2.5": -0.02, "p5": 0.0, "p50": 0.11, "p95": 0.22, "p97.5": 0.26}}},
       "size_matched": {"hand_speed_peak": [[1.75, 80, 6.0]] * 30 + [[1.95, 100, 8.0]] * 30}}


def test_elite_percent_rules():
    from swinglab import elite
    assert elite.part_percent("hand_speed_peak", 5.6, "ratio", REF, 1.95, 100)[0] == 70.0        # 5.6 / 8.0
    assert elite.part_percent("hand_speed_peak", 5.4, "ratio", REF, 1.75, 80)[0] == 90.0         # size-matched: / 6.0
    assert elite.part_percent("lead_knee_flex_contact", 18.0, "lower", REF)[0] == 100.0
    # zero point = elite p75 + max(2 x (lab p97.5 - p75), elite IQR) = 18 + 32 = 50
    assert elite.part_percent("lead_knee_flex_contact", 26.0, "lower", REF)[0] == 75.0
    assert elite.part_percent("lead_knee_flex_contact", 50.0, "lower", REF)[0] == 0.0
    assert elite.part_percent("stride_length", 0.11, "range", REF)[0] == 100.0
    assert elite.part_percent("stride_length", 0.40, "range", REF)[0] == 0.0
    assert 0 < elite.part_percent("stride_length", 0.30, "range", REF)[0] < 50
    assert elite.part_percent("rear_foot_turn", 20.0, "heel_turn", REF)[0] == 50.0               # coaching rule
    assert elite.part_percent("rear_foot_turn", float("nan"), "heel_turn", REF)[0] is None


@needs_models
def test_elite_match_headline_count_and_ai_factor():
    from swinglab import elite
    ref = elite.load_reference()
    med = {f: m["elite"]["p50"] for f, m in ref["measurements"].items()}
    el = elite.match({**med, "rear_foot_turn": 40.0}, height_m=1.8, mass_kg=85, ref=ref)
    assert el["n_judged"] == len(elite.FACTORS) and el["weighted_average"] >= 95 and el["at_elite"] >= len(elite.FACTORS) - 1
    assert el["match"] == el["weighted_average"] - elite.SCORE_DEDUCTION      # the fixed deduction
    weak = elite.match({**med, "rear_foot_turn": 40.0, "lead_knee_extension": -20.0, "lead_knee_flex_contact": 60.0}, ref=ref)
    fc, part = elite.weakest(weak)
    assert fc["key"] == "front_leg_block" and fc["verdict"] == "work on" and weak["match"] < el["match"]
    elite.with_ai(weak, 20, "slow and stiff", {"model": "m"})
    assert weak["n_judged"] == len(elite.FACTORS) + 1 and weak["factors"][-1]["label"].startswith("Explosiveness")
    assert elite.weakest(weak)[0]["key"] == "front_leg_block"          # the AI factor is never the focus
    assert elite.match({}, ref=ref)["match"] is None


def test_blind_rating_and_video_report_with_a_fake_client():
    from swinglab.coach.schemas import BlindRating
    calls = []

    class Resp:
        def __init__(self, obj):
            self.parsed, self.text, self.model_version = obj, obj.model_dump_json(), "fake"

    class Models:
        def generate_content(self, model, contents, config):
            calls.append(contents)
            if config.response_schema is BlindRating:
                return Resp(BlindRating(rating=72, upper_body="full turn", legs="soft plant", standout="quick hands"))
            return Resp(SwingReport.model_validate(json.loads(cues.report(make_payload()).model_dump_json())))
    fake = SimpleNamespace(models=Models())
    video = SimpleNamespace(uri="files/x", mime_type="video/mp4")
    r, info = gemini.blind_rating(video, client=fake)
    assert r.rating == 72 and info["source"] == "gemini"
    assert not any(isinstance(c, str) and "{" in c for c in calls[0])      # blind: no numbers sent
    rep, info = gemini.report(make_payload(), client=fake, video=video)
    assert info["source"] == "gemini" and info["video"] and calls[1][0].video_metadata.fps == gemini.VIDEO_FPS
