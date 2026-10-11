"""Export the app's reference data and the "Synthetic" profile (Driveline lab swings).

Run: python scripts/export_demo.py [--ids 103_1 ...] [--refresh-gemini]
Writes
  web/data/app.json                      reference data the viewer needs (lab ranges, top-third bands,
                                         fastest lab swing as the ghost, feature explanations, status)
  <userdata>/swings/synthetic/<id>.json  4 lab swings (+ annotated videos) as the "Synthetic" profile;
                                         user data, so shipping it is a choice (see profiles.py)
With no --ids, picks swings across the score range (strong overall, weak sequencing, weak posture,
average) so every UI state shows.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from swinglab import config, features, kinematics as K, model, obp, profiles, scoring, skeleton
from swinglab.coach import gemini, payload

ap = argparse.ArgumentParser()
ap.add_argument("--ids", nargs="*")
ap.add_argument("--skip-tests", action="store_true", help="keep the last test result in the status tab")
ap.add_argument("--refresh-gemini", action="store_true",
                help="call Gemini even when a cached Gemini report exists for the same payload (free tier: 20/day/model)")
a = ap.parse_args()

models, scorers = model.load(), joblib.load(config.MODELS / "scorers.joblib")
tab = model.load_table()
meta = obp.load_metadata()

# analyze every swing once (fast) to choose examples
res = {sid: scoring.analyze(r[model.MECHANICS].to_dict(), models, scorers, r[model.COVARIATES].to_dict())
       for sid, r in tab.iterrows()}
S = pd.DataFrame({k: v["scores"] for k, v in res.items()}).T
if a.ids:
    picks = a.ids
else:
    picks = [S.overall.idxmax(),
             S[(S.power > 50) & (S.posture > 40)].transfer.idxmin(),
             S[(S.power > 40)].posture.idxmin(),
             (S.overall - 50).abs().idxmin()]
ref_id = tab["bat_speed_mph_max_x"].idxmax()
wanted = set(picks) | {ref_id}

df = pd.read_parquet(obp.landmarks_parquet())
df = df[df.session_swing.isin(wanted)]
out = {"swings": [], "reference": None}
# Gemini's free tier allows 20 requests/day/model and is sometimes overloaded (503/504).
# A swing whose payload is unchanged reuses its last Gemini report without calling the API,
# and a failed call never replaces a good Gemini report with a template ("cached").
SYN = "synthetic"
syn_dir = profiles.swings_dir(SYN)
OLD_GEMINI = {}
for f in list(syn_dir.glob("*.json")) + ([config.WEB_DATA / "demo.json"] if (config.WEB_DATA / "demo.json").exists() else []):
    d = json.loads(f.read_text(encoding="utf-8"))
    for r in d.get("swings", [d]):
        if (r.get("report_source") or {}).get("source") == "gemini" and r["id"] not in OLD_GEMINI:
            OLD_GEMINI[r["id"]] = (r["report"], r["report_source"], r.get("payload"))


def coach(key, call, pl):
    """(report dict, info): cached Gemini report for an identical payload, else a new call;
    on a failed call, the old Gemini report if there is one."""
    old = OLD_GEMINI.get(key)
    same = old is not None and json.dumps(old[2], sort_keys=True, default=str) == json.dumps(pl, sort_keys=True, default=str)
    if same and not a.refresh_gemini:
        return old[0], {**old[1], "cached": True}
    rep, info = call(pl)
    if info["source"] != "gemini" and old is not None:
        return old[0], {**old[1], "cached": True, "latest_attempt": info.get("reason", "")[:120]}
    return rep.model_dump(), info
r3 = lambda x: np.round(np.asarray(x, float), 3).tolist()
for sw in obp.iter_swings(df):
    m = meta.loc[sw.id]
    sk = skeleton.from_obp(sw, m.hitter_side, m.height_m, fps=config.PHONE_FPS)
    s = K.series(sk)
    ev = K.detect_events(sk, s)
    f = features.compute(sk, ev, s)
    rec = {
        "id": sw.id, "bats": m.hitter_side, "height_m": round(float(m.height_m), 2),
        "level": m.highest_playing_level, "fps": config.PHONE_FPS,
        "time": r3(sk.time), "joints": sk.names, "J": r3(sk.J),
        "bat": [r3(sk.extras["bat_handle"]), r3(sk.extras["bat_sweet_spot"])] if "bat_handle" in sk.extras else None,
        "events": {k: round(v, 3) for k, v in ev.items() if isinstance(v, float)},
        "events_true": {"foot_plant": round(sw.events.get("fp_10_time", np.nan) - sw.time[0], 3),
                        "contact": round(sw.events["contact_time"] - sw.time[0], 3)},
        "series": {k: r3(s[k]) for k in ("pelvis_vel", "torso_vel", "hand_speed", "separation", "lead_knee_flex")},
        "measured": {"bat_speed_mph": round(float(m.bat_speed_mph_max_x), 1),
                     "exit_velo_mph": None if pd.isna(m.exit_velo_mph_x) else round(float(m.exit_velo_mph_x), 1)},
        "features": {k: round(float(v), 3) for k, v in f.items()},
    }
    if sw.id == ref_id:
        out["reference"] = rec
    if sw.id in picks:
        an = {**res[sw.id], "mode": "lab"}
        pl = payload.build(an, f, {"bats": m.hitter_side, "height_m": round(float(m.height_m), 2),
                                   "level": m.highest_playing_level})
        rep, info = coach(sw.id, gemini.report, pl)
        rec.update(analysis=an, payload=pl, report=rep, report_source=info, mode="lab", profile=SYN,
                   hitter=f"Lab hitter {sw.id.split('_')[0]}", recorded_at=None,
                   mode_evidence="Driveline motion capture: a real swing at a ~65 mph pitching machine")
        out["swings"].append(rec)
out["swings"].sort(key=lambda r: picks.index(r["id"]))
# The Synthetic profile: lab swings saved like any hitter's swings (with annotated videos).
if profiles.get(SYN) is None:
    profiles.create({"name": "Synthetic", "bats": "R", "height_m": 1.83, "level": "college"}, synthetic=True, pid=SYN)
from swinglab import pipeline
for r in out["swings"]:
    pth = pipeline.save_result(r, SYN, out_root=config.USERDATA)   # renders <id>.mp4 next to it
    print("synthetic swing", pth.name, (r.get("overlay") or {}).get("codec"))
out["feature_info"] = {k: {"group": g, "label": lab, "unit": u, "help": features.FEATURE_HELP[k][0],
                           "better": features.FEATURE_HELP[k][1], "pct_height": features.FEATURE_HELP[k][2]}
                       for k, (g, lab, u) in features.FEATURES.items()}
out["groups"] = {g: {"label": features.GROUP_LABELS[g], "help": features.GROUP_HELP[g]} for g in features.GROUPS}
out["population"] = {k: r3(np.nanpercentile(tab[k], [10, 25, 50, 75, 90])) for k in features.FEATURES}
# "Good form" reference: the top third of hitters by size-adjusted bat speed.
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
dd = tab.dropna(subset=["bat_speed_mph_max_x"])
adj = dd["bat_speed_mph_max_x"] - make_pipeline(SimpleImputer(), RidgeCV()).fit(
    dd[model.COVARIATES], dd["bat_speed_mph_max_x"]).predict(dd[model.COVARIATES])
top = dd[adj >= adj.quantile(2 / 3)]
out["target"] = {k: r3(np.nanpercentile(top[k], [25, 50, 75])) for k in features.FEATURES}

# pipeline status: CV metrics, event validation, test results
metrics = json.loads((config.MODELS / "metrics.json").read_text(encoding="utf-8"))
ev = pd.read_csv(config.PROCESSED / "event_validation.csv")
evs = ev[ev.fps == config.PHONE_FPS]
APP = config.WEB_DATA / "app.json"
old_tests = (json.loads(APP.read_text(encoding="utf-8")).get("status") or {}).get("tests") if APP.exists() else None
tests = None if a.skip_tests else subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                                                 capture_output=True, text=True, cwd=config.ROOT)
out["status"] = {
    "cv": metrics,
    "events": {"n": int(len(evs)),
               "contact_within_50ms": round(float((evs.contact_err_ms.abs() < 50).mean()), 3),
               "contact_median_abs_ms": round(float(evs.contact_err_ms.abs().median()), 1),
               "fp_within_50ms": round(float((evs.fp_err_ms.abs() < 50).mean()), 3),
               "fp_median_abs_ms": round(float(evs.fp_err_ms.abs().median()), 1)},
    "tests": (tests.stdout.strip().splitlines()[-1] if tests.stdout else "not run") if tests else (old_tests or "not run"),
}
config.WEB_DATA.mkdir(parents=True, exist_ok=True)
app = {k: v for k, v in out.items() if k != "swings"}
APP.write_text(json.dumps(app, separators=(",", ":")), encoding="utf-8")
print("wrote", APP, round(APP.stat().st_size / 1e6, 2), "MB; synthetic profile:", syn_dir)
for r in out["swings"]:
    src = r["report_source"]
    print(r["id"], r["analysis"]["scores"], "focus:", (r["analysis"]["focus"] or {}).get("feature"), "|", r["report"]["headline"])
    print("   source:", src["source"] + (" (cached)" if src.get("cached") else ""), src.get("model", src.get("reason", "")[:80]),
          "| numbers not in payload:", [b["number"] for b in src.get("unsupported_numbers", [])] or "none")
print("tests:", out["status"]["tests"])
