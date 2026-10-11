"""Re-score saved swings after a scoring change, from their stored measurements (no pose, no Gemini).

  python scripts\\rescore.py                 # every profile's swings in the user-data folder
  python scripts\\rescore.py --profile wilson
  python scripts\\rescore.py --reports        # also rewrite the coach reports (Gemini if a key is set)

Recomputes analysis (scores, elite match, focus, grades) from rec["features"], keeps the hit/miss/dry mode and
the ball-contact score, and refreshes the recording summaries' numbers. Without --reports the coach text is not
rewritten and may quote the old numbers.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swinglab import config, pipeline, profiles, scoring, session  # noqa: E402


def rescore(rec: dict, hitter: dict) -> dict:
    models, scorers = pipeline._models()
    partial = bool((rec.get("window") or {}).get("partial"))
    old = rec["analysis"]
    an = scoring.analyze(rec["features"], models, scorers, pipeline.covariates(hitter),
                         unreliable=dict.fromkeys(scoring.NEEDS_STANCE, scoring.PARTIAL_REASON) if partial else None)
    an["mode"] = old.get("mode", rec.get("mode", "dry"))
    scoring.with_contact(an, old.get("contact_detail"))
    rec["analysis"] = an
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile")
    ap.add_argument("--reports", action="store_true", help="also rewrite each swing's coach report")
    ap.add_argument("--no-gemini", action="store_true", help="with --reports: template reports only")
    ap.add_argument("--render", action="store_true", help="draw the annotated videos again (they show the scores)")
    a = ap.parse_args()
    for pr in profiles.list_profiles():
        if pr.get("synthetic") or (a.profile and pr["id"] != a.profile):
            continue
        hitter = profiles.hitter(pr)
        d = profiles.swings_dir(pr["id"])
        by_rec: dict[str, list] = {}
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("_"):
                continue
            rec = json.loads(f.read_text(encoding="utf-8"))
            if "analysis" not in rec or "features" not in rec:
                continue
            before = rec["analysis"]["scores"].get("overall")
            rescore(rec, hitter)
            if a.reports:      # Gemini watches the clip again when a key is set (see pipeline._coach)
                pipeline._coach(rec, hitter, rec.get("measured"), not a.no_gemini)
            if a.render:       # the annotated video shows the scores, so it is drawn again
                pipeline.render_overlay(rec, f.with_suffix(""), config.USERDATA)
            f.write_text(pipeline.dump_json(rec), encoding="utf-8")
            by_rec.setdefault(rec.get("recording") or rec["id"], []).append((f, rec))
            print(f"{rec['id']:34s} {before} -> {rec['analysis']['scores']['overall']}")
        for name, items in by_rec.items():          # recording summaries: same text, new numbers
            recs = [r for _, r in sorted(items, key=lambda x: (x[1].get("window") or {}).get("peak") or 0)]
            pipeline.rotate_focus(recs)
            for f, r in items:
                f.write_text(pipeline.dump_json(r), encoding="utf-8")
            sp = next(iter(d.glob(f"_summary_*{Path(name).stem}.json")), None)
            if sp is None or len(recs) < 2:
                continue
            old = json.loads(sp.read_text(encoding="utf-8"))
            new = session.summarize(recs, source=old.get("source"), quality=old.get("quality"))
            for k in ("report", "report_source", "recording_video", "notes", "slowmo_matching", "hitter", "warnings", "recording"):
                if k in old:
                    new[k] = old[k]
            sp.write_text(pipeline.dump_json(new), encoding="utf-8")
            print(f"  summary {sp.name}: form mean {new['form_score'].get('mean')}")
    pipeline.rebuild_index(config.USERDATA)


if __name__ == "__main__":
    main()
