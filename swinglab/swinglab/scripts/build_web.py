"""Build a single self-contained HTML viewer (for sharing/publishing): web/index.html with the app's
reference data (web/data/app.json) and the Synthetic profile's swings inlined.

Run: python scripts/build_web.py [--profile synthetic]  ->  web/dist/swinglab-viewer.html
Read-only (no profiles editing, upload, chat or videos: those need `python api\\server.py`).
Only the profile you name is embedded, so personal swings never end up in the shared file.
"""
import argparse
import json

from swinglab import config, profiles

ap = argparse.ArgumentParser()
ap.add_argument("--profile", default="synthetic")
a = ap.parse_args()

src = (config.ROOT / "web" / "index.html").read_text(encoding="utf-8")
app = json.loads((config.WEB_DATA / "app.json").read_text(encoding="utf-8"))
prof = profiles.get(a.profile)
swings = []
if prof:
    for x in profiles.swing_summaries(a.profile):
        rec = json.loads((profiles.swings_dir(a.profile) / x["file"]).read_text(encoding="utf-8"))
        rec.pop("overlay", None)          # videos are not embedded
        swings.append(rec)
data = {"app": app, "profiles": [prof] if prof else [], "swings": {a.profile: swings} if prof else {}}
out = config.ROOT / "web" / "dist" / "swinglab-viewer.html"
out.parent.mkdir(parents=True, exist_ok=True)
marker = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
assert marker in src
out.write_text(src.replace(marker, f"<script>window.SWINGLAB_DATA={json.dumps(data, separators=(',', ':'))};</script>\n{marker}"),
               encoding="utf-8")
print(out, round(out.stat().st_size / 1e3), "KB,", len(swings), "swings of profile", a.profile)
