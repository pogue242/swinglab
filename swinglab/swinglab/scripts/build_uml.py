"""Render the Mermaid diagrams in docs/SPEC.md into a viewable page: docs/uml.html."""
import html
import re

from swinglab import config

spec = (config.ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
blocks = re.findall(r"```mermaid\n(.*?)```", spec, re.S)
titles = [("Components", "How the phones, cloud services and laptop modules connect."),
          ("Class diagram", "Data classes, modules and how they depend on each other."),
          ("Sequence: one swing", "From both phones to the viewer for a single swing."),
          ("Sequence: batting-cage session", "One long recording split into swings, one Gemini session report.")]
sections = "\n".join(
    f'<section><h2>{t}</h2><p>{d}</p><div class="dia"><pre class="mermaid">{html.escape(b)}</pre></div></section>'
    for (t, d), b in zip(titles, blocks))
page = f"""<title>SwingLab UML</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500&display=swap">
<style>
/* Layout: one column of diagrams, each in its own scroll box */
:root{{--bg:#f3f4f1;--panel:#ffffff;--ink:#1b2027;--muted:#5d6670;--line:#dcdfd8;--seam:#c2362d;
  --diagram:#ffffff;--display:"Barlow Condensed","Arial Narrow",sans-serif;--body:"IBM Plex Sans",system-ui,sans-serif}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#13171c;--panel:#1b2128;--ink:#e6e9ec;--muted:#9aa4ae;--line:#2b333c;--seam:#ff6b5e;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#13171c;--panel:#1b2128;--ink:#e6e9ec;--muted:#9aa4ae;--line:#2b333c;--seam:#ff6b5e;color-scheme:dark}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 var(--body);margin:0}}
.wrap{{max-width:1200px;margin:0 auto;padding-inline:20px;padding-block:20px 48px}}
h1{{font:700 32px/1 var(--display);text-transform:uppercase;letter-spacing:.02em;margin:0 0 6px}}
h1 span{{color:var(--seam)}}
.lead{{color:var(--muted);margin:0 0 20px;max-width:70ch}}
section{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:16px}}
h2{{font:600 20px/1.2 var(--display);letter-spacing:.06em;text-transform:uppercase;margin:0 0 4px;color:var(--muted)}}
section p{{margin:0 0 10px;color:var(--muted);font-size:14px}}
.dia{{overflow-x:auto}}
pre.mermaid{{background:var(--diagram);border-radius:8px;padding:12px;margin:0;min-width:640px}}
</style>
<div class="wrap">
<h1>Swing<span>Lab</span> UML</h1>
<p class="lead">Architecture of the 3D swing analyzer as built on 2026-10-09: phone capture, Driveline-trained models, Gemini coach, sessions and the shared-folder pipeline. Source: docs/SPEC.md.</p>
{sections}
</div>"""
out = config.ROOT / "docs" / "uml.html"
out.write_text(page, encoding="utf-8")
print(out, len(blocks), "diagrams")
