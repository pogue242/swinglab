# Prompt: UI session

Paste everything below the line into the new Claude Code session (on the second device).

---

You are the **UI [UI]** session for SwingLab. A second session on another laptop does functionality work
(scoring, detection, Gemini, server) at the same time. Repo:
https://github.com/pogue242/baseball-batting-analyze, branch `swinglab-handoff`; app in `swinglab/swinglab/`.

1. Setup on this device: clone the repo (or `git pull origin swinglab-handoff`), `git checkout -b ui/home`.
   In `swinglab/swinglab/`: `py -3.12 -m venv .venv`, then
   `.venv\Scripts\python -m pip install -e ".[phone,gemini,server,dev]"`. Unzip
   `swinglab_userdata_for_ui.zip` (I'll copy it over) into `swinglab/swinglab/userdata/` so the Synthetic, Matteo
   and Wilson profiles show. Start the site: `.venv\Scripts\python api\server.py --port 8000`, open
   http://localhost:8000. No Gemini key is needed for UI work (the chat falls back to templates).
2. Read `swinglab/swinglab/docs/NEXT_SESSION_PROMPT.md` first (shared context, file ownership, data contract,
   labeled issue list), then `docs/HANDOFF.md` section 8 (UI) and `docs/TEAM_QA.md`.
3. Work only on items marked **UI** or the UI part of **BOTH**: 9 UI fixes (too many measurements first), 10 match
   `potential_redesign_donotread.html` (repo root) and hide pipeline status from users, 8 app-like upload flow
   (`web/index.html` dialog + `web/upload.html` phone page), 4 sequence view (whole recording + its swings,
   individual scores on the right), 5 kinematic-sequence display (rule verdicts with sources), 7 a suggested
   "My lower back hurts when I swing" chat chip.
4. You own `web/index.html`, `web/upload.html` and the look of `swinglab/render.py`. Don't edit the analysis
   code or `api/server.py`; if you need a new field or endpoint, write the request in
   `docs/CONTRACT_CHANGES.md` under "Requested by UI" and build the UI to hide that element until the data
   exists. Read all JSON fields defensively.
5. Check every change in the browser at desktop and phone width (375 px), with all swing states: hit, miss,
   dry, lab (Synthetic), low-quality data warnings. Show me screenshots before big layout changes.
6. After each item: `.venv\Scripts\python -m pytest -q` (server tests check the pages load), update your
   subsection "14 UI log" in `docs/HANDOFF.md`, commit small, `git pull origin swinglab-handoff` often, merge
   into `swinglab-handoff` when tests pass. The scores in the test data are known to be wrong (the FUNC session
   is fixing them); design for the layout, not the numbers.
