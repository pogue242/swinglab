# Prompt: FUNCTIONALITY session

Paste everything below the line into the new Claude Code session (on the laptop that has `TestSet2/`,
`userdata/` and `.env`).

---

You are the **FUNCTIONALITY [FUNC]** session for SwingLab. A second session on another device does UI/look
work at the same time. Repo: https://github.com/pogue242/baseball-batting-analyze, branch `swinglab-handoff`;
app in `swinglab/swinglab/`. Windows 11, PowerShell, Python 3.12 venv at `swinglab/swinglab/.venv`.

1. `git pull origin swinglab-handoff`, then `git checkout -b func/scoring`.
2. Read `swinglab/swinglab/docs/NEXT_SESSION_PROMPT.md` first (shared context, file ownership, data contract,
   labeled issue list), then `docs/HANDOFF.md` sections 6 and 13, `docs/TEAM_QA.md`, `docs/FILES.md`.
3. Work only on items marked **FUNC** or the FUNC part of **BOTH**, in this order: 1 scoring, 2 swing detection,
   3 ball detection, 6 Gemini text, 7 pain questions in chat, 4 and 5 FUNC parts (sequence endpoint,
   kinematic-sequence rules in the swing JSON), 8 hosted-upload plan (plan only, no build).
4. Don't edit `web/` (UI owns it). Adding fields to the swing JSON / API is fine; any rename or removal goes
   in `docs/CONTRACT_CHANGES.md`. Check that file for requests from the UI session.
5. Ground truth: TestSet2 is all dry swings (no ball); Wilson has 9 swings across lidar 1 + lidar 2. Before
   changing the scoring, ask me to rank the 10 swings by eye, and show me a table of phone-measured vs lab-range
   values per feature so we can see where the numbers go wrong.
6. After each item: `.venv\Scripts\python -m pytest -q`, re-run TestSet2 (`scripts\analyze_capture.py` /
   the drop folders), update `docs/HANDOFF.md` section 6 and your subsection "14 FUNC log". Never mark
   something "verified on real data" unless it ran on real recordings. Commit small and merge into
   `swinglab-handoff` when tests pass.
7. The Gemini key in `.env` will be replaced with a better one; don't print or commit it. Save Gemini quota
   (`--no-gemini` while tuning).
