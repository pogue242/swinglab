# Next sessions: two parallel tracks

The work is split across **two Claude Code sessions on two devices**:

- **[FUNC] Functionality**: analysis, scoring, detection, Gemini, server/API. Prompt: `docs/NEXT_SESSION_FUNC.md`.
- **[UI] UI / look**: the viewer, the upload pages, the rendered video's look. Prompt: `docs/NEXT_SESSION_UI.md`.

Items marked **[BOTH]** have a FUNC part (data/API) and a UI part (display); each track does only its part.
This file is the shared context both sessions read first.

## Project in one paragraph
**SwingLab** rates a baseball swing from an iPhone LiDAR recording (Record3D `.r3d`) plus an optional 240 fps
slow-mo clip. MediaPipe Pose → 3D joints from LiDAR depth (bone-length fallback) → swings found → 23
measurements → scores from models trained on 677 Driveline lab swings → ball/bat tracking (hit, miss or dry
swing) → Gemini coach text (numbers checked) → annotated MP4 → FastAPI server + single-page viewer. App lives in
`swinglab/swinglab/`. Windows 11, Python 3.12. Deeper docs: `docs/HANDOFF.md` (status table section 6,
decisions section 13), `docs/TEAM_QA.md`, `docs/FILES.md`.

## Who owns which files (to avoid merge conflicts)

| Track | Owns (edits freely) | Must not edit (ask the other track) |
|---|---|---|
| FUNC | `swinglab/**/*.py`, `swinglab/coach/**`, `api/server.py`, `scripts/`, `tests/`, `data/models/` | `web/` |
| UI | `web/index.html`, `web/upload.html`, `swinglab/render.py` (video look only), UI notes | analysis code, `api/server.py` |

- **Data contract**: the UI reads the swing JSON files (`userdata/swings/<profile>/<id>.json`,
  `_summary_<recording>.json`) and the `/api/*` endpoints. FUNC may **add** fields freely; renaming or removing
  a field must be written in `docs/CONTRACT_CHANGES.md` (create it) with the date. UI reads fields
  defensively (missing field → hide that element, never crash).
- If UI needs a new field or endpoint, write the request in `docs/CONTRACT_CHANGES.md` under "Requested by UI";
  FUNC builds it.
- **Git**: base branch `swinglab-handoff`. FUNC works on `func/<topic>`, UI on `ui/<topic>`; commit small,
  `git pull origin swinglab-handoff` often, merge back into `swinglab-handoff` when tests pass.
  Each track logs its progress in its own subsection of `docs/HANDOFF.md` section 14 (not each other's).
- Run `.venv\Scripts\python -m pytest -q` before every merge (last run: 119 passed, 1 deselected).

## Not in git (laptop only)
`.env` (Gemini key; a new key is coming, never print or commit it), `userdata/` (profiles, analyzed swings,
drop folders), `TestSet2/` (1.1 GB recordings), `data/raw`, `data/processed`, `data/captures`. The analyzed swings
(13 MB, no raw recordings) are zipped at `hack knights\swinglab_userdata_for_ui.zip` for the UI device: unzip
into `swinglab/swinglab/userdata/`. It contains Matteo's and Wilson's swing videos, so share it privately
only. Never mark anything "verified on real data" unless it ran on real recordings.

## Test data (TestSet2, on the laptop)
Matteo lidar 6/7/8 (1 swing each), Wilson lidar 1 + lidar 2 (**9 swings total**). All are **dry swings, no ball**.

| Recording | Swings found | Labeled | Form (Power / Sequencing / Posture) |
|---|---|---|---|
| Matteo lidar 6 | 1 | hit | 19 (16 / 40 / 0) |
| Matteo lidar 7 | 1 | hit | 49 (45 / 100 / 3) |
| Matteo lidar 8 | 1 | hit | 1 (0 / 0 / 2) |
| Wilson lidar 1 | 5 | hit, hit, dry, hit, miss | 58, 62, 11, 9, 18 (Power 0 on all) |
| Wilson lidar 2 | 2 | hit, hit | 36, 54 (Power 0 on both) |

9 of 10 swings wrongly labeled hit/miss, 2 of Wilson's 9 swings missed, Matteo (the better hitter) scores lower
than Wilson, Power 0 on every Wilson swing.

## The team's issues, labeled

| # | Issue | Track |
|---|---|---|
| 1 | Scoring is wrong; must point out real weak points (team issue 2) | **FUNC** |
| 2 | Swing detection misses swings: 7 of Wilson's 9 (team issue 5) | **FUNC** |
| 3 | False ball detection: dry swings labeled hit/miss (team issue 4) | **FUNC** |
| 4 | Sequence view: whole multi-swing recording + its swings, scores on the right (team issue 3) | **BOTH** |
| 5 | Grounded kinematic-sequence rules with sources (team issue 7) | **BOTH** |
| 6 | Gemini text generic/repetitive (team issue 1) | **FUNC** |
| 7 | Ask Gemini about pain while swinging (team issue 8) | **FUNC** (+ small UI: suggested question chip) |
| 8 | Upload should feel like an app, not local (team issue 9) | **BOTH** |
| 9 | UI fixes: too many measurements, etc. (team issue 6) | **UI** |
| 10 | App look from `potential_redesign_donotread.html` (MLB colors; home + upload done), hide pipeline status from users | **UI** |
| 11 | Live streaming from the phone over USB (`record3d` 1.4.1 has a Windows cp312 wheel; paid streaming feature) | **FUNC**, later |

### FUNC details
1. **Scoring.** Current rules (`swinglab/scoring.py`, `features.py`): 23 measurements in 3 groups:
   **Power** (stance/stride, hip–shoulder separation, hip turn, top hip/chest rotation speed, top hand speed),
   **Sequencing** (`seq_pelvis_to_torso_ms`, `seq_torso_to_hands_ms`, `gain_torso_over_pelvis`, `seq_order_ok`,
   foot plant → contact time, shoulders open at plant), **Posture** (front knee, trunk tilt, head/hip drift,
   hands out front). Per group a ridge model trained on the 677 lab swings predicts an outcome from that
   group only (Power, Sequencing → bat speed; Posture → exit velo), size-adjusted; the score is that
   prediction's **percentile among the lab swings**. **Form** = mean of the 3. **Focus** = measurement costing the
   most mph (≥ 0.3) in the weakest group. **Contact** (hits only, provisional) = 70% smash factor + 30% launch
   angle; miss = 0. **Overall** = Form (dry) or mean(Form, Contact).
   Hypotheses (not verified): our pose runs at 30 Hz (every 2nd frame of 60 fps) vs 360 Hz lab markers, plus depth
   noise → peak speeds underestimated → Power 0; timing at 33 ms steps is noisy and a linear model can reward
   noise (Sequencing 100 on a weak swing); percentiles against college/pro lab swings squash amateurs to 0;
   posture ratios depend on camera angle, handedness and the forward axis. Plan: compare phone features with
   the lab ranges one by one, check `bats` and the forward axis, try full-rate pose around the swing, consider
   rule-based grading per feature (lab-normal ranges) instead of / alongside ridge percentiles, and get the
   team to rank the 10 TestSet2 swings by eye as ground truth.
2. **Swing detection** (`session.find_swings`: hand-speed peak ≥ 3.5 m/s AND shoulder turn ≥ 80°): find which 2
   Wilson swings were dropped and why (low speeds from issue 1, or two swings merged); tune on TestSet2.
3. **Ball detection**: evidence strings say "the ball's path broke…" from the LiDAR video
   (`ballflight.ball_hit_in_frames`, `pipeline.decide_mode`); too permissive (bat/hands/clothing blobs). Make dry
   the default unless the evidence is strong; YOLO is the fallback (TEAM_QA "Free YOLO models").
4. **Sequence view, FUNC part**: an endpoint (e.g. `/api/sessions/<profile>`) listing recordings with their
   swings, full-recording video and `_summary_<recording>.json`; keep per-swing files as now.
5. **Kinematic sequence, FUNC part**: explicit rules with sources: proximal-to-distal order, peak timing gaps
   and speed gains graded against the 677 lab swings' distribution; put each rule's verdict, value, lab range
   and source text into the swing JSON.
6. **Gemini text**: after scoring is fixed, give Gemini distinct per-swing facts (focus, best measurement,
   comparison with the hitter's other swings) and forbid repeated headlines within a session
   (`swinglab/coach/prompts/`).
7. **Pain questions in chat**: link the pain to the measured mechanics where reasonable, give safe general
   advice, no diagnosis, and always say to stop and see a doctor / athletic trainer if pain persists, is sharp,
   or comes with swelling or numbness. Tests for it.
8. **Upload, FUNC part**: write a short plan for a hosted path (phone upload to cloud storage, processing
   queue, team accounts) for the team to decide on before building.

### UI details
4. **Sequence view, UI part**: a "whole recording" view: full video with each swing marked, the recording's
   summary, and the individual swing scores listed on the right (click one → that swing). Until FUNC's endpoint
   lands, read `_summary_<recording>.json` files directly.
5. **Kinematic sequence, UI part**: show each rule's verdict, the hitter's value, the lab range and where the rule
   comes from; keep the chart (hip speed, chest speed, hand speed).
7. Add a suggested question chip in the chat, e.g. "My lower back hurts when I swing".
8. **Upload, UI part**: make upload feel like an app: clear steps, progress, "analyzing…" state, the result
   opening by itself; phone page `web/upload.html` (served at `/upload`) matching the app look.
9. **UI fixes**: too many measurements (show the few that matter, the rest behind "All measurements"), and
   collect further fixes from the team.
10. Match `potential_redesign_donotread.html` (repo root) more closely; hide pipeline/system status from users
    (developer-only).
