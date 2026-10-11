# Handoff prompt for the next session

Paste everything below the line into the new session.

---

You are continuing work on **SwingLab** (baseball swing analysis from an iPhone LiDAR recording + optional
240 fps slow-mo clip). Repo: `baseball-batting-analyze`, app in `swinglab/swinglab/`. Windows 11, PowerShell,
Python 3.12 venv at `swinglab/swinglab/.venv`. Run tests with `.venv\Scripts\python -m pytest -q` from
`swinglab/swinglab` (last run: 119 passed, 1 deselected live-Gemini test). Start the site with
`.venv\Scripts\python api\server.py --port 8000` and open http://localhost:8000.

Read first: `docs/HANDOFF.md` (Saturday checklist at top, status table in section 6, decisions/tasks in
section 13), `docs/TEAM_QA.md` (answers to the team's questions), `docs/FILES.md` (file map).

Not in git (on the laptop only): `swinglab/swinglab/.env` (Gemini key; a new key is coming, don't print it),
`userdata/` (profiles, analyzed swings, drop folders), `TestSet2/` (1.1 GB test recordings), `data/captures`,
`data/raw`, the MediaPipe `.task` model. Never mark anything "verified on real data" unless it ran on real recordings.

## Test data on the laptop
`TestSet2/`: Matteo lidar 6/7/8 (1 swing each), Wilson lidar 1 + lidar 2 (**9 swings total**). All are
**dry swings: no ball**. They were copied into `userdata/drop/<Name>/` and analyzed. Current results:

| Recording | Swings found | Labeled | Form (Power / Sequencing / Posture) |
|---|---|---|---|
| Matteo lidar 6 | 1 | hit | 19 (16 / 40 / 0) |
| Matteo lidar 7 | 1 | hit | 49 (45 / 100 / 3) |
| Matteo lidar 8 | 1 | hit | 1 (0 / 0 / 2) |
| Wilson lidar 1 | 5 | hit, hit, dry, hit, miss | 58, 62, 11, 9, 18 (Power 0 on all) |
| Wilson lidar 2 | 2 | hit, hit | 36, 54 (Power 0 on both) |

So: 9 of 10 swings were wrongly labeled hit/miss, 2 of Wilson's 9 swings were missed, Matteo (the
better hitter) scores lower than Wilson, and Power is 0 for every Wilson swing.

## Issues to fix (from the team, in priority order)

1. **Scoring is wrong** (team issue 2). Matteo's swings should score higher; some of Wilson's weak swings
   score high. It must point out real weak points. Current rules (see `swinglab/scoring.py`, `features.py`):
   - 23 measurements in 3 groups: **Power** (stance/stride, hip–shoulder separation, hip turn, top hip/chest
     rotation speed, top hand speed), **Sequencing** (`seq_pelvis_to_torso_ms`, `seq_torso_to_hands_ms`,
     `gain_torso_over_pelvis`, `seq_order_ok`, foot plant → contact time, shoulders open at plant),
     **Posture** (front knee bend/extension, trunk tilt, head and hip drift, hands out front).
   - Per group, a ridge model trained on the 677 Driveline OpenBiomechanics lab swings predicts an outcome
     from that group's measurements only (Power, Sequencing → bat speed; Posture → exit velo), adjusted for
     height/weight; the score is the **percentile of that prediction among the lab swings**.
   - **Form** = mean of the 3 group scores. **Focus** = the measurement costing the most mph (≥ 0.3 mph) in the
     weakest group. **Contact** (hits only, provisional) = 70% smash factor + 30% launch angle; miss = 0.
     **Overall** = Form (dry) or mean(Form, Contact).
   - Likely causes to investigate (hypotheses, not verified): lab data is 360 Hz marker capture, ours is pose
     every 2nd frame of 60 fps (30 Hz) + LiDAR depth noise, so peak speeds are underestimated → Power 0;
     timing features at 33 ms resolution are noisy, and a linear model can reward noise (Sequencing 100 on a weak
     swing); percentiles against pro/college lab swings compress amateurs to 0; posture ratios depend on
     camera angle and handedness/forward direction. Plan: compare phone-measured features with lab ranges per
     feature, check `bats` (L/R) and the forward axis, consider full-rate pose (60 fps) around the swing,
     consider rule-based grading per feature (lab-normal ranges) instead of / alongside ridge percentiles, and
     get the team to rank the 10 TestSet2 swings by eye as ground truth.
2. **Swing detection misses swings** (issue 5): Wilson had 9 swings, we found 7. Rule in `session.find_swings`:
   hand-speed peak ≥ 3.5 m/s AND shoulder turn ≥ 80°. Check which 2 were dropped and why (likely the
   underestimated speeds above, or two swings merged), tune on TestSet2.
3. **False ball detection** (issue 4): all TestSet2 swings had no ball, but 9/10 got hit/miss. Evidence
   strings say "the ball's path broke…" from the LiDAR video (`ballflight.ball_hit_in_frames`,
   `pipeline.decide_mode`). The LiDAR-video path is too permissive (probably bat/hands/clothing blobs). Make
   dry the default unless the evidence is strong; consider a YOLO model (see TEAM_QA "Free YOLO models").
4. **Sequence view** (issue 3): don't only cut 5 s clips per swing. A multi-swing recording should keep the
   individual swings and also offer a grouped "whole sequence" view: full video + analysis of all swings in it,
   with the individual swing scores listed on the right. `save_session` already writes
   `_summary_<recording>.json`; the UI doesn't show it as a sequence yet.
5. **Grounded kinematic sequence** (issue 7): the chart plots pelvis/torso angular velocity + hand speed; the
   only "rules" are the sequencing features above, graded only through the ridge model. Add explicit grounded
   rules with sources (e.g. proximal-to-distal order, peak timing gaps and speed gains from the Driveline lab
   distribution — percentiles from our own 677 swings), and say in the UI where each rule comes from.
6. **Gemini text is generic/repetitive** (issue 1): most headlines are "hip rotation speed", because Power is 0
   everywhere (issue 1 feeds it). After scoring is fixed, give Gemini more distinct per-swing facts (focus
   feature, comparison with the hitter's other swings, what was best) and forbid repeating the same headline
   within a session. Prompts: `swinglab/coach/prompts/`. A new Gemini key is coming.
7. **Ask Gemini about pain** (issue 8): the chat must handle "my lower back / shoulder / wrist hurts when I
   swing": link it to the measured mechanics where reasonable, give safe general advice, and always say to stop
   and see a doctor / athletic trainer for pain that persists, is sharp, or comes with swelling/numbness. No
   diagnosis. Add tests for this in the chat prompt.
8. **Upload should feel like an app, not local** (issue 9): today it's a laptop server + drop folders + a
   phone page on the same Wi-Fi. Plan a hosted path (cloud storage upload from the phone, server-side
   processing queue, accounts per team) and decide with the team before building.
9. **UI fixes later** (issue 6): e.g. too many measurements shown. Collect, don't start yet.

Also open: live streaming from Record3D over USB (`record3d` 1.4.1 has a Windows cp312 wheel; needs the
paid streaming feature), LiDAR depth caps at ~3.1 m in these files (stand the phone 2–3 m away).

Work style the team expects: run the tests after each item, update HANDOFF section 6, keep user data out of
git, explain answers in plain language.
