# SwingLab — Fresh-Start Build Brief (Session B)

For a Claude Code session that builds the project **from scratch**, in
parallel with Session A (which continues the existing `swinglab` repo). This
brief is self-contained: product goals, verified facts about the data, the
parameters and pitfalls already discovered, a data contract so both builds can
share recordings and a viewer, a phased plan, and acceptance tests.

**Coordination rules**
- Work in a **separate folder/repo** (e.g. `swinglab-b`). Do not edit Session
  A's repo; no shared branches.
- Read the same recordings (`SwingLab/` shared folder, read-only) and write
  results in the **same JSON contract** (section 6) to `swinglab-b/web/data/`,
  so either build's viewer can show either build's results.
- Purpose of B: an independent implementation to cross-check A's numbers on
  real swings, and room to try different choices (other pose model, other
  learner, other UI). Where A and B disagree on a real swing, investigate.
- Report numbers, not impressions; mark anything unverified.

---

## 1. Product (what we're building)

Film a baseball swing with an **iPhone 17 Pro** (video + LiDAR depth, app
**Stray Scanner**, free) and a **second phone at 240 fps** (slow motion). The
system rebuilds the hitter in 3D, measures the mechanics that drive bat speed
and contact, rates the swing against **677 lab swings from Driveline
Baseball's OpenBiomechanics** dataset, measures the ball off a tee, and uses
**Gemini** to explain what to fix (cue, drill, diagram choice, training plan).
A long batting-cage recording is split into swings and summarized over time.
Team: Adrian & Wilson, Hack Knight hackathon, both on **Windows** laptops
(Python 3.12). Gemini API key available.

Tracks: Best Use of Gemini API (core), Best Overall, Best Health Hack,
ElevenLabs and Presage (stretch), GoDaddy domain.

### Features (from the team's outline, expanded)

**F1 Swing analysis (core).** Body tracking (MediaPipe Pose or similar) +
LiDAR depth → 3D joints in meters → swing events (load, foot plant, contact)
→ mechanics: hip and shoulder rotation and speed, hip–shoulder separation,
kinematic sequence / power transfer (pelvis → torso → hands), stride, front-leg
block, trunk tilt, head and hip drift, hands out in front at contact → model
trained on Driveline → scores (Power, Transfer, Contact, Overall), predicted
bat speed and exit velocity, the mechanic costing the most mph.

**F2 Contact and ball (second camera).** From 240 fps side video: contact
frame, exit speed off the tee, launch angle, projected trajectory line drawn on
the video; bat speed at contact from a taped bat tip; contact point (in front
of / behind the body; where on the bat); an early/late proxy (contact point
relative to front foot, since a tee has no pitch).

**F3 Gemini coach.** Readable analysis from the measured numbers only;
one cue, one drill; diagrams (our code draws from real data, Gemini chooses
which and captions); 3D views (replay, ghost of a top lab swing, target pose);
**training plan** of exercises chosen from a curated baseball S&C list
(rotational med-ball throws, landmine rotations, Pallof press, open books,
90/90 hips, single-leg RDL, split squats, trap-bar deadlift, depth drops,
lateral bounds, hip-lead turns, step-back drill, overload/underload bats…) with
a safety disclaimer; follow-up chat; summaries of how the player has been doing.

**F4 Sessions / long video.** One long recording (batting cage, ~50 swings)
→ swings found automatically → per-swing analysis → trends (bat speed, scores,
fatigue), consistency, recurring issue, best swing, Gemini session report;
progress across sessions.

**F5 Easy capture pipeline.** Both phones upload to one shared Google Drive
folder; the laptop (Google Drive for desktop) pairs each LiDAR recording with
its slow-motion clip and analyzes automatically; results appear in a dashboard.
Later: an upload page for phones on the same Wi-Fi.

**F6 UI.** Demo opens like the inspirations (annotated video with skeleton,
angles, stats HUD and ball trajectory), then a dashboard: 3D replay, scores,
measured vs predicted speeds, coach card, training plan, sequencing chart,
measurements vs good-form range, session trends, pipeline status (what works).

**F7 Stretch.** ElevenLabs voice coach; Presage heart/breathing rate from a
webcam **between** swings (fatigue, composure); pitching mode with Driveline's
411 fastballs.

### The hitter profile (required input)
Bats R/L, **height**, **weight**, age, playing level (HS / college /
independent / MiLB). Body size strongly affects bat speed in Driveline's data
(about +0.4 mph per inch, +0.85 mph per 10 lb, +0.2 mph per year, +1.5 mph per
level), so it must be an input and outcomes must be size-adjusted (section 4).

---

## 2. Data — verified facts (2026-10-09)

**Driveline OpenBiomechanics** — https://github.com/drivelineresearch/openbiomechanics
- Free, no login. License: data CC BY-NC-SA 4.0 + exclusion for people tied to
  pro sports orgs or financial-analysis firms; credit Driveline.
- Small tables (raw GitHub URLs, `baseball_hitting/data/…`): `metadata.csv`
  (677 swings: `session_swing`, `user`, height in, mass lb, age, playing level,
  hitter side, bat size, `bat_speed_mph_max_x`, `blast_bat_speed_mph_x`,
  `exit_velo_mph_x`), `poi/poi_metrics.csv` (Driveline's own metrics, 127
  columns), `poi/hittrax.csv` (604 rows: launch angle, distance, result, point
  of impact), `data_dictionary.csv`.
- Large files on the GitHub release `dataset-v1`, direct URLs
  `https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/<name>`:
  `hitting_landmarks.zip` (66 MB → 221 MB CSV; **the one you need**),
  `hitting_joint_angles.zip`, `hitting_joint_velos.zip`, `hitting_force_plate.zip`,
  `hitting_c3d.zip` (432 MB), `pitching_landmarks.zip`, `pitching_joint_velos.zip`.
  SHA-256 checksums in the repo at `scripts/release_checksums.sha256`. Their
  own downloader needs the `gh` CLI; plain HTTP download works.
- Swings were real swings at a pitching machine (~65 mph, ~40 ft); 98 hitters,
  mostly college (517 college, 83 HS, 47 MiLB, 30 independent); 496 R, 181 L.
- `landmarks.csv`: 360 Hz, meters, lab axes **+x toward pitcher, +y toward the
  right-handed batter's box, +z up**. Columns `<name>_x/_y/_z` + event times
  `fp_10_time`, `fp_100_time` (lead-foot force > 10/100 N = foot plant),
  `contact_time` (some swings lack `fp_10_time`).
- **Pitfall:** landmark names are not in the data dictionary.
  `lsjc/rsjc` shoulders, `lejc/rejc` elbows, `lwjc/rwjc` wrists, `lkjc/rkjc`
  knees, `lajc/rajc` ankles are joint centers, **but `lhjc/rhjc` are the HANDS**
  (≈7.6 cm from the wrist). **Hips are `left_hip/right_hip`.** Also
  `thorax_prox` (base of neck), `centerofmass`, `sweet_spot`, `blast_hand`
  (bat). Verify by geometry (thigh ≈ 0.40 m, pelvis width ≈ 0.28 m).
- No video exists for these 677 swings. Driveline's separate **OBP-CV**
  collection (index:
  https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q)
  has multi-camera and iPhone 240 fps videos of other trials; contents not yet
  checked; optional for validating pose accuracy.
- Pitching: 411 fastballs, 100 pitchers, `pitch_speed_mph`; landmarks named
  `shoulder_jc`, `elbow_jc`, `glove_*`, `lead_hip`, `rear_ankle_jc`, ….

**Tools**
- **Stray Scanner** (iOS, free, MIT): folder with `rgb.mp4`,
  `depth/000000.png` (16-bit **millimeters**, 256×192), `confidence/*.png`
  (0/1/2), `odometry.csv` (`timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy,
  cx, cy` per frame — camera pose in ARKit world coords, **Y up**, camera looks
  down −Z), `imu.csv`. Depth frame rate not documented: measure it.
- Record3D is installed on the iPhone but RGBD export needs a $2.99 in-app
  pack; prefer Stray Scanner.
- **MediaPipe** current releases removed `mp.solutions`; use
  `mediapipe.tasks.python.vision.PoseLandmarker` with a downloaded
  `pose_landmarker_heavy.task` model, `RunningMode.VIDEO`. Landmarks 11–16
  shoulders/elbows/wrists, 23–28 hips/knees/ankles (person's left = `l_`).
- **Gemini**: `google-genai` SDK; structured output via
  `GenerateContentConfig(response_mime_type="application/json",
  response_schema=<Pydantic model>)`; free tier is Flash-only; list models
  with the key and pin one.

---

## 3. Lessons and parameters already found (reuse or re-derive)

1. **Train at phone frame rate.** Resample Driveline 360 Hz to the phone's
   rate *before* computing features, and resample phone data to the same rate.
2. **Canonical frame from the stance:** x = rear→lead ankle direction at
   stance, z = up (lab z, or ARKit +Y), floor height = ankle height at stance
   − 0.08 m; mirror left-handed hitters; name joints lead/rear. Features must
   not change under any rotation/translation of the input (test this).
3. **Contact** = time of peak mid-wrist speed **+ 65 ms** (very consistent:
   100% of 677 swings within 50 ms at 30 fps, median error 9 ms).
4. **Foot plant** = last frame the lead ankle is > 3 cm above its stance
   height before peak hand speed, **− 28 ms** (70% within 50 ms, median 22 ms
   at 30 fps). Fallbacks: ankle settles after its fastest motion; else contact
   − 190 ms.
5. **Occlusion:** hands crossing the hips put the hand's depth under the hip
   for a frame (30 cm error). Reject samples > 12 cm from a 5-frame rolling
   median, interpolate gaps < 150 ms. Without this, peak pelvis speed came out
   2.4× too high. Sample depth as the median of the nearer half of a 5×5 window.
6. **Sequencing at 30 fps** is quantized to 33 ms; refine peaks with a parabola
   through the 3 samples around each peak; expect ±20 ms on phone data.
7. **Ball scale from ball size:** a fixed high threshold makes the ball ~20%
   too small (speeds ~20% too high); use a half-way threshold between local
   background and ball brightness. Prefer a yardstick at the tee for scale.
8. **Swing finder for long videos:** peaks of 3D mid-wrist speed ≥ 3.5 m/s at
   least 2 s apart; window −1.6 s … +0.8 s around the peak (lab peaks at 30 fps:
   5.1–8.6 m/s, median 6.9). Finds exactly one swing in each of 677 lab clips.
9. **Validation targets:** your pelvis peak rotation speed should correlate
   ≈ 0.95 with Driveline's `pelvis_angular_velocity_swing_max_x` (medians
   ≈ 715 deg/s); lead-knee max bend before foot plant vs
   `lead_knee_stride_max_x` r ≈ 0.99. Static rotation *angles* agree only
   weakly with Driveline's K-Vest convention (open issue, worse for lefties).
10. **Models that worked:** ridge regression (with imputation + scaling) on 23
    mechanics + height, weight, age, level; athlete-grouped 5-fold CV:
    bat speed R² ≈ 0.31, MAE ≈ 3.1 mph (predict-the-mean 3.8); size only
    R² ≈ 0.20; exit velocity R² ≈ 0.15, MAE ≈ 5.1 (5.7). Gradient boosting was
    no better. Peak hand speed predicts well but isn't coachable: use it to
    predict, not to coach.
11. **Size bias check:** within Driveline's range the CV error shows no
    meaningful size bias (height thirds +0.3 / −0.2 / +0.3 mph; weight thirds
    +0.8 / −0.7 / +0.4). Outside 1.60–1.96 m, 63.5–118 kg, ages 17–31 there is
    no data: warn.
12. **Good-form reference:** the top third of hitters by size-adjusted bat
    speed differ from the bottom third most in peak pelvis speed (+0.66 SD),
    pelvis rotation range (+0.51), torso speed (+0.51), hands out in front at
    contact (+0.46), lead-knee straightening into contact (+0.37).
13. **Gemini:** send numbers only; require structured output; drop any
    exercise id not in the offered list; one session report per session (not
    one per swing) to respect free-tier limits; always have a template fallback.

---

## 4. Required approach for size

- Divide distance features by height.
- Use height, weight, age, level as model inputs.
- Scores = percentile of the **size-adjusted** outcome (residual after
  regressing the outcome on size covariates), so mechanics drive the score.
- Explanations and good-form ranges built on size-adjusted outcomes.
- Warn outside the training range; flag missing weight/age.
- Later: calibrate with our own measured bat speeds (slow-motion bat tip).

---

## 5. Recording setup (same for both sessions)

- Camera A (iPhone 17 Pro, Stray Scanner): facing the hitter's chest, 3–4 m,
  angled 30–45° toward the catcher's side, hip height, landscape, tripod; whole
  body + bat path in frame; start with the hitter still in the stance ~1 s.
- Camera B (240 fps): side view perpendicular to the ball's path, level with
  the tee, 4–5 m, tee in one third of the frame; yardstick at the tee per setup;
  bright tape on the bat end and above the hands.
- Sync on the ball leaving the tee. Real baseballs (field). One swing per
  recording, or one long recording for a cage session.
- Shared folder: `SwingLab/hitters.csv`; `SwingLab/sessions/<date>/lidar/`,
  `/slowmo/`, optional `log.csv` (swing, hitter, set, lidar, slowmo, contact,
  notes). Shot list per hitter: 20 normal, 5 each of 5 deliberate flaws (arms
  only, shoulders open early, lunge, front leg stays bent, no stride), 5 dry,
  tee heights, 3 handheld, optional pitching, demo takes.

---

## 6. Data contract (shared JSON so viewers interoperate)

`web/data/index.json`
```json
{"swings": [{"path": "swings/<session>/<id>.json", "session": "...", "id": "...", "hitter": "...",
             "overall": 61.0, "headline": "...", "stages": {"lidar": true, "slowmo": false, "coach": true}}],
 "sessions": [{"path": "swings/<session>/_session.json", "session": "...", "n_swings": 12, "headline": "..."}]}
```

`web/data/swings/<session>/<id>.json` (one swing)
```json
{"id": "001", "hitter": "Adrian", "bats": "R", "height_m": 1.78, "level": "college", "fps": 30,
 "time": [0.0, 0.033, ...],
 "joints": ["lead_shoulder","rear_shoulder","lead_elbow","rear_elbow","lead_wrist","rear_wrist",
            "lead_hip","rear_hip","lead_knee","rear_knee","lead_ankle","rear_ankle","neck"],
 "J": [[[x,y,z], ... 13 joints], ... per frame],          // meters, canonical frame, z up
 "events": {"load": 0.9, "foot_plant": 1.3, "contact": 1.5},
 "series": {"pelvis_vel": [...], "torso_vel": [...], "hand_speed": [...], "separation": [...], "lead_knee_flex": [...]},
 "features": {"separation_fp": 21.0, ...},
 "analysis": {"predicted": {"bat_speed": 66.2, "exit_velo": 88.0},
              "scores": {"power": 41, "transfer": 22, "contact": 63, "overall": 42},
              "weakest_group": "transfer", "focus": {"feature": "...", "label": "...", "value": 0, "typical": 0,
              "effect_mph": -0.9, "unit": "..."}, "contributions": {...}, "tree_path": {...}},
 "measured": {"bat_speed_mph": null, "exit_velo_mph": 68.8},
 "report": {"headline": "...", "what_happened": "...", "why_it_matters": "...", "cue": "...",
            "drill": {"name": "...", "steps": ["..."]}, "focus_joint": "lead_hip", "focus_phase": "foot_plant",
            "diagram": "sequencing", "caption": "...", "training_focus": "...",
            "training_plan": [{"exercise_id": "...", "why": "...", "prescription": "..."}]},
 "report_source": {"source": "gemini"},
 "payload": {"exercise_options": [{"id": "...", "name": "...", "kind": "power", "how": "...", "prescription": "..."}],
             "training_disclaimer": "..."},
 "stages": {"lidar": {"ok": true}, "slowmo": {"ok": true}, "coach": {"ok": true, "source": "gemini"}},
 "warnings": ["..."]}
```
`_session.json`: `n_swings`, `per_swing` [{id, bat_speed, exit_velo, overall,
focus}], `bat_speed` {mean, first_third, last_third, trend_per_10_swings},
`overall_score` {mean, trend_per_10_swings}, `consistency` {feature: {label,
mean, sd, unit}}, `recurring_issue`, `best_swing`, `report` (headline, trend,
consistency, recurring_issue, best_swing_id, next_session_focus, fatigue_note).

---

## 7. Phased build plan for Session B

| Phase | Build | Done when |
|---|---|---|
| 0 Setup | Repo, venv (Python 3.12, Windows), dependencies, data download with checksum, Gemini ping | data loads; key works |
| 1 Lab skeleton | Loader, landmark mapping (pitfall above), canonical frame, resampling | geometry checks pass; rotation invariance test |
| 2 Events | Contact and foot plant from skeleton | ≥ 95% contact and ≥ 60% foot plant within 50 ms at phone fps |
| 3 Features | The F1 mechanics | pelvis speed r ≥ 0.9 vs Driveline |
| 4 Models | Size-aware predictor, CV by athlete, scores, focus, good-form ranges | bat speed beats predict-the-mean and size-only |
| 5 Phone path | Stray Scanner loader, pose, depth lift, occlusion repair | rendered synthetic recording reproduces ground-truth features |
| 6 Slow motion | Ball detect/track/contact/launch, trajectory, bat tip | synthetic within 5%; real clips: ball found > 80% |
| 7 Coach | Payload, Gemini structured report, training plan, session report, fallback | schema-valid; no invented numbers or exercises |
| 8 Sessions + inbox | Swing finder, summaries, shared-folder processing | correct swing counts; folder → results |
| 9 UI | Overlay video + dashboard per F6 (or reuse A's viewer via the contract) | demo runs end to end offline |

Test and verify at every phase; keep a status table of what passed and on
what data; never mark real-data items done from synthetic tests alone.

### First prompt for Session B
> Read docs/FRESH_START_BRIEF.md. Build in a new folder `swinglab-b`, on
> Windows with Python 3.12. Do Phase 0 and Phase 1: download the Driveline
> hitting data (direct release URLs, verify checksums), confirm the landmark
> mapping by geometry, build the canonical skeleton with resampling to 30 fps,
> and write tests for rotation invariance and lefty mirroring. Report what
> passed with numbers, then continue phase by phase.
