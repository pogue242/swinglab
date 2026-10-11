# SwingLab — Project Handoff

Hack Knight · Adrian & Wilson · written 2026-10-09 (Friday night, hackathon start)

This is the single source of truth for any Claude Code session picking up the
project. **The full product specification (goals, user interaction, data,
models, status, UML) is `docs/SPEC.md`; diagrams render in `docs/uml.html`.** Read it top to bottom before touching code. Section 10 has the next
session prompts, in order. Section 11 is a standing directive: **test and
verify everything, and keep the status table in section 6 honest.**

---

## ☀️ Saturday morning checklist (first real recordings, in order)

All commands from the repo folder in PowerShell, venv active
(`cd swinglab\swinglab` then `.venv\Scripts\Activate.ps1`; never plain `python3`).
The Gemini key is read from `swinglab\.env`.

**Before leaving (phones):**
1. Record3D: look for a depth range / max distance setting and set it to **≥ 5 m**. Both test files stopped at 3.09 m, which cuts off a hitter at the planned 3–4 m.
2. Both phones: Settings → General → Date & Time → **Set automatically** (clips are matched to swings by clock).
3. Bat: two bands of bright tape (end + just above the hands). **Measure from the knob** to the center of each band and note the bat length.
4. Hitter profiles: create them in the viewer (+ New profile: name, bats, height, weight, age, level). For the
   Drive folder, `hitters.csv` rows become profiles automatically; the optional columns `bat_length_m,tip_band_m,hand_band_m`
   give the tape positions.
5. Gemini free tier = **20 calls per model per day** (resets midnight Pacific). For ~90 swings, enable billing in AI Studio, or accept template reports when the quota runs out (the app falls back on its own).

**At the field, first setup (laptop):**
```powershell
python api\server.py --watch "G:\My Drive\SwingLab"   # phones: the printed http://<laptop IP>:8000/upload (allow Python in the firewall prompt); --watch also analyzes the Drive folder
```
1. Record 3 s of the yardstick upright at the tee (slow-mo phone) → upload as **Yardstick**.
2. Record one normal swing on both phones → upload **LiDAR** (.r3d) and **Slow motion** (.MOV, the original).
3. Check them (paths as printed on the upload page):
```powershell
python scripts\check_recording.py "userdata\inbox\sessions\<date>\lidar\<name>.r3d"   # depth_fps, farthest_depth_m >= 4, warnings
python scripts\check_recording.py "userdata\inbox\sessions\<date>\slowmo\<IMG>.MOV"  # 240 fps (or pass --slowmo-fps 240 later)
python scripts\pose_overlay.py "userdata\inbox\sessions\<date>\lidar\<name>.r3d"     # eyeball the skeleton (green = used)
python scripts\analyze_capture.py --lidar "<…>.r3d" --slowmo "<…>.MOV" --yardstick "<…yardstick….MOV>" --profile Wilson --tip-band 0.815 --hand-band 0.22 --bat-length 0.84
```
4. Read the printout before recording more:
   - `quality.bones.worst_variation` < 0.10?
   - Is any joint `placed_by_bone_length` > 0.3? That means no depth on it.
   - `framing`: feet and hands in frame ≥ 0.9?
   - bat: are `bands found`, `contact from ball leaving the tee` and the scale source `yardstick` all reported?
   - Exit speed and bat speed plausible?

   Fix framing, light or distance now (RECORDING_PLAN section 15). The annotated video is next to the result in `web\data\swings\single\`.
5. Open the viewer at `http://localhost:8000/`: the swing is at the top, with the annotated video and the "Ask the coach" box.

**After the first good clip:**
```powershell
python scripts\process_inbox.py userdata\inbox --list    # pairing (sessions show their clip lists)
python scripts\process_inbox.py "G:\My Drive\SwingLab"   # or let the server (--watch) do it
python -m pytest -q                                  # real-data tests now also run on any .r3d in data\captures
```
- Copy one good `.r3d` to `data\captures\`: `tests\test_capture.py` and `tests\test_pose.py` then check it on every run.
- Depth is 60 fps (Record3D). Retraining at 60 fps (`$env:SWINGLAB_PHONE_FPS=60; python scripts\setup_data.py`, ~3 min) makes timing finer but changes every model number. Decide once, early, and record the before/after in section 6.

---

## 0. Where things stand right now

- Repo: `swinglab/` (Python package + scripts + tests + web viewer). One command
  rebuilds everything from a fresh clone: `python scripts/setup_data.py`
  (~1.5 min after the ~900 MB download).
- **Working and verified tonight:** Driveline data download, lab skeleton,
  swing-event detection, 23 swing features, bat-speed and exit-velocity models,
  three 0–100 scores, coaching focus, Gemini coach module with template
  fallback, ball-flight math, phone-capture loader, 3D web viewer.
  42/42 automated tests pass. **Newest:** recordings are examined for how many swings they hold (1 → single, 2+ →
  session, regardless of length) and split into swings with session trends and a Gemini session report (Session tab), and
  the app warns when a hitter's height/weight/age is outside Driveline's
  range (1.60–1.96 m, 63.5–118 kg, 17–31 years). **New 2026-10-09 evening:** Gemini training
  plans from a curated exercise library, and a shared-folder pipeline (both
  phones upload to one synced folder; one command pairs, analyzes, runs
  Gemini and lists results in the viewer), tested end to end on synthetic
  recordings.
- **Built 2026-10-10 (night session, Windows laptop; details and numbers in section 6):**
  Windows fixes; live Gemini (pinned `gemini-3.6-flash`, fallback chain, number check on
  every report and chat answer); MediaPipe on real phone frames; recording quality checks
  (bone lengths, framing) and a bone-length fallback for limbs without depth; Record3D
  format confirmed on two real night test files; `swinglab/render.py` annotated video,
  synced in the viewer; bat speed from two tape bands + yardstick scale (`ballflight.py`);
  per-swing slow-motion clips matched to the swings of long sessions; `api/server.py`
  (phone upload page, viewer, coach chat). **104 tests pass** (+1 live Gemini test: `pytest -m live`).
- **Ran on real data so far:** only two short night test recordings (no field swing):
  file format, orientation, K, 60 fps depth, MediaPipe, swing finder, rendering. Their 3D
  numbers are flagged as unreliable (arms had no depth at night). **Not yet on real data:**
  field swings, slow-motion bat/ball (synthetic only), session clip matching, phones uploading.
- **Not started:** pitching, ElevenLabs, Presage, domain.

Quick start (both team laptops run **Windows**; use PowerShell and Python
**3.12** from python.org, the safest version for MediaPipe wheels; if only the Microsoft
Store Python exists: `winget install Python.Python.3.12 --scope user`):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[phone,gemini,server,dev]"
python scripts\setup_data.py        # downloads Driveline data (~220 MB), validates, trains, builds viewer + Excel sample
python -m pytest -q                 # expect all green
python scripts\get_pose_model.py    # MediaPipe model (~30 MB)
# Gemini key: put GEMINI_API_KEY=... in swinglab\.env (git-ignored), or set it in the environment
python -m swinglab.coach.gemini --ping
python api\server.py                # viewer at http://localhost:8000, phone upload at /upload
```
(macOS/Linux: same commands with `source .venv/bin/activate` and `export`.)
The downloader is pure Python (no `curl`, `gh` or login needed).

**To look at the data:** open `data\sample\driveline_sample.xlsx` (built by
`setup_data.py`, or `python scripts\make_data_sample.py`). Tabs: README,
metadata (677 swings), poi_metrics (Driveline's own metrics), hittrax (604
ball-flight rows), our_features_30fps (the training table), one swing's raw
360 Hz joints, the same swing after our pipeline, and Driveline's data
dictionary. The full raw tables are in `data\raw\` after setup.

---

## 1. The project (corrected and expanded outline)

**One-line pitch:** film a swing with an iPhone Pro; SwingLab rebuilds the
hitter in 3D, measures the mechanics that drive bat speed and contact, scores
them against 677 lab-recorded swings, and a Gemini coach explains what to fix
with a drill and a diagram.

### Goal 1 — Measure and rate the swing (core)

What the team wrote: train a model to analyze the swing; body movement, angles,
power transfer, geometry; LiDAR + camera; MediaPipe; Driveline to learn what
good form looks like.

**Corrections and detail**
- Driveline does **not** label form as good or bad. It labels **outcomes**:
  bat speed (mocap and Blast sensor) and exit velocity for every swing. The
  model learns *which mechanics go with faster bats and harder contact for a
  hitter of a given size and level*. "Good form" in the app means "associated
  with better outcomes in the lab data," and the copy must say so.
- Body size matters: bigger/older/higher-level hitters swing harder regardless
  of mechanics. Height, mass, age and level are model inputs, and the scores
  and coaching use size-adjusted outcomes so advice is about mechanics.
- **The model is trained entirely on Driveline. Our own videos are for testing
  transfer, tuning thresholds and the demo, not for training.**
- LiDAR's job is turning MediaPipe's 2D keypoints into true 3D joint positions
  in meters. 3D is the project's main edge over 2D apps (see section 2).
- Measured mechanics (implemented, `swinglab/features.py`): stance width,
  stride length, hip–shoulder separation (at foot plant and max), pelvis
  rotation range and peak speed, torso peak speed, hand speed, kinematic
  sequence timing and speed gain (power transfer), foot plant → contact time,
  lead-knee bend and straightening, trunk tilt, head and hip drift, hands out
  in front at contact.
- Scores (implemented, `swinglab/scoring.py`): **Power**, **Transfer**
  (sequencing), **Contact**, each a percentile vs. the 677 lab swings, plus an
  overall score.

**How "good form" is defined (and why we can define it)**

No one in the dataset graded swings as good or bad, so "good form" is defined
from the data in three complementary ways:

1. **Outcome-based (the models):** which mechanics go with higher bat speed
   and exit velocity, after adjusting for height, mass, age and level.
2. **Reference group (the viewer's green bands):** the **top third of hitters
   by size-adjusted bat speed**. Each measurement shows where the hitter sits
   vs. that group's middle 50%. A pro-level subgroup (MiLB + independent: 77
   swings, 12 hitters) is a second reference.
3. **Coaching knowledge (rules and templates):** e.g. the kinematic sequence
   should run pelvis → torso → hands with speed building at each step.

What separates the top third from the bottom third (size-adjusted, 30 fps
features; effect in standard deviations):

| Mechanic | Top third | Bottom third | Effect |
|---|---|---|---|
| Peak pelvis rotation speed | 709 deg/s | 674 deg/s | +0.66 SD |
| Pelvis rotation, load → contact | 121° | 114° | +0.51 |
| Peak torso rotation speed | 1086 deg/s | 1025 deg/s | +0.51 |
| Hands out in front at contact (÷ height) | 0.15 | 0.13 | +0.46 |
| Lead knee straightening, plant → contact | 33° | 29° | +0.37 |
| Trunk tilt toward plate at contact | 22° | 20° | +0.29 |
| Max hip–shoulder separation | 40° | 37° | +0.28 |
| Lead knee bend at contact | 13° | 17° | −0.27 |

The pro subgroup shows the same pattern for a firmer front leg at contact
(−0.47 SD knee bend) and more trunk tilt (+0.35). These are associations in
98 hitters, mostly college; the copy says "hitters who swing hardest for
their size tend to…", never "this will add X mph".

**Do the 677 Driveline swings have video? No.** They are marker motion
capture only (joint positions, force plates, bat markers), so MediaPipe can't
be run on them, and it doesn't need to be: we train on the mocap skeleton and
run MediaPipe + LiDAR only on *our* phone video to produce the same skeleton.
Driveline separately publishes a small **OBP-CV video collection** (see
section 3) that could be used to measure how accurate phone-style pose is
against lab-grade data; that is optional validation, not training data.

### Goal 2 — Contact, bat and ball (second camera)

What the team wrote: where on the bat contact happens, ball exit velocity,
early/late, trajectory with a line drawn on the video; LiDAR + camera; ball
detection; a math engine for speeds.

**Corrections and detail**
- **LiDAR cannot see the ball or the bat at speed.** Depth is 256×192 at about
  video rate; a bat head moves ~1 m between frames at 30 fps. Speeds come from
  a **second phone filming 240 fps slow motion from the side** (~12–15 cm of
  bat travel per frame). This is where "two cameras" comes in.
- Use a **tee**. With no pitch, "early/late" can't be measured directly;
  measure the **contact point relative to the body and front foot** (out in
  front vs. deep), which is what timing changes. Driveline has a point of
  impact per swing (HitTrax `poi_x/y/z`) to compare against. True early/late
  needs live pitches (stretch, outdoors).
- **Exit velocity:** we record **on a baseball field with real baseballs**, so
  exit speed off the tee is a real number. It is still lower than Driveline's,
  whose balls came in from a ~65 mph machine (incoming pitch speed adds to exit
  speed), so compare tee swings with each other, or apply a correction later.
  **Bat speed at contact is the directly comparable number** (Driveline
  `bat_speed_mph_contact_x`) and doubles as ground truth for the Power model
  ("predicted 62, measured 60").
- Outdoors: LiDAR range shrinks in bright sun. Keep the LiDAR phone 3–4 m from
  the hitter, test a clip first, prefer shade or overcast if depth is patchy.
- **Where on the bat:** at the contact frame, distance from the bat tip to the
  ball along the bat line. Needs the bat's two ends tracked in slo-mo. Rough
  (a few cm) but demo-able.
- **Trajectory:** exit speed + launch angle from the first ~8 frames after
  contact, then ballistic projection drawn on the video. The math is
  implemented and tested (`swinglab/ballflight.py`).
- One side camera measures speed in the image plane only (motion toward or
  away from it reads low). A second slo-mo phone in stereo fixes this; stretch.

### Goal 3 — Gemini coach, diagrams, 3D (core)

- Gemini **explains**; it never computes. It receives a JSON payload of
  measured numbers, scores, the focus mechanic and its cost in mph, and returns
  a structured report (headline, explanation, cue, drill, joint and phase to
  highlight, which diagram to show, caption). Implemented with a schema and a
  template fallback (`swinglab/coach/`).
- **Diagrams: Gemini chooses, our code draws**, so every angle in a picture is
  measured. Built: 3D replay with a ghost of the fastest lab swing aligned at
  contact, kinematic-sequence chart, measurements vs. lab range. Next: an
  annotated video overlay (section 8) and a "target pose" ghost.
- Also planned: follow-up chat about one swing (`gemini.chat`), a multimodal
  "video second opinion" (`gemini.video_read`: grip, bat lag, head, finish,
  plus an independent contact-time estimate to cross-check ours), and a session
  summary across swings.

### Training plans (added 2026-10-09)

Gemini also prescribes **workouts** for the physical side of the weakness:
`swinglab/coach/exercises.py` holds 22 exercises common in baseball strength
and conditioning (med-ball rotational throws, landmine rotations, Pallof
press, open books, 90/90 hip switches, single-leg RDL, Bulgarian split squat,
trap-bar deadlift, depth drop to stick, lateral bounds, hip-lead turns,
step-back drill, overload/underload bats…), each tagged with the areas it
trains (hip power, hip mobility, trunk rotation, thoracic mobility, lead-leg
block, sequencing, stability, lower-body strength) and a default prescription.
Each measured mechanic maps to areas (`FEATURE_AREAS`). The payload carries
the hitter's weakest areas and the matching options; Gemini picks 3–4 and
explains why; ids not in the list are dropped (`gemini.validate_plan`); the
template fallback builds a plan too. A safety disclaimer is shown with every
plan (general information, train with a coach under 18, stop if it hurts).
Limit: Driveline's force-plate strength data can't be linked to the swing IDs,
so the link between these exercises and bat speed rests on coaching practice,
not on our data.

### Goal 4 — Stretch: ElevenLabs voice + Presage vitals

- ElevenLabs reads the headline and cue aloud after each swing.
- Presage measures heart and breathing rate from a camera on the face. It is
  unreliable during the swing (motion, distance), so measure **between swings**
  with a laptop webcam: composure before a swing, fatigue across a session,
  heart rate vs. bat speed chart.

### Pitching

Team decision (2026-10-09): pitching **after goals 1–3 work**, if time
allows. Recording a few pitching clips at the field tomorrow costs little, so
capture them anyway and process later.

Recommendation: **Phase 2, after hitting works end to end.** Driveline has 411
pitches from 100 pitchers with pitch speed and the same landmark format
(already named lead/rear/throwing arm), so the pipeline mostly carries over
(new joint mapping, events: foot plant / max external rotation / release,
features, target = pitch speed). Costs: a second set of captures and tests,
and throwing indoors needs a net and permission. Plan ~6–10 hours once hitting
is done; a dry "towel drill" throw is the safe capture.

### Scope tiers

| Tier | Includes |
|---|---|
| **MVP (must demo)** | Phone capture → 3D skeleton → scores + focus → coach report (Gemini or template) → 3D viewer. Pre-recorded demo swings cached. |
| **Should** | Slo-mo bat speed at contact (measured vs. predicted), tee contact point, trajectory line on video, annotated overlay video, Gemini chat + video read, upload page. |
| **Stretch** | Pitching, ElevenLabs voice, Presage between-swing vitals, stereo slo-mo, session progress view, domain. |

---

## 2. Inspiration and prior art

| Project | What it does | What we take | How we differ |
|---|---|---|---|
| **jeremyipark/vision-demos** (GitHub, Apache-2.0) | 7 demos via the VLM Run Gateway (ViTPose+, SAM 3.1, Gemma): **rock_climbing_3d** (iPhone 15 Pro LiDAR capture → wall, holds and climber in 3D, route in meters; camera pose solved per frame), rock_climbing (2D holds + order), **deadlift** (reps from the plate, back straight/rounded via a VLM per frame), **pull_ups** (full/partial vs. your own calibrated dead hang), running (cadence, knee at foot strike), chin_ups, dance_sync (similarity score between dancers). Output is an **annotated MP4** with a side panel, drawn with OpenCV (`render.py`, `panel.py`). | Depth-to-3D lifting idea, edge-robust depth sampling, the annotated-video-with-panel output, angles in a body frame, compare-to-reference. | Nothing in it is trained on outcomes or gives advice; "good" is hand-written thresholds. No interactive UI. We learn from 677 labeled swings and coach. |
| **Pickleball CV pipeline** (Qazi Ammar Arshad, ML Bench; LinkedIn post, Oct 2026) | Phone match video → player detection + tracking IDs, full-body pose, per-side elbow/shoulder/hip/knee angles, stance width and arm span normalized by height, ball detection and hit counting, on-screen stats HUD. States the hard part is turning noisy keypoints into stable metrics. | HUD layout for our overlay video, per-joint angle set, height normalization, ball detection + event counting. | Its angles are 2D image-plane angles, distorted when limbs point toward the camera; we measure in 3D in meters. It reports numbers for a coach to interpret; we score against outcomes and suggest a fix. |
| **Driveline OpenBiomechanics** itself | Besides the data: a browser `swing_visualizer` for C3D swings, a `computer_vision` area (camera calibration, splat reconstruction). | Data, events, metric definitions to validate against. | — |
| **HomeCourt** (basketball) | Phone shot tracking, release angle/time, leg bend, jump. | Proof that phone-only sports CV ships. | 2D, basketball. |
| **Lifting form apps** (FormVision, Lift App with "3D form replay", Metric, Qwik) | Bar path, velocity, rep scores, mostly 2D side view. | Score/report UX patterns. | Baseball swing, true 3D from LiDAR, lab outcomes. |
| **Research** (AIFit/Fit3D, EC3D) | 3D pose feedback for fitness, compare to instructor. | Reference-comparison framing. | — |

**Pitch line:** tools already turn video into angles. 2D angles are distorted
by camera position, and raw angles don't tell a hitter what to change. We
measure in 3D and learn from 677 lab swings what produces bat speed and solid
contact.

---

## 3. Datasets (verification status)

| Dataset | Status | Contents we use | Access / license |
|---|---|---|---|
| **Driveline OBP — hitting** | **Verified 2026-10-09**: downloaded with plain `curl` (no login), SHA-256 matches the repo's `scripts/release_checksums.sha256`, loaded and modeled. | 677 swings, 98 hitters, real swings at a pitching machine (~65 mph, ~40 ft). `metadata.csv` (height, mass, age, level, handedness, bat speed mocap + Blast, exit velo); `poi/poi_metrics.csv` (Driveline's metrics, used only to validate ours); `poi/hittrax.csv` (604 swings: launch angle, distance, result, point of impact); release `hitting_landmarks.zip` (360 Hz joint centers in meters + `fp_10_time`, `fp_100_time`, `contact_time`). Also available: joint angles/velocities, force plates, C3D (432 MB). | Free. Data CC BY-NC-SA 4.0 **plus an exclusion** for people tied to pro sports orgs or financial-analysis firms; code MIT. Non-commercial student use is fine; credit Driveline in the submission. |
| **Driveline OBP — pitching** | **Verified downloadable** (landmarks, joint velocities); not processed yet. | 411 pitches, 100 pitchers, `pitch_speed_mph`; landmarks named `shoulder_jc`, `elbow_jc`, `glove_*`, `lead_hip`, `rear_ankle_jc`… | Same as above. |
| Driveline OBP — high_performance | Exists in repo, not needed | 1,934 force-plate assessments; no link to swing IDs. | Same. |
| **Driveline OBP-CV video collection** | **Exists, contents not yet checked** (index is a [Google Sheet](https://docs.google.com/spreadsheets/d/1NhpF8DnfBdio_xsU7B44KNuHghtePDp-d3juvCYNm9Q/edit?usp=sharing) linking a Google Drive folder, with a Mega backup) | Separate from the 677 swings. Folders: 8 synchronized OptiTrack Prime Color cameras, 4 Edgertronic high-speed cameras, **iPhone 13 Pro 240 fps slow-motion videos**, sample Theia3D markerless C3D outputs, calibration objects. One known trial is a 62.7 mph throw (`OBP_movement_BaseballThrow_001`). Shot-list tabs say which movements (throws, swings?) are included. | Optional: test MediaPipe accuracy against lab-grade pose, and test our ball tracker on Driveline's iPhone 240 fps clips. |
| *Direct links (browser download, no login)* | | Small tables: [metadata.csv](https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/metadata.csv), [poi_metrics.csv](https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/poi_metrics.csv), [hittrax.csv](https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/poi/hittrax.csv), [data_dictionary.csv](https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/baseball_hitting/data/data_dictionary.csv). Large: [hitting_landmarks.zip](https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/hitting_landmarks.zip) (66 MB → 221 MB CSV), [hitting_c3d.zip](https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/hitting_c3d.zip) (432 MB, raw mocap; view in Driveline's `swing_visualizer` or the free Mokka viewer), all assets on the [dataset-v1 release](https://github.com/drivelineresearch/openbiomechanics/releases/tag/dataset-v1). | |
| Statcast via `pybaseball` | **Not verified** | MLB bat-tracking averages for context lines only. | Public. |
| Roboflow Universe bat/ball sets | **Not verified** | Only if the simple ball detector fails on real footage. | Free account; license per dataset. |
| **Our own captures** | Recording tomorrow | Testing transfer, thresholds, demo. Plan in next chat. | Ours. |

---

## 4. Tools, apps and costs

| Item | Cost | Used for | Notes |
|---|---|---|---|
| **Stray Scanner** (iOS, open source, MIT) | **Free** | LiDAR capture (goal 1). Either app works; compare a test recording from each with `check_recording.py`. | Exports `rgb.mp4`, `depth/*.png` (16-bit mm, 256×192), `confidence/*.png`, `odometry.csv` (per-frame camera pose + intrinsics), `imu.csv`. Loader built (`swinglab/capture.py`). Frame rate not documented: **measure tomorrow.** |
| Record3D (iOS) | **Installed on Wilson's phone.** Free download, 3 trial recordings; in-app: Basic $3.99, Full $4.99, **"Wi-Fi Streaming & RGBD Videos" $2.99 is what unlocks RGBD export** (per developer reply on the App Store). | LiDAR capture (goal 1), equal option. | **Loader built** (`capture.load_record3d`, `.r3d` zip or extracted folder: `metadata` JSON with K, w, h, fps, poses; `rgbd/<i>.jpg`; `rgbd/<i>.depth` LZFSE float32 m; `rgbd/<i>.conf`). **Verified on two real exports (2026-10-10, night test):** portrait 1440x1920 color + 192x256 depth, aligned, K correct, 60 fps; loader also reads `dw`/`dh` and per-frame intrinsics. Depth stopped at 3.09 m: check the app's range setting. Same ARKit LiDAR as Stray Scanner, so depth range and resolution are the same; pick whichever exports cleaner data per `check_recording.py`. |
| Built-in camera slo-mo (second phone) | Free | 240 fps bat/ball video (goal 2). | **Team has one.** Check the exported file really is 240 fps (some apps export at 30 fps playback). |
| iPhone 17 Pro with LiDAR | Have (Wilson) | Capture | Check the depth frame rate in the first recording. |
| MediaPipe Pose Landmarker | Free | 2D keypoints | Current releases (0.10.3x) **removed `mp.solutions`**; we use the Tasks API + a `.task` model file (`scripts/get_pose_model.py`). |
| Gemini API (Google AI Studio key) | Free tier | Coach report, chat, video read | Free tier covers Flash models; check rate limits in AI Studio. Default model alias `gemini-flash-latest`; run `python -m swinglab.coach.gemini --list-models` and pin one. |
| ElevenLabs | Free plan has limited monthly credits | Voice (stretch) | Verify credits; ask MLH whether sponsor credits exist. |
| Presage | Free trial + API key | Vitals (stretch) | SDKs for iOS/Android/macOS/Windows/Linux. |
| GoDaddy Registry domain | Paid (varies) | Domain track | Ask MLH for a promo code before buying. |
| three.js, scikit-learn, OpenCV, FastAPI | Free | — | — |
| Hardware | — | — | 2 tripods (or phone clamps), tee, net, foam/wiffle balls (real balls only if venue allows), bat, tape measure + tape marks for scale. Hardware lab: Logitech webcam for Presage. |

---

## 5. Architecture and repo map

```
LAB (training)                                  PHONE (inference)
Driveline landmarks 360 Hz                      Stray Scanner: rgb.mp4 + depth + poses
   obp.py                                          capture.py: MediaPipe 2D → depth lift → world 3D
        \                                          /
         ──► skeleton.py  CANONICAL SKELETON  ◄───
             lead/rear names, body frame from the stance (x toward pitcher, z up,
             floor at 0), lefties mirrored, resampled to PHONE_FPS
                            │
             kinematics.py  series (rotation, speeds, knee…) + events (load, foot plant, contact)
                            │
             features.py    23 mechanics, one row per swing
                            │
             model.py       ridge models: bat speed, exit velo (+ size covariates); explanation trees
             scoring.py     Power / Transfer / Contact percentiles; focus = costliest actionable mechanic
                            │
             coach/         payload.py → gemini.py (report, chat, video_read) | cues.py fallback
                            │
             web/           3D viewer (three.js), sequencing chart, measurements, status tab

SLOW-MO CAMERA (goal 2): ballflight.py  detect ball → track → contact frame → exit speed,
launch angle → trajectory; tip_speed for the bat. Scale from ball size or tape marks.
```

| Path | What | Status |
|---|---|---|
| `swinglab/config.py` | Paths; `PHONE_FPS` (env `SWINGLAB_PHONE_FPS`, default 30) | ✅ |
| `swinglab/obp.py` | Download + checksum, metadata/POI/HitTrax loaders, landmark parquet, `iter_swings` | ✅ verified |
| `swinglab/skeleton.py` | Canonical skeleton, body frame, floor, mirroring, resampling | ✅ tested |
| `swinglab/kinematics.py` | Signals + event detection (fitted offsets) | ✅ validated vs. force plates |
| `swinglab/features.py` | 23 features + labels/units/groups | ✅ |
| `swinglab/model.py`, `scoring.py` | Models, CV, explanation tree, scores, focus | ✅ |
| `swinglab/coach/` | schemas, prompts/*.md, payload, gemini, cues, **exercises.py** (training library) | ✅ offline; live call untested |
| `swinglab/pipeline.py` | analyze_lidar / analyze_slowmo / analyze_swing / save_result + index | ✅ end-to-end on synthetic recordings |
| `swinglab/inbox.py`, `scripts/process_inbox.py` | Shared-folder pairing + processing (`--init`, `--list`, `--watch`) | ✅ synthetic tests + end-to-end |
| `swinglab/capture.py` | Stray Scanner loader, depth sampling, back-projection, ARKit world frame, MediaPipe Tasks | ⚠️ synthetic tests only |
| `swinglab/ballflight.py` | Ball detection, tracking, launch, trajectory, bat tip speed | ⚠️ synthetic tests only |
| `scripts/setup_data.py` | Full rebuild | ✅ ran end to end |
| `scripts/validate_events.py`, `validate_features.py` | Validation reports | ✅ |
| `scripts/train.py`, `export_demo.py`, `build_web.py` | Train, export viewer data, single-file viewer | ✅ |
| `scripts/analyze_capture.py`, `get_pose_model.py` | Phone end to end | ⚠️ not run on real data |
| `scripts/check_recording.py` | Checks a recording right after transfer (depth fps, coverage, slo-mo real fps) | ✅ synthetic test |
| `scripts/make_data_sample.py` | Excel workbook to look at the data | ✅ |
| `web/index.html` | Viewer (fetches `web/data/demo.json`) | ✅ logic smoke-tested in jsdom |
| `tests/` | 42 tests (synthetic + real-data) | ✅ 42/42 |
| `swinglab/session.py` | Swing finder for long recordings, per-swing analysis, session summary | ✅ synthetic + lab |

---

## 6. Verification status (keep this table current)

Labels: **synthetic** = generated test data · **lab data** = Driveline mocap ·
**real recording** = our own phone files. Nothing is "verified on real data"
unless it ran on real recordings.

| Check | Result |
|---|---|
| Unit + integration tests | 2026-10-10 Windows laptop (Python 3.12.10 venv): **47 passed** after item 1 (42 original + 5 Windows regressions); **57** after item 2 (+1 live, deselected by default); 61 after item 3; 71 after item 4; 74 after item 5; 89 after item 6; 97 after item 7; **104 passed** after item 8 (final run 3 min 22 s, through `export_demo.py` too). Real-data tests (`test_real_record3d_*`) ran on the 2 night test files in `data\captures`. |
| Windows setup (2026-10-10) | Fresh venv, `pip install -e ".[phone,gemini,server,dev]"` OK (mediapipe 1.1.0, opencv 5.0, lzfse 0.4.2, numpy 2.5, pandas 3.0, scikit-learn 1.9). `setup_data.py` ran end to end. The laptop only had the Microsoft Store Python 3.11 (`python`/`python3` in WindowsApps); python.org 3.12 installed with `winget install Python.Python.3.12 --scope user`. **Always run through `.venv\Scripts\python.exe` (or activate the venv)**: plain `python3` is the Store stub. Fixed: text files without an encoding (cp1252 on Windows: `build_uml.py` crashed on SPEC.md, Gemini prompts garbled "—"), piped output crashing on "→"/emoji, `export_demo.py` running pytest with `python3` (reported "14 failed" from the wrong interpreter). Guarded by `tests/test_windows.py`. |
| Driveline download | all hitting assets fetched, checksums match (re-verified on Windows 2026-10-10) |
| Landmark meanings (not in Driveline's data dictionary) | checked by geometry: `lhjc`/`rhjc` are **hands** (7.6 cm from wrist); hips are `left_hip`/`right_hip` |
| Contact detection at 30 fps (all 677 swings) | 100% within 50 ms; median error 9 ms (peak hand speed + 65 ms); reproduced on Windows (9.4 ms) |
| Foot plant detection at 30 fps | 70% within 50 ms; median abs error 22 ms (< 1 frame) |
| Our pelvis peak speed vs. Driveline | r = 0.97 at 360 Hz, 0.93 at 30 fps; medians 721 vs 714 deg/s |
| Our lead-knee max bend vs. Driveline | r = 0.99 (same definition) |
| Static rotation angles vs. Driveline (pelvis/torso/separation at foot plant) | **weak**: r ≈ 0.35–0.49 for right-handed hitters, ≈ 0 for left-handed. Likely Driveline's K-Vest angle convention; speeds agree. Open issue. |
| Bat speed model (athlete-grouped 5-fold CV, 30 fps features) | R² 0.32, MAE 3.1 mph (guess-average 3.8). Size only R² 0.20, mechanics only 0.17. Windows rebuild 2026-10-10 (newer sklearn/pandas): R² 0.314, MAE 3.14 (guess-average 3.81); size only 0.197, mechanics only 0.163. |
| Exit velocity model | R² 0.16, MAE 5.1 mph (guess-average 5.7). Windows rebuild: R² 0.154, MAE 5.11 (5.70). |
| Scores on training swings | spread 0–100, median 50 |
| Viewer | four demo swings render; switching swings, phase jumps, status tab, top-third target bands work (jsdom); published as artifact |
| Clean-download rebuild | `setup_data.py` from empty `data/` in 90 s; pure-Python downloader also verified on pitching assets |
| Excel data sample | 8 tabs written and re-read |
| Phone capture on real recording | ⚠️ **ran on 2 real night test recordings (Record3D, iPhone 17 Pro, 2026-10-09 22:17; one hitter standing with a bat 5.1 s, one dry swing 2.85 s); no field swing yet.** `check_recording.py`: both load, **depth 60.0 fps**, no dropped frames (max gap 16.7 ms), depth 192x256, confidence maps present, `set_SWINGLAB_PHONE_FPS_to: 60`. **Record3D format confirmed** (`tests/test_capture.py::test_real_record3d_orientation_and_K` runs on any `.r3d` in `data\captures`): recordings were **portrait** (color 1440x1920); depth is stored in the same orientation as the JPGs (not mirrored, silhouettes line up); `K` fits the portrait JPG (fx 1337, cx 720.7 ≈ w/2, cy 956.8 ≈ h/2; column-major reshape correct); poses are in the same portrait camera frame (floor normal along ARKit world +Y, 0.98–0.998; camera height 1.3–1.4 m). Loader now uses the metadata's `dw`/`dh` and per-frame `perFrameIntrinsicCoeffs` [fx, fy, cx, cy] (fx varies 1336–1340). **Depth range: no reading beyond 3.09 m in either file** (floor cut at a hard line): likely a Record3D range setting — raise it to ≥ 5 m, or keep hitter and bat path < 3 m. `check_recording.py` now reports `farthest_depth_m` and warns. **Arms had no depth during the swing** (85–95% NaN in the file itself around the hands at the finish; dark hoodie, night, fast motion); head too. `analyze_capture.py --lidar`: the dry swing crashed (`array must not contain infs or NaNs`, lead wrist 3D in 1% of frames) → fixed with a plain-language refusal when a joint has 3D in < 50% of frames, and a **bone-length fallback** (`capture.lift_keypoints`: a limb joint with missing/inconsistent depth is placed on its camera ray at bone length from its parent; length from the recording's own consistent depth, else Driveline ratio × height). Now both run; swing finder finds **exactly 1 swing** in the dry swing (none in the standing clip). Numbers are **not trustworthy yet** and are flagged: worst bone varies 21–22%, peak hand speed 12.2 m/s (lab 5.1–8.6), load→foot plant 70 ms; the warnings list which joints were estimated. |
| Recording quality checks (item 4) | ✅ synthetic: `quality.bone_lengths` (thigh, shin, upper arm, forearm; sd ÷ median < 10%; worst bone in `quality.bones`) and `quality.framing` (person found, all 13 joints visible, feet in frame, hands in frame; warn < 90%) in every record, with plain warnings. Tests inject errors into rendered recordings: a knee keypoint slipping 60 px for 0.5 s is caught and named (`lead_thigh`), 4 cm wrist jitter → `lead_forearm`, feet cut off / wrists blurred / person lost → exact fractions. Bone fallback test: **no arm depth for 1.25 s through the whole swing** → arms rebuilt, events within 1 frame, all features not sampled at an event instant match ground truth, peak hand speed within 6%. On the real night clips the checks fire (bones 21–22%, all joints visible 0–42%, feet in frame 78% in the standing clip). **71 passed** after item 4. |
| MediaPipe on real video | ✅ **real phone frames, not a field swing** (2026-10-10): model downloaded on the laptop (30.7 MB, mediapipe 1.1.0). Ran on the two night Record3D test recordings (a hitter standing with a bat 5.1 s; a dry swing 2.85 s, both 60 fps, portrait, street-lit): person found in **172/172 and 307/307 frames**, ~12 fps on CPU. Overlay videos: `python scripts\pose_overlay.py <.r3d|video|folder\%d.jpg>` (green = visibility ≥ 0.5, orange = weaker). Shoulders/hips visibility 1.0; **during the swing, wrists/elbows/rear knee often < 0.5** (lead wrist ≥ 0.5 in only 2% of frames) although the overlay shows them roughly in place (night, motion blur, occlusion). The 3D lift drops joints < 0.5 (`lift_recording(min_vis=...)`): threshold to be decided with the bone-length check (item 4). **JPG sequence:** `mediapipe_keypoints` reads `rgbd\%d.jpg` (all 172 frames, path with spaces); MP4 vs JPGs cut from it agree to a median 2.9 px (p95 23 px, low-visibility wrists). **Bug fixed:** OpenCV reports 25 fps for any JPG sequence, so MediaPipe got wrong timestamps; the capture's true fps is now passed. Tests: `tests/test_pose.py` (3 synthetic + 1 real-data, skipped when no capture/model). **61 passed** after item 3. |
| Ball/bat speed on real 240 fps video | **not yet** (synthetic: math within 5%; from a rendered 240 fps video file through `analyze_slowmo`: exit speed within 2.5%, launch angle within 1°) |
| Bat-tip tracking (item 6, `ballflight.py`) | ✅ **synthetic only** (no real slow-motion clip yet). Two bright tape bands found as blobs brighter than the clip's median background (motion blur dims them below any fixed threshold); streak centroids = mid-exposure positions; velocity linear along the rigid bat → **bat speed at the sweet spot** (15 cm from the end, the Blast/Driveline-comparable number), end-band speed, bat angle and attack angle at contact. Test clips are rendered in 3D with perspective like camera B (side, 4.5–6 m, tee at the image center) and real motion blur (16 or 8 sub-frames over a 1/480 s exposure): the bat points at the camera around contact, so the bands look < half as far apart (94 vs 238 px) — the tracker follows them by prediction instead of spacing. **Results (contact frame from the ball): bat speed −0.4…+1.1% (4 swings, 46–82 mph, camera 4.5/6 m), attack angle ±1.3°; with ball + yardstick through `analyze_slowmo`: bat speed within 0.2%, exit speed within 0.5%, launch angle exact, contact frame exact (4 shots, −5° to 25°, 67–101 mph).** Yardstick scale from a clip: within 0.2–1.3% (profile along the stick, not Hough endpoints). Dry swing (no ball): contact = peak bat speed. Bugs found and fixed on the way: the ball detector locked onto a tape band at rest (now prefers the expected ball size and rejects non-ballistic "flights"); the ball vanishes for frames at contact (bat cover + blur) → contact extrapolated from the fitted flight; the fit used clip time for gravity (+3.6° launch error) → time since contact; the low line drive was taken for a tape band → the fitted ball path is masked out of band tracking. Wired into `analyze_slowmo` (two streaming passes), the record (`measured.bat_speed_mph` → payload "measured" for Gemini, viewer "measured"), `analyze_capture.py --yardstick/--bat-length/--tip-band/--hand-band`, the inbox (slow-mo files named `*yardstick*`/`*calib*` or log rows with set `calibration` = the session's scale clip; optional `bat_length_m, tip_band_m, hand_band_m` columns in `hitters.csv`), and the replay (bat line + "bat … mph"). **Limits:** image-plane only; end-band speed reads ~1–2.5% high (it is nearer the camera than the tee); real tape/lighting/compression untested. **89 passed** after item 6. |
| Phone LiDAR path (lift → clean → skeleton → features → scores → report) | ✅ on a synthetic Stray Scanner recording rendered from known 3D joints: all features match ground truth (timing features within 20 ms) |
| Annotated video `swinglab/render.py` (item 5) | ✅ synthetic + lab data; ran on the real night dry swing (not a field swing). Skeleton on the RGB video, hip (blue) and shoulder (red) lines labeled with the separation **measured in 3D**, focus-joint ring around its phase, side panel (phase, pelvis/torso rotation speed, separation, hand speed, kinematic-sequence chart with cursor), 3 s score card at the end (scores, predicted vs measured bat speed, exit speed, Gemini headline), then a slow-motion replay with the ball track and projected flight when a slow-mo clip exists. `save_result` renders `<id>.mp4` next to each swing (stage `render`, non-fatal); `export_demo.py` renders the 4 lab demo swings (`web/data/overlays/`). **Codec:** OpenCV's Media Foundation H.264 has no bitrate control (57 MB for a 2.85 s clip) and FFmpeg's H.264 needs a missing OpenH264 DLL, so videos are written with x264 through `imageio-ffmpeg` (added to the `phone` extra): **0.47 MB** for the real clip, 120–155 KB per lab swing; fallbacks MSMF H.264 → VP9 WebM → mp4v. Tests (`tests/test_render.py`): lab swing frame count/timing/layout; synthetic phone recording with an `rgb.mp4` — hip and shoulder lines land on the hips and shoulders at contact; slow-mo replay draws the flight (exit 68.8 mph vs 67.1 true). **Viewer:** "Annotated video" panel under the 3D replay, synced both ways, checked in the browser on the real swing and a lab swing: phase buttons/scrub seek the video (foot plant → contact = 1.742 → 1.842 s), scrubbing the video moves the 3D figure (within one frame), Play runs the video at ¼ speed and the 3D replay follows it. Bugs fixed on the way: Python's `http.server` sends no `Accept-Ranges`, so Chrome couldn't seek (videos now load as blobs); `index.html` had no `<meta charset>` (served locally, "·" showed as "Â·"). **74 passed** after item 5. |
| Long sessions + per-swing slow-mo clips (item 7) | ✅ **synthetic only**. `session.match_clips` matches N clips to the swings of one long LiDAR recording: **by when the ball left the tee** when both clocks are known (Record3D file names carry the start time; videos their QuickTime `creation_time`, read with the bundled ffmpeg), finding the clock offset that lines up the most swings (ties: smaller offset), so a missing clip or extra practice swing doesn't shift every pair; otherwise **by order**. Count mismatches are flagged in plain words (`summary.notes`, printed by the inbox with `!!`, shown in the viewer's Session tab). Found: video creation times have **1 s resolution** (MP4 header / iPhone metadata), so matching tolerance is 1.5 s; swings in a cage session are 10+ s apart. Inbox: clips go to the LiDAR recording that was running when they were filmed (by time), or all to the single LiDAR recording of a session folder, or via log.csv rows naming the same LiDAR recording. Each matched clip's ball/bat result is attached to its swing (measured bat/exit speed → swing record, coach payload, replay video, session summary). Tests (`tests/test_session_slowmo.py`): matcher order/offset/missing/extra/regular-spacing cases; times read back from a remuxed MP4 and a Record3D name; **through `inbox.process`**: a 3-swing Record3D session with 2 clips on a clock 2 s off → matched by time, swing 2 flagged "no slow-motion clip", bat speeds within 5%; a Stray Scanner session (no clock) with 3 clips → by order. **97 passed** after item 7. |
| Upload server + coach chat (item 8) | ✅ synthetic files + live in the browser on this laptop (phones not tried yet). `python api\server.py` (FastAPI/uvicorn) serves the viewer at `/` (static files with byte ranges: videos seek), a phone upload page at `/upload` (session, hitter, kind: LiDAR `.r3d`/`.zip` · slow motion · yardstick; progress bar; fits a 375 px phone screen), `POST /api/upload` → `<inbox>/sessions/<session>/lidar|slowmo/` (streamed to `.part` then renamed; names sanitized, `../` stripped; wrong file types → 400; 4 GB cap; yardstick clips renamed `yardstick_*`; chosen hitter saved as the session's `hitter.txt`, read by the inbox), processing in a background thread **15 s after the last upload** (the two phones upload separately) or via "Process now", `GET /api/status` (log + results), `POST /api/chat` → `coach.gemini.chat` with the swing's payload; the answer is number-checked like the reports. Viewer: "Ask the coach" box under the coach card (plain-text answers, per-swing history; explains when opened without the server). **Live check:** asked "What one thing should I change first, and why?" on lab swing 42_1 → Gemini answered with only payload numbers (−14.345°, 12.126°, 293 ms, the open-book prescription). Tests (`tests/test_server.py`): pages served, safe uploads, wrong types refused, yardstick naming, **zipped Stray recording + slow-mo clip uploaded → paired, analyzed, listed in index.json with a rendered video**, chat with a fake client and without a key. Prints the phone URL (`http://<laptop IP>:8000/upload`); **Windows asks once to let Python through the firewall: allow Private networks**, or phones can't connect. No login: LAN-only hackathon tool. **104 passed** after item 8. |
| Rework 2026-10-10 evening (team feedback, see section 13) | ✅ **111 passed** (+1 live). **Bug fixed:** the "Contact" score was a *mechanics* group (body position at contact), so dry swings got 0/99 → renamed **Posture**; **Form** = mean of Power, Sequencing (transfer), Posture; **Contact** = ball contact only (provisional rule: smash factor + launch angle), only when a ball is seen leaving the tee; **Overall** = Form (dry) or mean of Form and Contact (hit). Hit vs dry decided automatically (slow-mo clip, else the LiDAR phone's own video: a ball resting then flying off; a still light is not a hit). Both night recordings re-analyzed: **1 swing each, dry, no contact score** (Matteo, Wilson profiles; heights/weights are placeholders to edit). One analysis path for any recording (no clip/session split). Swing finder now also needs a **shoulder turn ≥ 80°** around the hand-speed peak: lab swings turn 133–260° (all 677), noise peaks from rebuilt arms reached 51° (Matteo's recording had given 3 false swings). Hidden joints (rear arm behind the body, MediaPipe visibility < 0.1 in 54% of the swing) are rebuilt from MediaPipe's 2D guess + bone length and flagged. Pose on every 2nd frame of 60 fps captures (21.0 → 10.9 s per clip). **Profiles** with create/edit/delete (delete moves swings to `userdata/trash`), stored in `userdata/` outside the app (git-ignored; `SWINGLAB_USERDATA`); Synthetic profile = the 4 lab swings. Viewer rebuilt: profile sidebar, swing history strip + slider, Progress tab (form/contact/overall and bat speed over swings), Form/Contact verdict with the reason for the mode, data-quality notes collapsed, measurements with plain explanations, units in words, direction hints and out-of-range markers, `<!doctype html>` (the page ran in quirks mode, where table text can lose its color), chart labels no longer collide, auto-show of new swings (checked: touching a swing file → viewer switched to it with a toast). Server: profile API, `/api/latest`, `--watch <Drive folder>`. |
| Round 2, 2026-10-10 night (team review) | ✅ synthetic + browser, **119 passed**. **Toss and miss**: `ballflight.ball_events` (tee hit / toss hit = path breaks / miss / none) with ball-likeness (round, steady size, travels) and **gravity** checks (3–20 m/s² from the ball's own size); synthetic toss hit: contact frame, exit speed, launch exact; miss detected; bat speed within 1% (tossed ball masked from tape tracking); the two night videos: a lit window and a drifting light first passed as balls → now both dry. **Miss → contact 0.** Recording summaries count hits/misses/dry, form/contact stats; **Gemini writes every swing and every 2+ swing summary** (`GEMINI_MAX_SWINGS` to cap). **Drop folders** `userdata\drop\<Name>\` checked every 15 s. **Upload**: in-app dialog (drag and drop, auto file type, progress, QR code for the phone page), phone page in MLB colors. **UI**: home screen ("Choose a hitter" cards with latest overall + sparkline, New hitter, Driveline reference), navy top bar with hitter menu (switch, edit, delete, all hitters), Upload / Ask coach, MLB palette (navy #0C2340, red #BF0D3E, steel #4A7FB5), system status moved to a footer link. `record3d` 1.4.1 has a Windows cp312 wheel (live USB stream feasible; not built). |
| Shared-folder flow | ✅ synthetic session folder → pairing → analysis → `index.json` → served viewer lists the swing with per-stage ✓ |
| Training plan | ✅ chosen only from the library; invented exercises dropped (test) |
| Swing finder (long recordings) | ✅ exactly 1 swing found in each of 677 lab clips; 5/5 on a synthetic 21 s session; none when the hitter is still |
| Size handling | height normalizes distances; height/weight/age/level are model inputs; scores use size-adjusted outcomes; out-of-range warning (test) |
| Gemini live call | ✅ **live, 2026-10-10** (lab data payloads). Key read from the git-ignored `swinglab/.env` (new `config.load_dotenv`; real env vars win). Pinned **`gemini-3.6-flash`** (`--ping`: OK in 1.4 s, served as itself). Google re-routes names: `gemini-flash-latest` and `gemini-3.7-flash` were served by 3.8-flash, `gemini-3.5-flash` by 3.6; `gemini-2.5-flash` is listed but returns 404; 3.8/3.5 returned 503 "high demand" at times. **Free tier = 20 requests/day per model** (both 3.6-flash and flash-latest ran out during this session) → fallback chain 3.6-flash → flash-latest → 3.1-flash-lite → flash-lite-latest on 429/5xx, then the template. Default thinking made one report take > 60 s (504) on every model; `thinking_level="low"` did it in 2.8 s. Structured reports took 10–20 s with default thinking. All 4 demo swings + the demo session now have Gemini reports (3× 3.6-flash/flash-latest, 1× 3.1-flash-lite), **0 numbers outside the payload** after fixes. `export_demo.py` reuses a cached Gemini report when the payload is unchanged (`--refresh-gemini` to force) so reruns don't spend quota. `pytest -m live` runs the live test (passes); normal runs never call the API. **For Saturday: ~90 swings exceed the free quota → enable billing on the AI Studio project, or accept template reports for some swings.** |
| Gemini safeguards (2026-10-10) | **57 passed** after item 2. Number check found two real issues in live output, both fixed: Gemini wrote HTML entities ("3 &#215; 5" → now unescaped), and "per 10 swings" comes from a payload key (keys now count). `coach/numbers.py`: every number in a report's text must trace to the payload (rounding, sign, ms→s, ratio→% allowed; small counts allowed only in drill steps/prescriptions); result stored in `report_source.unsupported_numbers`. Checked: the template reports of ~100 lab swings come out clean (no false alarms), invented numbers are caught. **Bad key → template** verified against the real API (400 API_KEY_INVALID, fallback in ~2 s), also for session report and chat. 60 s request timeout (`GEMINI_TIMEOUT_S`). |

---

## 7. Findings and gotchas

1. **Driveline landmark names**: `lhjc` is the left *hand*, not hip. Fixing
   this took pelvis speed from nonsense to r = 0.97. Geometry test guards it.
2. **Peak hand speed comes ~65 ms before contact**, very consistently; contact
   = peak + 65 ms.
3. **Foot plant** = last frame the lead ankle is > 3 cm above its stance
   height, minus 28 ms; falls back to ankle settling, then to "contact − 190 ms"
   for no-stride hitters.
4. **Train at phone frame rate.** All training features are computed after
   resampling 360 Hz to `PHONE_FPS`. When tomorrow's captures show the real
   depth frame rate, set `SWINGLAB_PHONE_FPS` and rerun `setup_data.py`.
   Optional `--noise` in `build_dataset.py` jitters joints to mimic phone depth.
5. **Peak hand speed is a strong predictor but not actionable**, so it's used
   for prediction and excluded from coaching.
6. **Strong swings get praise, not a fix**: focus is shown only if it costs
   ≥ 0.3 mph and overall < 85.
7. **Phone world frame**: ARKit is Y-up and the world origin is where the phone
   started; the canonical frame measures height from the floor (ankles at
   stance − 8 cm). A test confirms features are identical under any rotation
   and translation.
8. **MediaPipe API change**: use `mediapipe.tasks.python.vision.PoseLandmarker`
   with a downloaded `.task` model; `mp.solutions` is gone.
9. **Depth at body edges** bleeds to background; `sample_depth` takes the
   nearer half of a 5×5 window.
10. Ridge regression matched or beat gradient boosting in CV, so it's the
    shipped predictor (stable, explainable).
11. **Sequencing timing is coarse at 30 fps**: peak-to-peak times come in
    33 ms steps, so `seq_*_ms` features are nearly constant. If the iPhone 17
    Pro records depth at 60 fps, set `SWINGLAB_PHONE_FPS=60` and retrain.
    Sub-frame peak interpolation (parabola through the 3 samples around each
    peak) is a cheap improvement for the next session.
12. Driveline's own downloader needs `gh`; ours doesn't (pure Python, works on
    Windows).
13. **Occlusion:** when the hands pass in front of the hips, the depth under
    the hip belongs to the hands for a frame (30 cm error in the synthetic
    test). `capture.clean_joints` drops samples > 12 cm from a 5-frame rolling
    median and interpolates gaps < 150 ms. Without it, peak pelvis speed came
    out 2.4× too high.
14. **Phone data is resampled to `PHONE_FPS` (the training rate)** before
    features, whatever the capture rate.
15. **Ball size as scale:** a fixed brightness threshold made the ball look
    ~20% small and speeds ~20% high; `_refine_radius` uses a half-way
    threshold. Prefer the yardstick clip for scale anyway.
16. **Sequencing timing** now uses sub-frame peak interpolation
    (`kinematics.peak_offset`); expect about ±20 ms noise on phone data.


### 7b. Size correction: what's done and how to test it on real hitters

Done: distances ÷ height; height/weight/age/level as model inputs; scores,
explanation trees and top-third bands on size-adjusted outcomes; warnings
outside 1.60–1.96 m, 63.5–118 kg, 17–31 years. Check on Driveline
(athlete-grouped CV): no meaningful size bias inside that range (height thirds
+0.3 / −0.2 / +0.3 mph mean error; weight thirds +0.8 / −0.7 / +0.4; level
groups larger but small samples: independent −3.0 (n=30), MiLB +2.0 (n=47)).

To test on our own swings (after bat-tip tracking works):
1. For each swing, compare predicted bat speed with slow-motion **measured**
   bat speed. Log the error per hitter.
2. If one hitter is consistently off by a similar amount, that is a size /
   strength offset the lab data can't capture: add a per-hitter calibration
   (subtract that hitter's mean error after ≥ 10 swings; show "calibrated").
3. Sensitivity check: rerun one swing with height ±5 cm and weight ±5 kg;
   predictions should move ~0.8 mph and ~0.9 mph, scores should barely move
   (they are size-adjusted). Add as a test.
4. Youth / out-of-range hitters: keep mechanics feedback, show speed
   predictions as ranges or hide them; never compare raw speeds to college
   hitters without the size adjustment.
5. If bias appears across many hitters, try normalizing bat speed by
   body size before modeling or adding height×mechanics interaction terms,
   and compare in athlete-grouped CV.

---

## 8. UI

**What the references do:** vision-demos and the pickleball pipeline both
produce a **video with graphics drawn on top** (skeleton, joint angle arcs, a
stats panel) rendered offline with OpenCV. Neither has an interactive UI.

**What we have:** `web/index.html`, an interactive viewer: swing picker, 3D
replay you can orbit (ghost of the fastest lab swing aligned at contact, focus
joint ring, phase timeline, ¼-speed playback), Power/Transfer/Contact scores,
predicted vs. measured speeds, coach card with a "Gemini / template" badge,
kinematic-sequence chart synced to playback, measurements vs. the lab 10–90%
range, and a **Pipeline status** tab listing what passes and what's pending.
It is the team's main tool for knowing the pipeline works.

**Demo structure (team decision 2026-10-09):** lead with the same format as
the inspirations, a **swing video with tracking and stats drawn on screen**,
then open the **SwingLab viewer** for the depth: 3D replay, ghost of a top
swing, scores, measurements vs. the top third, coach report. Best version: the
viewer shows the annotated video in a panel, synced to the 3D replay's
timeline (scrub one, the other follows).

**Next UI work:**
1. `swinglab/render.py`: annotated MP4 overlay (skeleton, separation arc, HUD
   panel with phase and live numbers, trajectory line from the slo-mo camera).
   Shareable; plays during judging. Then embed it in the viewer, synced.
2. `api/server.py` (FastAPI): upload a capture folder or zip → run
   `analyze_capture` → viewer opens the result; `/chat` endpoint for Gemini.
3. Viewer: load any `web/data/*.json`, chat box, diagram chosen by Gemini,
   session view (all swings, trend, vitals).

---

## 9. Recording plan (baseball field, Saturday morning)

**The full, current version is `docs/RECORDING_PLAN.md` (clips and long
sessions). The summary below is kept for reference.**

**MediaPipe is not needed to record.** Everything is processed afterwards on
the laptop. Record both cameras for every swing and keep the raw files; one
swing = one recording on each phone.

### Before the field (optional, 15 minutes)
1. LiDAR app: **Record3D** or **Stray Scanner**; the pipeline reads both
   (Record3D `.r3d` files or extracted folders; Stray Scanner folders).
   Record3D needs its RGBD export unlocked (see RECORDING_PLAN section 2).
2. Make a 10 s test recording of someone standing 3–4 m away, and a 5 s
   slow-motion clip on the second phone.
3. Move both to a laptop (see "Getting files off the phones") and run
   `python scripts\check_recording.py <folder, .r3d or video>`. Note the depth
   fps and the slo-mo file's real fps. Fix any warning before the field.

### Equipment
iPhone 17 Pro (Record3D or Stray Scanner) · second phone (240 fps slo-mo) · 2 tripods or
clamp mounts (a fence clamp works) · tee · bucket of real baseballs · bat ·
**bright tape** (white or neon) wrapped around the bat end and a second band
just above the hands (lets the slow-motion camera find the bat in blurry
frames: bat speed at contact, bat angle, contact point) · tape measure · yardstick or a 1 m stick · something to
note the log (phone notes) · power bank.

### Measure once per hitter (the model needs these)
Height (measure it), weight, age, bats R/L, playing level (high school /
college / other), bat length and weight.

### Camera A — LiDAR body capture (iPhone 17 Pro, Record3D or Stray Scanner)
- **Where:** across the plate, **facing the hitter's chest**, **3–4 m** from
  the hitter, turned about **30–45° toward the catcher's side** so the camera
  is behind the line of any batted ball. Never put a phone in front of the tee.
- **Height:** about hip height (~1 m). **Landscape.** Phone still on a tripod.
- **Framing:** whole body and the full bat path with margin: feet, hands at
  the top of the load, and the bat at full extension must stay in frame.
- **Light:** LiDAR range drops in direct sun. Avoid the sun behind the hitter
  or shining straight into the lens; shade or overcast is best. Check
  `valid_depth_fraction` with the checker on the first clip.
- **Timing:** start recording with the hitter **already set in the stance
  and still for ~1 s** (the pipeline uses the first frames to define the
  hitter's stance and direction), swing, hold the finish, stop 1–2 s later.

### Camera B — slow motion bat and ball (second phone, 240 fps)
- **Where:** side view **perpendicular to the direction the ball travels**,
  level with the tee, about **4–5 m** away, on the same side as camera A
  (facing the hitter) but further toward first/third so it looks straight
  across the tee. Use the backstop or a fence as protection if possible.
- **Height:** tee height. **Landscape.** Still on a mount, never handheld.
- **Framing:** the tee in the left/right third of the frame so the first
  ~1.5 m of ball flight stays in frame; bat end visible through contact.
- **Scale:** at the start of each camera setup, record 3 s of the yardstick
  held upright **at the tee**, in the same plane as the ball. If the camera
  moves, record it again. (The ball's 7.4 cm diameter is a backup scale.)
- Bright daylight is ideal; slow motion gets dark and blurry in shade.

### Sync
Start camera B, then camera A, then the swing. The ball leaving the tee is
visible in both and is the sync point. For a backup, have the hitter tap the
bat on the plate once before setting up.

### Shot list (per hitter; ~45 swings each, ~1.5 hours for two hitters)
| Set | Count | Purpose |
|---|---|---|
| Calibration clips | 1 per camera setup | yardstick at the tee (B), 10 s empty scene + hitter standing in stance (A) |
| Normal tee swings | 20 | main test set, model transfer, demo |
| Deliberate flaws | 5 each: **arms only** (no hip turn), **shoulders open early**, **lunge / hips slide forward**, **front leg stays bent**, **no stride** | check that scores and the focus react the right way |
| Dry swings (no ball) | 5 | compare with tee swings |
| Different tee heights | 3 low, 3 high | contact-point variety |
| Camera A handheld, slowly walking | 3 | tests the moving-camera feature (stretch) |
| Pitching (optional) | 5–10 throws into a net or to a partner, camera A oblique, camera B on the arm side | Phase 2 data |
| Demo takes | 3–5 of each hitter's best swings, clean background | judging video |

### Log every swing (one line each)
`swing # | hitter | set (normal / flaw type / dry / pitch) | camera A folder | camera B file | contact (line drive / grounder / fly / miss) | notes`.
Save it as `data\captures\log.csv`. The flaw labels let us test the scores.

### Getting files off the phones (Windows)
- **Stray Scanner** saves each recording as a folder. Export it via the iOS
  Files app (Stray Scanner documents) or by plugging the iPhone into the
  laptop and copying from the Apple Devices / iTunes **File Sharing** section.
  Zipping on the phone and uploading to Google Drive also works. Put folders
  in `data\captures\`.
- **Slow motion:** export the **unmodified original**. On an iPhone, AirDrop
  or share with "All Photos Data" / Photos "Export Unmodified Original";
  Google Photos and messaging apps re-encode and can drop the 240 fps. Check
  with `check_recording.py`; if it reports 30 fps for a slo-mo clip, the
  frames are all there and we pass `--fps 240`.
- HEVC videos on Windows may need the HEVC Video Extension or a conversion
  (`ffmpeg -i in.mov -c:v libx264 out.mp4`).

### 9b. From both phones to analysis: the shared folder

**How data flows now:**
```
iPhone (Stray Scanner) ──► Files app ──► Google Drive: SwingLab/sessions/<date>/lidar/
Phone 2 (240 fps)      ──► Drive app  ──► Google Drive: SwingLab/sessions/<date>/slowmo/
                                         (Google Drive for desktop syncs it to the laptop, e.g. G:\My Drive\SwingLab)
laptop: python scripts\process_inbox.py "G:\My Drive\SwingLab" --watch
          pairs each LiDAR recording with its slow-motion video
          → check → 3D analysis → ball speed → Gemini report + training plan
          → web\data\swings\<session>\<swing>.json + web\data\index.json
viewer: python -m http.server -d web 8000 → new swings appear at the top of the picker
```
- Set up once: install **Google Drive for desktop** on the laptop, create the
  folder, run `process_inbox.py <folder> --init` (creates `hitters.csv`,
  `sessions/example_session/lidar|slowmo/`, an example `log.csv`). Fill in
  `hitters.csv` (name, bats, height_m, mass_kg, age, level).
- **Pairing:** if `log.csv` names both files for a swing, that's used (most
  reliable; also gives hitter and set, e.g. "flaw_arms_only"). Otherwise
  recordings are paired by order; mismatched counts are flagged.
  `--list` shows the pairing before anything runs.
- Zipped Stray Scanner exports are unzipped automatically. Already processed
  swings are skipped (`.swinglab_processed.json`); `--force` redoes all.
- Each result records which stages ran (`lidar`, `slowmo`, `coach`) and why
  any failed; the viewer shows ✓/✗ per stage.
- Uploading from the field over cellular is fine but large (LiDAR folders are
  tens of MB per swing); otherwise upload on Wi-Fi afterwards.
- **Next (for users):** an upload page on the laptop (FastAPI) so a phone
  browser can send files directly over the same Wi-Fi, and a "Process"
  button in the viewer. The shared folder works now and needs no app.

### Run on site if a laptop is there
`check_recording.py` on the first swing of each setup only (depth coverage,
framing, frame rate). Full analysis happens later.

---

## 10. Next Claude Code session prompts (in order)

**Two parallel sessions (team decision 2026-10-09):**
- **Session A** continues this repo with the prompts below.
- **Session B** builds from scratch in a separate folder (`swinglab-b`) from
  `docs/FRESH_START_BRIEF.md`, writing results in the same JSON contract
  (brief section 6) so either viewer can show either build's swings.
- Don't let both sessions edit the same repo. Both read the same shared
  recording folder (read-only). Compare their numbers on the same real swings.

**Session A prompts:**

**Tonight (Friday night) — paste this first. We have NO swing videos yet;
build and test everything on synthetic data, Driveline lab data, or any
ordinary video of a person.**
> Read docs/HANDOFF.md, docs/FILES.md and docs/RECORDING_PLAN.md. We're on
> Windows (PowerShell, Python 3.12). We have no swing recordings yet (they come
> Saturday). Build and test everything that doesn't need them, in order. After
> each item run `python -m pytest -q`, report what passed with numbers, and
> update HANDOFF section 6. Never mark anything "verified on real data" unless
> it ran on real recordings; say "synthetic" or "lab data" instead.
> 1. Setup: venv, `pip install -e ".[phone,gemini,server,dev]"` (includes
>    `lzfse` for Record3D), `python scripts\setup_data.py`, tests. Fix any
>    Windows-specific issue (paths, encodings, wheels) and add a test for it.
> 2. Gemini live: `$env:GEMINI_API_KEY` is set. Run
>    `python -m swinglab.coach.gemini --list-models`, pin a Flash model in
>    `GEMINI_MODEL`, `--ping`, then rerun `scripts\export_demo.py` so the demo
>    swings and demo session get real Gemini reports. Add an automated check
>    that every number in a Gemini report's text appears in its payload, and a
>    test that a bad key falls back to the template.
> 3. MediaPipe sanity without swing videos: `python scripts\get_pose_model.py`,
>    then run pose on any phone video of a person moving (a few seconds of
>    someone walking or shadow-swinging, filmed on a laptop webcam or phone).
>    Write a 2D keypoint overlay video to eyeball. Also make sure
>    `mediapipe_keypoints` reads a JPG image sequence (`rgbd\%d.jpg`, how
>    Record3D stores color): test with a folder of frames cut from that video.
> 4. Recording quality checks (no real data needed): add a bone-length check
>    to `pipeline.analyze_joints` (thigh, shin, upper arm, forearm vary < 10%
>    across frames; report the worst bone in `quality`) and a framing check
>    (fraction of frames with all 13 joints visible, feet and hands in frame).
>    Test both on synthetic recordings with injected errors. If we drop a
>    Record3D or Stray Scanner sample in data\captures (even of someone
>    standing), run `check_recording.py` and `analyze_capture.py --lidar` on it
>    and confirm the depth orientation and K for Record3D (the loader is
>    written from the format description and tested only on synthetic files).
> 5. Build `swinglab/render.py`: an annotated MP4 like our inspirations
>    (skeleton on the RGB video, hip and shoulder lines with the separation
>    angle, a HUD with phase, live rotation speeds, scores at the end, and the
>    ball's trajectory when a slow-motion result exists). Test it on a
>    synthetic recording (render_stray in tests) and on a Driveline lab swing
>    drawn on a blank background. Embed it in the viewer, synced to the 3D
>    replay timeline.
> 6. Bat-tip tracking in `ballflight.py`: find the bright tape bands on the
>    bat (end and above the hands) in 240 fps frames, track them around
>    contact, compute bat speed at contact and the bat angle with the yardstick
>    scale. Test on synthetic videos with motion blur (draw a blurred bat
>    with two bright bands moving on an arc at known speed); target < 5% error.
> 7. Long sessions with per-swing slow-motion clips: match N slow-motion clips
>    to the N swings found in one long LiDAR recording by order (and by
>    ball-off-tee time when available); flag count mismatches. Test with
>    synthetic data through `inbox.process`.
> 8. FastAPI upload page (`api/server.py`: phone browser uploads a file over
>    the same Wi-Fi → inbox → analysis) and a Gemini chat box in the viewer
>    (`coach.gemini.chat` with the swing's payload). Test with synthetic files.
> 9. Write a short "Saturday morning" checklist at the top of HANDOFF: the
>    exact commands to run on the first real recordings, in order.

**Saturday (after the field videos):** Prompts 2–3 below on the real
recordings, then 4–8.

**Final phase — cleanup, simplify, fix, polish (after the videos are
analyzed and every feature works):**
> Read docs/HANDOFF.md and docs/FILES.md. All features now work on real data.
> Clean up without changing results: (1) run the full tests and save the
> numbers in section 6 as the baseline; (2) delete dead code, unused scripts,
> parameters and env vars nobody uses; merge small modules that belong
> together; simplify long functions; give names that say what things are;
> (3) list every known bug (from section 12 and the real-data runs), fix them
> one at a time, each with a test; (4) UI pass on the viewer: consistent
> layout, readable on a phone and a projector, loading and error states,
> empty states, no jargon in labels, every number has a unit; (5) rerun the
> tests and the real recordings and confirm every number matches the baseline
> (or explain each change); (6) update FILES.md, SPEC.md and the UML.

Copy one at a time. Each ends with the standing directive in section 11.

**Prompt 1 — Re-verify setup and external facts**
> Read docs/HANDOFF.md. We are on Windows (PowerShell, Python 3.12). Follow
> the section 0 quick start: venv, `pip install -e ".[phone,gemini,server,dev]"`,
> `python scripts\setup_data.py`, `python -m pytest -q`. Fix anything that
> breaks on Windows (paths, encodings, MediaPipe wheels) and add a test for
> it. Confirm the numbers in section 6 reproduce. Open the OBP-CV Google Sheet
> (section 3) in a browser, read the shot-list tabs, and record which
> movements and frame rates exist; if it has swings, note them as an optional
> validation set. Then re-confirm section 3 and 4 facts that may have
> changed: Driveline license text (LICENSE-DATA.md), Gemini free-tier models
> and limits (list models with our key, pin `GEMINI_MODEL`), ElevenLabs free
> credits, Stray Scanner availability, Record3D pricing. Run
> `python scripts/get_pose_model.py` on the laptop. Run
> `python -m swinglab.coach.gemini --ping`. Update sections 3, 4 and 6.

**Prompt 2 — First real LiDAR capture**
> Put the first real recordings in the shared folder (section 9b) and run
> `python scripts\process_inbox.py <folder> --list`, then without `--list`
> (or one recording: `scripts\analyze_capture.py --lidar <folder> --slowmo
> <video> --bats R --height 1.78`). Report each stage's printed check. Measure the
> real depth frame rate; if not 30, set `SWINGLAB_PHONE_FPS` and rerun
> `setup_data.py`. Visualize the lifted skeleton in the viewer. Fix depth
> sampling, gaps and smoothing until bone lengths are stable (add a test that
> thigh/shin/forearm lengths vary < 10% across frames). Compare our features'
> ranges to the Driveline 10–90% range and log outliers.

**Prompt 3 — Slow-motion bat and ball**
> Using a 240 fps side-view tee clip, run ball detection and tracking, contact
> frame, exit speed, launch angle, and draw the trajectory. Add bat detection
> (bright tape on the bat tip, or a small detector) and bat speed at contact.
> Sync with the LiDAR capture via the ball leaving the tee. Show measured vs.
> predicted bat speed in the viewer. Write tests with the real clip's
> hand-labeled frames.

**Prompt 4 — Gemini live + upload server** (the shared-folder pipeline already exists: test it with the real Drive folder first)
> Wire `gemini.report` with the real key, then `chat` and `video_read`. Build
> `api/server.py` (FastAPI) and connect the viewer: upload → analyze → show.
> Show the Gemini-vs-template badge and timing. Test: schema validation on 10
> swings, fallback on a bad key, no invented numbers (check every number in the
> report text appears in the payload).

**Prompt 5 — Overlay video and diagrams**
> Build `swinglab/render.py` (annotated MP4: skeleton, separation arc, HUD,
> trajectory). Add the Gemini-chosen diagram to the viewer and a "target pose"
> ghost. Session summary across swings.

**Prompt 6 — Pitching (Phase 2)**
> Extend `obp.py` for pitching landmarks, add pitching events (foot plant,
> max external rotation, release) and features, target `pitch_speed_mph`,
> validate the same way, add a pitching mode to the viewer.

**Prompt 7 — Stretch**
> ElevenLabs voice for headline + cue. Presage between-swing vitals with a
> webcam, joined to swings by time; heart rate vs. bat speed chart.

**Prompt 8 — Demo hardening**
> Cache demo swings and Gemini responses, offline mode, backup video, slides,
> domain, Devpost write-up crediting Driveline and the reference projects.

---

## 11. Standing directive: test and verify

- Nothing is "done" until it has run on real input and a test covers it.
- Every module gets a synthetic test (known input → known output) and, once
  data exists, a real-data test.
- After any change: `python -m pytest -q`; after model/feature changes:
  `python scripts/setup_data.py` and compare section 6 numbers.
- Keep section 6 and the viewer's Pipeline status tab in sync with reality.
  Mark things ✅ only when verified; say what was not verified.
- Validate against ground truth whenever it exists (Driveline events and POI,
  slo-mo bat speed vs. prediction, hand-labeled frames).

---

## 12. Team answers and open questions

Answered 2026-10-09:
1. Wilson has an **iPhone 17 Pro** with **Record3D installed**; a second phone
   records **240 fps** slow motion.
2. Recording **on a baseball field** (outdoors, real baseballs, tee).
3. Both laptops run **Windows**.
4. Pitching only after goals 1–3 work; record a few pitching clips anyway.
5. **Gemini API key exists.** Set `GEMINI_API_KEY` and run `--ping`.

Answered 2026-10-09 (evening):
6. Record3D's live outline looked better than Stray Scanner's in testing.
   Both use the same ARKit LiDAR; the preview difference is rendering. Both
   are now supported; the team may buy Record3D's RGBD export ($2.99).
7. No indoor test tonight. Field recordings Saturday outdoors (clips and long
   sessions), possibly one night test with a Record3D free recording first.
8. After videos and features work: a cleanup/simplify/bug-fix/UI phase
   (section 10, "Final phase").

Still open:
- ~~First real Record3D export: confirm depth orientation, K, real fps~~ → confirmed 2026-10-10 on two night test files (section 6): portrait, aligned, K correct, **60 fps**. Decide: set `SWINGLAB_PHONE_FPS=60` and rerun `setup_data.py` (lab contact timing improves from 9.4 to 4.5 ms median error at 60 fps) — not done yet; it retrains the models.
- Record3D depth stops at 3.09 m in both test files: find the range / max-distance setting in the app before the field.
- Arms lost depth in the night test (dark sleeves, motion): see whether daylight / lighter clothing fixes it on the first field clip.
- Recording plan: done (docs/RECORDING_PLAN.md).
- ElevenLabs / Presage keys.
- Project name and domain (SwingLab is a placeholder).

---

## 13. Decisions and tasks from the team review (2026-10-10 evening)

**Decided and built**
- **Form vs Contact.** Form (Power, Sequencing, Posture) rates how you move on every swing. Contact rates
  how squarely a ball was hit and exists only for hits. Overall = Form for a dry swing, else the mean.
  The contact rule is provisional (smash factor = exit ÷ bat speed, 1.2 = square off a tee, 70%;
  launch angle 8–25° best, 30%) until we have our own tee data to calibrate it.
- **Two modes, chosen automatically** from ball detection (`pipeline.decide_mode`): a ball seen
  resting and then flying off → hit; otherwise dry swing. Users never pick the mode.
- **One way to analyze a video**: find every swing in it, analyze each the same way. A recording with
  2+ swings also gets a summary. Gemini writes reports for the first 3 swings per recording (quota).
- **Profiles and user data outside the app**: `userdata/` (git-ignored). Shipping the Synthetic profile
  is a choice: copy `userdata/swings/synthetic` + its `profiles.json` entry.

**Done since: drop folders + in-app/phone upload (section 6, round 2).** **Task — hands-free demo capture (next; `record3d` has a Windows wheel):** today the fastest path is Record3D → Share → Files → the
Google Drive folder, with `python api\server.py --watch "G:\My Drive\SwingLab"` and the viewer's
auto-show. Phone taps remain. The hands-free version streams the LiDAR phone live:
1. Record3D's paid "Wi-Fi Streaming & RGBD Videos" / USB streaming with the `record3d` Python package
   (pip; USB on Windows needs the iTunes/Apple Devices drivers) gives RGB + depth + camera pose per frame.
2. A capture loop keeps the last ~6 s in a ring buffer, runs pose at ~15–30 fps (MediaPipe *lite* or
   *full* model for speed), and triggers on the swing detector (hand speed + shoulder turn).
3. The ~2.5 s window around the swing goes through the same `lift_keypoints → analyze_joints →
   decide_mode → coach` path and is saved to the hitter's profile; the viewer's auto-show pops it up
   (~10–20 s after the swing, mostly Gemini).
   Effort: about a day, plus testing with the phone. Risk: whether Record3D's streaming works with the free
   version and on Windows; Stray Scanner has no streaming.

**Task — LiDAR phone only (no 240 fps camera): feasibility and plan (not started)**
- What works with one phone: body 3D (now); hit vs dry (now, from its color video); **ball exit
  speed and launch angle roughly**: at 60 fps a 35 m/s ball moves 58 cm per frame, so it is seen in
  ~2–4 frames before leaving the frame; its depth from LiDAR at 256x192 is unreliable (small, fast), so
  use the color frames + the known ball size for distance. Expect ±10–15% exit speed, not ±1%.
  **Bat speed** is hard: the bat is a blur at 60 fps and too thin for the depth map; the model's
  predicted bat speed stays the main number.
- Plan: (1) record 10 tee hits with both cameras, (2) build a LiDAR-video ball tracker (same blob +
  ballistic fit, scale from ball size and K), (3) compare with the 240 fps numbers, (4) switch the
  default to LiDAR-only if the error is acceptable, keeping the second camera as an optional upgrade.
- Recording tip found today: filmed from the chest side, the **rear arm hides behind the body** in
  much of the swing. The planned 30–45° turn toward the catcher's side should help; check it in the
  first field clip (`placed_by_bone_length` for rear elbow/wrist).
