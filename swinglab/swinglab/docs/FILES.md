# SwingLab — File Guide

What every file in the repo does, how far along it is, and what comes next.
Status: ✅ built and tested · ⚠️ built, not yet run on real recordings ·
❌ planned. Snapshot: 2026-10-10 night (Windows laptop), 104 tests passing (+1 live Gemini test, `pytest -m live`).

## Top level

| File | Purpose | Status / next |
|---|---|---|
| `README.md` | Three-command quick start | ✅ |
| `pyproject.toml` | Package + dependencies. Extras: `phone` (OpenCV, MediaPipe, lzfse, imageio-ffmpeg for H.264 video), `gemini`, `server` (FastAPI, for the upload page), `dev` (pytest). Marker `live` (real Gemini calls) is skipped by default | ✅ installed on Windows / Python 3.12 |
| `.gitignore` | Keeps raw data, processed data, captures, the inbox, `.env` (Gemini key), `.venv` and the MediaPipe model out of git | ✅ |
| `.env` | `GEMINI_API_KEY=...` (and optionally `GEMINI_MODEL`), read by `config.load_dotenv`; git-ignored | yours |
| `api/server.py` | FastAPI: viewer at `/`, phone upload page `/upload` → inbox → background analysis, `/api/status`, `/api/chat` (Gemini chat with a swing's payload) | ✅ synthetic + local browser · ⚠️ real phones |

## `docs/`

| File | Purpose |
|---|---|
| `SPEC.md` | Product specification: goals 1–7, user interaction, data, models, status, UML |
| `FILES.md` | This guide |
| `TEAM_QA.md` | Answers to the team's questions (rounds 1 and 2) |
| `uml.html` | The SPEC's diagrams rendered as a page (built by `scripts/build_uml.py`) |

## `swinglab/` — the Python package

| Module | What it does | Status | Next |
|---|---|---|---|
| `config.py` | Paths; `PHONE_FPS` (env `SWINGLAB_PHONE_FPS`, default 30); loads `.env` | ✅ | Record3D depth is 60 fps: decide on 60 |
| `obp.py` | Driveline data: pure-Python download with checksum check, loaders for metadata / POI / HitTrax, landmark parquet, `iter_swings` | ✅ | Pitching loader (goal 7) |
| `skeleton.py` | Canonical skeleton: lead/rear names, body frame from the stance (x toward pitcher, z up, floor at 0), lefties mirrored, resampling | ✅ | — |
| `kinematics.py` | Per-frame signals (pelvis/torso rotation, separation, speeds, knee bend, trunk tilt); event detection (load, foot plant, contact) with offsets fitted on Driveline; sub-frame peak timing | ✅ validated vs force plates | Re-check offsets on real phone data |
| `features.py` | 23 mechanics per swing, grouped Power / Transfer / Contact, with labels and units | ✅ | Bat features once the bat is tracked |
| `model.py` | Bat-speed and exit-velocity ridge models (mechanics + height, weight, age, level); athlete-grouped CV; size-adjusted explanation trees | ✅ trained | Calibrate on our own measured bat speeds |
| `scoring.py` | Three 0–100 scores as percentiles of size-adjusted outcomes; coaching focus in mph | ✅ | — |
| `capture.py` | Stray Scanner and Record3D (`.r3d`) loaders (`load_any`; Record3D reads `dw`/`dh` and per-frame intrinsics), depth sampling, back-projection, ARKit world frame, **bone-length fallback** for limbs without depth, occlusion repair (`clean_joints`), MediaPipe Tasks pose (videos and `%d.jpg` sequences), wall-clock start times of recordings and videos | ✅ Record3D format on 2 real night files · ⚠️ field swings |
| `quality.py` | Recording checks: bone-length stability (worst bone), framing (all joints visible, feet/hands in frame), fallback warnings | ✅ synthetic, fires on the real night files |
| `render.py` | Annotated swing video (skeleton, hip/shoulder lines + 3D separation, HUD, sequence chart, score card, slow-mo replay with ball flight and bat); 2D keypoint overlay | ✅ synthetic, lab, real night clip |
| `ballflight.py` | Ball: rest on the tee → flight (blur- and occlusion-tolerant), exit speed, launch angle, trajectory. Bat: two tape bands → sweet-spot bat speed, attack and bat angle. Yardstick scale | ⚠️ synthetic video only (bat speed within ~1%) | Real 240 fps clips |
| `pipeline.py` | Stages as functions: `analyze_recording` (entry point: counts swings → single or session; per-swing slow-mo clips), `lift_recording`, `analyze_joints` (+ quality checks), `analyze_lidar`, `analyze_slowmo` (ball + bat), `analyze_swing`, `analyze_session`, `match_slowmo`; saving results + annotated video + viewer index | ✅ synthetic end to end | Real field data |
| `session.py` | Long recordings: swing finder, per-swing cut, session summary; `match_clips` (slow-mo clips ↔ swings by ball-off-tee time with clock offset, else order; mismatch notes) | ✅ synthetic + lab | Real cage recording |
| `inbox.py` | Shared folder: scan, unzip, pair LiDAR recordings with slow-motion clips (log.csv, time, session, order), yardstick clips, bat tape columns, processed-state tracking | ✅ synthetic | Real Google Drive sync |
| `coach/payload.py` | Packs a swing into the JSON Gemini sees (numbers, focus, costs, exercise options, disclaimer) | ✅ | Add vitals when Presage is in |
| `coach/schemas.py` | Pydantic schemas for structured output: `SwingReport`, `TrainingItem`, `SessionReport`, `VideoRead` | ✅ | — |
| `coach/gemini.py` | Gemini calls: report, session report, chat, video read; pinned `gemini-3.6-flash` + fallback chain on 429/5xx; low thinking; HTML entities unescaped; exercise validation; fallback on any error | ✅ live (lab payloads) | video_read not tried live |
| `coach/numbers.py` | Checks that every number in a Gemini text is in its payload | ✅ |
| `coach/cues.py` | Template reports (swing + session) for when Gemini is off or fails | ✅ | More templates as focus types appear |
| `coach/exercises.py` | 22-exercise training library, training areas, mapping from mechanics to areas | ✅ | Review list with a coach if possible |
| `coach/prompts/*.md` | System prompts: `report`, `chat`, `video_read`, `session` | ✅ | Tune wording after live runs |

## `scripts/` — command-line entry points

| Script | Run it to… | Status |
|---|---|---|
| `setup_data.py` | Do everything from a fresh clone: download, validate, build features, train, export viewer data, build Excel sample (~1.5 min) | ✅ |
| `validate_events.py` | Compare detected foot plant / contact with Driveline's force plates | ✅ |
| `build_dataset.py` | Build the training table (features at a chosen fps; optional noise) | ✅ |
| `validate_features.py` | Compare our measurements with Driveline's own metrics | ✅ |
| `train.py` | Cross-validate and train models + scorers | ✅ |
| `export_demo.py` | Write `web/data/demo.json` (lab demo swings, ghost, ranges, demo session, status) | ✅ |
| `build_web.py` | Single-file viewer `web/dist/swinglab-viewer.html` | ✅ |
| `make_data_sample.py` | `data/sample/driveline_sample.xlsx` to look at the data in Excel | ✅ |
| `build_uml.py` | `docs/uml.html` from SPEC diagrams | ✅ |
| `get_pose_model.py` | Download the MediaPipe pose model (~30 MB) | ✅ ran on the laptop |
| `pose_overlay.py` | MediaPipe 2D keypoint overlay video of any video / `.r3d` / `%d.jpg` folder (sanity check) | ✅ real night clips |
| `check_recording.py` | Check a new recording: depth fps, coverage, farthest depth (range limit), slow-motion real fps | ✅ real Record3D files |
| `analyze_capture.py` | Analyze one recording or pair without the shared folder (`--yardstick`, tape positions) | ✅ real night clips (LiDAR) · synthetic (slow-mo) |
| `process_inbox.py` | Shared folder: `--init`, `--list`, process once, `--watch` | ⚠️ real Drive |

## `tests/` — 104 tests (+1 live)

| File | Covers |
|---|---|
| `test_kinematics.py` | Angles, rotation speed, lefty mirroring, resampling, world-frame invariance, sub-frame peaks |
| `test_obp_real.py` | Landmark meanings by geometry, events vs force plates, pelvis speed vs Driveline, model beats baseline |
| `test_capture.py` | Stray Scanner loading, back-projection, world frame, recording checker |
| `test_ballflight.py` | Exit speed/angle recovery, contact frame, trajectory, ball detection |
| `test_pipeline.py` | **Full phone path on a rendered recording** (every feature matches ground truth), slow-motion from a rendered video, inbox pairing, index, Record3D export gives the same features as Stray Scanner |
| `test_session.py` | Swing finder (synthetic + all lab clips), long-recording analysis, inbox auto-split, size warnings |
| `test_coach.py` | Payload, template report, Gemini mocked (success, failure, no key, busy model, quota chain), real bad key → template, number check, HTML entities, training plan, live test (`-m live`) |
| `test_windows.py` | UTF-8 everywhere, piped output, no `python3`, forward-slash index paths |
| `test_pose.py` | JPG sequences, MediaPipe timestamps, real Record3D frames (skipped without a capture) |
| `test_quality.py` | Bone-length and framing checks with injected errors; bone-length fallback; plain error for missing joints |
| `test_render.py` | Annotated video: lab swing, synthetic phone recording (lines on the body), slow-mo replay |
| `test_bat.py` (+ `bat_synth.py`) | Tape-band bat tracking on 3D-rendered blurred clips, yardstick, ball vs tape, full slow-mo path, dry swing, inbox calibration |
| `test_session_slowmo.py` | Clip↔swing matching (order, time, offset, missing/extra), times from metadata, sessions through the inbox |
| `test_server.py` | Upload server, safe file handling, end-to-end upload → analysis → index, chat |

## `web/` — viewer

| File | Purpose | Status |
|---|---|---|
| `index.html` | Viewer: Swing tab (3D replay, ghost, **annotated video synced with the replay**, scores, speeds, coach card, training plan, **Ask the coach chat**, sequencing chart, measurements vs top third), Session tab (incl. slow-mo matching notes), Pipeline status tab. Loads `data/demo.json` and, when served, `data/index.json` | ✅ (checked in the browser) |
| `upload.html` | Phone upload page, served at `/upload` (owned by the UI session) | ✅ |
| `data/overlays/*.mp4` | Annotated videos of the demo swings | ✅ |
| `data/demo.json` | Demo data (4 lab swings, reference ghost, ranges, demo session, status) | ✅ |
| `dist/swinglab-viewer.html` | Single-file version (published as the artifact) | ✅ |
| `data/index.json`, `data/swings/…` | Created by the pipeline for real swings | created at runtime |

## `data/`

| Path | Contents | In the zip? |
|---|---|---|
| `data/models/` | Trained models (`swing_models.joblib`, `scorers.joblib`), `metrics.json` | ✅ yes — works without retraining |
| `data/sample/` | Excel workbook + CSVs of one swing | ✅ yes |
| `data/raw/` | Driveline downloads (~900 MB unzipped) | ❌ no — `setup_data.py` downloads |
| `data/processed/` | Parquet + feature tables + validation CSVs | ❌ no — rebuilt by `setup_data.py` |
| `data/captures/` | Your recordings | created by you |

## Not built yet (by priority)

1. Run everything on real field recordings (goal 1–2), decide 60 fps, retrain.
2. Contact point on the bat; early/late proxy.
3. Gemini video read in the pipeline, diagram switching.
4. Size calibration with our own measured bat speeds (see HANDOFF section 7b).
5. ElevenLabs voice, Presage vitals, pitching, cross-session progress.
