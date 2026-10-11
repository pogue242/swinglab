# Data contract changes between the FUNC and UI sessions

The UI reads `userdata/swings/<profile>/<id>.json`, `_summary_<recording>.json` and `/api/*`.
FUNC may add fields freely. Renames, removals and changed meanings go here, newest first.

## Changes by FUNC

### 2026-10-11 (branch func/scoring): additions only, nothing renamed or removed
For the whole-recording view (issue 4, UI part):
- **`GET /api/profiles/{id}/recordings`**: the profile's swings grouped by recording, oldest first:
  `{"recordings": [{"recording", "n_swings", "recorded_at", "summary", "summary_url", "headline",
  "form" ({mean, best, worst, ...} or null), "video" ({"file", "url", "t0"} or null),
  "swings": [{"id", "file", "url", "video_file", "video_url", "t" ({start, peak, end} or null),
  "partial", "mode", "scores", "focus"}]}]}`.
  `video` is the whole recording as a plain MP4; `swings[].t` are **seconds into that video**, so
  marking the swings on its timeline needs no arithmetic. Swings analyzed before this change have
  no recording info: each comes as its own one-swing "recording" with `t: null` and `video: null`
  (re-analyzing fixes that).
- New file `userdata/swings/<profile>/_recording_<recording>.mp4` (H.264, 30 fps, 960 px high) for
  recordings with 2+ swings.
- `_summary_<recording>.json` gains `recording_video`: `{"source", "fps", "t0", "path" (relative to
  userdata), "swings": [{"id", "start", "peak", "end", "partial"}]}` (times in video seconds).
- Swing JSON gains `recording_t0` (the recording clock at its first frame; `window.* - recording_t0`
  = seconds into the whole-recording video).

Other additions:
- `window.partial` (bool): the recording starts mid-swing (less than 0.6 s before the peak), so the
  stance and load were not filmed. A warning says so too. Suggest: show the swing with a "partial"
  tag.
- `ball_check` (only when the blob tracker reported a hit or miss): `{"tracker", "confirmed",
  "frames_checked", "frames_with_ball", "near_track", "best_score"}` from the YOLOX ball detector.
  `mode` keeps its meaning (hit | miss | dry) but is now dry unless the detector confirms a ball.
- `quality.lift`: `"mediapipe_3d_with_lidar_anchor"` (new default) or `"lidar_depth"`; plus
  `quality.mediapipe_scale`, `quality.anchor_frames`. In the new mode `quality.placed_by_bone_length`
  is renamed `quality.depth_placed_by_bone_length` (it no longer shapes the joints). The viewer does
  not read either.

## Requested by UI
- 2026-10-11 (after re-analyzing Adam with func/scoring): swing 1 is `window.partial` and its warning says
  "stance, stride and timing numbers for this swing are unreliable", yet its **focus is stance width** and the
  headline is "Work on: stance width". Please skip measurements the swing's own warnings call unreliable when
  choosing the focus. Also `mode_evidence` carries the full local path of the missing YOLOX model; users see
  that text in the "data notes" popover, so a short "ball detector not installed: treated as a dry swing" would
  read better (the path can go in the log).
- 2026-10-11 (bug, FUNC side): swing JSON files contain bare `NaN` (e.g. `"...r_pelvis":NaN`), which is not
  valid JSON; browsers refuse the file. Seen in Adam's swings 1 and 2 of 3 (`Adam__2026-10-10--22-25-12-01/-02`).
  The viewer now reads NaN/Infinity as null, but please write `null` (`json.dumps(..., allow_nan=False)` after
  replacing NaN), so other readers work too.
- 2026-10-11 (question): `series.pelvis_vel` / `torso_vel` go negative (e.g. −199 °/s hips at contact). The
  chart clips at 0 and the phase table shows the signed value. Is negative "turning back"? If so, fine; if it is
  a sign/handedness issue, please fix at the source.
- 2026-10-10 (bug, FUNC side): a **failed** analysis is still recorded in `drop/.swinglab_processed.json`
  (`"swings": 0`), so the recording is never retried, even after the cause is fixed (seen: missing
  `pose_landmarker_heavy.task`). Re-uploading the same file is then silently skipped and the viewer can only say
  "no swing came out of this upload". Also the log says `done: 1 new swing(s)` when 0 swings were saved.
  Please: don't mark failed recordings as processed (or retry them on re-upload), and count only saved swings.
- 2026-10-10: the upload dialog now follows `GET /api/status` (`running`, `last_run`, `results[].id/stages`,
  `log` lines with `!!` / `ERROR`) to show queued → analyzing → ready/failed. Please keep those fields; a
  `results[].error` (plain reason) and `results[].file` (the saved swing JSON) would let the UI stop parsing
  log lines.
- 2026-10-10 (optional, low priority): `app.json` → `key_features`: an ordered list of 4–6 feature keys the
  swing view shows by default ("Key measurements"). Until it exists the viewer uses a built-in list (top hip
  speed, top chest speed, biggest separation, hands out front, front leg straightening). If scoring changes
  which mechanics matter most, send the list here so the default view follows the model.
