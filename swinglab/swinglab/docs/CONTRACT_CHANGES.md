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
(none yet)
