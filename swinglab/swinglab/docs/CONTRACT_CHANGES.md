# Data contract changes between the FUNC and UI sessions

The UI reads `userdata/swings/<profile>/<id>.json`, `_summary_<recording>.json` and `/api/*`.
FUNC may add fields freely. Renames, removals and changed meanings go here, newest first.

## Changes by FUNC
(none yet)

## Requested by UI
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
