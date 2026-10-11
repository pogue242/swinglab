# Data contract changes between the FUNC and UI sessions

The UI reads `userdata/swings/<profile>/<id>.json`, `_summary_<recording>.json` and `/api/*`.
FUNC may add fields freely. Renames, removals and changed meanings go here, newest first.

## Changes by FUNC
(none yet)

## Requested by UI
- 2026-10-10 (optional, low priority): `app.json` → `key_features`: an ordered list of 4–6 feature keys the
  swing view shows by default ("Key measurements"). Until it exists the viewer uses a built-in list (top hip
  speed, top chest speed, biggest separation, hands out front, front leg straightening). If scoring changes
  which mechanics matter most, send the list here so the default view follows the model.
