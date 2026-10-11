# SwingLab — Recording Plan (Saturday, baseball field)

Everything you need to record usable data for single swings and long
sessions. **MediaPipe and the laptop are not needed while recording**;
everything is analyzed afterwards. Keep every raw file.

The app decides **from the video itself** whether a recording is one swing or
a session: it counts the swings in it (1 → single-swing analysis, 2 or more →
session with trends). So record clips and long videos the same way; only the
number of swings differs.

---

## 1. What tomorrow's recordings must prove

| Recording | Proves |
|---|---|
| Single clips (both cameras) | Phone 3D pipeline works on real data; measured exit speed; scores react to form |
| Deliberate-flaw clips | Each flaw lowers the right score and triggers the right coaching focus |
| Long session videos | Swing finder counts swings correctly; trends and fatigue show up |
| Calibration clips | Scale for ball/bat speed; depth frame rate; depth quality outdoors |
| Demo takes | Clean footage for judging |

---

## 2. Before the field

### 2a. LiDAR app: Record3D or Stray Scanner (both supported)

- Both apps record the **same iPhone LiDAR through Apple's ARKit**: same
  depth resolution (~256×192), same range (reliable to ~5 m; we use 3–4 m),
  and both save the **color video** too. The nicer "outline" in Record3D's
  live view is its rendering; what matters is the exported data.
- **Record3D:** the free version allows 3 recordings. Exporting the depth
  video (`.r3d`) needs the **"Wi-Fi Streaming & RGBD Videos" ($2.99)**
  purchase (per the developer; check in the app before buying anything
  bigger). If it records at 60 fps, good: smoother timing.
- **Distance:** neither app should cut off a hitter at 3–4 m. If Record3D has
  a depth-range or "max distance" filter in its settings, set it to at least
  5 m so the hitter and bat are never clipped.
- Export from Record3D as the **`.r3d` file** (or the extracted folder); drop
  it in `lidar/` as-is. Stray Scanner: the whole recording folder.
- **Decide with data:** make one 10 s recording in each app of the same scene
  and run `check_recording.py` on both; pick the one with higher
  `valid_depth_fraction`, steady frame rate and the right fps. Use one app for
  the whole day.

### 2b. Night test (optional, using one Record3D free recording)

Good for checking the **file format and transfer**, not for judging quality:
- **LiDAR depth works in the dark** (it has its own laser; it actually does
  better than in direct sun).
- **But the pose model (MediaPipe) reads the color video**, and ARKit's
  camera tracking needs a visible scene. In darkness the skeleton will be
  poor. Under bright field or porch lights it can work; dim light won't.
- **Slow motion at night is unreliable:** 240 fps needs strong light, and
  stadium/LED lights often flicker visibly at 240 fps.
- So: record 10 s of someone standing and doing a slow practice swing under
  the brightest light available, transfer it, run
  `python scripts\check_recording.py "<file.r3d>"`. If it loads and depth
  coverage is above 0.5, the pipeline is ready. Save the real swings for
  daylight.

### 2c. Checklist

1. Phone storage: Stray Scanner/Record3D about 0.2–0.4 GB per minute; 1080p
   240 fps slow motion about 0.4–0.5 GB per minute. Bring a power bank.
2. Slow motion setting on the second phone: Settings → Camera → Record Slo-mo
   → 1080p at 240 fps.
3. Set up the shared folder and fill in `hitters.csv` (section 11).
4. Bright tape on the bat (section 3).

---

## 3. Equipment checklist

- iPhone 17 Pro (Record3D or Stray Scanner) + second phone (240 fps)
- 2 tripods or clamp mounts (a fence clamp works); phone holders
- Tee, a bucket of real baseballs, bat, net or open field
- **Bright tape** (white or neon): one ~5 cm band around the bat's end, one
  just above the hands. **Why:** at 240 fps the bat is a blur, and a plain bat
  blends into the background. Two bright bands give the slow-motion camera two
  points it can find in every frame, so we can measure **bat speed at contact**
  (the same number Driveline's model predicts, so we can check our
  prediction), the **bat angle** through the zone, and **where on the bat**
  the ball was hit. Without tape we still get the ball's exit speed and launch
  angle, just not bat speed. Use a contrasting color (white tape on a black
  bat, neon on a silver one).
- Yardstick or 1 m stick (scale), tape measure
- Phone for notes (the log), power bank, towel (sweat/glare on lenses)
- Optional: chalk or tape to mark tripod and tee positions

---

## 4. Hitter profile (once per hitter, before recording)

Measure, don't guess: **height** (shoes off, cm or in), **weight**, age,
bats R/L, playing level (high school / college / other), bat length and
weight. These go in `hitters.csv`; the model needs height, weight, age and
level to adjust for body size.

---

## 5. Field layout (top view, right-handed hitter)

```
                           ↑ toward the field (ball goes this way)
                           │
                     ·  ·  ·  ·  ·   ball flight
                           │
   [Camera B] ────────── (tee) ●          B: 4–5 m to the side, perpendicular
   240 fps, level            │               to the ball's path, tee height,
   with the tee         HITTER ▲             looking straight across the tee
                       (faces plate,
                        chest toward A)
                           │
                    3–4 m  │   30–45° toward the catcher's side
                           ▼
                      [Camera A]   iPhone 17 Pro, LiDAR app, hip height
                         (behind the line of any batted ball)
```

For a left-handed hitter, mirror the layout. Never place a phone in front of
the tee or along the ball's path. Use a fence or net between Camera B and the
hitter if possible.

---

## 6. Camera A — LiDAR body capture (iPhone 17 Pro, Record3D or Stray Scanner)

- **Distance:** 3–4 m from the hitter (LiDAR range drops beyond ~5 m and in
  bright sun).
- **Angle:** facing the hitter's chest, turned 30–45° toward the catcher's
  side.
- **Height:** about hip height (~1 m). Phone level, **landscape**, on a
  tripod; never moves during a recording.
- **Framing:** the whole body and the whole bat path with margin: feet, hands
  at the top of the load, bat at full extension. Check the first recording.
- **Light:** avoid the sun directly behind the hitter or into the lens; shade
  or overcast is best. If depth coverage is low, move closer or rotate the
  setup.
- **Start/stop:** start with the hitter **already set in the stance and
  still for about 1–2 seconds** before each swing (the app uses those frames to
  find the hitter's stance and direction). Hold the finish 1 s.

## 7. Camera B — slow motion bat and ball (second phone, 240 fps)

- **Position:** side view, perpendicular to the ball's path, level with the
  tee, 4–5 m away, tee in the left or right third of the frame (so the first
  ~1.5 m of flight stays in view).
- **Settings:** 1080p 240 fps, no zoom (1×). Lock focus and exposure on the
  tee (iPhone: press and hold on the tee until "AE/AF LOCK").
- **Scale:** at the start of each setup, record 3 s of the yardstick held
  upright **at the tee**, in the same plane as the ball. Re-record it if the
  camera moves.
- **Bat:** the taped bat end must be visible through contact.
- Bright daylight; slow motion gets dark and blurry in shade.

## 8. Sync (every recording)

Start Camera B, then Camera A. Before the first swing, the hitter **taps
the bat on the plate once** (visible and audible to both). The ball leaving
the tee is the precise sync point for each swing.

---

## 9. Two kinds of recordings

### 9a. Single clips (one swing per recording)

1. Both cameras start; tap the bat; hitter sets up still 1–2 s; swings; holds
   the finish 1 s; both cameras stop.
2. One line in the log (section 11).
3. Reset the ball on the tee in the same spot.

### 9b. Long session videos (many swings per recording)

The batting-cage use case: one continuous Camera A recording.

1. **Length:** 10–20 swings per recording (about 3–5 minutes). Shorter
   recordings are safer (file size, transfers, a dropped phone loses less).
2. Camera A records continuously and **never moves**.
3. For **each swing**: reset the ball, hitter steps back into the **same
   spot**, sets up **still for 1–2 s**, swings, holds the finish. Walking
   around between swings is fine.
4. **Camera B in sessions:** start and stop slow motion **for each swing**
   (one slow-motion clip per swing, in order). This keeps files small and lets
   us match clips to swings by order. (Matching many swings inside one long
   slow-motion video isn't built yet.)
5. Practice/half swings count as swings for the app: say "practice" out loud
   and note them in the log so we can label them.
6. Log one line per recording with the swing count, and one line per swing
   if you can (section 11).

---

## 10. Shot list (per hitter; ~1.5–2 hours for two hitters)

| Block | What | Count | Cameras | Notes |
|---|---|---|---|---|
| 0 Calibration | Yardstick at the tee (B); 10 s of the hitter standing still in the stance (A) | 1 per setup | A, B | Redo when cameras move |
| 1 Clips — normal | Normal tee swings | 15–20 | A + B | Main test set |
| 2 Clips — flaws | **Arms only** (no hip turn) · **shoulders open early** · **lunge / hips slide forward** · **front leg stays bent** · **no stride** | 3–5 each | A + B | Label the flaw in the log |
| 3 Clips — variety | Low tee ×3, high tee ×3, dry swings (no ball) ×3 | 9 | A + B | Contact point variety |
| 4 Session — steady | One long recording, normal swings | 15 | A (+ B per swing) | Consistency, swing count |
| 5 Session — fatigue | One long recording, 20 swings with little rest | 20 | A (+ B per swing) | Should show a late drop |
| 6 Moving camera (optional) | Camera A handheld, walking slowly | 3 clips | A | Stretch feature |
| 7 Pitching (optional) | Throws into a net or to a partner; A oblique, B on the arm side | 5–10 | A, B | Phase 2 data |
| 8 Demo takes | Best swings, clean background, good light | 3–5 | A + B | For judging video |

Do Block 0 and one Block 1 clip first, then **check them on a laptop** if one
is there (section 12) before recording the rest.

---

## 11. Log and naming

Shared folder layout:
```
SwingLab/
  hitters.csv          name,bats,height_m,mass_kg,age,level
  sessions/
    2026-10-10_field/
      lidar/           Record3D .r3d files, or Stray Scanner folders (zips OK)
      slowmo/          slow-motion videos
      log.csv
```
`hitters.csv` example:
```
name,bats,height_m,mass_kg,age,level
Adrian,R,1.78,75,20,college
Wilson,R,1.80,80,20,college
```
`log.csv`, one row per recording (and per swing where you can):
```
swing,hitter,set,lidar,slowmo,contact,notes
001,Adrian,calibration,,IMG_1001.MOV,,yardstick
002,Adrian,normal,8f3a2c1d,IMG_1002.MOV,line drive,
003,Adrian,flaw_arms_only,1b9e7a40,IMG_1003.MOV,grounder,
010,Wilson,session_steady,c77e1f02,,,15 swings; slowmo IMG_1010–1024
```
`lidar` = the Record3D file name or Stray Scanner folder name; `slowmo` =
the video file name. If writing names on the field is too slow, record in a
strict order and write the order down; the app can pair by order.

---

## 12. Quick on-site check (if a laptop is there)

On the first recording of each setup only:
```powershell
python scripts\check_recording.py "<.r3d file or lidar folder>"
python scripts\check_recording.py "<slow-motion file>"
```
Look for: depth fps; `valid_depth_fraction` above 0.5; video readable; slow
motion at 240 fps (or the re-timed warning). Fix framing/light before
recording the rest.

---

## 13. Getting files off the phones (Windows)

- **Record3D:** export the recording (share → `.r3d`) and save it to Files or
  Google Drive; AirDrop to a Mac also works.
- **Stray Scanner:** export each recording folder via the iOS Files app
  (Stray Scanner's documents) or plug the iPhone into the laptop and copy from
  Apple Devices / iTunes **File Sharing**. Zipping on the phone and uploading
  to Google Drive also works; zips are unzipped automatically.
- **Slow motion:** export the **unmodified original** (iPhone: AirDrop with
  "All Photos Data", or Photos "Export Unmodified Original"). Google Photos and
  messaging apps re-encode and can drop the 240 fps.
- Put files in `SwingLab/sessions/<date>/lidar` and `/slowmo`. Then:
  ```powershell
  python scripts\process_inbox.py "G:\My Drive\SwingLab" --list   # check pairing
  python scripts\process_inbox.py "G:\My Drive\SwingLab"          # analyze
  python -m http.server -d web 8000                                # open http://localhost:8000
  ```
- HEVC video on Windows may need the HEVC Video Extension, or convert:
  `ffmpeg -i in.mov -c:v libx264 out.mp4`.

---

## 14. Safety

- Phones and people stay out of the ball's path; Camera B behind a fence or
  net when possible; nobody stands in front of the tee.
- Check the swing area; at night, only swing where the area is well lit.
- Hydrate; the fatigue block is about swing speed, not exhaustion.

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| Depth patchy / low coverage | Move closer (3 m), get out of direct sun, avoid shiny backgrounds |
| Feet or bat cut off | Step the tripod back or use landscape; recheck framing |
| Slow-motion file says 30 fps | Frames are fine; pass `--fps 240`; next time export the original |
| Ball blurry | More light; lock exposure; avoid shade |
| Swing count wrong in a session | Pause still 1–2 s before each swing; log practice swings |
| Phone storage full | Shorter session recordings (10 swings); transfer between blocks |
