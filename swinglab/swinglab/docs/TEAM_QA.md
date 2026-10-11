# Team questions and answers (2026-10-10 evening)

## 1. How does the Google Drive pipeline work?
Google Drive for desktop syncs a Drive folder to the laptop (e.g. `G:\My Drive\SwingLab`). Layout:
```
SwingLab/
  hitters.csv            name,bats,height_m,mass_kg,age,level[,bat_length_m,tip_band_m,hand_band_m]
  sessions/<date>/lidar/   Record3D .r3d files (or zipped Stray Scanner folders)
  sessions/<date>/slowmo/  240 fps clips (+ a clip with "yardstick" in its name = scale)
  sessions/<date>/log.csv  optional: which LiDAR file goes with which clip
```
Run `python api\server.py --watch "G:\My Drive\SwingLab"` (or once: `python scripts\process_inbox.py "G:\My Drive\SwingLab"`).
Every 15 s it looks for new recordings, pairs them (log.csv, else recording times, else order), analyzes
them, and saves each swing into the hitter's profile. Rows in hitters.csv become profiles automatically.
Already-processed files are skipped (`.swinglab_processed.json` in the folder).

## 2. From video to MediaPipe to Gemini: what happens
1. Load the recording (Record3D `.r3d`: color frames, depth maps, camera poses, intrinsics).
2. MediaPipe Pose (BlazePose heavy) on the color frames → 12 body joints in 2D per frame (every 2nd frame of 60 fps).
3. Lift to 3D: each joint's depth from LiDAR; joints without usable depth are rebuilt from 2D + bone length.
4. Quality checks (bone lengths, framing) → warnings.
5. Find the swings (hand-speed peak + shoulder turn ≥ 80°); for each swing: canonical skeleton (same as the
   Driveline lab data), events (load, foot plant, contact), 23 measurements, predicted bat speed, Form scores.
6. Hit or dry swing: look for a ball leaving the tee (slow-mo clip, else the LiDAR video). Hits get a contact score.
7. Annotated video rendered.
8. Gemini gets a JSON payload of the numbers (never the video) and writes the report; it is checked so it
   uses only numbers from the payload. Fallback: template report.

## 3. Fast pipeline from the LiDAR phone to the result (demo)
Built now: `api\server.py --watch <Drive folder>` + the viewer's "Auto-show new swings" (the newest swing pops
up with a notice) + pose on every 2nd frame. Remaining manual step: on the phone, Record3D → Share → save to
the Drive folder (or the /upload page). Time after the file lands: ~10 s pose + ~3–15 s Gemini.
Task (about a day, not built): live streaming from Record3D (paid "Wi-Fi Streaming & RGBD Videos" / USB, `record3d`
Python package) → ring buffer → automatic swing trigger → same analysis → viewer. Plugging the phone in
over USB does not by itself expose the files on Windows; streaming is the real hands-free path. Details: HANDOFF §13.

## 4. Clip vs long video
Removed. Every recording is analyzed the same way: find all swings, analyze each. A recording with 2+ swings
also gets a summary; history across recordings is the profile's Progress tab.

## 5. Two modes, Form vs Contact
Built. The old "Contact" score measured the *body* at contact, which is why your dry swings got 0 and 99; it is
now **Posture**. **Form** = Power + Sequencing + Posture. **Contact** = ball contact (smash factor + launch angle,
provisional rule), only for hits. **Overall** = Form for a dry swing, mean of Form and Contact for a hit. The mode
is decided automatically by ball detection; both night recordings now come out as dry swings with no contact score.

## 6. What object detection is used?
No trained detector. Body: MediaPipe Pose Landmarker (BlazePose heavy), a neural network. Ball and bat:
classic OpenCV image processing: bright-blob detection (threshold, connected components, roundness, size
from second moments), background subtraction (median frame), tracking with a constant-velocity prediction,
and a ballistic fit to confirm a real ball flight. Fine for a white ball / bright tape against a darker scene;
if real field footage is too busy, the next step is a trained detector (e.g. YOLO-family or a Roboflow model; check licenses).

## 7. LiDAR phone only (no slow-mo camera): feasibility
Body 3D and hit-vs-dry already work with one phone. Exit speed and launch angle: possible but rougher
(60 fps sees the ball in ~2–4 frames; use color frames + known ball size; expect ±10–15%). Bat speed at contact:
hard (too fast for 60 fps, too thin for the depth map); the model's predicted bat speed stays the main number.
Plan: record ~10 tee hits with both cameras → build a LiDAR-video ball tracker → compare with the 240 fps
numbers → switch to LiDAR-only if the error is acceptable. Not started (HANDOFF §13).

## 8. How much of the side panel is Gemini?
Gemini writes only the coach card text: headline, "what happened", "why it matters", cue, drill, which
exercises to pick from our library and why. Everything else is computed: scores, speeds, measurements,
hit/dry decision, data-quality notes, exercise prescriptions from the library. The badge on the card says
which wrote it. Your two night swings showed "Template report" because I had analyzed them with `--no-gemini`
to save the 20-calls-per-day quota; they are now re-analyzed with Gemini. Within one recording, swings after the
3rd get templates (quota). Exercise links and highlighting weak rotations on the video are for the next round.

---

# Round 2 (2026-10-10 night)

## Folder workflow instead of Google Drive (built)
Every hitter has a folder: `userdata\drop\<Name>\` (made when the hitter is created). Copy downloaded
recordings into it (`.r3d`, zipped Stray Scanner folders, slow-motion `.mov/.mp4`, a clip named
`yardstick…` for scale). `python api\server.py` checks it every 15 s, analyzes each new file once, and the
viewer opens the swing by itself (Auto-show). No sessions, CSV or log needed. Drive still works with `--watch`.

## Did we already build an upload feature? Yes.
1. **In the app**: "Upload swing" (top bar) → pick the hitter → drag files in (or browse) → progress bar →
   the swing appears when analyzed. File types are recognized automatically.
2. **From a phone**: the same dialog shows a QR code / link to `http://<laptop IP>:8000/upload` (phone and laptop
   on the same Wi-Fi). The phone page picks the hitter and the files; they go into that hitter's folder.
   On Windows, allow Python through the firewall (Private networks) the first time.

## Live stream with the phone plugged in (next task, not built)
Feasible on Windows: Record3D's Python package `record3d` 1.4.1 has a ready Windows build for Python 3.12
(checked). Needs: Record3D's streaming purchase in the app, Apple's USB drivers (iTunes / Apple Devices app),
then a capture loop: buffer the last ~6 s of RGB + depth + pose, run pose, detect the swing (hand speed +
shoulder turn), analyze that window, save to the hitter. About a day of work plus testing with the phone.

## Thrown-up ball, sometimes missed (built)
The ball logic no longer assumes a tee. Outcomes: **hit** (tee: ball leaves its spot fast; toss: the ball's
path breaks — much faster, new direction), **miss** (a ball that falls like a ball, g ≈ 9.8 m/s², with no
break: contact score 0), **dry** (no ball). Checked on synthetic toss clips (hit: contact frame, exit speed and
launch exact; miss detected; bat speed within 1% in both). On the two night videos the first version found
false "balls" (a lit window, a light drifting as the phone moved); a ball must now be round, steady in size,
really travel, and fall under gravity — both night swings are dry again.

## Free YOLO models
- **Ultralytics YOLOv8 / YOLO11** pretrained on COCO include "sports ball" and "baseball bat": free to
  download, `pip install ultralytics` (brings PyTorch). License **AGPL-3.0**: fine for an open-source
  hackathon project; a closed commercial app needs their paid license.
- **YOLOX** (Apache-2.0) and **RT-DETR** from PaddleDetection (Apache-2.0): COCO weights, permissive.
- **Roboflow Universe**: community baseball ball/bat models and datasets (license per dataset, often CC BY 4.0;
  free account).
COCO models struggle with a small, motion-blurred ball, especially at 60 fps. Our tracker + physics checks
should do for a white ball; YOLO is the fallback if real clips show it failing.

## Gemini for every swing (built)
Counts and statistics are computed first, without Gemini: hits, misses, dry swings, hit rate, form and
contact averages, bat speeds, trends. Gemini then writes every swing's report and the summary of any
recording with 2+ swings, from that data. Free tier: 20 calls per model per day (4 models in the fallback
chain); set `GEMINI_MAX_SWINGS` to cap it, or enable billing.
