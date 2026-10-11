"""MediaPipe sanity check: run pose on any video and write a 2D keypoint overlay to eyeball.

  python scripts\\pose_overlay.py <video.mp4 | recording.r3d | Stray Scanner folder | folder\\%d.jpg> [--out file.mp4] [--fps N]

Green joints/bones = visibility >= 0.5 (used by the 3D lift); orange = weaker (dropped).
Prints how often each joint was found and saves the keypoints as JSON next to the video.
Run `python scripts\\get_pose_model.py` once first.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from swinglab import capture, config, render

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--out")
ap.add_argument("--fps", type=float, help="true frame rate (image sequences: OpenCV assumes 25)")
ap.add_argument("--max-height", type=int, default=960, help="output video height (smaller file)")
a = ap.parse_args()

src = Path(a.src)
fps = a.fps
if capture.is_recording(src):
    cap = capture.load_any(src)
    video, fps = cap.video, fps or cap.fps
else:
    video = src
out = Path(a.out) if a.out else src.parent / f"{src.stem.replace('%d', 'seq')}_pose.mp4"

t0 = time.time()
kps, (w, h) = capture.mediapipe_keypoints(video, config.MODELS / "pose_landmarker_heavy.task", fps=fps)
fps = fps or cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 30.0
print(f"pose: {len(kps)} frames with a person, {w}x{h}, {time.time() - t0:.1f} s")

s = min(1.0, a.max_height / h)
size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
vc = cv2.VideoCapture(str(video))
i = 0
while True:
    ok, frame = vc.read()
    if not ok:
        break
    img = cv2.resize(frame, size)
    kp = kps.get(i)
    if kp:
        render.draw_keypoints_2d(img, kp, scale=s)
    n_ok = sum(v[2] >= 0.5 for v in kp.values()) if kp else 0
    cv2.putText(img, f"frame {i}  t={i / fps:.2f}s  joints >=0.5: {n_ok}/12", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(img)
    i += 1
vw.release()

names = list(next(iter(kps.values())).keys()) if kps else []
vis = np.array([[f[n][2] for n in names] for f in kps.values()]) if kps else np.zeros((0, 0))
print(f"frames read {i}; person found in {len(kps) / max(i, 1):.0%}")
for k, n in enumerate(names):
    print(f"  {n:11s} mean visibility {vis[:, k].mean():.2f}   >= 0.5 in {np.mean(vis[:, k] >= 0.5):.0%} of frames")
kp_json = out.with_suffix(".json")
kp_json.write_text(json.dumps({"fps": fps, "size": [w, h], "frames": {str(k): v for k, v in kps.items()}}), encoding="utf-8")
print("wrote", out, "and", kp_json.name)
