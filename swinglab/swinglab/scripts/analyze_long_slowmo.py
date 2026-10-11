"""A LiDAR recording of several swings + ONE long slow-motion video of the same session (no clock to sync).

  python scripts\\analyze_long_slowmo.py --lidar "TestSet4\\angle 1- lidar 1.r3d" --slowmo "TestSet4\\angle 1- vid 1.mp4" ^
      --profile matteo --name "toss 1"

1. lift the LiDAR recording and find its swings;
2. MediaPipe on the slow-motion video (every 3rd frame) -> swings there (pipeline.slowmo_swing_times);
3. fit video_time = offset + slowdown x lidar_time on the swings both saw (pipeline.fit_time_map); the slowdown
   gives the capture rate (x8 of a 30 fps export = 240 fps);
4. cut one clip per LiDAR swing around its contact, saved under userdata/swings/<profile>/_slowmo/;
5. analyze everything like an upload (ball check with the object detector, hit / miss / dry, scores, coach, video).
Slow-motion clips are matched to swings by order (one clip per swing, cut in swing order).
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swinglab import capture, config, pipeline, profiles, session  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lidar", required=True)
ap.add_argument("--slowmo", required=True)
ap.add_argument("--profile", required=True)
ap.add_argument("--name", required=True, help="recording label, e.g. 'toss 1'")
ap.add_argument("--video-fps", type=float, default=30.0, help="frame rate of the slow-motion FILE (playback)")
ap.add_argument("--lidar-cache", help="pickle of pipeline.lift_recording output (skip the lift)")
ap.add_argument("--pose-cache", help="pickle {'world': {...}, 'every': n} of the slow-motion pose (skip MediaPipe)")
ap.add_argument("--no-gemini", action="store_true")
a = ap.parse_args()

prof = profiles.get(a.profile) or profiles.by_name(a.profile)
if not prof:
    sys.exit(f"no profile {a.profile}")
hitter = profiles.hitter(prof)
lifted = pickle.load(open(a.lidar_cache, "rb")) if a.lidar_cache else pipeline.lift_recording(a.lidar, height_m=float(hitter["height_m"]))
t, J, q = lifted
wins = session.find_swings(t, J, bats=hitter["bats"])
tl = [w.peak - t[0] for w in wins]
print(f"LiDAR: {len(wins)} swings at {[round(x, 2) for x in tl]} s")

if a.pose_cache:
    world = pickle.load(open(a.pose_cache, "rb"))["world"]
else:
    world = {}
    capture.mediapipe_keypoints(a.slowmo, config.MODELS / "pose_landmarker_heavy.task", every=3, fps=a.video_fps, world=world)
tv = pipeline.slowmo_swing_times(world, a.video_fps, hitter["bats"])
fit = pipeline.fit_time_map(tl, tv)
if not fit:
    sys.exit("could not line up the slow-motion video with the LiDAR swings")
k, off = fit["slowdown"], fit["offset"]
print(f"slow-motion: {len(tv)} swings; matched {len(fit['pairs'])}/{len(tl)}; slowdown x{k:.2f} "
      f"(capture ~{a.video_fps * k:.0f} fps); residuals {fit['residuals_s']} s")

clip_dir = profiles.swings_dir(prof["id"]) / "_slowmo"
clips = []
for i, x in enumerate(tl):
    c_v = off + k * (x + 0.065)                               # contact ~ peak hand speed + 65 ms
    f0, f1 = max(0, int((c_v - 0.45 * k) * a.video_fps)), int((c_v + 0.45 * k) * a.video_fps)
    clips.append(pipeline.cut_clip(a.slowmo, f0, f1, clip_dir / f"{prof['name']}__{a.name}-{i + 1:02d}.mp4", a.video_fps))

out = pipeline.analyze_recording(Path(a.lidar), None, hitter, use_gemini=not a.no_gemini, slowmo_clips=clips,
                                 slowmo_fps=a.video_fps * k, lifted=lifted)
for r in out["recs"]:
    r["id"] = f"{prof['name']}__{a.name}" + (r["id"][len(Path(a.lidar).stem):] if r["id"].startswith(Path(a.lidar).stem) else "")
    r["recording"] = a.name
    r["profile"] = prof["id"]
    if "analysis" in r:
        p = pipeline.save_result(r, prof["id"])
        s = r.get("slowmo") or {}
        print(f"{r['id']}: {r['mode']} ({r.get('mode_evidence', '')[:90]}) elite {r['analysis']['elite']['match']} "
              f"rough exit {s.get('exit_speed_rough_mph')}")
if out.get("summary"):
    out["summary"]["recording"] = f"{prof['name']}__{a.name}"
    out["summary"]["slowmo_long"] = {"slowdown": k, "capture_fps": round(a.video_fps * k), "offset_s": off,
                                     "matched": len(fit["pairs"]), "residuals_s": fit["residuals_s"]}
    print("summary", pipeline.save_session(out["summary"], prof["id"]))
