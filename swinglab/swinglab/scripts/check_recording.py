"""Check a recording right after transferring it to the laptop (no MediaPipe needed).

  python scripts\\check_recording.py <Stray Scanner folder | Record3D .r3d>   # LiDAR body capture
  python scripts\\check_recording.py <slow-motion video file>    # 240 fps bat/ball capture

Prints what the pipeline needs to know and flags problems:
  LiDAR: frame count, depth frame rate, depth/confidence present, camera poses,
         duration, video resolution; how much of the frame has valid depth.
  Slo-mo: true capture rate (frames ÷ duration), resolution, duration. Some
         exports re-time slow motion to 30 fps playback: frames are all there
         but the file's fps is wrong; we flag it so you can pass --fps 240.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def check_video(p: Path, expect_fps: float | None = None) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return {"ok": False, "problem": f"cannot open {p.name} (HEVC on Windows may need the HEVC codec or convert with ffmpeg)"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    info = {"ok": ok, "frames": n, "file_fps": round(fps, 2), "duration_s": round(n / fps, 2) if fps else None,
            "resolution": f"{w}x{h}", "mean_brightness": round(float(frame.mean()), 1) if ok else None}
    if expect_fps and fps and fps < expect_fps * 0.8:
        info["warning"] = (f"file says {fps:.0f} fps; if this is a {expect_fps:.0f} fps slow-motion capture it was "
                           f"re-timed for playback. Real duration ≈ {n / expect_fps:.2f} s. Pass --fps {expect_fps:.0f}.")
    if ok and frame.mean() < 40:
        info["warning_light"] = "very dark video; slow motion needs bright light"
    return info


def check_stray(root: Path) -> dict:
    from swinglab import capture
    cap = capture.load_any(root)
    dts = np.diff(cap.timestamps)
    depth_files = sorted((root / "depth").glob("*.png")) if cap.kind == "stray" else sorted(cap.depth_dir.glob("*.depth"))
    d0 = cap.depth(len(depth_files) // 2) if depth_files else None
    sample = [cap.depth(i) for i in np.linspace(0, len(depth_files) - 1, min(10, len(depth_files))).astype(int)] if depth_files else []
    far = float(np.nanmax([np.nanmax(d) for d in sample if np.isfinite(d).any()])) if sample else None
    info = {"app": cap.kind, "frames_odometry": cap.n, "depth_frames": len(depth_files),
            "depth_fps": round(cap.fps, 1), "max_gap_ms": round(float(dts.max() * 1000), 1) if len(dts) else None,
            "duration_s": round(float(cap.timestamps[-1] - cap.timestamps[0]), 2),
            "confidence_maps": cap.conf_dir is not None or (cap.kind == "record3d" and (cap.depth_dir / "0.conf").exists()),
            "camera_moved_m": round(float(np.linalg.norm(cap.cam_pos.max(0) - cap.cam_pos.min(0))), 3),
            "depth_resolution": None if d0 is None else f"{d0.shape[1]}x{d0.shape[0]}",
            "valid_depth_fraction_mid_frame": None if d0 is None else round(float(np.isfinite(d0).mean()), 2),
            "median_depth_m_mid_frame": None if d0 is None else round(float(np.nanmedian(d0)), 2),
            "farthest_depth_m": None if far is None else round(far, 2),
            "video": check_video(cap.video) if cap.kind == "stray" else {"ok": (cap.depth_dir / "0.jpg").exists(), "frames": "jpg sequence", "resolution": f"{cap.rgb_size[0]}x{cap.rgb_size[1]}"}}
    warn = []
    if len(depth_files) != cap.n:
        warn.append("depth frame count differs from odometry rows")
    if info["depth_fps"] < 25:
        warn.append("depth below 25 fps: swing timing will be coarse")
    if d0 is not None and info["valid_depth_fraction_mid_frame"] < 0.5:
        if info["median_depth_m_mid_frame"] > 3.5:
            warn.append("less than half the frame has confident depth: move closer (3–4 m), avoid direct sun on the hitter")
        else:
            warn.append(f"less than half the frame has depth, but what is measured is close (median "
                        f"{info['median_depth_m_mid_frame']} m): the far background is out of range, which is fine "
                        "if the hitter is fully covered (check the pose overlay)")
    if far is not None and far < 3.5:
        warn.append(f"no depth beyond {far:.2f} m anywhere in the clip: the app (or light) limits range. "
                    "If the app has a max-distance / depth-range setting, raise it to 5 m; otherwise keep the "
                    "hitter AND the bat path closer than this")
    if info["duration_s"] < 3:
        warn.append("short clip: start recording with the hitter already set in the stance, end 1–2 s after the finish")
    info["warnings"] = warn
    info["set_SWINGLAB_PHONE_FPS_to"] = round(info["depth_fps"])
    return info


if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = Path(sys.argv[1])
    fps = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else 240.0
    is_lidar = target.is_dir() or target.suffix.lower() == ".r3d"
    res = check_stray(target) if is_lidar else check_video(target, expect_fps=fps)
    print(json.dumps(res, indent=2))
