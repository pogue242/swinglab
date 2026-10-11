"""Live capture: the iPhone streams color + LiDAR depth + camera pose over USB (Record3D's USB streaming, Python package
``record3d``); the last few seconds stay in memory, and after each swing they are written as a Stray Scanner
recording (rgb.mp4, depth/*.png in mm, confidence/*.png, odometry.csv) into the hitter's drop folder, where the
server (run it with --demo) analyzes it like any other recording. No export from the phone, no copying.

  python scripts\\live_capture.py --hitter Matteo              # Enter after each swing

Needs: Record3D on the iPhone with USB streaming unlocked, the phone plugged in by USB (on Windows, Apple's device
driver: iTunes or "Apple Devices"), ``pip install record3d``. The buffer/saving part (LiveBuffer) is tested without a
phone; the streaming part has not been run against a real phone yet.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LiveFrame:
    t: float                    # seconds (stream clock)
    rgb_jpg: bytes              # color frame, JPEG (keeps the buffer small)
    depth_mm: np.ndarray        # uint16 (h, w) depth in mm
    conf: np.ndarray | None     # uint8 (h, w) ARKit confidence 0-2
    K: np.ndarray               # (3, 3) intrinsics for the color frame
    pose: np.ndarray            # (7,) x, y, z, qx, qy, qz, qw: camera to world (ARKit)


class LiveBuffer:
    """The last ``seconds`` of the stream. Thread-safe: frames are added from the streaming callback."""

    def __init__(self, seconds: float = 8.0, jpeg_quality: int = 88):
        self.seconds, self.q = seconds, jpeg_quality
        self.frames: collections.deque[LiveFrame] = collections.deque()
        self.lock = threading.Lock()

    def add(self, t: float, rgb: np.ndarray, depth_m: np.ndarray, conf: np.ndarray | None, K: np.ndarray, pose) -> None:
        import cv2
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, self.q])
        d = np.nan_to_num(np.asarray(depth_m, np.float32), nan=0.0, posinf=0.0)
        f = LiveFrame(float(t), jpg.tobytes(), np.clip(d * 1000.0, 0, 65535).astype(np.uint16),
                      None if conf is None else np.asarray(conf, np.uint8), np.asarray(K, float), np.asarray(pose, float))
        with self.lock:
            self.frames.append(f)
            while self.frames and self.frames[-1].t - self.frames[0].t > self.seconds:
                self.frames.popleft()

    def window(self, t0: float, t1: float) -> list[LiveFrame]:
        with self.lock:
            return [f for f in self.frames if t0 <= f.t <= t1]

    def latest_t(self) -> float | None:
        with self.lock:
            return self.frames[-1].t if self.frames else None


def save_stray(frames: list[LiveFrame], out_dir: Path) -> Path:
    """Write frames as a Stray Scanner recording (what capture.load_stray reads). Written under <out>.part and
    renamed at the end, so the drop-folder watcher never sees half a recording."""
    import cv2
    from .render import open_writer
    if len(frames) < 10:
        raise ValueError(f"only {len(frames)} frames to save")
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + ".part")
    (tmp / "depth").mkdir(parents=True, exist_ok=True)
    (tmp / "confidence").mkdir(exist_ok=True)
    fps = (len(frames) - 1) / max(1e-6, frames[-1].t - frames[0].t)
    writer = None
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy"]
    try:
        for i, f in enumerate(frames):
            bgr = cv2.imdecode(np.frombuffer(f.rgb_jpg, np.uint8), cv2.IMREAD_COLOR)
            if writer is None:
                writer, _, _ = open_writer(tmp / "rgb", fps, (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
            cv2.imwrite(str(tmp / "depth" / f"{i:06d}.png"), f.depth_mm)
            cv2.imwrite(str(tmp / "confidence" / f"{i:06d}.png"),
                        f.conf if f.conf is not None else np.full(f.depth_mm.shape, 2, np.uint8))
            x, y, z, qx, qy, qz, qw = f.pose
            rows.append(f"{f.t - frames[0].t:.6f}, {i}, {x}, {y}, {z}, {qx}, {qy}, {qz}, {qw}, "
                        f"{f.K[0, 0]}, {f.K[1, 1]}, {f.K[0, 2]}, {f.K[1, 2]}")
    finally:
        if writer is not None:
            writer.release()
    (tmp / "odometry.csv").write_text("\n".join(rows), encoding="utf-8")
    tmp.replace(out_dir)
    return out_dir


def rgb_intrinsics(K: np.ndarray, rgb_w: int, depth_w: int) -> np.ndarray:
    """The pipeline needs intrinsics for the COLOR frame. If the optical center sits near the middle of the depth
    image rather than the color image, the matrix was for the depth image: scale it up."""
    if rgb_w != depth_w and abs(2 * K[0, 2] - depth_w) < abs(2 * K[0, 2] - rgb_w):
        K = K.copy()
        K[:2] *= rgb_w / depth_w
    return K


def stream(buffer: LiveBuffer, on_frame=None, device_index: int = 0):
    """Connect to the iPhone over USB with the ``record3d`` package and feed ``buffer``; ``on_frame(t, rgb)`` is
    called for each frame. Returns the session (keep a reference). Not yet run on a real phone.
    (An automatic swing trigger from frame-to-frame motion was tried on TestSet2 and caught only 4 of 7 swings:
    the hitter fills too little of the frame, so the trigger is a key press.)"""
    from record3d import Record3DStream
    devs = Record3DStream.get_connected_devices()
    if not devs:
        raise RuntimeError("no iPhone found: plug it in by USB, open Record3D and start USB streaming")
    s = Record3DStream()

    def new_frame():
        t = time.time()
        rgb, depth = s.get_rgb_frame(), s.get_depth_frame()
        conf = s.get_confidence_frame() if hasattr(s, "get_confidence_frame") else None
        c = s.get_intrinsic_mat()
        K = np.array([[c.fx, 0, c.tx], [0, c.fy, c.ty], [0, 0, 1.0]])
        K = rgb_intrinsics(K, rgb.shape[1], depth.shape[1])
        p = s.get_camera_pose()
        buffer.add(t, rgb, depth, conf if conf is not None and getattr(conf, "size", 0) else None, K,
                   [p.tx, p.ty, p.tz, p.qx, p.qy, p.qz, p.qw])
        if on_frame:
            on_frame(t, rgb)

    s.on_new_frame = new_frame
    s.on_stream_stopped = lambda: print("stream stopped")
    s.connect(devs[device_index])
    return s
