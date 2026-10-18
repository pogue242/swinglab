"""Live capture: the iPhone streams color + LiDAR depth + camera pose over USB (Record3D's USB streaming, Python package
``record3d``). LiveSession (the viewer's Record swing, through the server) records a whole session, every swing in
it, and ends it once the hitter has stood still for a few seconds; R3dWriter packs it as a Record3D .r3d into the
hitter's drop folder, where the server analyzes it like any other recording. No export from the phone, no copying.

Needs: Record3D on the iPhone with USB streaming unlocked, the phone plugged in by USB (on Windows, Apple's device
driver: iTunes or "Apple Devices"), ``pip install record3d`` and ``pip install lzfse`` (or ``pyliblzfse``). The
streaming part has not been run against a real phone yet.
"""

from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path

import numpy as np


class R3dWriter:
    """A whole session (one or more swings) written to disk while it streams, then packed as a Record3D ``.r3d``
    (what capture.load_record3d reads): ``rgbd/<i>.jpg``, ``rgbd/<i>.depth`` (LZFSE float32 meters),
    ``rgbd/<i>.conf`` (LZFSE uint8 0-2) and a ``metadata`` JSON. Frames go to ``staging`` (keep it outside the drop
    folder); finish() zips them to ``<final>.part`` and renames it, so the drop-folder watcher never sees half a
    session. add() runs in the stream's callback and only queues a copy of the frame; a thread writes it. When the
    writer falls behind, frames are dropped and counted rather than queued up (a backlog would delay the stream).
    Frames past ``max_s`` seconds are not kept (``is_full``)."""

    def __init__(self, staging: Path, final: Path, max_s: float = 90.0, queue_frames: int = 30, jpeg_quality: int = 90):
        self.staging, self.final, self.max_s, self.jpeg_quality = Path(staging), Path(final), max_s, jpeg_quality
        if self.final.exists():
            raise FileExistsError(f"{self.final} already exists")
        shutil.rmtree(self.staging, ignore_errors=True)          # left over from a crashed session
        (self.staging / "rgbd").mkdir(parents=True)
        self.q: queue.Queue = queue.Queue(maxsize=queue_frames)   # ~1 s of full-resolution frames
        self.timestamps: list[float] = []
        self.poses: list[list[float]] = []                        # [qx, qy, qz, qw, tx, ty, tz] per frame
        self.coeffs: list[list[float]] = []                       # [fx, fy, cx, cy] per frame
        self.K0: np.ndarray | None = None
        self.rgb_wh: tuple[int, int] | None = None
        self.depth_wh: tuple[int, int] | None = None
        self.written = self.dropped = 0
        self.t0: float | None = None
        self.t_last: float | None = None
        self.error: str | None = None
        self.is_full = False
        self._closed = False
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    @property
    def elapsed(self) -> float:
        return 0.0 if self.t0 is None or self.t_last is None else self.t_last - self.t0

    def add(self, t: float, rgb: np.ndarray, depth_m: np.ndarray, conf: np.ndarray | None, K: np.ndarray, pose) -> bool:
        """Queue one frame (False when it was not kept: finished, past max_s, or the writer is behind)."""
        if self._closed or self.error:
            return False
        if self.t0 is None:
            self.t0 = t
        if t - self.t0 > self.max_s:
            self.is_full = True
            return False
        frame = (t, np.array(rgb, copy=True), np.array(depth_m, np.float32, copy=True),
                 None if conf is None else np.array(conf, np.uint8, copy=True), np.array(K, float), np.array(pose, float))
        try:
            self.q.put_nowait(frame)
        except queue.Full:
            self.dropped += 1
            return False
        self.t_last = t
        return True

    def _write_loop(self) -> None:
        rgbd = self.staging / "rgbd"
        try:
            import cv2
            try:
                from lzfse import compress
            except ImportError:
                from liblzfse import compress
            while (frame := self.q.get()) is not None:
                t, rgb, depth_m, conf, K, pose = frame
                i = self.written
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    raise RuntimeError(f"could not encode frame {i} as JPEG")
                (rgbd / f"{i}.jpg").write_bytes(jpg.tobytes())
                d = np.ascontiguousarray(np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0), np.float32)
                (rgbd / f"{i}.depth").write_bytes(compress(d.tobytes()))
                if conf is not None and conf.shape == d.shape:   # the loader reshapes it to the depth map
                    (rgbd / f"{i}.conf").write_bytes(compress(np.ascontiguousarray(conf, np.uint8).tobytes()))
                if self.K0 is None:
                    self.K0, self.rgb_wh, self.depth_wh = K, (rgb.shape[1], rgb.shape[0]), (d.shape[1], d.shape[0])
                x, y, z, qx, qy, qz, qw = (float(v) for v in pose)
                self.poses.append([qx, qy, qz, qw, x, y, z])
                self.coeffs.append([float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])])
                self.timestamps.append(t - self.t0)
                self.written += 1
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"

    def _stop_writer(self) -> None:
        self._closed = True
        while self._thread.is_alive():          # the writer may have died on an error with the queue full
            try:
                self.q.put(None, timeout=0.5)
                break
            except queue.Full:
                continue
        self._thread.join()

    def _metadata(self) -> dict:
        ts = self.timestamps
        w, h = self.rgb_wh
        dw, dh = self.depth_wh
        return {"frameTimestamps": ts, "K": [float(v) for v in self.K0.T.flatten()],   # column-major
                "w": w, "h": h, "dw": dw, "dh": dh, "cameraType": 1,
                "fps": round((len(ts) - 1) / max(1e-6, ts[-1] - ts[0])), "initPose": [0, 0, 0, 1, 0, 0, 0],
                "poses": self.poses, "perFrameIntrinsicCoeffs": self.coeffs, LIVE_MARKER: True}

    def finish(self) -> dict:
        """Write the rest and pack the session as ``final`` (.r3d): {"path", "frames", "dropped", "seconds"}."""
        import json
        import zipfile
        if self._closed:
            raise RuntimeError("this session is already finished")
        self._stop_writer()
        if self.error:
            self.abort()
            raise RuntimeError(f"could not write the session: {self.error}")
        if self.written < 10:
            self.abort()
            raise ValueError(f"only {self.written} frames recorded")
        self.final.parent.mkdir(parents=True, exist_ok=True)
        part = self.final.with_name(self.final.name + ".part")
        with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:   # JPEG and LZFSE are compressed already
            z.writestr("metadata", json.dumps(self._metadata()))
            for f in sorted((self.staging / "rgbd").iterdir()):
                z.write(f, f"rgbd/{f.name}")
        part.replace(self.final)
        shutil.rmtree(self.staging, ignore_errors=True)
        return {"path": str(self.final), "frames": self.written, "dropped": self.dropped,
                "seconds": round(self.elapsed, 1)}

    def abort(self) -> None:
        """Stop and delete what was written."""
        if not self._closed or self._thread.is_alive():
            self._stop_writer()
        shutil.rmtree(self.staging, ignore_errors=True)


def rgb_intrinsics(K: np.ndarray, rgb_w: int, depth_w: int) -> np.ndarray:
    """The pipeline needs intrinsics for the COLOR frame. If the optical center sits near the middle of the depth
    image rather than the color image, the matrix was for the depth image: scale it up."""
    if rgb_w != depth_w and abs(2 * K[0, 2] - depth_w) < abs(2 * K[0, 2] - rgb_w):
        K = K.copy()
        K[:2] *= rgb_w / depth_w
    return K


def stream(buffer, on_frame=None, device_index: int = 0, on_stopped=None):
    """Connect to the iPhone over USB with the ``record3d`` package and feed ``buffer`` (anything with
    ``add(t, rgb, depth_m, conf, K, pose)``, e.g. LiveSession); ``on_frame(t, rgb)`` is called for each frame,
    ``on_stopped()`` when the phone stops streaming. Returns the session (keep a reference).
    Record3D's callback only flags that a frame arrived; a thread then reads the NEWEST frame and handles it, so
    frames that come in while it is busy are skipped, not queued. (Handling every frame inside the callback fell
    behind on a real phone, 2026-10-11: ~22 of 60 fps handled, the backlog grew to 10+ s and a session recorded
    the seconds before the swings instead of the swings.)
    (An automatic swing trigger from frame-to-frame motion was tried on TestSet2 and caught only 4 of 7 swings:
    the hitter fills too little of the frame. LiveSession only uses motion to end a whole session after a long
    stillness, and records everything in between.)"""
    from record3d import Record3DStream
    devs = Record3DStream.get_connected_devices()
    if not devs:
        raise RuntimeError("no iPhone found: plug it in by USB, open Record3D and start USB streaming")
    s = Record3DStream()
    has_frame = threading.Event()
    is_running = threading.Event()
    is_running.set()

    def handle_newest():
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

    def consume():
        while is_running.is_set():
            if not has_frame.wait(timeout=0.5):
                continue
            has_frame.clear()                    # a frame arriving from here on is picked up next round
            try:
                handle_newest()
            except Exception as e:               # one bad frame must not end the stream
                print(f"live: frame skipped: {type(e).__name__}: {e}")

    def stopped():
        is_running.clear()
        has_frame.set()
        (on_stopped or (lambda: print("stream stopped")))()

    s.on_new_frame = has_frame.set
    s.on_stream_stopped = stopped
    threading.Thread(target=consume, daemon=True).start()
    if s.connect(devs[device_index]) is False:   # the phone refuses a new connection until its stream is restarted
        is_running.clear()
        has_frame.set()
        raise RuntimeError("the iPhone refused the connection: in Record3D, stop and restart USB streaming, "
                           "then try again")
    return s


LIVE_MARKER = "swinglabLive"     # metadata key in .r3d files written by R3dWriter
# Swing minimums for live recordings (session.find_swings min_peak / min_turn): the first real live test
# (2026-10-11, a half-speed indoor swing filmed head-on) peaked at 2.3 m/s hand speed and 77 deg shoulder turn,
# under the 3.5 m/s / 80 deg tuned on full-speed lab swings, so it was not counted. Uploads keep the strict ones.
LIVE_SWING_MINIMUMS = {"min_peak": 2.0, "min_turn": 60.0}


def is_live_recording(path: str | Path) -> bool:
    """True for a .r3d written by Record swing (R3dWriter marks its metadata)."""
    import json
    import zipfile
    path = Path(path)
    try:
        if path.is_file() and path.suffix.lower() == ".r3d":
            with zipfile.ZipFile(path) as z:
                return bool(json.loads(z.read("metadata")).get(LIVE_MARKER))
        meta = path / "metadata"
        return meta.exists() and bool(json.loads(meta.read_text(encoding="utf-8")).get(LIVE_MARKER))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


MOTION_W = 64          # motion is measured on frames this wide (grayscale)


def small_gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    h, w = rgb.shape[:2]
    small = cv2.resize(np.ascontiguousarray(rgb), (MOTION_W, max(1, round(h * MOTION_W / w))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def motion_score(prev: np.ndarray | None, cur: np.ndarray, pixel_threshold: int = 25) -> float:
    """Fraction of pixels (0-1) that changed by more than ``pixel_threshold`` gray levels between two small_gray
    frames. With the phone on a tripod: ~0 while the hitter stands still, a clear spike during a swing."""
    if prev is None or prev.shape != cur.shape:
        return 0.0
    return float(np.mean(np.abs(prev.astype(np.int16) - cur.astype(np.int16)) > pixel_threshold))


class LiveSession:
    """The phone stream and one recording session at a time (the server's Record swing). start() writes every frame
    to an R3dWriter; the session ends by itself once the hitter has been still for ``idle_s`` seconds (armed only
    after the first movement, so setting up in the box does not end it), at ``max_s`` seconds, or when the phone
    stops streaming; or by stop(), which keeps recording ``tail_s`` more seconds first: the phone's video reaches the
    laptop 2-3 s late over USB (measured 2026-10-11), so a swing just before Stop is still on its way. The .r3d
    lands in the hitter's drop folder and ``on_saved(info)`` is called. preview_jpeg() is the newest frame, small,
    for the browser (refreshed every PREVIEW_EVERY_S, as is motion)."""

    PREVIEW_H = 480
    PREVIEW_EVERY_S = 0.1
    STALE_S = 3.0          # no frame for this long: the phone stopped streaming

    def __init__(self, staging_root: Path, *, idle_s: float = 8.0, max_s: float = 90.0, tail_s: float = 3.0,
                 motion_threshold: float = 0.005, on_saved=None, log=print, connect=None):
        self.staging_root, self.idle_s, self.max_s, self.tail_s = Path(staging_root), idle_s, max_s, tail_s
        self.stop_at: float | None = None             # stop() was pressed: end the session at this time
        self._stop_reason_pending: str | None = None
        self.motion_threshold, self.on_saved, self.log = motion_threshold, on_saved, log
        self._connect = connect or stream
        self.lock = threading.Lock()
        self._stream = None
        self._connected_at = 0.0
        self._stream_id = 0
        self.is_streaming = False
        self.state = "idle"                 # idle | recording | saving | saved | error
        self.hitter: str | None = None
        self.writer: R3dWriter | None = None
        self.started_at = 0.0
        self.last_frame_t = 0.0
        self.fps = 0.0
        self.motion = 0.0
        self.has_moved = False
        self.last_motion_t = 0.0
        self.reason: str | None = None
        self.last_saved: dict | None = None
        self.error: str | None = None
        self._preview: bytes | None = None
        self._preview_t = 0.0
        self._prev_small: np.ndarray | None = None

    def connect(self) -> None:
        """Start the phone stream if it is not running (raises RuntimeError when no phone is plugged in)."""
        with self.lock:
            self._ensure_stream()

    def _ensure_stream(self) -> None:
        """Connect, or reconnect when the stream stopped or has sent nothing for STALE_S (a connection that
        stalled or was refused would otherwise count as running forever)."""
        is_fresh = time.time() - max(self.last_frame_t, self._connected_at) < self.STALE_S
        if self._stream is not None and self.is_streaming and is_fresh:
            return
        self._close_stream()
        self._stream_id += 1
        sid = self._stream_id               # a stop reported late by an old stream must not end the new one
        self._stream = self._connect(self, on_stopped=lambda: self._on_stream_stopped(sid))
        self._connected_at = time.time()
        self.is_streaming = True

    def _close_stream(self) -> None:
        s, self._stream, self.is_streaming = self._stream, None, False
        close = getattr(s, "disconnect", None)
        if close:
            try:
                close()
            except Exception as e:
                self.log(f"live: disconnect failed: {type(e).__name__}: {e}")

    def disconnect(self) -> None:
        """Release the phone (on server shutdown): it then accepts the next connection."""
        with self.lock:
            self._close_stream()

    def _on_stream_stopped(self, sid: int | None = None) -> None:
        if sid is not None and sid != self._stream_id:
            return
        self.is_streaming = False
        self.log("live: phone stream stopped")

    def add(self, t: float, rgb: np.ndarray, depth_m: np.ndarray, conf, K: np.ndarray, pose) -> None:
        """Stream thread: one frame."""
        dt = t - self.last_frame_t
        self.fps = self.fps * 0.9 + 0.1 / dt if 0 < dt < 1 else self.fps
        self.last_frame_t = t
        if t - self._preview_t >= self.PREVIEW_EVERY_S:
            self._preview_t = t
            self._update_preview(t, rgb)
        w = self.writer
        if self.state == "recording" and w is not None:
            w.add(t, rgb, depth_m, conf, K, pose)

    def _update_preview(self, t: float, rgb: np.ndarray) -> None:
        import cv2
        h, w = rgb.shape[:2]
        small = cv2.resize(np.ascontiguousarray(rgb), (round(w * self.PREVIEW_H / h), self.PREVIEW_H),
                           interpolation=cv2.INTER_AREA)
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self._preview = jpg.tobytes()
        gray = small_gray(small)
        self.motion = motion_score(self._prev_small, gray)
        self._prev_small = gray
        if self.state == "recording" and self.motion >= self.motion_threshold:
            self.has_moved, self.last_motion_t = True, t

    def preview_jpeg(self) -> bytes | None:
        return self._preview

    def start(self, hitter: str, drop_dir: Path) -> dict:
        """Start recording ``hitter`` into ``drop_dir`` (RuntimeError if a session is running or no phone)."""
        with self.lock:
            if self.state in ("recording", "saving"):
                raise RuntimeError(f"a session is already {self.state}")
            self._ensure_stream()
            name = time.strftime("%Y-%m-%d--%H-%M-%S")       # Record3D's naming: the start time is read from it
            self.writer = R3dWriter(self.staging_root / name, Path(drop_dir) / f"{name}.r3d", max_s=self.max_s)
            self.hitter, self.started_at = hitter, time.time()
            self.has_moved, self.last_motion_t, self.reason, self.error = False, 0.0, None, None
            self.stop_at, self._stop_reason_pending = None, None
            self.state = "recording"
        threading.Thread(target=self._watch, daemon=True).start()
        self.log(f"live: recording {hitter} -> {self.writer.final}")
        return self.status()

    def _stop_reason(self) -> str | None:
        w, now = self.writer, time.time()
        if w is None:
            return None
        if w.error:
            return f"write error: {w.error}"
        if w.is_full:
            return f"{self.max_s:.0f} s limit"
        if not self.is_streaming or now - max(self.last_frame_t, self.started_at) > self.STALE_S:
            return "phone stream stopped"
        if self.stop_at is not None and now >= self.stop_at:
            return self._stop_reason_pending
        if self.has_moved and now - self.last_motion_t >= self.idle_s:
            return f"still for {self.idle_s:.0f} s"
        return None

    def _watch(self) -> None:
        while self.state == "recording":
            time.sleep(0.25)
            reason = self._stop_reason()
            if reason:
                self._end(reason)

    def stop(self, reason: str = "stopped by user") -> dict:
        """End the session after ``tail_s`` more seconds of recording (the video arrives late); then it is packed
        in a background thread (state "saving", then "saved" or "error")."""
        with self.lock:
            if self.state != "recording" or self.stop_at is not None:
                return self.status()
            if self.tail_s > 0:
                self.stop_at, self._stop_reason_pending = time.time() + self.tail_s, reason
                return self.status()
        return self._end(reason)

    def _end(self, reason: str) -> dict:
        with self.lock:
            if self.state != "recording":
                return self.status()
            self.state, self.reason = "saving", reason
            w = self.writer
        threading.Thread(target=self._save, args=(w, reason), daemon=True).start()
        return self.status()

    def _save(self, w: R3dWriter, reason: str) -> None:
        try:
            info = {**w.finish(), "hitter": self.hitter, "reason": reason, "saved_at": time.time()}
        except Exception as e:
            with self.lock:
                self.state, self.error = "error", f"{type(e).__name__}: {e}"
            self.log(f"live: session for {self.hitter} not saved ({reason}): {self.error}")
            return
        with self.lock:
            self.state, self.last_saved = "saved", info
        self.log(f"live: session ended ({reason}): {self.hitter}, {info['seconds']} s, {info['frames']} frames"
                 + (f", {info['dropped']} dropped" if info["dropped"] else "") + f" -> {info['path']}")
        if self.on_saved:
            try:
                self.on_saved(info)
            except Exception as e:
                self.log(f"live: on_saved failed: {type(e).__name__}: {e}")

    def status(self) -> dict:
        w, now = self.writer, time.time()
        is_live = self.state == "recording" and w is not None
        idle_left = max(0.0, self.idle_s - (now - self.last_motion_t)) if is_live and self.has_moved else None
        is_finishing = is_live and self.stop_at is not None
        return {"state": self.state, "hitter": self.hitter, "finishing": is_finishing,
                "tail_left_s": round(max(0.0, self.stop_at - now), 1) if is_finishing else None,
                "connected": self.is_streaming and now - self.last_frame_t < self.STALE_S,
                "elapsed_s": round(w.elapsed, 1) if w else None, "frames": w.written if w else None,
                "dropped": w.dropped if w else None, "fps": round(self.fps, 1), "motion": round(self.motion, 4),
                "motion_threshold": self.motion_threshold,
                "has_moved": self.has_moved, "idle_s": self.idle_s,
                "idle_left_s": None if idle_left is None else round(idle_left, 1), "max_s": self.max_s,
                "reason": self.reason, "saved": self.last_saved, "error": self.error}
