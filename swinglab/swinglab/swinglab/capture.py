"""Phone capture: load a LiDAR recording and lift 2D keypoints to 3D (goal 1, phone side).

Supported format: **Stray Scanner** (free iOS app, MIT-licensed,
github.com/strayrobots/scanner). One recording is a folder with:
  rgb.mp4              HEVC video
  depth/000000.png     16-bit depth in millimeters, 256x192, one per frame
  confidence/*.png     0/1/2 per depth pixel (2 = most confident)
  odometry.csv         timestamp, frame, x,y,z, qx,qy,qz,qw, fx,fy,cx,cy (per frame)
  imu.csv              accelerometer + gyroscope
Record3D exports can be converted to the same in-memory ``Capture``.

Not yet verified on a real recording (see HANDOFF "Verification status").
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Capture:
    root: Path
    video: Path
    timestamps: np.ndarray        # (N,) seconds
    K_rgb: np.ndarray             # (N, 3, 3) intrinsics for the RGB frame size
    cam_pos: np.ndarray           # (N, 3) camera position, meters (session frame)
    cam_quat: np.ndarray          # (N, 4) qx, qy, qz, qw
    depth_dir: Path
    conf_dir: Path | None
    rgb_size: tuple[int, int] | None = None   # (w, h), filled by video probe
    kind: str = "stray"                       # "stray" | "record3d"
    depth_reader: object = None               # callable(i) -> depth in meters (Record3D)

    @property
    def n(self) -> int:
        return len(self.timestamps)

    @property
    def fps(self) -> float:
        return 1.0 / float(np.median(np.diff(self.timestamps)))

    def depth(self, i: int) -> np.ndarray:
        """Depth in meters (H, W); 0/NaN = no reading."""
        import cv2
        if self.depth_reader is not None:
            return self.depth_reader(i)
        d = cv2.imread(str(self.depth_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(self.depth_dir / f"{i:06d}.png")
        d = d.astype(np.float32) / 1000.0
        if self.conf_dir is not None:
            c = cv2.imread(str(self.conf_dir / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            if c is not None:
                d[c < 1] = np.nan
        d[d <= 0] = np.nan
        return d


def load_stray(root: str | Path) -> Capture:
    root = Path(root)
    od = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    od.columns = [c.strip() for c in od.columns]
    K = np.zeros((len(od), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = od.fx, od.fy, od.cx, od.cy, 1
    conf = root / "confidence"
    return Capture(root, root / "rgb.mp4", od.timestamp.to_numpy(float), K,
                   od[["x", "y", "z"]].to_numpy(float), od[["qx", "qy", "qz", "qw"]].to_numpy(float),
                   root / "depth", conf if conf.exists() else None)


def _lzfse_decompress(b: bytes) -> bytes:
    try:
        import lzfse
        return lzfse.decompress(b)
    except ImportError:
        import liblzfse  # pip install pyliblzfse (Windows wheel exists for Python 3.12)
        return liblzfse.decompress(b)


def _r3d_shape(n_values: int, rgb_wh: tuple[int, int]) -> tuple[int, int]:
    """Depth map shape (h, w) with the same orientation as the RGB frames."""
    for h, w in ((192, 256), (256, 192), (384, 512), (512, 384), (480, 640), (640, 480)):
        if h * w == n_values and (w >= h) == (rgb_wh[0] >= rgb_wh[1]):
            return h, w
    side = int(round(np.sqrt(n_values * rgb_wh[1] / rgb_wh[0])))
    return side, n_values // side


def load_record3d(path: str | Path) -> Capture:
    """Record3D export: a ``.r3d`` file (zip) or its extracted folder.

    Contents: ``metadata`` (JSON: ``K`` flattened column-major for the RGB size,
    ``w``/``h`` RGB size, ``dw``/``dh`` depth size, ``fps``, ``poses`` = one
    ``[qx, qy, qz, qw, tx, ty, tz]`` camera-to-world pose per frame in ARKit world
    coordinates, ``perFrameIntrinsicCoeffs`` = ``[fx, fy, cx, cy]`` per frame,
    ``frameTimestamps``), ``rgbd/<i>.jpg`` color frames, ``rgbd/<i>.depth``
    LZFSE-compressed float32 depth in meters (NaN = no reading), ``rgbd/<i>.conf``
    LZFSE-compressed uint8 confidence (0–2), ``sound.m4a``.

    Checked on two real iPhone 17 Pro exports (Record3D, 2026-10-09, portrait):
    color 1440x1920 and depth 192x256 share one orientation (not mirrored, the
    person lines up), K matches the portrait JPG (cx ≈ w/2, cy ≈ h/2), and the
    poses are in the same portrait camera frame: back-projected floor points come
    out with a normal along ARKit world +Y (0.98–0.998). fps 60, timestamps even.
    """
    import json
    import zipfile
    path = Path(path)
    if path.is_file():
        root = path.with_name(path.stem + "_r3d")
        if not (root / "metadata").exists():
            with zipfile.ZipFile(path) as z:
                z.extractall(root)
    else:
        root = path
    meta = json.loads((root / "metadata").read_text(encoding="utf-8"))
    w, h = int(meta["w"]), int(meta["h"])
    K = np.array(meta["K"], float).reshape(3, 3).T
    poses = np.array(meta["poses"], float)
    n = len(poses)
    Ks = np.repeat(K[None], n, 0)
    coeffs = np.array(meta.get("perFrameIntrinsicCoeffs") or [], float)
    if coeffs.shape == (n, 4):   # [fx, fy, cx, cy] per frame (autofocus changes fx by ~0.3%)
        Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2] = coeffs.T
    dshape = (int(meta["dh"]), int(meta["dw"])) if meta.get("dw") and meta.get("dh") else None
    fps = float(meta.get("fps", 60))
    ts = np.array(meta["frameTimestamps"], float)[:n] if meta.get("frameTimestamps") else np.arange(n) / fps
    rgbd = root / "rgbd"

    def depth(i: int) -> np.ndarray:
        d = np.frombuffer(_lzfse_decompress((rgbd / f"{i}.depth").read_bytes()), np.float32).copy()
        d = d.reshape(dshape if dshape and dshape[0] * dshape[1] == d.size else _r3d_shape(d.size, (w, h)))
        cf = rgbd / f"{i}.conf"
        if cf.exists():
            c = np.frombuffer(_lzfse_decompress(cf.read_bytes()), np.uint8).reshape(d.shape)
            d[c < 1] = np.nan
        d[~(d > 0)] = np.nan
        return d

    return Capture(root, rgbd / "%d.jpg", ts - ts[0], Ks, poses[:, 4:7], poses[:, 0:4],
                   rgbd, None, (w, h), "record3d", depth)


def load_any(path: str | Path) -> Capture:
    """Stray Scanner folder, Record3D .r3d file, or extracted Record3D folder."""
    path = Path(path)
    if path.is_dir() and (path / "odometry.csv").exists():
        return load_stray(path)
    if path.suffix.lower() == ".r3d" or (path.is_dir() and (path / "metadata").exists()):
        return load_record3d(path)
    raise ValueError(f"not a Stray Scanner folder or Record3D export: {path}")


def is_recording(path: Path) -> bool:
    return (path.is_dir() and ((path / "odometry.csv").exists() or ((path / "metadata").exists() and (path / "rgbd").exists()))) \
        or (path.is_file() and path.suffix.lower() == ".r3d")


def sample_depth(depth: np.ndarray, uv_rgb: np.ndarray, rgb_size: tuple[int, int], win: int = 2) -> np.ndarray:
    """Median depth (m) in a small window around each RGB pixel (u, v), robust to edge bleed.

    Uses the nearer half of the window's values, because a joint near the body
    outline can see background depth behind it.
    """
    h, w = depth.shape
    sx, sy = w / rgb_size[0], h / rgb_size[1]
    out = np.full(len(uv_rgb), np.nan)
    for k, (u, v) in enumerate(uv_rgb):
        if not np.isfinite(u):
            continue
        x, y = int(round(u * sx)), int(round(v * sy))
        patch = depth[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1].ravel()
        patch = np.sort(patch[np.isfinite(patch)])
        if len(patch):
            out[k] = float(np.median(patch[: max(1, len(patch) // 2 + 1)]))
    return out


def backproject(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N,2) + depth (N,) -> camera-frame points (N,3), meters. X right, Y down, Z forward."""
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.c_[x, y, z]


def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# Bone length ÷ height in the Driveline lab skeletons (median of 226 swings; lead/rear averaged).
# Used when the recording's own depth can't measure a bone consistently.
LIMBS = [("shoulder", "elbow", 0.180), ("elbow", "wrist", 0.146), ("hip", "knee", 0.241), ("knee", "ankle", 0.248)]


def limb_lengths(P: dict[str, np.ndarray], height_m: float | None, *, min_frames: int = 15,
                 max_spread: float = 0.10) -> dict[tuple[str, str], tuple[float, str]]:
    """Length (m) and source ("depth" | "lab") for each parent->child bone, both sides.
    The recording's own median when depth measured the bone in >= min_frames frames with
    IQR/median < max_spread; otherwise the lab ratio x height (None if height unknown)."""
    out = {}
    for side in ("l", "r"):
        for a, b, ratio in LIMBS:
            pa, pb = P.get(f"{side}_{a}"), P.get(f"{side}_{b}")
            L = np.linalg.norm(pa - pb, axis=1) if pa is not None and pb is not None else np.array([])
            L = L[np.isfinite(L)]
            if len(L) >= min_frames and (np.subtract(*np.percentile(L, [75, 25])) / np.median(L)) < max_spread:
                out[(f"{side}_{a}", f"{side}_{b}")] = (float(np.median(L)), "depth")
            elif height_m:
                out[(f"{side}_{a}", f"{side}_{b}")] = (ratio * float(height_m), "lab")
    return out


def place_on_ray(parent: np.ndarray, ray: np.ndarray, L: float, hint: np.ndarray | None) -> np.ndarray:
    """Point on the camera ray s*ray (s > 0) at distance L from ``parent`` (camera frame).
    Two solutions in general (in front of / behind the parent): take the one nearer ``hint``
    (measured depth or the previous frame), else the one nearer the camera. If the ray
    misses the sphere (2D error), take the ray point closest to the parent."""
    b = float(ray @ parent)
    disc = b * b - float(parent @ parent) + L * L
    if disc < 0:
        return max(b, 1e-3) * ray
    roots = [s for s in (b - np.sqrt(disc), b + np.sqrt(disc)) if s > 0] or [max(b, 1e-3)]
    if hint is not None and np.all(np.isfinite(hint)):
        return min((s * ray for s in roots), key=lambda p: float(np.linalg.norm(p - hint)))
    return roots[0] * ray


def lift_keypoints(cap: Capture, kps_uv: dict[int, dict[str, tuple[float, float, float]]],
                   rgb_size: tuple[int, int], world: bool = True, min_vis: float = 0.5,
                   height_m: float | None = None, bone_fallback: bool = True, fallback_min_vis: float = 0.0,
                   tol: float = 0.15, info: dict | None = None) -> dict[str, np.ndarray]:
    """2D keypoints per frame -> 3D joints (T, 3) per name.

    ``kps_uv[frame][name] = (u, v, visibility)``. ARKit camera convention is
    X right, Y up, Z backward; we flip our Y/Z to match before applying the
    camera pose so joints land in the session (world) frame, which is fixed
    even if the phone moves. Gravity is +Y up in ARKit world coordinates.

    Bone-length fallback (``bone_fallback``): in the first real recordings the
    moving arms often had NO depth (dark sleeves at night, motion), or depth from
    the background. A joint's pixel still fixes its camera ray, so an elbow, wrist,
    knee or ankle whose depth is missing or makes its bone > ``tol`` off the bone
    length (see ``limb_lengths``) is placed on its ray at that length from its parent,
    if its 2D visibility is at least ``fallback_min_vis`` (default 0: MediaPipe still guesses a hidden
    joint, e.g. the rear arm behind the body when filmed from the chest side; such joints are flagged
    through the placed fraction). ``info`` receives the
    fraction of frames each joint was placed this way and the bone lengths used.
    """
    names = sorted({n for f in kps_uv.values() for n in f})
    P = {n: np.full((cap.n, 3), np.nan) for n in names}          # camera frame (OpenCV axes)
    UV = {n: np.full((cap.n, 2), np.nan) for n in names}
    VIS = {n: np.zeros(cap.n) for n in names}
    for i in range(cap.n):
        if i not in kps_uv:
            continue
        d = cap.depth(i)
        f = kps_uv[i]
        for n in names:
            if n in f:
                UV[n][i], VIS[n][i] = f[n][:2], f[n][2]
        uv = np.array([UV[n][i] if VIS[n][i] >= min_vis else [np.nan, np.nan] for n in names])
        Pi = backproject(uv, sample_depth(d, uv, rgb_size), cap.K_rgb[i])
        for k, n in enumerate(names):
            P[n][i] = Pi[k]
    placed = {}
    lengths = limb_lengths(P, height_m) if bone_fallback else {}
    for (pa, ch), (L, _src) in lengths.items():   # LIMBS order: parents are fixed before their children
        if ch not in P:
            continue
        moved = np.zeros(cap.n, bool)
        prev = prev2 = None   # last two known positions -> constant-velocity guess for this frame

        def known(p):
            nonlocal prev, prev2
            if np.all(np.isfinite(p)):
                prev, prev2 = p, prev
        for i in range(cap.n):
            parent, meas = P[pa][i], P[ch][i]
            if not np.all(np.isfinite(parent)) or not (VIS[ch][i] >= fallback_min_vis and np.all(np.isfinite(UV[ch][i]))):
                known(meas)
                continue
            if np.all(np.isfinite(meas)) and abs(np.linalg.norm(meas - parent) - L) <= tol * L:
                known(meas)
                continue
            K = cap.K_rgb[i]
            ray = np.array([(UV[ch][i][0] - K[0, 2]) / K[0, 0], (UV[ch][i][1] - K[1, 2]) / K[1, 1], 1.0])
            ray /= np.linalg.norm(ray)
            guess = prev if prev2 is None else 2 * prev - prev2 if prev is not None else None
            hint = guess if guess is not None else (meas if np.all(np.isfinite(meas)) else None)   # bad depth is a poor guide
            P[ch][i] = place_on_ray(parent, ray, L, hint)
            known(P[ch][i])
            moved[i] = True
        placed[ch] = round(float(moved.mean()), 3)
    if info is not None:
        info["placed_by_bone_length"] = placed
        info["bone_lengths_m"] = {f"{a}-{b}": [round(L, 3), src] for (a, b), (L, src) in lengths.items()}
    if not world:
        return P
    flip = np.diag([1.0, -1.0, -1.0])
    out = {}
    for n in names:
        Wn = np.full((cap.n, 3), np.nan)
        for i in range(cap.n):
            if np.all(np.isfinite(P[n][i])):
                Wn[i] = quat_to_R(cap.cam_quat[i]) @ (flip @ P[n][i]) + cap.cam_pos[i]
        out[n] = Wn
    return out


def clean_joints(joints: dict[str, np.ndarray], t: np.ndarray, *, jump_m: float = 0.12,
                 window: int = 5, max_gap_s: float = 0.15) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Remove single-frame depth errors and fill short gaps.

    When one body part passes in front of another (hands across the hips during
    the swing), the depth under the hidden joint belongs to the occluder for a
    frame or two. Those samples sit far from the joint's rolling median; they are
    set to NaN and, with other gaps up to ``max_gap_s``, linearly interpolated.
    Returns cleaned joints and the fraction of samples replaced per joint.
    """
    from scipy.ndimage import median_filter
    out, replaced = {}, {}
    for n, P in joints.items():
        P = P.copy()
        ok = np.isfinite(P[:, 0])
        if ok.sum() >= window:
            med = np.full_like(P, np.nan)
            idx = np.where(ok)[0]
            med[idx] = median_filter(P[idx], size=(window, 1), mode="nearest")
            bad = ok & (np.linalg.norm(P - med, axis=1) > jump_m)
            P[bad] = np.nan
        ok = np.isfinite(P[:, 0])
        replaced[n] = float(1 - ok.mean())
        if ok.sum() >= 2:
            for a in range(3):
                filled = np.interp(t, t[ok], P[ok, a])
                # only fill gaps shorter than max_gap_s
                gap_ok = np.ones(len(t), bool)
                missing = np.where(~ok)[0]
                for i in missing:
                    prev = t[ok][t[ok] < t[i]]
                    nxt = t[ok][t[ok] > t[i]]
                    gap_ok[i] = len(prev) and len(nxt) and (nxt[0] - prev[-1]) <= max_gap_s
                P[:, a] = np.where(ok | gap_ok, filled, np.nan)
        out[n] = P
    return out, replaced


TORSO_JOINTS = ("l_shoulder", "r_shoulder", "l_hip", "r_hip")
MIN_ANCHOR_FRAMES = 0.3     # lift_world needs LiDAR torso positions in at least this share of the posed frames


def lift_world(cap: Capture, world: dict[int, dict[str, tuple[float, float, float]]],
               depth_joints: dict[str, np.ndarray], *, smooth_s: float = 0.25,
               info: dict | None = None) -> dict[str, np.ndarray] | None:
    """MediaPipe's metric 3D landmarks, placed in the session (world) frame by the LiDAR.

    Why: the camera usually faces the hitter's chest, so in the swing the shoulders and hips turn
    toward / away from the camera and one of each pair hides behind the body. The depth under a
    hidden joint belongs to the near side, so depth-only joints flip the shoulder and hip lines by
    ~180° (TestSet2: impossible 198° separations, "hip turn" of -22°, 2 of 9 swings missed).
    MediaPipe's 3D pose keeps the body's shape through the turn, but it is centered on the hips and
    has no position in the room. So: body shape = MediaPipe 3D, rotated into the world by the camera
    pose; scale = LiDAR torso length / MediaPipe torso length; position = the median over the torso
    joints of (LiDAR position - MediaPipe position), smoothed over ``smooth_s`` (the torso is the
    best-measured part). Returns None when there are too few LiDAR torso positions to anchor it."""
    from scipy.ndimage import median_filter
    if not world:
        return None
    from .skeleton import MEDIAPIPE
    names = list(MEDIAPIPE.values())
    flip = np.diag([1.0, -1.0, -1.0])
    rel = {n: np.full((cap.n, 3), np.nan) for n in names}
    for i, f in world.items():
        R = quat_to_R(cap.cam_quat[i]) @ flip
        for n in names:
            if n in f:
                rel[n][i] = R @ np.asarray(f[n], float)

    def torso_len(P):
        return np.linalg.norm(0.5 * (P["l_shoulder"] + P["r_shoulder"]) - 0.5 * (P["l_hip"] + P["r_hip"]), axis=1)
    if any(n not in depth_joints for n in TORSO_JOINTS):
        return None
    ld, lm = torso_len(depth_joints), torso_len(rel)
    both = np.isfinite(ld) & np.isfinite(lm)
    if both.sum() < max(5, MIN_ANCHOR_FRAMES * len(world)):
        return None
    scale = float(np.median(ld[both]) / np.median(lm[both]))
    for n in names:
        rel[n] *= scale
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # frames with no torso position: all-NaN
        anchor = np.nanmedian(np.stack([depth_joints[n] - rel[n] for n in TORSO_JOINTS]), axis=0)
    ok = np.isfinite(anchor[:, 0])
    idx = np.arange(cap.n)
    anchor = np.c_[[np.interp(idx, idx[ok], anchor[ok, a]) for a in range(3)]].T
    anchor = median_filter(anchor, size=(int(smooth_s * cap.fps) | 1, 1), mode="nearest")
    from .kinematics import smooth
    anchor = smooth(anchor, cap.fps, smooth_s)
    if info is not None:
        info["lift"] = "mediapipe_3d_with_lidar_anchor"
        info["mediapipe_scale"] = round(scale, 3)
        info["anchor_frames"] = round(float(ok[list(world)].mean()), 3)
    return {n: np.where(np.isfinite(rel[n][:, :1]), rel[n] + anchor, np.nan) for n in names}


WORLD_UP_ARKIT = np.array([0.0, 1.0, 0.0])  # pass as ``up`` to skeleton.build for world-frame joints


def iter_frames(video: str | Path):
    """BGR frames from a video file or an image sequence pattern (``rgbd/%d.jpg``, numbered from 0)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")


def mediapipe_keypoints(video: str | Path, model_path: str | Path, every: int = 1, fps: float | None = None,
                        world: dict | None = None):
    """MediaPipe Pose Landmarker (Tasks API, mediapipe>=0.10.30) on a video or an
    image sequence such as Record3D's ``rgbd/%d.jpg``.

    Returns ({frame: {name: (u, v, visibility)}}, (w, h)). Download the model
    once with ``python scripts/get_pose_model.py`` (POSE_MODEL_URL).
    The legacy ``mp.solutions.pose`` API no longer exists in current releases.
    ``fps``: the true capture rate. Pass it for image sequences: OpenCV reports
    25 fps for any ``%d.jpg`` sequence, which skews MediaPipe's time-based tracking.
    ``world``: a dict that receives MediaPipe's own metric 3D landmarks, {frame: {name: (x, y, z)}}
    (camera-aligned axes, x right, y down, z away from the camera, origin between the hips): see lift_world.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    from .skeleton import MEDIAPIPE
    opts = vision.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
                                        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    fps = fps or cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 60.0
    out, size = {}, None
    with vision.PoseLandmarker.create_from_options(opts) as lm_model:
        for i, frame in enumerate(iter_frames(video)):
            size = (frame.shape[1], frame.shape[0])
            if i % every == 0:
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                r = lm_model.detect_for_video(img, int(i * 1000 / fps))
                if r.pose_landmarks:
                    lm = r.pose_landmarks[0]
                    out[i] = {name: (lm[k].x * size[0], lm[k].y * size[1], lm[k].visibility or 1.0)
                              for k, name in MEDIAPIPE.items()}
                    if world is not None and r.pose_world_landmarks:
                        wl = r.pose_world_landmarks[0]
                        world[i] = {name: (wl[k].x, wl[k].y, wl[k].z) for k, name in MEDIAPIPE.items()}
    return out, size


# ---------------------------------------------------------------- wall-clock times of recordings
#
# Used to match per-swing slow-motion clips to the swings of one long LiDAR recording by when the
# ball left the tee. Both phones set their clocks from the network, so their times agree to ~1 s.

def recording_start_time(path: str | Path) -> float | None:
    """Wall-clock start (epoch seconds, local time) of a LiDAR recording, if it can be known:
    Record3D names exports by their start time ("2026-10-09--22-17-12.r3d"). Stray Scanner
    folders carry no wall-clock time (odometry is time since boot): None."""
    import re
    import time as _time
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})--(\d{2})-(\d{2})-(\d{2})", Path(path).name)
    if not m:
        return None
    y, mo, d, h, mi, s = map(int, m.groups())
    return _time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


def video_start_time(path: str | Path) -> float | None:
    """Wall-clock start (epoch seconds) of a video from its container metadata (QuickTime
    ``creation_time``, which iPhones write in UTC), read with the ffmpeg that imageio-ffmpeg ships.
    None if unavailable (e.g. files written by OpenCV, or messaging apps that strip metadata)."""
    import re
    import subprocess
    from datetime import datetime
    try:
        import imageio_ffmpeg
        r = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
    except Exception:
        return None
    m = re.search(r"com\.apple\.quicktime\.creationdate\s*:\s*(\S+)", r.stderr) or re.search(r"creation_time\s*:\s*(\S+)", r.stderr)
    if not m:
        return None
    s = m.group(1).replace("Z", "+00:00")
    s = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s)            # -0400 -> -04:00
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None
