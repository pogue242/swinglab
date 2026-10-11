"""Synthetic checks of the geometry: known motion in, known numbers out."""
import numpy as np

from swinglab import features, kinematics as K, skeleton


def synthetic_swing(fps=30.0, bats="R", T=2.0, hip_rate=600.0, smooth=False):
    """Hitter whose hips rotate at `hip_rate` deg/s between 1.0 and 1.2 s.
    smooth=True: same total turn with a bell-shaped speed curve (one clear peak), like a real swing."""
    t = np.arange(0, T, 1 / fps)
    n = len(t)
    if smooth:
        u = np.clip((t - 1.0) / 0.3, 0, 1)
        yaw = np.radians(hip_rate * 0.2 * (3 * u**2 - 2 * u**3))
    else:
        yaw = np.radians(np.clip((t - 1.0) * hip_rate, 0, hip_rate * 0.2))
    return t, body_from_yaw(t, yaw, bats)


def body_from_yaw(t, yaw, bats="R"):
    """Synthetic hitter whose hips, shoulders and arms rotate by `yaw` (radians, per frame)."""
    n = len(t)
    def ring(r, z, ang0):
        a = ang0 + yaw
        return np.c_[r * np.cos(a), r * np.sin(a), np.full(n, z)]
    # lab frame: +x toward pitcher; right-handed hitter's lead (left) side toward +x
    J = {
        "l_hip": ring(0.15, 0.9, 0.0), "r_hip": ring(0.15, 0.9, np.pi),
        "l_shoulder": ring(0.2, 1.4, 0.0), "r_shoulder": ring(0.2, 1.4, np.pi),
        "l_knee": np.c_[np.full(n, 0.2), np.zeros(n), np.full(n, 0.5)],
        "r_knee": np.c_[np.full(n, -0.2), np.zeros(n), np.full(n, 0.5)],
        "l_ankle": np.c_[np.full(n, 0.3), np.zeros(n), np.full(n, 0.08)],
        "r_ankle": np.c_[np.full(n, -0.3), np.zeros(n), np.full(n, 0.08)],
        "l_elbow": ring(0.3, 1.2, 0.0), "r_elbow": ring(0.3, 1.2, np.pi),
        "l_wrist": ring(0.35, 1.1, -0.3), "r_wrist": ring(0.3, 1.1, -0.4),
    }
    if bats == "L":  # mirror the whole hitter across the x-z plane and swap sides
        J = {("r" if k[0] == "l" else "l") + k[1:]: v * np.array([1, -1, 1]) for k, v in J.items()}
    return J


def test_joint_angle_straight_and_right():
    a, b = np.array([[0, 0, 1.0]]), np.array([[0, 0, 0.0]])
    assert np.isclose(K.joint_angle_deg(a, b, np.array([[0, 0, -1.0]]))[0], 180)
    assert np.isclose(K.joint_angle_deg(a, b, np.array([[1.0, 0, 0]]))[0], 90)


def test_pelvis_rotation_speed_recovered():
    t, J = synthetic_swing(fps=120, hip_rate=600)
    sk = skeleton.build(t, J, "R")
    s = K.series(sk)
    assert abs(np.nanmax(s["pelvis_vel"]) - 600) < 60  # within 10% after smoothing
    assert s["pelvis_yaw"][-1] - s["pelvis_yaw"][0] > 100  # opening = positive


def test_lefty_mirrors_to_righty():
    t, JR = synthetic_swing(bats="R")
    _, JL = synthetic_swing(bats="L")
    sR, sL = K.series(skeleton.build(t, JR, "R")), K.series(skeleton.build(t, JL, "L"))
    for k in ("pelvis_yaw", "torso_yaw", "separation", "lead_knee_flex"):
        assert np.allclose(sR[k], sL[k], atol=1e-6), k


def test_resample_rate():
    t, J = synthetic_swing(fps=360)
    sk = skeleton.build(t, J, "R", fps=30)
    assert abs(sk.fps - 30) < 1e-6


def test_features_run_on_synthetic():
    t, J = synthetic_swing(fps=30)
    sk = skeleton.build(t, J, "R", height_m=1.8)
    f = features.compute(sk)
    assert set(features.FEATURES) <= set(f)
    assert abs(f["stance_width"] - 0.6 / 1.8) < 0.02


def test_world_frame_with_y_up_gives_same_features():
    """Phone joints arrive in ARKit world coords (Y up). Rotating the lab swing into
    that frame (and an arbitrary heading) must not change any feature."""
    t, J = synthetic_swing(fps=30)
    base = features.compute(skeleton.build(t, J, "R", height_m=1.8))
    th = np.radians(37)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]]) @ \
        np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])  # z-up -> y-up, then yaw 37°
    Jw = {k: v @ R.T + np.array([2.0, -1.0, 5.0]) for k, v in J.items()}
    rot = features.compute(skeleton.build(t, Jw, "R", up=np.array([0, 1.0, 0]), height_m=1.8))
    for k in base:
        assert np.isclose(base[k], rot[k], atol=1e-6, equal_nan=True), k


def test_peak_offset_subframe():
    x = np.arange(10.0)
    y = -(x - 4.3) ** 2
    i = int(np.argmax(y))
    assert abs(i + K.peak_offset(y, i) - 4.3) < 1e-9
