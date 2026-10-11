"""Synthetic 240 fps tee shot: known speed and angle in, recovered out."""
import numpy as np

from swinglab import ballflight as B


def synth(v=30.0, ang=20.0, fps=240.0, m_per_px=0.004, rest=20, n=40, noise_px=0.7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    vx, vy = v * np.cos(np.radians(ang)), v * np.sin(np.radians(ang))
    x = vx * t
    y = vy * t - 0.5 * B.G * t**2
    start = np.array([400.0, 500.0])
    flight = np.c_[start[0] + x / m_per_px, start[1] - y / m_per_px]
    xy = np.vstack([np.tile(start, (rest, 1)), flight[1:]])
    return xy + rng.normal(0, noise_px, xy.shape), fps, m_per_px


def test_launch_recovers_speed_and_angle():
    xy, fps, s = synth()
    l = B.launch(xy, fps, s)
    assert l is not None
    assert abs(l.exit_speed_mps - 30.0) / 30.0 < 0.05
    assert abs(l.launch_angle_deg - 20.0) < 2.5
    assert l.direction == 1


def test_contact_frame():
    xy, *_ = synth(rest=20, noise_px=0.3)
    assert abs(B.contact_frame(xy) - 19) <= 1


def test_trajectory_starts_at_ball_and_rises():
    xy, fps, s = synth()
    l = B.launch(xy, fps, s)
    path = B.trajectory(l, (400.0, 500.0), s, t_max=0.5)
    assert np.allclose(path[0], [400, 500])
    assert path[20, 1] < 500  # image y goes up (smaller) as the ball rises


def test_detect_ball_on_synthetic_frame():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (300, 200), 10, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (60, 300), (255, 255, 255), -1)  # a bat-like bright bar
    x, y, r = B.detect_ball(img)
    assert abs(x - 300) < 2 and abs(y - 200) < 2 and 8 < r < 12


def test_scale_from_ball():
    assert np.isclose(B.scale_from_ball(10, 0.0737), 0.0737 / 20)
