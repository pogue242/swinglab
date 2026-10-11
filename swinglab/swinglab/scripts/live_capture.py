"""Live demo capture: iPhone (Record3D, USB streaming) -> the hitter's drop folder, one recording per swing.

  python scripts\\live_capture.py --hitter Matteo            # press Enter right after each swing

Run the server in demo mode next to it (python api\\server.py --demo): each saved swing shows up in the viewer
with its scores in about 20 s, and the ball check, AI report and video follow. See swinglab/live.py for what it
needs (Record3D USB streaming, the phone on USB, pip install record3d).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swinglab import config, live, profiles  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hitter", required=True, help="profile name (its drop folder receives the recordings)")
ap.add_argument("--before", type=float, default=2.5, help="seconds kept before the swing")
ap.add_argument("--after", type=float, default=1.0, help="seconds kept after the swing")
a = ap.parse_args()

prof = profiles.by_name(a.hitter) or profiles.get(a.hitter)
if not prof:
    sys.exit(f"no profile '{a.hitter}' (create it in the viewer first)")
drop = profiles.drop_dir(prof["name"], config.USERDATA)
drop.mkdir(parents=True, exist_ok=True)
buf = live.LiveBuffer(seconds=max(8.0, a.before + a.after + 2))
lock = threading.Lock()


def save(peak_t: float | None = None):
    """Save the window around the swing (Enter right after it: the swing is taken to have peaked `after` s ago)."""
    with lock:
        now = buf.latest_t()
        if now is None:
            print("no frames yet")
            return
        end = (peak_t + a.after) if peak_t else now
        if end > now:                          # wait for the follow-through to arrive
            time.sleep(end - now)
        start = end - a.before - a.after
        frames = buf.window(start, end)
        name = time.strftime("live %Y-%m-%d--%H-%M-%S")
        try:
            out = live.save_stray(frames, drop / name)
            print(f"saved {len(frames)} frames -> {out}  (the server picks it up in a few seconds)")
        except Exception as e:
            print(f"could not save: {e}")


session = live.stream(buf)
print(f"streaming. Recordings go to {drop}")
print("press Enter right after each swing (Ctrl+C to stop)")
try:
    while True:
        input()
        save()
except KeyboardInterrupt:
    print("stopped")
