"""Download the MediaPipe Pose Landmarker model (heavy, ~30 MB) into data/models/."""
import urllib.request

from swinglab import config
from swinglab.capture import POSE_MODEL_URL

out = config.MODELS / "pose_landmarker_heavy.task"
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists():
    urllib.request.urlretrieve(POSE_MODEL_URL, out)
print(out, out.stat().st_size, "bytes")
