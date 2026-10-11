"""Download the YOLOX-S object detector (COCO, ~35 MB, Apache-2.0, OpenCV model zoo) into data/models/.
Used by ballflight.yolo_balls to confirm a ball is really in view before a swing is called a hit or a miss."""
import urllib.request

from swinglab import config
from swinglab.ballflight import YOLO_MODEL_FILE, YOLO_MODEL_URL

out = config.MODELS / YOLO_MODEL_FILE
out.parent.mkdir(parents=True, exist_ok=True)
if not out.exists():
    urllib.request.urlretrieve(YOLO_MODEL_URL, out)
print(out, out.stat().st_size, "bytes")
