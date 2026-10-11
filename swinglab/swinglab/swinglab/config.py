"""Paths and constants shared by the whole pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path = ROOT / ".env") -> list[str]:
    """KEY=VALUE lines from ``.env`` (git-ignored) into os.environ; real environment variables win.
    Lets the team keep GEMINI_API_KEY in one file instead of setting it in every terminal."""
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded.append(k)
    return loaded


load_dotenv()
DATA = Path(os.environ.get("SWINGLAB_DATA", ROOT / "data"))
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = DATA / "models"
WEB_DATA = ROOT / "web" / "data"
# Hitters' profiles, swings, uploads: kept OUT of the app (git-ignored) so the app can be shared
# without anyone's data. See swinglab/profiles.py.
USERDATA = Path(os.environ.get("SWINGLAB_USERDATA", ROOT / "userdata"))

OBP_RELEASE_URL = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/{name}"
OBP_REPO_RAW = "https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/main/{path}"

# Driveline motion capture rate and the rate we pretend the phone records at.
# Training features are computed AFTER downsampling to PHONE_FPS so that the
# model sees the same timing coarseness as the phone. Change PHONE_FPS once the
# real capture rate is measured (see docs/HANDOFF.md, "Open questions").
MOCAP_FPS = 360.0
PHONE_FPS = float(os.environ.get("SWINGLAB_PHONE_FPS", 30.0))

M_PER_IN = 0.0254
KG_PER_LB = 0.45359237
MPS_TO_MPH = 2.2369363
