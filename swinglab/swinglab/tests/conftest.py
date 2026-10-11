import pytest

from swinglab import config, obp

HAS_OBP = obp.landmarks_parquet().exists()
HAS_MODELS = (config.MODELS / "swing_models.joblib").exists()
needs_obp = pytest.mark.skipif(not HAS_OBP, reason="run scripts/setup_data.py first")
needs_models = pytest.mark.skipif(not HAS_MODELS, reason="run scripts/train.py first")
# Real phone recordings dropped in data/captures (git-ignored) and the MediaPipe model file.
REAL_R3D = sorted((config.DATA / "captures").glob("*.r3d"))
POSE_MODEL = config.MODELS / "pose_landmarker_heavy.task"
needs_real_r3d = pytest.mark.skipif(not REAL_R3D, reason="no Record3D recording in data/captures")
needs_pose_model = pytest.mark.skipif(not POSE_MODEL.exists(), reason="run scripts/get_pose_model.py first")


@pytest.fixture(autouse=True)
def _own_userdata(tmp_path_factory, monkeypatch):
    """Profiles and saved swings go to a temporary folder, never the real userdata/."""
    monkeypatch.setattr(config, "USERDATA", tmp_path_factory.mktemp("userdata"))


@pytest.fixture(autouse=True)
def _no_live_gemini(request, monkeypatch):
    """Tests never call the real Gemini API just because a key is in .env;
    tests marked ``live`` keep it (run them with ``pytest -m live``)."""
    if "live" not in request.keywords:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
