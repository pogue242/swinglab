"""Upload server and coach chat (item 8), with FastAPI's test client and synthetic files."""
import importlib.util
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import needs_models

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("swinglab_server", ROOT / "api" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def make(tmp_path, **kw):
    from swinglab import profiles
    ud = tmp_path / "userdata"
    for name in ("Adrian", "Wilson"):
        if not profiles.by_name(name, ud):
            profiles.create({"name": name, "bats": "R", "height_m": 1.8}, ud)
    app = server.create_app(tmp_path / "inbox", ud, debounce_s=0, watch_every_s=0, **kw)
    return TestClient(app), app


def test_pages_are_served(tmp_path):
    c, _ = make(tmp_path)
    assert "SwingLab Viewer" in c.get("/").text
    assert "SwingLab Upload" in c.get("/upload").text
    h = c.get("/api/hitters").json()
    assert h["hitters"] == ["Adrian", "Wilson"] and h["today"]            # template hitters.csv
    video = c.get("/data/overlays/42_1.mp4", headers={"Range": "bytes=0-99"})
    if (ROOT / "web" / "data" / "overlays" / "42_1.mp4").exists():          # the viewer can seek: ranges work
        assert video.status_code == 206 and video.headers.get("accept-ranges") == "bytes"


def test_upload_saves_files_safely_and_triggers_processing(tmp_path):
    calls = []
    fake = lambda l, v, h, **kw: (calls.append((l, v, h["name"])) or {"recs": [{"stages": {}, "analysis": {"scores": {}}}], "summary": None})  # noqa: E731
    c, app = make(tmp_path, analyze_recording=fake)
    r = c.post("/api/upload", data={"session": "2026-10-10 field", "kind": "slowmo", "hitter": "Wilson"},
               files=[("files", ("../../evil IMG_1.MOV", b"x" * 1000, "video/quicktime"))])
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["hitter"] == "Wilson" and d["saved"][0]["name"] == "evil IMG_1.MOV"
    f = tmp_path / "userdata" / "drop" / "Wilson" / "evil IMG_1.MOV"            # the hitter's drop folder
    assert f.exists() and f.stat().st_size == 1000 and not list(f.parent.glob("*.part"))
    assert not (tmp_path / "evil IMG_1.MOV").exists()
    assert calls and calls[-1][1].endswith("evil IMG_1.MOV") and calls[-1][2] == "Wilson"   # the chosen hitter
    st = c.get("/api/status").json()
    assert not st["running"] and any("received 1 file" in line for line in st["log"])


def test_upload_rejects_the_wrong_kind_of_file(tmp_path):
    c, _ = make(tmp_path)
    r = c.post("/api/upload", data={"kind": "lidar"}, files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert r.status_code == 400 and "lidar upload must be" in r.json()["error"]
    r = c.post("/api/upload", data={"kind": "hack"}, files=[("files", ("a.mov", b"x", "video/quicktime"))])
    assert r.status_code == 400


def test_upload_without_a_hitter_goes_to_a_dated_session(tmp_path):
    c, _ = make(tmp_path)
    r = c.post("/api/upload", data={"session": "2026-10-10 field", "process": "false"},
               files=[("files", ("IMG_1.MOV", b"x", "video/quicktime")), ("files", ("rec.r3d", b"x", "application/zip"))])
    d = r.json()
    assert [x["kind"] for x in d["saved"]] == ["slowmo", "lidar"]                 # sorted by file type
    assert (tmp_path / "inbox" / "sessions" / "2026-10-10_field" / "lidar" / "rec.r3d").exists()
    assert c.post("/api/upload", data={"process": "false"}, files=[("files", ("notes.txt", b"x", "text/plain"))]).status_code == 400


def test_drop_folder_is_analyzed_and_creates_no_stray_pairs(tmp_path):
    """Download recordings into userdata/drop/<Hitter>/ and they are analyzed for that hitter."""
    calls = []
    fake = lambda l, v, h, **kw: (calls.append((l, v, h["name"])) or {"recs": [{"stages": {}, "analysis": {"scores": {}}}], "summary": None})  # noqa: E731
    c, app = make(tmp_path, analyze_recording=fake)
    d = tmp_path / "userdata" / "drop"
    assert (d / "Adrian").is_dir() and (d / "Wilson").is_dir()                   # every hitter has a folder
    (d / "Wilson" / "2026-10-11--10-00-00.r3d").write_bytes(b"x")
    (d / "Somebody else").mkdir()
    (d / "Somebody else" / "a.r3d").write_bytes(b"x")                             # not a hitter: ignored
    app.state.processor.run()
    assert [(Path(l).name, v, h) for l, v, h in calls] == [("2026-10-11--10-00-00.r3d", None, "Wilson")]
    app.state.processor.run()
    assert len(calls) == 1                                                        # done once


def test_yardstick_upload_is_named_as_calibration(tmp_path):
    c, _ = make(tmp_path)
    r = c.post("/api/upload", data={"session": "s1", "kind": "yardstick", "process": "false"},
               files=[("files", ("IMG_0099.MOV", b"x", "video/quicktime"))])
    assert r.json()["saved"][0]["name"] == "yardstick_IMG_0099.MOV"
    from swinglab import inbox
    assert inbox.is_yardstick(tmp_path / "inbox" / "sessions" / "s1" / "slowmo" / "yardstick_IMG_0099.MOV")


@needs_models
def test_phone_upload_to_viewer_end_to_end(tmp_path):
    """A zipped Stray Scanner recording and a slow-motion clip uploaded from 'phones' are paired,
    analyzed and listed in the viewer's index."""
    from swinglab import pipeline
    from test_pipeline import H, W, render_stray, write_ball_video
    from test_quality import rigid_swing
    t, J = rigid_swing()
    rec_dir = tmp_path / "rec"
    kps = render_stray(rec_dir, t, J)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in rec_dir.rglob("*"):
            if p.is_file():
                z.write(p, Path("rec_0001") / p.relative_to(rec_dir))
    shot = tmp_path / "IMG_0001.mp4"
    write_ball_video(shot)
    analyze = lambda l, v, h, **kw: pipeline.analyze_recording(  # noqa: E731
        l, v, h, use_gemini=False, keypoints_fn=lambda k: (kps, (W, H)), **kw)
    c, _ = make(tmp_path, analyze_recording=analyze)
    r1 = c.post("/api/upload", data={"session": "s1", "kind": "lidar", "process": "false"},
                files=[("files", ("rec_0001.zip", buf.getvalue(), "application/zip"))])
    r2 = c.post("/api/upload", data={"session": "s1", "kind": "slowmo"},
                files=[("files", ("IMG_0001.mp4", shot.read_bytes(), "video/mp4"))])
    assert r1.status_code == r2.status_code == 200
    sw = c.get("/api/profiles/adrian/swings").json()["swings"]       # first profile = the session's hitter
    assert len(sw) == 1 and sw[0]["mode"] == "hit" and sw[0]["exit_speed"]
    rec = c.get("/" + sw[0]["url"]).json()
    assert rec["stages"]["lidar"]["ok"] and rec["stages"]["slowmo"]["ok"] and rec["overlay"]["path"].startswith("swings/adrian/")
    assert rec["analysis"]["scores"]["contact"] is not None                     # a hit gets a ball-contact score
    assert c.get("/userdata/" + rec["overlay"]["path"]).status_code == 200      # the video is served
    assert c.get("/api/latest").json()["profile"] == "adrian"


def test_chat_answers_from_the_payload(tmp_path):
    seen = {}

    def gen(model, contents, config):
        seen["system"] = config.system_instruction
        seen["turns"] = [c.parts[0].text for c in contents]
        return SimpleNamespace(text="Turn your hips first.", model_version=model)
    c, _ = make(tmp_path, gemini_client=SimpleNamespace(models=SimpleNamespace(generate_content=gen)))
    payload = {"scores_0_100": {"overall": 42.0}, "focus": {"label": "Shoulders open at foot plant"}}
    r = c.post("/api/chat", json={"payload": payload, "question": "Why 42?",
                                  "history": [{"role": "user", "text": "hi"}, {"role": "model", "text": "hello"}]})
    assert r.status_code == 200 and r.json()["answer"] == "Turn your hips first." and r.json()["unsupported_numbers"] == []
    assert "Shoulders open at foot plant" in seen["system"] and seen["turns"] == ["hi", "hello", "Why 42?"]
    assert c.post("/api/chat", json={"payload": payload, "question": ""}).status_code == 422


def test_chat_without_a_key_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    c, _ = make(tmp_path)
    r = c.post("/api/chat", json={"payload": {"x": 1}, "question": "Hi?"})
    assert r.status_code == 200 and "GEMINI_API_KEY" in r.json()["answer"]


def test_profiles_create_edit_delete(tmp_path):
    c, _ = make(tmp_path)
    r = c.post("/api/profiles", json={"name": "Matteo", "bats": "L", "height_m": 1.75, "mass_kg": 72})
    assert r.status_code == 200 and r.json()["id"] == "matteo"
    assert c.post("/api/profiles", json={"name": "matteo"}).status_code == 400          # duplicate name
    assert c.post("/api/profiles", json={"name": "X", "height_m": 175}).status_code == 400   # cm, not m
    r = c.put("/api/profiles/matteo", json={"mass_kg": 74.5, "age": 21})
    assert r.json()["mass_kg"] == 74.5 and r.json()["bats"] == "L"
    names = [p["name"] for p in c.get("/api/profiles").json()["profiles"]]
    assert names == ["Adrian", "Wilson", "Matteo"]
    (tmp_path / "userdata" / "swings" / "matteo" / "a.json").write_text("{}", encoding="utf-8")
    d = c.delete("/api/profiles/matteo").json()
    assert d["deleted"] == "matteo" and "trash" in d["swings_moved_to"] and Path(d["swings_moved_to"], "a.json").exists()
    assert c.get("/api/profiles/matteo/swings").status_code == 404


def test_recordings_group_swings_with_the_whole_recording_video(tmp_path):
    """Whole-recording view (issue 4): swings grouped by recording, times in the recording's video."""
    c, _ = make(tmp_path)
    d = tmp_path / "userdata" / "swings" / "wilson"
    d.mkdir(parents=True, exist_ok=True)
    for k, peak in ((1, 10.3), (2, 12.4)):
        rec = {"id": f"lidar 1-0{k}", "recording": "lidar 1", "recording_t0": 10.0, "mode": "dry",
               "window": {"start": peak - 1.6, "peak": peak, "end": peak + 0.8, "partial": k == 1},
               "analysis": {"scores": {"form": 40.0 + k}, "focus": {"label": "Hip turn"}},
               "overlay": {"path": f"swings/wilson/lidar 1-0{k}.mp4"}, "recorded_at": "2026-10-10 10:00:00"}
        (d / f"lidar 1-0{k}.json").write_text(json.dumps(rec), encoding="utf-8")
    (d / "old.json").write_text(json.dumps({"id": "old", "analysis": {"scores": {}}}), encoding="utf-8")
    (d / "_summary_lidar 1.json").write_text(json.dumps({"n_swings": 2, "report": {"headline": "Two swings"},
                                                         "recording_video": {"t0": 10.0}}), encoding="utf-8")
    (d / "_recording_lidar 1.mp4").write_bytes(b"x")
    recs = c.get("/api/profiles/wilson/recordings").json()["recordings"]
    rec = next(r for r in recs if r["recording"] == "lidar 1")
    assert rec["n_swings"] == 2 and rec["headline"] == "Two swings"
    assert rec["video"]["url"] == "userdata/swings/wilson/_recording_lidar 1.mp4" and rec["video"]["t0"] == 10.0
    assert rec["summary_url"] == "userdata/swings/wilson/_summary_lidar 1.json"
    s1, s2 = rec["swings"]
    assert s1["t"] == {"start": -1.3, "peak": 0.3, "end": 1.1} and s1["partial"] and not s2["partial"]
    assert s2["video_url"] == "userdata/swings/wilson/lidar 1-02.mp4" and s2["focus"] == "Hip turn"
    assert any(r["recording"] == "old" and r["video"] is None and r["swings"][0]["t"] is None for r in recs)
    assert c.get("/api/profiles/nobody/recordings").status_code == 404


def test_recordings_find_the_inbox_named_summary(tmp_path):
    """Drop-folder swings are saved as '<session>__<id>' with '_summary_<session>__<recording>.json'."""
    c, _ = make(tmp_path)
    d = tmp_path / "userdata" / "swings" / "wilson"
    d.mkdir(parents=True, exist_ok=True)
    for k in (1, 2):
        rec = {"id": f"Wilson__lidar 1-0{k}", "recording": "lidar 1", "recording_t0": 0.0,
               "window": {"start": 1.0 * k, "peak": 2.0 * k, "end": 3.0 * k}, "analysis": {"scores": {}}}
        (d / f"Wilson__lidar 1-0{k}.json").write_text(json.dumps(rec), encoding="utf-8")
    (d / "_summary_Wilson__lidar 1.json").write_text(json.dumps({"report": {"headline": "H"}}), encoding="utf-8")
    (d / "_recording_Wilson__lidar 1.mp4").write_bytes(b"x")
    rec = c.get("/api/profiles/wilson/recordings").json()["recordings"][0]
    assert rec["headline"] == "H" and rec["video"]["url"].endswith("_recording_Wilson__lidar 1.mp4")


def test_failed_recording_is_retried_when_the_file_changes(tmp_path):
    """UI request 2026-10-10: a failed analysis must not mark the recording as processed for good."""
    import os
    from swinglab import inbox, profiles
    ud = tmp_path / "userdata"
    profiles.create({"name": "Wilson", "bats": "R", "height_m": 1.8}, ud)
    drop = ud / "drop"
    rec_dir = drop / "Wilson"
    rec_dir.mkdir(parents=True, exist_ok=True)
    f = rec_dir / "swing.r3d"
    f.write_bytes(b"broken")
    calls = []

    def fail(lidar, slowmo, hitter, **kw):
        calls.append(lidar)
        return {"recs": [{"id": "x", "stages": {"lidar": {"ok": False, "error": "no pose model"}}}], "summary": None}
    out = inbox.process(drop, analyze_recording=fail, userdata=ud, log=lambda *a: None)
    assert len(calls) == 1 and out[0]["error"] == "no pose model"
    inbox.process(drop, analyze_recording=fail, userdata=ud, log=lambda *a: None)
    assert len(calls) == 1                                       # unchanged file: not retried every 15 s
    f.write_bytes(b"uploaded again")
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 5))
    inbox.process(drop, analyze_recording=fail, userdata=ud, log=lambda *a: None)
    assert len(calls) == 2                                       # changed file: retried
