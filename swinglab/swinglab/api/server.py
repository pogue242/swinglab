"""SwingLab server: the viewer, a phone upload page and the coach chat, on the laptop.

  python api\\server.py                                   # http://<laptop IP>:8000 (viewer), /upload (phones)
  python api\\server.py --watch "G:\\My Drive\\SwingLab"     # also analyze whatever lands in the Drive folder

Phones on the same Wi-Fi open http://<laptop IP>:8000/upload, choose the session, hitter and
kind of file (LiDAR recording: Record3D .r3d or a zipped Stray Scanner folder; slow-motion clip;
yardstick clip) and upload. Files land in the shared-folder layout (inbox.py):
<inbox>/sessions/<session>/lidar|slowmo/, so the same pairing and analysis run as for the synced
folder. Processing starts automatically once uploads have stopped for ``--debounce`` seconds
(the two phones upload the two halves of a swing separately), or with the Process button.

  GET  /                  viewer (web/index.html; new swings appear from data/index.json)
  GET  /upload            upload page for phones
  POST /api/upload        multipart: files, session, kind (lidar | slowmo | yardstick), hitter?
  POST /api/process       start processing now
  GET  /api/status        processing state, recent log lines, last results
  GET  /api/hitters       profile names (upload page)
  GET  /api/profiles      profiles with swing counts;  POST: create;  PUT/DELETE /api/profiles/{id}
  GET  /api/profiles/{id}/swings   the profile's swing history (oldest first)
  GET  /api/profiles/{id}/recordings  swings grouped by recording + whole-recording video and summary
  GET  /api/latest        the most recently analyzed swing (the viewer pops new swings up)
  GET  /userdata/swings/… swing records and annotated videos (user data, outside the app)
  POST /api/chat          {"payload": swing payload, "history": [...], "question": "..."} -> {"answer"}
  POST /api/live/connect  start the iPhone's USB stream (Record3D) for the preview
  POST /api/live/start    {"hitter"}: record a session; it ends after --live-idle s of stillness (or /api/live/stop)
                          and lands as one .r3d in the hitter's drop folder, analyzed right away
  GET  /api/live/status   recording state, elapsed, frames, motion, auto-stop countdown, last saved session
  GET  /api/live/preview  MJPEG of what the phone sees

Windows asks once to allow Python through the firewall (choose Private networks): phones need it.
LAN tool for the hackathon: no login; anyone on the Wi-Fi can upload recordings.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from swinglab import config, features, inbox, live, pipeline, profiles  # noqa: E402

ALLOWED = {"lidar": {".r3d", ".zip"}, "slowmo": {".mov", ".mp4", ".m4v"}, "yardstick": {".mov", ".mp4", ".m4v"}}
MAX_BYTES = 4 * 1024 ** 3          # 4 GB per file
CHUNK = 1024 * 1024


def _rel(path: str | None, root: Path) -> str | None:
    """``path`` relative to ``root`` with forward slashes (as the viewer fetches it), else as given."""
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(path)


def safe_name(name: str) -> str:
    """A plain file name: no folders (``../``), no odd characters."""
    base = re.split(r"[\\/]", name or "")[-1].strip().lstrip(".")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:120]
    return base or "upload"


def safe_session(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]", "_", (name or "").strip())[:60].strip("._")
    return s or dt.date.today().isoformat()


class ChatIn(BaseModel):
    payload: dict
    history: list[dict] = Field(default_factory=list, max_length=40)
    question: str = Field(min_length=1, max_length=1000)


class Processor:
    """Runs inbox.process in one background thread at a time, debounced after uploads."""

    def __init__(self, inbox_root: Path, userdata: Path, analyze_recording=None, debounce_s: float = 15.0,
                 watch: list[Path] | None = None, demo: bool = False):
        self.inbox_root, self.userdata, self.watch = inbox_root, userdata, list(watch or [])
        self.analyze_recording, self.debounce_s = analyze_recording, debounce_s
        # demo mode: phase 1 saves the scores as soon as the pose is measured (no ball check, template text, no
        # video); phase 2 (complete()) adds the ball check, the AI report and the annotated video to the same swing
        self.demo = demo
        if demo and analyze_recording is None:
            from functools import partial
            self.analyze_recording = partial(pipeline.analyze_recording, quick=True)
        self.lock = threading.Lock()
        self.running = False
        self.log: list[str] = []
        self.results: list[dict] = []
        self.last_run: str | None = None
        self._timer: threading.Timer | None = None

    def say(self, *msg):
        line = " ".join(str(m) for m in msg)
        self.log = (self.log + [f"{time.strftime('%H:%M:%S')} {line}"])[-200:]

    def schedule(self):
        if self._timer:
            self._timer.cancel()
        if self.debounce_s <= 0:
            self.run()
            return
        self._timer = threading.Timer(self.debounce_s, self.run)
        self._timer.daemon = True
        self._timer.start()

    def run(self):
        if not self.lock.acquire(blocking=False):
            self.say("already processing; will not start a second run")
            return
        self.running = True
        try:
            done = []
            for root in [self.inbox_root, self.userdata / "drop", *self.watch]:
                if not root.exists():
                    continue
                kw = dict(save=lambda r, folder: pipeline.save_result(r, folder, out_root=self.userdata,
                                                                      render_video=False)) if self.demo else {}
                done += inbox.process(root, analyze_recording=self.analyze_recording, log=self.say,
                                      userdata=self.userdata, **kw,
                                      # uploads land complete (.part, then renamed); dropped files may still be copying
                                      settle_s=0.0 if root == self.inbox_root else inbox.SETTLE_S)
            finished_at = time.time()
            self.results = [{"id": r.get("id"), "hitter": r.get("hitter"),
                             "stages": {k: v.get("ok") for k, v in (r.get("stages") or {}).items()},
                             "headline": (r.get("report") or {}).get("headline"),
                             "file": _rel(r.get("saved_file"), self.userdata),
                             "error": r.get("error"), "finished_at": finished_at} for r in done] or self.results
            if done:
                n_ok = sum(1 for r in done if r.get("saved_file"))
                self.say(f"done: {n_ok} new swing(s)" + (f", {len(done) - n_ok} failed" if n_ok < len(done) else ""))
            for r in done:
                if r.get("pending") and r.get("saved_file"):
                    self.complete(r)
        except Exception as e:   # keep the server up; show the error on the page
            self.say(f"ERROR {type(e).__name__}: {e}")
        finally:
            self.running = False
            self.last_run = time.strftime("%Y-%m-%d %H:%M:%S")
            self.lock.release()


    def complete(self, rec: dict):
        """Demo mode phase 2: ball check, AI coach report, annotated video, saved over the phase-1 record."""
        t0 = time.time()
        try:
            prof = profiles.get(rec.get("profile"), self.userdata) or {}
            hitter = profiles.hitter(prof) if prof else {"bats": rec.get("bats", "R"), "height_m": rec.get("height_m", 1.78)}
            pipeline.complete_swing(rec, hitter)
            pipeline.save_result(rec, Path(rec["saved_file"]).parent.name, out_root=self.userdata, render_video=True)
            self.say(f"      -> {rec['id']}: {rec.get('mode')} swing, report and video ready ({time.time() - t0:.0f} s)")
        except Exception as e:
            self.say(f"      !! {rec.get('id')}: finishing failed: {type(e).__name__}: {e}")

    def watch_loop(self, every_s: float = 15.0):
        """Poll the watched folders (e.g. the synced Google Drive folder) for new recordings."""
        while True:
            time.sleep(every_s)
            if not self.running:
                self.run()


class LiveStartIn(BaseModel):
    hitter: str = Field(min_length=1, max_length=80)


class ProfileIn(BaseModel):
    name: str | None = None
    bats: str | None = None
    height_m: float | None = None
    mass_kg: float | None = None
    age: float | None = None
    level: str | None = None


def create_app(inbox_root: str | Path | None = None, userdata: str | Path | None = None, *,
               analyze_recording=None, debounce_s: float = 15.0, gemini_client=None,
               watch: list[str | Path] | None = None, watch_every_s: float = 15.0, demo: bool = False,
               live_idle_s: float = 8.0, live_tail_s: float = 3.0, live_connect=None) -> FastAPI:
    userdata = Path(userdata or config.USERDATA)
    inbox_root = Path(inbox_root or (userdata / "inbox"))
    (userdata / "swings").mkdir(parents=True, exist_ok=True)
    (inbox_root / "sessions").mkdir(parents=True, exist_ok=True)
    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.live.disconnect()                   # released, the phone accepts the next server's connection

    app = FastAPI(title="SwingLab", lifespan=lifespan)
    proc = Processor(inbox_root, userdata, analyze_recording, debounce_s, [Path(w) for w in (watch or [])], demo=demo)
    app.state.processor = proc
    (userdata / "drop").mkdir(parents=True, exist_ok=True)
    for pr in profiles.list_profiles(userdata):         # every hitter has a drop folder
        if not pr.get("synthetic"):
            profiles.drop_dir(pr["name"], userdata).mkdir(parents=True, exist_ok=True)
    if watch_every_s:                                 # the drop folder (and --watch folders) are checked all the time
        threading.Thread(target=proc.watch_loop, args=(watch_every_s,), daemon=True).start()

    def _live_saved(info: dict) -> None:
        settled = time.time() - inbox.SETTLE_S - 1    # packed complete (.part, then renamed): no copy-settle wait
        os.utime(info["path"], (settled, settled))
        threading.Thread(target=proc.run, daemon=True).start()

    live_session = live.LiveSession(userdata / "live_staging", idle_s=live_idle_s, tail_s=live_tail_s,
                                    on_saved=_live_saved, log=proc.say, connect=live_connect)
    app.state.live = live_session

    def _swing_list(pid: str) -> list[dict]:
        return [{**x, "url": f"userdata/swings/{pid}/{x['file']}"} for x in profiles.swing_summaries(pid, userdata)]

    @app.get("/api/profiles")
    def list_profiles():
        out = []
        for pr in profiles.list_profiles(userdata):
            sw = profiles.swing_summaries(pr["id"], userdata)
            trend = [x["scores"].get("overall") for x in sw if x["scores"].get("overall") is not None][-12:]
            out.append({**pr, "n_swings": len(sw), "last_swing": sw[-1]["recorded_at"] if sw else None,
                        "latest_overall": trend[-1] if trend else None, "trend": trend,
                        "drop_folder": None if pr.get("synthetic") else str(profiles.drop_dir(pr["name"], userdata))})
        return {"profiles": out}

    @app.get("/api/info")
    def info():
        port = app.state.port if hasattr(app.state, "port") else 8000
        return {"phone_upload_url": f"http://{lan_ip()}:{port}/upload", "drop_root": str(userdata / "drop"),
                "key_features": list(features.KEY_FEATURES)}

    @app.post("/api/profiles")
    def create_profile(body: ProfileIn):
        try:
            return profiles.create(body.model_dump(exclude_none=True), userdata)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/profiles/{pid}")
    def update_profile(pid: str, body: ProfileIn):
        try:
            return profiles.update(pid, body.model_dump(exclude_none=True), userdata)
        except KeyError:
            raise HTTPException(404, "no such profile")
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/profiles/{pid}")
    def delete_profile(pid: str):
        try:
            moved = profiles.delete(pid, userdata)
        except KeyError:
            raise HTTPException(404, "no such profile")
        return {"deleted": pid, "swings_moved_to": str(moved) if moved else None}

    @app.get("/api/profiles/{pid}/swings")
    def profile_swings(pid: str):
        if profiles.get(pid, userdata) is None:
            raise HTTPException(404, "no such profile")
        return {"swings": _swing_list(pid)}

    @app.get("/api/profiles/{pid}/recordings")
    def profile_recordings(pid: str):
        if profiles.get(pid, userdata) is None:
            raise HTTPException(404, "no such profile")
        base = f"userdata/swings/{pid}/"
        recs = profiles.recordings(pid, userdata)
        for rc in recs:
            rc["summary_url"] = base + rc["summary"] if rc["summary"] else None
            if rc["video"]:
                rc["video"]["url"] = base + rc["video"]["file"]
            for sw in rc["swings"]:
                sw["url"] = base + sw["file"]
                sw["video_url"] = base + sw["video_file"] if sw["video_file"] else None
        return {"recordings": recs}

    @app.get("/api/latest")
    def latest():
        best = None
        for pr in profiles.list_profiles(userdata):
            for f in profiles.swings_dir(pr["id"], userdata).glob("*.json"):
                if not f.name.startswith("_") and (best is None or f.stat().st_mtime > best[0]):
                    best = (f.stat().st_mtime, pr["id"], f.name)
        return {"profile": best[1], "file": best[2], "modified": best[0]} if best else {"profile": None}

    @app.get("/upload", response_class=HTMLResponse)
    def upload_page():
        return (config.ROOT / "web" / "upload.html").read_text(encoding="utf-8")

    @app.get("/api/hitters")
    def hitters():
        return {"hitters": [pr["name"] for pr in profiles.list_profiles(userdata) if not pr.get("synthetic")],
                "today": dt.date.today().isoformat(),
                "sessions": sorted(p.name for p in (inbox_root / "sessions").glob("*") if p.is_dir())}

    @app.post("/api/upload")
    async def upload(files: list[UploadFile] = File(...), session: str = Form(""), kind: str = Form("auto"),
                     hitter: str = Form(""), process: bool = Form(True)):
        """kind: auto (by file type: .r3d/.zip = LiDAR recording, .mov/.mp4 = slow-motion clip,
        "yardstick" in the name = scale clip) | lidar | slowmo | yardstick. With a hitter that has a
        profile, files go to that hitter's drop folder; otherwise to a dated session in the inbox."""
        if kind not in ALLOWED and kind != "auto":
            raise HTTPException(400, f"kind must be auto or one of {sorted(ALLOWED)}")
        received_at = time.time()          # server clock: the viewer waits for results finished after this
        prof = profiles.by_name(hitter, userdata) if hitter else None
        if prof is not None and not prof.get("synthetic"):
            sdir = profiles.drop_dir(prof["name"], userdata)
            dest_for = lambda k: sdir  # noqa: E731   flat: the inbox sorts by file type
        else:
            sdir = inbox_root / "sessions" / safe_session(session)
            dest_for = lambda k: sdir / ("lidar" if k == "lidar" else "slowmo")  # noqa: E731
        saved = []
        for f in files:
            name = safe_name(f.filename)
            ext = Path(name).suffix.lower()
            k = kind
            if k == "auto":
                k = ("lidar" if ext in ALLOWED["lidar"] else
                     "yardstick" if "yardstick" in name.lower() else "slowmo" if ext in ALLOWED["slowmo"] else None)
                if k is None:
                    raise HTTPException(400, f"{name}: not a recording (.r3d or zipped Stray Scanner folder) "
                                             f"or a video (.mov, .mp4)")
            if ext not in ALLOWED[k]:
                raise HTTPException(400, f"{name}: a {k} upload must be {', '.join(sorted(ALLOWED[k]))}")
            if k == "yardstick" and "yardstick" not in name.lower():
                name = f"yardstick_{name}"
            dest = dest_for(k)
            dest.mkdir(parents=True, exist_ok=True)
            target, part = dest / name, dest / (name + ".part")
            size = 0
            with open(part, "wb") as out:          # .part until complete: the inbox never sees half a file
                while chunk := await f.read(CHUNK):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        out.close()
                        part.unlink(missing_ok=True)
                        raise HTTPException(413, f"{name} is larger than {MAX_BYTES // 1024 ** 3} GB")
                    out.write(chunk)
            part.replace(target)
            settled = time.time() - inbox.SETTLE_S - 1      # complete on arrival: no copy-settle wait
            os.utime(target, (settled, settled))
            saved.append({"name": name, "kind": k, "bytes": size,
                          "path": str(target.relative_to(userdata if prof else inbox_root)).replace("\\", "/")})
        if prof is None and hitter and profiles.by_name(hitter, userdata):
            _note_hitter(sdir, hitter)
        proc.say(f"received {len(saved)} file(s) for {sdir.name}: " + ", ".join(x["name"] for x in saved))
        if process:
            proc.schedule()
        return {"saved": saved, "session": sdir.name, "hitter": prof["name"] if prof else None,
                "processing_in_s": proc.debounce_s if process else None, "received_at": received_at}

    @app.post("/api/process")
    def process_now():
        threading.Thread(target=proc.run, daemon=True).start()
        return {"started": True}

    @app.get("/api/status")
    def status():
        return {"running": proc.running, "last_run": proc.last_run, "log": proc.log[-40:], "results": proc.results[-20:]}

    def _no_phone(e: Exception) -> HTTPException:
        msg = str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"
        return HTTPException(503, f"Could not reach the iPhone: {msg}")

    @app.post("/api/live/connect")
    def live_connect_phone():
        try:
            live_session.connect()
        except Exception as e:
            raise _no_phone(e)
        return live_session.status()

    @app.post("/api/live/start")
    def live_start(body: LiveStartIn):
        prof = profiles.by_name(body.hitter, userdata)
        if prof is None or prof.get("synthetic"):
            raise HTTPException(404, f"no hitter named '{body.hitter}'")
        if live_session.state in ("recording", "saving"):
            raise HTTPException(409, f"a session is already {live_session.state}")
        drop = profiles.drop_dir(prof["name"], userdata)
        drop.mkdir(parents=True, exist_ok=True)
        try:
            return live_session.start(prof["name"], drop)
        except Exception as e:
            raise _no_phone(e)

    @app.post("/api/live/stop")
    def live_stop():
        return live_session.stop()

    @app.get("/api/live/status")
    def live_status():
        return live_session.status()

    @app.get("/api/live/preview")
    async def live_preview():
        """MJPEG: the newest preview frame, for an <img> in the Record swing popup."""
        async def frames():
            last = None
            while True:
                jpg = live_session.preview_jpeg()
                if jpg is not None and jpg is not last:
                    last = jpg
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n" % (len(jpg), jpg)
                await asyncio.sleep(0.05)
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.post("/api/chat")
    def chat(body: ChatIn):
        from swinglab.coach import gemini
        if len(json.dumps(body.payload)) > 200_000:
            raise HTTPException(413, "payload too large")
        hist = [{"role": "model" if h.get("role") == "model" else "user", "text": str(h.get("text", ""))[:2000]}
                for h in body.history]
        from swinglab.coach import numbers
        answer = gemini.chat(body.payload, hist, body.question, client=gemini_client)
        bad = [x["number"] for x in numbers.unsupported({"answer": answer}, body.payload)]
        return {"answer": answer, "unsupported_numbers": bad}   # numbers not in the payload (should be none)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    app.mount("/userdata/swings", StaticFiles(directory=str(userdata / "swings")), name="swings")
    app.mount("/", StaticFiles(directory=str(config.ROOT / "web"), html=True), name="web")
    return app


def _note_hitter(sdir: Path, hitter: str) -> None:
    """Remember who hit in this session (the inbox uses the session's log.csv, else the first hitter)."""
    f = sdir / "hitter.txt"
    if not f.exists():
        f.write_text(hitter, encoding="utf-8")


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inbox", default=None, help="where phone uploads go (default: <userdata>/inbox)")
    ap.add_argument("--watch", action="append", default=[],
                    help="also analyze new recordings in this shared folder (e.g. the synced Google Drive folder); repeatable")
    ap.add_argument("--userdata", default=None, help="profiles and swings (default: <repo>/userdata or SWINGLAB_USERDATA)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debounce", type=float, default=15.0, help="seconds without uploads before processing starts")
    ap.add_argument("--demo", action="store_true",
                    help="fast demo: check the drop folders every 2 s; show the scores as soon as the pose is measured, "
                         "then add the ball check, AI report and video to the same swing")
    ap.add_argument("--live-idle", type=float, default=8.0,
                    help="Record swing: end the session after the hitter has been still this many seconds")
    ap.add_argument("--live-tail", type=float, default=3.0,
                    help="Record swing: keep recording this many seconds after Stop (the phone's video arrives late)")
    a = ap.parse_args()
    import uvicorn
    app = create_app(a.inbox, a.userdata, debounce_s=1.0 if a.demo else a.debounce, watch=a.watch,
                     watch_every_s=2.0 if a.demo else 15.0, demo=a.demo, live_idle_s=a.live_idle,
                     live_tail_s=a.live_tail)
    app.state.port = a.port
    ud = Path(a.userdata or config.USERDATA)
    print(f"viewer:   http://localhost:{a.port}/\nphones:   http://{lan_ip()}:{a.port}/upload   (same Wi-Fi)\n"
          f"drop:     {ud / 'drop'} (one folder per hitter: put downloaded recordings in it)\n"
          f"userdata: {ud}" + "".join(f"\nwatching: {w}" for w in a.watch))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
