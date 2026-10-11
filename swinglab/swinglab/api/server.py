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
  GET  /api/latest        the most recently analyzed swing (the viewer pops new swings up)
  GET  /userdata/swings/… swing records and annotated videos (user data, outside the app)
  POST /api/chat          {"payload": swing payload, "history": [...], "question": "..."} -> {"answer"}

Windows asks once to allow Python through the firewall (choose Private networks): phones need it.
LAN tool for the hackathon: no login; anyone on the Wi-Fi can upload recordings.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
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
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from swinglab import config, inbox, pipeline, profiles  # noqa: E402

ALLOWED = {"lidar": {".r3d", ".zip"}, "slowmo": {".mov", ".mp4", ".m4v"}, "yardstick": {".mov", ".mp4", ".m4v"}}
MAX_BYTES = 4 * 1024 ** 3          # 4 GB per file
CHUNK = 1024 * 1024


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
                 watch: list[Path] | None = None):
        self.inbox_root, self.userdata, self.watch = inbox_root, userdata, list(watch or [])
        self.analyze_recording, self.debounce_s = analyze_recording, debounce_s
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
                done += inbox.process(root, analyze_recording=self.analyze_recording, log=self.say,
                                      userdata=self.userdata)
            self.results = [{"id": r.get("id"), "hitter": r.get("hitter"),
                             "stages": {k: v.get("ok") for k, v in (r.get("stages") or {}).items()},
                             "headline": (r.get("report") or {}).get("headline")} for r in done] or self.results
            if done:
                self.say(f"done: {len(done)} new swing(s)")
        except Exception as e:   # keep the server up; show the error on the page
            self.say(f"ERROR {type(e).__name__}: {e}")
        finally:
            self.running = False
            self.last_run = time.strftime("%Y-%m-%d %H:%M:%S")
            self.lock.release()


    def watch_loop(self, every_s: float = 15.0):
        """Poll the watched folders (e.g. the synced Google Drive folder) for new recordings."""
        while True:
            time.sleep(every_s)
            if not self.running:
                self.run()


class ProfileIn(BaseModel):
    name: str | None = None
    bats: str | None = None
    height_m: float | None = None
    mass_kg: float | None = None
    age: float | None = None
    level: str | None = None


def create_app(inbox_root: str | Path | None = None, userdata: str | Path | None = None, *,
               analyze_recording=None, debounce_s: float = 15.0, gemini_client=None,
               watch: list[str | Path] | None = None, watch_every_s: float = 15.0) -> FastAPI:
    userdata = Path(userdata or config.USERDATA)
    inbox_root = Path(inbox_root or (userdata / "inbox"))
    (userdata / "swings").mkdir(parents=True, exist_ok=True)
    (inbox_root / "sessions").mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="SwingLab")
    proc = Processor(inbox_root, userdata, analyze_recording, debounce_s, [Path(w) for w in (watch or [])])
    app.state.processor = proc
    (userdata / "drop").mkdir(parents=True, exist_ok=True)
    for pr in profiles.list_profiles(userdata):         # every hitter has a drop folder
        if not pr.get("synthetic"):
            profiles.drop_dir(pr["name"], userdata).mkdir(parents=True, exist_ok=True)
    if watch_every_s:                                 # the drop folder (and --watch folders) are checked all the time
        threading.Thread(target=proc.watch_loop, args=(watch_every_s,), daemon=True).start()

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
        return {"phone_upload_url": f"http://{lan_ip()}:{port}/upload", "drop_root": str(userdata / "drop")}

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
        return UPLOAD_HTML

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
            saved.append({"name": name, "kind": k, "bytes": size,
                          "path": str(target.relative_to(userdata if prof else inbox_root)).replace("\\", "/")})
        if prof is None and hitter and profiles.by_name(hitter, userdata):
            _note_hitter(sdir, hitter)
        proc.say(f"received {len(saved)} file(s) for {sdir.name}: " + ", ".join(x["name"] for x in saved))
        if process:
            proc.schedule()
        return {"saved": saved, "session": sdir.name, "hitter": prof["name"] if prof else None,
                "processing_in_s": proc.debounce_s if process else None}

    @app.post("/api/process")
    def process_now():
        threading.Thread(target=proc.run, daemon=True).start()
        return {"started": True}

    @app.get("/api/status")
    def status():
        return {"running": proc.running, "last_run": proc.last_run, "log": proc.log[-40:], "results": proc.results[-20:]}

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


UPLOAD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SwingLab Upload</title>
<style>
:root{--bg:#F6F6F4;--panel:#fff;--ink:#0C2340;--muted:#5C6670;--line:#E3E3E0;--seam:#BF0D3E;--good:#2E7D4F;--bad:#BF0D3E;color-scheme:light}
header{background:#0C2340;color:#fff;margin:-16px -16px 16px;padding:14px 16px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:560px;margin:0 auto;padding:16px}h1{font-size:24px;margin:4px 0 12px}h1 span{color:var(--seam)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:14px}
label{display:block;font-weight:600;margin:12px 0 6px}select,input[type=text]{width:100%;font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
.kinds{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.kinds label{margin:0;font-weight:500;border:1px solid var(--line);border-radius:8px;padding:10px;text-align:center}
.kinds input{display:none}.kinds input:checked+span{color:var(--seam);font-weight:700}
input[type=file]{width:100%;margin-top:4px}button{width:100%;font:600 17px system-ui;padding:14px;border:0;border-radius:10px;background:var(--seam);color:#fff;margin-top:14px}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line)}progress{width:100%;height:14px;margin-top:10px}
.hint{color:var(--muted);font-size:14px}.ok{color:var(--good)}.err{color:var(--bad)}pre{white-space:pre-wrap;font-size:12px;color:var(--muted);max-height:240px;overflow:auto}
</style></head><body><div class="wrap">
<header><h1 style="margin:0;color:#fff">Swing<span>Lab</span> upload</h1></header>
<div class="card">
<label for="session">Session</label><input type="text" id="session" list="sessions" autocomplete="off"><datalist id="sessions"></datalist>
<label for="hitter">Hitter</label><select id="hitter"></select>
<label>What is it?</label>
<div class="kinds" style="grid-template-columns:repeat(2,1fr)">
 <label><input type="radio" name="kind" value="auto" checked><span>Automatic<br><small>by file type</small></span></label>
 <label><input type="radio" name="kind" value="lidar"><span>LiDAR<br><small>.r3d or .zip</small></span></label>
 <label><input type="radio" name="kind" value="slowmo"><span>Slow motion<br><small>.mov / .mp4</small></span></label>
 <label><input type="radio" name="kind" value="yardstick"><span>Yardstick<br><small>scale clip</small></span></label>
</div>
<label for="files">Files</label><input type="file" id="files" multiple>
<p class="hint">LiDAR: Record3D → share the .r3d; Stray Scanner → zip the recording folder. Slow motion: the original from Photos (not a WhatsApp/Messages copy).</p>
<button id="go">Upload</button><progress id="bar" max="100" value="0" hidden></progress>
<p id="msg" class="hint" role="status"></p>
</div>
<div class="card"><b>Processing</b> <span id="state" class="hint"></span>
<button class="sec" id="proc">Process now</button><pre id="log"></pre>
<p class="hint">Results appear in the viewer: <a href="/">open the viewer</a>.</p></div>
</div>
<script>
const $=s=>document.querySelector(s);
fetch("/api/hitters").then(r=>r.json()).then(d=>{
  $("#session").value=d.today;$("#sessions").innerHTML=d.sessions.map(s=>`<option value="${s}">`).join("");
  $("#hitter").innerHTML=d.hitters.map(h=>`<option>${h}</option>`).join("")});
$("#go").onclick=()=>{
  const files=$("#files").files;if(!files.length){$("#msg").textContent="Choose a file first.";return}
  const fd=new FormData();for(const f of files)fd.append("files",f);
  fd.append("session",$("#session").value);fd.append("hitter",$("#hitter").value);
  fd.append("kind",document.querySelector("input[name=kind]:checked").value);
  const x=new XMLHttpRequest();x.open("POST","/api/upload");$("#bar").hidden=false;$("#go").disabled=true;
  x.upload.onprogress=e=>{if(e.lengthComputable)$("#bar").value=100*e.loaded/e.total};
  x.onload=()=>{$("#go").disabled=false;let d={};try{d=JSON.parse(x.responseText)}catch(e){}
    if(x.status===200){$("#msg").className="ok";$("#msg").textContent=`Saved ${d.saved.map(s=>s.name).join(", ")} for ${d.hitter||d.session}. Analysis starts in ${d.processing_in_s} s; it shows up in the viewer on the laptop.`;$("#files").value=""}
    else{$("#msg").className="err";$("#msg").textContent=d.error||("Upload failed ("+x.status+")")}};
  x.onerror=()=>{$("#go").disabled=false;$("#msg").className="err";$("#msg").textContent="Upload failed: is the laptop on the same Wi-Fi?"};
  x.send(fd)};
$("#proc").onclick=()=>fetch("/api/process",{method:"POST"});
function poll(){fetch("/api/status").then(r=>r.json()).then(s=>{$("#state").textContent=s.running?"· running…":(s.last_run?"· last run "+s.last_run:"");$("#log").textContent=s.log.join("\\n")}).catch(()=>{})}
poll();setInterval(poll,3000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inbox", default=None, help="where phone uploads go (default: <userdata>/inbox)")
    ap.add_argument("--watch", action="append", default=[],
                    help="also analyze new recordings in this shared folder (e.g. the synced Google Drive folder); repeatable")
    ap.add_argument("--userdata", default=None, help="profiles and swings (default: <repo>/userdata or SWINGLAB_USERDATA)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debounce", type=float, default=15.0, help="seconds without uploads before processing starts")
    a = ap.parse_args()
    import uvicorn
    app = create_app(a.inbox, a.userdata, debounce_s=a.debounce, watch=a.watch)
    app.state.port = a.port
    ud = Path(a.userdata or config.USERDATA)
    print(f"viewer:   http://localhost:{a.port}/\nphones:   http://{lan_ip()}:{a.port}/upload   (same Wi-Fi)\n"
          f"drop:     {ud / 'drop'} (one folder per hitter: put downloaded recordings in it)\n"
          f"userdata: {ud}" + "".join(f"\nwatching: {w}" for w in a.watch))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
