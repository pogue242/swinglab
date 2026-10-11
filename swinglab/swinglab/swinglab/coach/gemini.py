"""Gemini client: swing report, follow-up chat, video second opinion.

Uses the ``google-genai`` SDK. Set GEMINI_API_KEY (from Google AI Studio).
GEMINI_MODEL defaults to the pinned Flash model below (chosen 2026-10-10 from
``python -m swinglab.coach.gemini --list-models``); the key and model can live in
the git-ignored ``.env`` file. Google re-routes some names (gemini-3.5-flash was
served by 3.6, gemini-flash-latest by 3.8), so ``info["served_by"]`` records the
model that actually answered.

Free tier (checked 2026-10-10): **20 requests per day per model**
(GenerateRequestsPerDayPerProjectPerModel-FreeTier). Each quota is per model, so
a busy or used-up model (429/5xx) moves on to the next in FALLBACK_MODELS before
the template takes over. Thinking is set low: a coaching write-up from given
numbers needs little reasoning, and default thinking made some reports exceed
60 s (504) where low thinking took ~3 s.

Every call falls back to the template report (cues.py) on any failure, and
reports which path was used, so the UI can show "Gemini" vs "template".
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .. import config  # noqa: F401  (loads .env: GEMINI_API_KEY, GEMINI_MODEL)
from . import cues, numbers
from .schemas import SessionReport, SwingReport, VideoRead

PROMPTS = Path(__file__).parent / "prompts"
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-3.1-flash-lite,gemini-flash-lite-latest").split(",") if m.strip()]
THINKING = os.environ.get("GEMINI_THINKING", "low")   # "" = model default
RETRY_CODES = {429, 500, 503, 504}
TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", 60))   # a hung call must not stall the pipeline


def _prompt(name: str) -> str:
    """System prompt text. Always UTF-8: the Windows default (cp1252) garbles "—" and "→"."""
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


def _client():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(TIMEOUT_S * 1000)))


def _thinking():
    from google.genai import types
    return types.ThinkingConfig(thinking_level=THINKING) if THINKING else None


def _config(system: str, schema):
    from google.genai import types
    return types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                       response_schema=schema, temperature=0.4, thinking_config=_thinking())


def _generate(client, model: str, **kw):
    """generate_content on ``model``, then on each FALLBACK_MODELS entry while the answer is
    overload / quota / timeout (RETRY_CODES). Returns (response, model asked)."""
    chain = list(dict.fromkeys([model, *FALLBACK_MODELS]))
    for k, m in enumerate(chain):
        try:
            return _call(client, m, **kw), m
        except Exception as e:
            if getattr(e, "code", None) not in RETRY_CODES or k == len(chain) - 1:
                raise


def _call(client, model: str, **kw):
    try:
        return client.models.generate_content(model=model, **kw)
    except Exception as e:   # a model without thinking levels: ask again without one
        cfg = kw.get("config")
        if getattr(e, "code", None) == 400 and "think" in str(e).lower() and getattr(cfg, "thinking_config", None):
            kw["config"] = cfg.model_copy(update={"thinking_config": None})
            return client.models.generate_content(model=model, **kw)
        raise


def _info(resp, model: str, **extra) -> dict:
    return {"source": "gemini", "model": model, "served_by": getattr(resp, "model_version", None) or model, **extra}


def unescape(rep):
    """Gemini sometimes writes HTML entities ("3 &#215; 5"); return plain text in every string field."""
    import html
    def fix(o):
        if isinstance(o, str):
            return html.unescape(o)
        if isinstance(o, list):
            return [fix(x) for x in o]
        if isinstance(o, dict):
            return {k: fix(v) for k, v in o.items()}
        return o
    return type(rep).model_validate(fix(rep.model_dump()))


def validate_plan(rep: SwingReport, payload: dict) -> tuple[SwingReport, list[str]]:
    """Keep only exercises offered in the payload; fill from the fallback if too few remain."""
    allowed = {o["id"] for o in payload.get("exercise_options", [])}
    kept = [t for t in rep.training_plan if t.exercise_id in allowed]
    dropped = [t.exercise_id for t in rep.training_plan if t.exercise_id not in allowed]
    if len(kept) < 2 and allowed:
        kept += [t for t in cues.report(payload).training_plan if t.exercise_id not in {k.exercise_id for k in kept}]
    return rep.model_copy(update={"training_plan": kept[:5]}), dropped


def report(payload: dict, *, client=None, model: str = DEFAULT_MODEL) -> tuple[SwingReport, dict]:
    """Return (report, info). info["source"] is "gemini" or "template"."""
    client = client if client is not None else _client()
    t0 = time.time()
    if client is None:
        return cues.report(payload), {"source": "template", "reason": "no GEMINI_API_KEY"}
    try:
        resp, used = _generate(client, model, contents=json.dumps(payload),
                               config=_config(_prompt("report"), SwingReport))
        rep = resp.parsed if isinstance(resp.parsed, SwingReport) else SwingReport.model_validate_json(resp.text)
        rep = unescape(rep)
        rep, dropped = validate_plan(rep, payload)
        info = _info(resp, used, seconds=round(time.time() - t0, 2))
        if dropped:
            info["dropped_exercises"] = dropped
        bad = numbers.unsupported(rep, payload)
        if bad:   # flagged, not hidden: the UI and export_demo show it
            info["unsupported_numbers"] = bad
        return rep, info
    except Exception as e:  # network, quota, schema...
        return cues.report(payload), {"source": "template", "reason": f"{type(e).__name__}: {e}"[:300]}


def session_report(summary: dict, *, client=None, model: str = DEFAULT_MODEL) -> tuple[SessionReport, dict]:
    client = client if client is not None else _client()
    if client is None:
        return cues.session_report(summary), {"source": "template", "reason": "no GEMINI_API_KEY"}
    try:
        resp, used = _generate(client, model, contents=json.dumps(summary),
                               config=_config(_prompt("session"), SessionReport))
        rep = resp.parsed if isinstance(resp.parsed, SessionReport) else SessionReport.model_validate_json(resp.text)
        rep = unescape(rep)
        info = _info(resp, used)
        bad = numbers.unsupported(rep, summary)
        if bad:
            info["unsupported_numbers"] = bad
        return rep, info
    except Exception as e:
        return cues.session_report(summary), {"source": "template", "reason": f"{type(e).__name__}: {e}"[:300]}


def chat(payload: dict, history: list[dict], question: str, *, client=None, model: str = DEFAULT_MODEL) -> str:
    """One follow-up answer about one swing. ``history`` = [{"role": "user"|"model", "text": ...}].
    Never raises: on any failure the answer says the chat is unavailable."""
    client = client if client is not None else _client()
    if client is None:
        return "The coach chat needs a Gemini API key (set GEMINI_API_KEY)."
    try:
        from google.genai import types
        contents = [types.Content(role=h["role"], parts=[types.Part.from_text(text=h["text"])]) for h in history]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        system = _prompt("chat") + "\n\nPAYLOAD:\n" + json.dumps(payload)
        resp, _ = _generate(client, model, contents=contents,
                            config=types.GenerateContentConfig(system_instruction=system, temperature=0.5,
                                                                thinking_config=_thinking()))
        return resp.text
    except Exception as e:
        return f"The coach chat is unavailable right now ({type(e).__name__}); the written report still applies."


def video_read(video_path: str | Path, *, client=None, model: str = DEFAULT_MODEL) -> tuple[VideoRead | None, dict]:
    """Upload a short swing clip and ask for qualitative observations + contact time."""
    client = client if client is not None else _client()
    if client is None:
        return None, {"source": "none", "reason": "no GEMINI_API_KEY"}
    try:
        f = client.files.upload(file=str(video_path))
        for _ in range(60):  # videos are processed asynchronously
            f = client.files.get(name=f.name)
            if getattr(f.state, "name", str(f.state)) != "PROCESSING":
                break
            time.sleep(1)
        resp, used = _generate(client, model, contents=[f, "Analyze this swing."],
                               config=_config(_prompt("video_read"), VideoRead))
        rd = resp.parsed if isinstance(resp.parsed, VideoRead) else VideoRead.model_validate_json(resp.text)
        rd = unescape(rd)
        return rd, _info(resp, used)
    except Exception as e:
        return None, {"source": "none", "reason": f"{type(e).__name__}: {e}"[:300]}


if __name__ == "__main__":
    if "--list-models" in sys.argv:
        c = _client()
        if not c:
            sys.exit("set GEMINI_API_KEY first")
        for m in c.models.list():
            print(m.name)
    elif "--ping" in sys.argv:
        c = _client()
        if not c:
            sys.exit("set GEMINI_API_KEY first")
        t0 = time.time()
        r, used = _generate(c, DEFAULT_MODEL, contents="Reply with the word OK.")
        print(f"{r.text.strip()}  (asked {used}, served by {r.model_version}, {time.time() - t0:.1f} s)")
