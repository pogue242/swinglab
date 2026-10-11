"""Guard against invented numbers: every number Gemini writes must come from its payload.

Gemini explains; it never computes (HANDOFF goal 3). This check finds numbers in
a report's text that cannot be traced to the payload it was given.

A number in the text counts as supported when some payload number (a numeric
value, or a number inside a payload string such as "677 lab swings") equals it
after rounding to the precision written, ignoring sign ("14 degrees" for
-14.345), or after the two unit changes a coach naturally makes: ms -> s
(293 ms -> "0.29 s") and ratio -> percent (0.15 -> "15%").

Instruction fields (drill steps, prescriptions) may also hold small counts
("10 swings", "3 × 5", "80% effort") that are coaching, not measurements; numbers
up to INSTRUCTION_MAX are allowed there. Counting words 0–3 are allowed anywhere.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

NUMBER = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")
INSTRUCTION_FIELDS = {"steps", "prescription", "exercise_id"}
INSTRUCTION_MAX = 30
COUNTING = {0.0, 1.0, 2.0, 3.0}


def numbers_in(text: str) -> list[tuple[str, float]]:
    return [(m.group(0), float(m.group(0).replace(",", ""))) for m in NUMBER.finditer(text)]


def payload_numbers(obj) -> set[float]:
    out: set[float] = set()
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
    elif isinstance(obj, str):
        out |= {v for _, v in numbers_in(obj)}
    elif isinstance(obj, dict):
        for k, v in obj.items():   # keys too: "trend_per_10_swings" -> "per 10 swings"
            out |= payload_numbers(str(k).replace("_", " ")) | payload_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= payload_numbers(v)
    return out


def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def supported(tok: str, value: float, known: set[float], *, percent: bool = False) -> bool:
    tol = 0.5 * 10 ** -_decimals(tok) + 1e-9
    # whole numbers may also be written rounded to tens ("about 700 deg/s" for 709)
    candidates = list(known) + [k / 1000 for k in known] + ([k * 100 for k in known] if percent else [])
    if any(abs(value - k) <= tol for k in candidates):
        return True
    return _decimals(tok) == 0 and value >= 100 and value % 10 == 0 and any(abs(value - k) <= 5 for k in known)


def _texts(obj, path: str = ""):
    """(field path, text) for every string in a report."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _texts(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _texts(v, f"{path}[{i}]")


def unsupported(report, payload: dict) -> list[dict]:
    """Numbers in ``report``'s text that are not in ``payload``. Empty list = clean."""
    known = payload_numbers(payload)
    bad = []
    for path, text in _texts(report):
        leaf = re.sub(r"\[\d+\]", "", path).split(".")[-1]
        instruction = leaf in INSTRUCTION_FIELDS or ".steps" in path
        for m in NUMBER.finditer(text):
            tok = m.group(0)
            v = float(tok.replace(",", ""))
            if v in COUNTING or (instruction and v <= INSTRUCTION_MAX):
                continue
            percent = text[m.end():m.end() + 2].lstrip().startswith("%")
            if instruction and percent:
                continue   # "80% effort"
            if not supported(tok, v, known, percent=percent):
                bad.append({"field": path, "number": tok, "context": text[max(0, m.start() - 30):m.end() + 30]})
    return bad
