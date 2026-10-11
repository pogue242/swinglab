"""SwingLab: 3D baseball swing analysis from phone video + LiDAR, trained on Driveline OpenBiomechanics."""

import sys

__version__ = "0.1.0"

# Windows: when output is piped or redirected (logs, CI, subprocess), Python writes
# with the ANSI code page (cp1252), and printing "→", "✓" or a Gemini headline with
# an emoji raises UnicodeEncodeError. Write UTF-8 instead; real consoles are unaffected.
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure") and (getattr(_s, "encoding", "") or "").lower().replace("-", "") != "utf8":
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
