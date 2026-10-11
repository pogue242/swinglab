#!/usr/bin/env bash
# One-click local SwingLab: install, then open the viewer at http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/swinglab/swinglab"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "Install Python 3.10 or newer, then run this again."
  exit 1
fi
"$PY" -c "import sys; raise SystemExit(0 if sys.version_info>=(3,10) else 1)" || {
  echo "Python 3.10 or newer is required. This one is $("$PY" --version)."
  exit 1
}

if [[ ! -d .venv ]]; then
  echo "Creating a Python environment..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
echo "Installing SwingLab..."
python -m pip install -e ".[phone,gemini,server]"
[[ -f data/models/pose_landmarker_heavy.task ]] || python scripts/get_pose_model.py
[[ -f data/models/object_detection_yolox_2022nov.onnx ]] || python scripts/get_ball_model.py
[[ -f .env ]] || cp .env.example .env

echo
echo "Opening http://localhost:8000"
echo "Coach reports use a template until you put a Gemini key in swinglab/swinglab/.env"
echo "Press Ctrl+C to stop SwingLab."
( sleep 3
  if command -v open >/dev/null 2>&1; then open "http://localhost:8000"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:8000"
  fi
) >/dev/null 2>&1 &
exec python api/server.py
