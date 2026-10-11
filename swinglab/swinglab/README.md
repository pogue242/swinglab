# SwingLab

3D baseball swing analysis from an iPhone Pro (video + LiDAR), trained on
Driveline's OpenBiomechanics lab swings, with a Gemini coach.

Start with **[docs/SPEC.md](docs/SPEC.md)** (scope, architecture, status) and
**[docs/FILES.md](docs/FILES.md)** (what every file does).

```bash
pip install -e ".[phone,gemini,server,dev]"
python scripts/setup_data.py        # download Driveline data, validate, train, export viewer data
python -m pytest -q
python -m http.server -d web 8000   # viewer at http://localhost:8000
```

Data: Driveline OpenBiomechanics (CC BY-NC-SA 4.0 with exclusions; see their
LICENSE-DATA.md). Credit Driveline Baseball in any write-up.
