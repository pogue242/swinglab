# SwingLab

3D baseball swing analysis from an iPhone Pro (video + LiDAR), trained on
Driveline's OpenBiomechanics lab swings, with a Gemini coach.

Start with **[docs/HANDOFF.md](docs/HANDOFF.md)**: scope, datasets, costs,
architecture, verification status and next steps.

```bash
pip install -e ".[phone,gemini,server,dev]"
python scripts/setup_data.py        # download Driveline data, validate, train, export viewer data
python -m pytest -q
python -m http.server -d web 8000   # viewer at http://localhost:8000
```

Data: Driveline OpenBiomechanics (CC BY-NC-SA 4.0 with exclusions; see their
LICENSE-DATA.md). Credit Driveline Baseball in any write-up.
