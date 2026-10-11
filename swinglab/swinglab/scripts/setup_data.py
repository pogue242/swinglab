"""One command from a fresh clone to a working model + viewer.

  python scripts/setup_data.py
    1. download Driveline hitting data (pure Python, checksum-verified, ~220 MB zipped; no login)
    2. build the landmark parquet
    3. validate event detection against force plates
    4. build features at phone fps (and 360 Hz for validation)
    5. train + cross-validate
    6. export the demo swings and build the single-file viewer
    7. write data/sample/driveline_sample.xlsx (open in Excel to look at the data)
"""
import subprocess
import sys

from swinglab import config, obp

steps = [
    ("download", lambda: obp.download(("hitting",))),
    ("parquet", obp.build_landmarks_parquet),
]
for name, fn in steps:
    print(f"== {name}")
    fn()
for script, args in [("validate_events.py", []), ("build_dataset.py", ["--fps", str(int(config.PHONE_FPS))]),
                     ("build_dataset.py", ["--fps", "360"]), ("validate_features.py", []), ("train.py", []),
                     ("export_demo.py", []), ("build_web.py", []),
                     ("make_data_sample.py", [])]:
    print(f"== {script} {' '.join(args)}")
    subprocess.run([sys.executable, str(config.ROOT / "scripts" / script), *args], check=True)
print("done. run tests: python -m pytest")
