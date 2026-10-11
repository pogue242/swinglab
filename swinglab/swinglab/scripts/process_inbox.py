"""Analyze everything new in the shared SwingLab folder.

  python scripts\\process_inbox.py "G:\\My Drive\\SwingLab" --init      # create folders + example CSVs
  python scripts\\process_inbox.py "G:\\My Drive\\SwingLab"             # process new swings once
  python scripts\\process_inbox.py "G:\\My Drive\\SwingLab" --watch     # keep checking every 20 s
  python scripts\\process_inbox.py "G:\\My Drive\\SwingLab" --list      # show how files were paired

Each swing is saved in its hitter's profile (<userdata>/swings/<profile>/; profiles are created
from hitters.csv when missing). See them in the viewer: python api\server.py (or let the server do
all of this: python api\server.py --watch "G:\My Drive\SwingLab"). See docs/HANDOFF.md section 9b.
"""
import argparse
import time
from dataclasses import asdict

from swinglab import inbox

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--init", action="store_true")
ap.add_argument("--watch", action="store_true")
ap.add_argument("--list", action="store_true")
ap.add_argument("--force", action="store_true", help="re-analyze everything")
ap.add_argument("--interval", type=int, default=20)
a = ap.parse_args()

if a.init:
    inbox.template(a.root)
    print("created", a.root)
elif a.list:
    for p in inbox.scan(a.root):
        d = asdict(p)
        print(f"{p.session}/{p.swing:>4} [{p.method}] hitter={p.hitter} lidar={d['lidar'] and d['lidar'].split('/')[-1].split(chr(92))[-1]} "
              f"slowmo={d['slowmo'] and d['slowmo'].split('/')[-1].split(chr(92))[-1]} {' '.join(p.notes)}")
else:
    while True:
        done = inbox.process(a.root, force=a.force)
        if done or not a.watch:
            print(f"processed {len(done)} new swing(s)")
        if not a.watch:
            break
        a.force = False
        time.sleep(a.interval)
