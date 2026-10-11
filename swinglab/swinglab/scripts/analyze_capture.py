"""Analyze one recording (or a pair) without the inbox, into a hitter's profile.

  python scripts\\analyze_capture.py --lidar data\\captures\\<file.r3d> --profile Wilson
  python scripts\\analyze_capture.py --lidar <file.r3d> --slowmo <video.mov> --yardstick <yardstick.mov> --profile Wilson
  python scripts\\analyze_capture.py --lidar <file.r3d> --bats R --height 1.78     # no profile: saved under "unsorted"
  ... --bat-length 0.84 --tip-band 0.815 --hand-band 0.22     # where the tape is (m from the knob)

Every swing in the recording is analyzed (one or many); each is a hit or a dry swing, decided by
looking for a ball. Results go to <userdata>/swings/<profile>/ and show in the viewer
(python api\\server.py). Run `python scripts\\get_pose_model.py` once first.
"""
import argparse
import json

from swinglab import model, pipeline, profiles
from swinglab.ballflight import BatTape

ap = argparse.ArgumentParser()
ap.add_argument("--lidar")
ap.add_argument("--slowmo")
ap.add_argument("--profile", help="profile name or id (its bats/height/weight/age/level are used)")
ap.add_argument("--bats", choices=["R", "L"])
ap.add_argument("--height", type=float, help="meters")
ap.add_argument("--mass", type=float)
ap.add_argument("--age", type=float)
ap.add_argument("--level", choices=list(model.LEVELS))
ap.add_argument("--name", default="")
ap.add_argument("--slowmo-fps", type=float, default=240.0)
ap.add_argument("--m-per-px", type=float, help="scale in meters per pixel at the tee (overrides --yardstick)")
ap.add_argument("--yardstick", help="slow-motion clip of a yardstick held upright at the tee (scale); default: ball size")
ap.add_argument("--stick-length", type=float, default=0.9144, help="yardstick length, m (1 m stick: 1.0)")
ap.add_argument("--bat-length", type=float, help="bat length, m (default 0.84)")
ap.add_argument("--tip-band", type=float, help="center of the end tape band, m from the knob (default 0.815)")
ap.add_argument("--hand-band", type=float, help="center of the tape band above the hands, m from the knob (default 0.22)")
ap.add_argument("--no-gemini", action="store_true")
ap.add_argument("--no-save", action="store_true")
a = ap.parse_args()
if not (a.lidar or a.slowmo):
    ap.error("give --lidar and/or --slowmo")

prof = (profiles.get(a.profile) or profiles.by_name(a.profile)) if a.profile else None
if a.profile and prof is None:
    ap.error(f"no profile '{a.profile}' (create it in the viewer, or: python -c \"from swinglab import profiles; "
             f"profiles.create({{'name': '{a.profile}', 'bats': 'R', 'height_m': 1.78}})\")")
hitter = profiles.hitter(prof) if prof else {"name": a.name, "bats": "R", "height_m": 1.78, "level": "college"}
for k, v in (("bats", a.bats), ("height_m", a.height), ("mass_kg", a.mass), ("age", a.age), ("level", a.level)):
    if v is not None:
        hitter[k] = v
tape = BatTape(**{k: v for k, v in (("bat_length_m", a.bat_length), ("tip_band_m", a.tip_band),
                                    ("hand_band_m", a.hand_band)) if v is not None})
out = pipeline.analyze_recording(a.lidar, a.slowmo, hitter, use_gemini=not a.no_gemini, slowmo_fps=a.slowmo_fps,
                                 m_per_px=a.m_per_px, yardstick=a.yardstick, stick_length_m=a.stick_length, tape=tape)
folder = prof["id"] if prof else "unsorted"
print(f"{out['n_swings']} swing(s) found in the recording")
for rec in out["recs"]:
    print(f"\n== {rec.get('id')}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('bones', 'trace', 'kp2d')}
                      for k, v in (rec.get("stages") or {}).items()}, indent=1, default=str)[:3000])
    if "analysis" in rec:
        sc = rec["analysis"]["scores"]
        print(f"mode: {rec['mode']} ({rec['mode_evidence']})")
        print(f"scores: form {sc['form']} (power {sc['power']}, sequencing {sc['transfer']}, posture {sc['posture']}),"
              f" contact {sc['contact']}, overall {sc['overall']} | predicted bat speed {rec['analysis']['predicted']['bat_speed']} mph")
        print("coach:", rec["report"]["headline"], f"[{rec['report_source']['source']}]")
        for w in rec.get("warnings", []):
            print("  note:", w)
    s = rec.get("slowmo")
    if s:
        if s.get("yardstick"):
            print("yardstick:", s["yardstick"], f"-> {s['m_per_px'] * 1000:.3f} mm/px" if s.get("m_per_px") else "")
        print(f"slow-mo: contact frame {s['contact_frame']}, exit {s['exit_speed_mph']} mph, launch {s['launch_angle_deg']}°"
              f" (scale: {s.get('scale_source')}){' · ' + s['ball_note'] if s.get('ball_note') else ''}")
        if s.get("bat"):
            b = s["bat"]
            print(f"bat: {b['bat_speed_mph']} mph at the sweet spot (end band {b['tip_speed_mph']}), attack {b['attack_angle_deg']}°,"
                  f" bat angle {b['bat_angle_deg']}° · contact from {b['contact_from']} · bands found {b['bands_found_fraction']:.0%}")
        else:
            print("bat: not measured (no tape bands found, or no scale)")
    if "analysis" in rec and not a.no_save:
        if prof:
            rec["profile"] = prof["id"]
        print("saved", pipeline.save_result(rec, folder))
if out.get("summary") and not a.no_save:
    print("summary", pipeline.save_session(out["summary"], folder))
