"""One row of swing-mechanics features per swing, computed from the canonical skeleton.

Every feature is something the phone pipeline can also compute (body joints
only, no bat markers). Distances are divided by hitter height.

Groups map to the three form scores (together: the Form score):
  POWER     rotation range and speed, separation, stride
  TRANSFER  ("Sequencing") kinematic sequence (pelvis -> torso -> hands) timing and gains
  POSTURE   body position at the moment of contact: lead-leg block, trunk, head/hips drift, hands
            (renamed from "contact" 2026-10-10: it measures the body, not the ball; ball contact
            quality is a separate score, see scoring.contact_score)
"""

from __future__ import annotations

import numpy as np

from . import kinematics as K
from .skeleton import Skeleton

FEATURES = {
    # name: (group, plain-English label, unit). Groups: power, transfer (sequencing), posture.
    "stance_width": ("power", "Stance width", "ratio"),
    "stride_length": ("power", "Stride length", "ratio"),
    "separation_fp": ("power", "Hip–shoulder separation at foot plant", "deg"),
    "separation_max": ("power", "Biggest hip–shoulder separation", "deg"),
    "pelvis_yaw_fp": ("power", "Hips already open at foot plant", "deg"),
    "pelvis_rotation_range": ("power", "Hip turn, load → contact", "deg"),
    "pelvis_vel_peak": ("power", "Top hip rotation speed", "deg/s"),
    "torso_vel_peak": ("power", "Top chest rotation speed", "deg/s"),
    "hand_speed_peak": ("power", "Top hand speed", "m/s"),
    "seq_pelvis_to_torso_ms": ("transfer", "Hips peak → chest peak", "ms"),
    "seq_torso_to_hands_ms": ("transfer", "Chest peak → hands peak", "ms"),
    "gain_torso_over_pelvis": ("transfer", "Speed gain, hips → chest", "ratio"),
    "seq_order_ok": ("transfer", "Hips → chest → hands in order", "bool"),
    "fp_to_contact_ms": ("transfer", "Foot plant → contact time", "ms"),
    "torso_yaw_fp": ("transfer", "Shoulders already open at foot plant", "deg"),
    "lead_knee_flex_fp": ("posture", "Front knee bend at foot plant", "deg"),
    "lead_knee_flex_contact": ("posture", "Front knee bend at contact", "deg"),
    "lead_knee_extension": ("posture", "Front leg straightening, plant → contact", "deg"),
    "trunk_forward_tilt_contact": ("posture", "Chest tilt over the plate at contact", "deg"),
    "trunk_lateral_tilt_contact": ("posture", "Upper body lean back at contact", "deg"),
    "head_drift_fp_contact": ("posture", "Head drift toward the pitcher, plant → contact", "ratio"),
    "hips_drift_fp_contact": ("posture", "Hips slide toward the pitcher, plant → contact", "ratio"),
    "hands_forward_contact": ("posture", "Hands out in front at contact", "ratio"),
}
GROUPS = ("power", "transfer", "posture")
GROUP_LABELS = {"power": "Power", "transfer": "Sequencing", "posture": "Posture"}
GROUP_HELP = {
    "power": "How much your body turns and how fast: the engine of bat speed.",
    "transfer": "Whether the turn passes from hips to chest to hands in order, each faster than the last.",
    "posture": "Your body position when the bat reaches the ball: a firm front leg, a stable head and hips, hands out front.",
}

# What each measurement means, in plain words, and which way the lab data points.
# better: "higher" / "lower" = the top third of lab hitters (by size-adjusted bat speed) sit there;
# "range" = no clear direction, stay inside the green band. pct_height: shown as % of your height.
FEATURE_HELP = {
    "stance_width": ("Distance between your feet before you start, as a share of your height.", "range", True),
    "stride_length": ("How far your front foot travels toward the pitcher, as a share of your height.", "range", True),
    "separation_fp": ("How far your hips have turned ahead of your shoulders when your front foot lands. "
                      "This stretch across your core stores energy.", "range", False),
    "separation_max": ("The most your hips get ahead of your shoulders during the swing.", "higher", False),
    "pelvis_yaw_fp": ("How much your hips have already turned toward the pitcher when the front foot lands.", "range", False),
    "pelvis_rotation_range": ("How far your hips turn from the load to contact.", "higher", False),
    "pelvis_vel_peak": ("The fastest your hips spin, in degrees per second.", "higher", False),
    "torso_vel_peak": ("The fastest your chest and shoulders spin, in degrees per second.", "higher", False),
    "hand_speed_peak": ("The fastest your hands move. Strongly tied to bat speed.", "higher", False),
    "seq_pelvis_to_torso_ms": ("Time from your hips' top speed to your chest's top speed. Positive = hips first, "
                               "as in an efficient swing; zero or negative = they spin together.", "range", False),
    "seq_torso_to_hands_ms": ("Time from your chest's top speed to your hands' top speed.", "range", False),
    "gain_torso_over_pelvis": ("Chest top speed divided by hip top speed. Above 1 means the speed builds as it "
                               "passes up the body.", "range", False),
    "seq_order_ok": ("Whether hips, chest and hands reach top speed in that order.", "range", False),
    "fp_to_contact_ms": ("Time from your front foot landing to contact.", "range", False),
    "torso_yaw_fp": ("How much your shoulders have turned toward the pitcher when the front foot lands. "
                     "Negative = still closed, which is good.", "range", False),
    "lead_knee_flex_fp": ("How bent your front knee is when it lands (0° = straight).", "range", False),
    "lead_knee_flex_contact": ("How bent your front knee still is at contact (0° = straight). A firmer front "
                               "leg stops your body and whips the bat.", "lower", False),
    "lead_knee_extension": ("How much your front leg straightens between landing and contact.", "higher", False),
    "trunk_forward_tilt_contact": ("How much your chest leans over the plate at contact.", "higher", False),
    "trunk_lateral_tilt_contact": ("How much your upper body leans back, away from the pitcher, at contact.", "range", False),
    "head_drift_fp_contact": ("How far your head moves toward the pitcher between foot plant and contact, as a "
                              "share of your height. Less movement = steadier eyes.", "range", True),
    "hips_drift_fp_contact": ("How far your hips slide toward the pitcher between foot plant and contact, as a "
                              "share of your height. Turning, not sliding, is the goal.", "range", True),
    "hands_forward_contact": ("How far in front of your body your hands are at contact, as a share of your "
                              "height. Out front means meeting the ball early, not letting it get deep.", "higher", True),
}


def compute(sk: Skeleton, ev: dict | None = None, s: dict | None = None) -> dict[str, float]:
    s = s or K.series(sk)
    ev = ev or K.detect_events(sk, s)
    h = sk.height_m or float(np.nanmax(sk["neck"][:, 2]) / 0.82)  # neck ≈ 82% of height
    i_load, i_fp, i_c = K.idx(sk, ev["load"]), K.idx(sk, ev["foot_plant"]), K.idx(sk, ev["contact"])
    i_c = max(i_c, i_fp + 1)
    w = slice(0, max(3, int(0.08 * len(sk.time))))
    win = slice(i_load, i_c + 1)
    f: dict[str, float] = {}
    f["stance_width"] = float(np.linalg.norm(np.nanmean(sk["lead_ankle"][w] - sk["rear_ankle"][w], 0)) / h)
    f["stride_length"] = float((sk["lead_ankle"][i_fp, 0] - np.nanmean(sk["lead_ankle"][w, 0])) / h)
    f["separation_fp"] = float(s["separation"][i_fp])
    f["separation_max"] = float(np.nanmax(s["separation"][win]))
    f["pelvis_yaw_fp"] = float(s["pelvis_yaw"][i_fp])
    f["torso_yaw_fp"] = float(s["torso_yaw"][i_fp])
    f["pelvis_rotation_range"] = float(s["pelvis_yaw"][i_c] - np.nanmin(s["pelvis_yaw"][: i_fp + 1]))
    # kinematic sequence: peaks in the load -> contact window
    pv, tv, hs = s["pelvis_vel"][win], s["torso_vel"][win], s["hand_speed"][win]
    ip, it, ih = int(np.nanargmax(pv)), int(np.nanargmax(tv)), int(np.nanargmax(hs))
    f["pelvis_vel_peak"] = float(pv[ip])
    f["torso_vel_peak"] = float(tv[it])
    f["hand_speed_peak"] = float(hs[ih])
    dt = 1000.0 / sk.fps
    tp, tt, th = ip + K.peak_offset(pv, ip), it + K.peak_offset(tv, it), ih + K.peak_offset(hs, ih)
    f["seq_pelvis_to_torso_ms"] = float((tt - tp) * dt)
    f["seq_torso_to_hands_ms"] = float((th - tt) * dt)
    f["gain_torso_over_pelvis"] = float(tv[it] / pv[ip]) if pv[ip] > 1 else np.nan
    f["seq_order_ok"] = float(ip <= it <= ih)
    f["fp_to_contact_ms"] = float((ev["contact"] - ev["foot_plant"]) * 1000)
    f["lead_knee_flex_fp"] = float(s["lead_knee_flex"][i_fp])
    f["lead_knee_flex_contact"] = float(s["lead_knee_flex"][i_c])
    f["lead_knee_extension"] = f["lead_knee_flex_fp"] - f["lead_knee_flex_contact"]
    f["trunk_forward_tilt_contact"] = float(s["trunk_forward_tilt"][i_c])
    f["trunk_lateral_tilt_contact"] = float(s["trunk_lateral_tilt"][i_c])
    f["head_drift_fp_contact"] = float((sk["neck"][i_c, 0] - sk["neck"][i_fp, 0]) / h)
    f["hips_drift_fp_contact"] = float((sk.mid_hip[i_c, 0] - sk.mid_hip[i_fp, 0]) / h)
    f["hands_forward_contact"] = float((sk.mid_wrist[i_c, 0] - sk.mid_hip[i_c, 0]) / h)
    return f
