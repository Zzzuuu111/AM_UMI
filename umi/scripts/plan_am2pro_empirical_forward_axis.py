#!/usr/bin/env python3
"""Build one physical-forward candidate from observed AM2Pro axis tests.

Given a tested vector that truly raised the TCP and a tested vector that truly
moved it left, the tabletop-forward candidate is left x up.  This is an
empirical robot-frame calibration step, not a hand-held replay plan.
It never opens a serial port or controls hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import URDF_JOINTS, URDF_PATH, load_joint_limits, pose_error  # noqa: E402
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


def solve(backend, target, seeds, limits):
    best = None
    for seed in seeds:
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(180):
            q = backend.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=.35)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        pos_m, rot_deg = pose_error(target, actual)
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        score = (pos_m / .005) ** 2 + (rot_deg / 3.) ** 2 + max(0, 2 - margin) ** 2
        if best is None or score < best[0]:
            best = score, q, pos_m, rot_deg, margin
    return best


def unit(vector, name):
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError(f"{name} 是零向量")
    return value / norm


def main():
    parser = argparse.ArgumentParser(description="由实体观察到的上、左方向，离线生成桌面前方候选。")
    parser.add_argument("--observed-plan", required=True,
                        help="已执行的 4 cm 轴向计划 JSON")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--distance-m", type=float, default=.04)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 < args.distance_m <= .06:
        parser.error("distance-m 必须在 (0, 0.06] 内")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")
    observed = json.loads(Path(args.observed_plan).expanduser().read_text(encoding="utf-8"))
    # User observed: the prior 'forward' test physically lifted the TCP, and
    # 'left' physically moved it left.  In a right-handed table frame:
    # forward = left x up.
    up_base = unit(observed["axes"]["forward"]["planned_tcp_delta_base_m"], "observed up")
    left_base = unit(observed["axes"]["left"]["planned_tcp_delta_base_m"], "observed left")
    forward_base = unit(np.cross(left_base, up_base), "left cross up")
    reference_path = Path(args.reference).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("tcp_frame") != "right_tcp":
        raise ValueError("reference 不是 right_tcp")
    q_ref = np.asarray(reference["robot_state"]["ActualQ"][:6], dtype=float)
    target = pose_to_mat(np.asarray(reference["robot_state"]["ActualTCPPose"], dtype=float))
    target[:3, 3] += forward_base * args.distance_m
    limits = load_joint_limits(); limits[3, 1] = min(limits[3, 1], 85.)
    rng = np.random.default_rng(20260910)
    seeds = [q_ref]
    for j2, j3 in ((20, 20), (35, 35), (-15, 15)):
        seed = q_ref.copy(); seed[1] += j2; seed[2] += j3
        seeds.append(np.clip(seed, limits[:, 0], limits[:, 1]))
    seeds.extend(rng.uniform(limits[:, 0], limits[:, 1], size=(16, 6)))
    backend = create_kinematics_backend("ros2_dh", str(URDF_PATH), URDF_JOINTS, "right_tcp")
    _, q, pos_m, rot_deg, margin = solve(backend, target, seeds, limits)
    accepted = pos_m <= .005 and rot_deg <= 3 and margin >= 2
    report = {
        "schema": "am_umi_am2pro_empirical_forward_axis_v1",
        "safety": "offline only; no serial port or robot command was used",
        "reference": str(reference_path), "source_observed_plan": str(Path(args.observed_plan).expanduser().resolve()),
        "basis_observation": {"up_was_plan_axis": "forward", "left_was_plan_axis": "left",
                              "forward_construction": "normalized(left cross up)"},
        "axis_distance_m": args.distance_m,
        "axes": {"forward": {
            "planned_tcp_delta_base_m": (forward_base * args.distance_m).tolist(),
            "planned_distance_m": args.distance_m,
            "target_tcp_pose_base": target[:3, 3].tolist(),
            "planned_joints_deg": q.tolist(), "ik_position_error_mm": pos_m * 1000,
            "ik_rotation_error_deg": rot_deg, "minimum_joint_margin_deg": margin,
            "accepted_for_low_speed_physical_test": accepted,
        }},
        "limitations": [
            "The sign of this forward candidate is empirically inferred; if it moves backward, use its exact inverse.",
            "This candidate only calibrates a table-forward direction and is not a complete hand-held-to-robot transform.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("EMPIRICAL_FORWARD_AXIS_PLAN_" + ("OK" if accepted else "REJECT"))
    print("TCP delta base [mm]:", np.round(forward_base * args.distance_m * 1000, 2))
    print(f"IK position={pos_m * 1000:.3f} mm rotation={rot_deg:.3f} deg margin={margin:.2f} deg")
    print("report:", out)


if __name__ == "__main__":
    main()
