#!/usr/bin/env python3
"""Analyse a guided V-jaw hand-to-robot axis probe without controlling hardware.

The input is the fixed-Tag camera trajectory recorded with the 48-second
guided axis script.  It converts camera motion through the *same* candidate
camera->right_tcp transform used for V-jaw demos, then asks the calibrated
robot model whether each deliberate motion is reachable from its reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS, load_joint_limits, pose_error,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    mapping_from_config, model_safe_limits_from_config,
)
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


JOINT_NAMES = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")
STAGES = (
    ("forward", 8.0, 16.0, "translation"),
    ("up", 16.0, 24.0, "translation"),
    ("left", 24.0, 32.0, "translation"),
    ("pitch_down", 32.0, 40.0, "rotation"),
)


def pose_from_csv(row: dict[str, str]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = [float(row[key]) for key in ("x", "y", "z")]
    matrix[:3, :3] = Rotation.from_quat(
        [float(row[key]) for key in ("q_x", "q_y", "q_z", "q_w")]).as_matrix()
    return matrix


def average_pose(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("没有可用于建立静止起点的轨迹帧")
    result = np.eye(4)
    result[:3, 3] = np.median(np.asarray([item[:3, 3] for item in matrices]), axis=0)
    result[:3, :3] = Rotation.from_matrix(
        np.asarray([item[:3, :3] for item in matrices])).mean().as_matrix()
    return result


def solve_target(backend, target, q_start, limits, rng):
    """Bounded multi-start solve; no controller or serial-port import."""
    seeds = [q_start]
    for _ in range(30):
        seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))
    best = None
    for seed in seeds:
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(200):
            q = backend.inverse_kinematics(q, target, position_weight=1.0,
                                            orientation_weight=0.35)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        pos_m, rot_deg = pose_error(target, actual)
        margin = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        score = pos_m / .010 + rot_deg / 8.0 + max(0.0, 5.0 - float(margin.min()))
        if best is None or score < best[0]:
            best = score, q, pos_m, rot_deg, margin
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="离线分析 V 型夹爪手持→机器人轴向诊断；不会控制机械臂。")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--camera-tcp-geometry", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")
    rows = []
    with Path(args.trajectory).expanduser().open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("is_lost", "false").lower() != "true":
                rows.append((float(row["timestamp"]), pose_from_csv(row)))
    if not rows:
        parser.error("轨迹没有可用帧")
    geometry = json.loads(Path(args.camera_tcp_geometry).expanduser().read_text(encoding="utf-8"))
    if geometry.get("schema") != "am_umi_camera_tcp_geometry_v1":
        parser.error("camera-TCP 几何格式不正确")
    T_camera_tcp = pose_to_mat(np.asarray(geometry["pose_cam_tcp"], dtype=float))
    tcp_rows = [(timestamp, T_world_camera @ T_camera_tcp) for timestamp, T_world_camera in rows]
    T_hand_start = average_pose([T for timestamp, T in tcp_rows if 3.0 <= timestamp <= 7.5])

    reference = json.loads(Path(args.start_reference).expanduser().read_text(encoding="utf-8"))
    if reference.get("tcp_frame") != "right_tcp":
        parser.error("reference 必须是 right_tcp 基准")
    state = reference["robot_state"]
    q_start = np.asarray(state["ActualQ"][:6], dtype=float)
    T_robot_start = pose_to_mat(np.asarray(state["ActualTCPPose"], dtype=float))
    config_path = Path(args.robot_config).expanduser()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    mapping = mapping_from_config(robot, ROOT)
    limits = load_joint_limits(urdf_path, model_safe_limits_from_config(robot, mapping))
    backend = create_kinematics_backend(robot.get("ik_backend", "placo"), str(urdf_path),
                                        URDF_JOINTS, "right_tcp")
    rng = np.random.default_rng(20260911)
    results = {}
    print("VJAW_HAND_TO_ROBOT_AXIS_ANALYSIS_STARTED")
    print("safety: offline only; no serial port or robot command is used")
    for name, begin, end, selection in STAGES:
        candidates = []
        for timestamp, T_hand in tcp_rows:
            if begin <= timestamp <= end:
                T_relative = np.linalg.inv(T_hand_start) @ T_hand
                translation = float(np.linalg.norm(T_relative[:3, 3]))
                rotation = float(np.degrees(Rotation.from_matrix(T_relative[:3, :3]).magnitude()))
                candidates.append((translation if selection == "translation" else rotation,
                                   timestamp, T_relative, translation, rotation))
        if not candidates:
            parser.error(f"{name} 阶段没有轨迹帧")
        # The preceding translation stage may spill a little into the next
        # 8-second slot.  A pitch probe is meaningful only after the operator
        # has returned near the starting point, so do not accidentally call a
        # leftover leftward translation a "pitch" failure.
        selection_constrained_to_near_start = False
        if selection == "rotation":
            near_start = [item for item in candidates if item[3] <= 0.06]
            if near_start:
                candidates = near_start
                selection_constrained_to_near_start = True
        _, timestamp, relative, translation, rotation = max(candidates, key=lambda item: item[0])
        target = T_robot_start @ relative
        _, q, pos_m, rot_deg, margins = solve_target(backend, target, q_start, limits, rng)
        reachable = pos_m <= .010 and rot_deg <= 8.0 and float(margins.min()) >= 5.0
        limiting_index = int(np.argmin(margins))
        results[name] = {
            "source_peak_timestamp_s": timestamp,
            "hand_relative_translation_in_start_tcp_m": relative[:3, 3].tolist(),
            "hand_relative_distance_m": translation,
            "hand_relative_rotation_deg": rotation,
            "selection_constrained_to_near_start": selection_constrained_to_near_start,
            "predicted_model_joint_deg": q.tolist(),
            "predicted_joint_margin_each_deg": {joint: float(margins[i]) for i, joint in enumerate(JOINT_NAMES)},
            "closest_joint": JOINT_NAMES[limiting_index],
            "ik_position_error_mm": pos_m * 1000.0,
            "ik_rotation_error_deg": rot_deg,
            "reachable_with_5deg_margin": bool(reachable),
        }
        print(f"{name}: hand Δ={translation * 100:.1f}cm / {rotation:.1f}deg; "
              f"IK={pos_m * 1000:.1f}mm; closest={JOINT_NAMES[limiting_index]} "
              f"margin={margins[limiting_index]:.1f}deg; "
              f"{'PASS' if reachable else 'REJECT'}")
    report = {
        "schema": "am_umi_vjaw_hand_to_robot_axis_analysis_v1",
        "safety": "offline only; no serial port or robot command was used",
        "trajectory": str(Path(args.trajectory).expanduser().resolve()),
        "geometry": str(Path(args.camera_tcp_geometry).expanduser().resolve()),
        "geometry_status": geometry.get("status"),
        "reference": str(Path(args.start_reference).expanduser().resolve()),
        "model_safe_limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "stages": results,
        "interpretation": (
            "A rejected small axis motion means the candidate hand TCP/right_tcp transform "
            "or the robot model/workspace still needs verification; it is not an authority to widen motor limits."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_HAND_TO_ROBOT_AXIS_ANALYSIS_OK")
    print("report:", out)


if __name__ == "__main__":
    main()
