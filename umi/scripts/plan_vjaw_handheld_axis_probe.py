#!/usr/bin/env python3
"""Build a tiny physical V-jaw TCP axis probe from a guided hand trajectory.

This program is intentionally *offline*.  It turns the measured directions of
the guided forward/up/left hand motions into three small, fixed-orientation
right_tcp translations from a saved robot reference.  The resulting candidate
plan is accepted by ``am2pro_replay_retarget_plan.py`` but this program never
opens a serial port or commands a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_vjaw_hand_to_robot_axis import (  # noqa: E402
    JOINT_NAMES, STAGES, average_pose, pose_from_csv, solve_target,
)
from scripts.filter_am2pro_replayability import URDF_JOINTS, load_joint_limits, pose_error  # noqa: E402
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_joint_mapping import mapping_from_config, model_safe_limits_from_config  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


PROBE_STAGES = tuple(stage for stage in STAGES if stage[3] == "translation")


def peak_relative_pose(tcp_rows, hand_start, name, begin, end):
    candidates = []
    for timestamp, pose in tcp_rows:
        if begin <= timestamp <= end:
            relative = np.linalg.inv(hand_start) @ pose
            distance = float(np.linalg.norm(relative[:3, 3]))
            candidates.append((distance, timestamp, relative))
    if not candidates:
        raise ValueError(f"{name} 阶段没有可用轨迹帧")
    return max(candidates, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成离线 V-jaw 2cm 手持三轴实体探针；不会连接或控制机械臂。")
    parser.add_argument("--trajectory", required=True, help="guided session 的 camera_trajectory_multi_tag.csv")
    parser.add_argument("--camera-tcp-geometry", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--axis-distance-m", type=float, default=0.02,
                        help="每个轴从基准位移出的距离，默认 0.02m")
    parser.add_argument("--only-stage", choices=tuple(item[0] for item in PROBE_STAGES),
                        help="只生成指定单轴的 基准→探针→基准 计划，便于实体观察")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0.005 <= args.axis_distance_m <= 0.05:
        parser.error("--axis-distance-m 必须在 0.005 到 0.05m 之间")

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"拒绝覆盖已有文件：{out}")
    trajectory_path = Path(args.trajectory).expanduser().resolve()
    geometry_path = Path(args.camera_tcp_geometry).expanduser().resolve()
    reference_path = Path(args.start_reference).expanduser().resolve()
    config_path = Path(args.robot_config).expanduser().resolve()

    rows = []
    with trajectory_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("is_lost", "false").lower() != "true":
                rows.append((float(row["timestamp"]), pose_from_csv(row)))
    if not rows:
        parser.error("轨迹没有可用帧")
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry.get("schema") != "am_umi_camera_tcp_geometry_v1":
        parser.error("camera-TCP 几何格式不正确")
    T_camera_tcp = pose_to_mat(np.asarray(geometry["pose_cam_tcp"], dtype=float))
    tcp_rows = [(timestamp, T_world_camera @ T_camera_tcp) for timestamp, T_world_camera in rows]
    T_hand_start = average_pose([pose for timestamp, pose in tcp_rows if 3.0 <= timestamp <= 7.5])

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("tcp_frame") != "right_tcp":
        parser.error("reference 必须是 right_tcp 基准")
    q_start = np.asarray(reference["robot_state"]["ActualQ"][:6], dtype=float)
    T_robot_start = pose_to_mat(np.asarray(reference["robot_state"]["ActualTCPPose"], dtype=float))

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    mapping = mapping_from_config(robot, ROOT)
    limits = load_joint_limits(urdf_path, model_safe_limits_from_config(robot, mapping))
    backend = create_kinematics_backend(robot.get("ik_backend", "placo"), str(urdf_path),
                                        URDF_JOINTS, "right_tcp")
    rng = np.random.default_rng(20260914)

    path = [q_start.tolist()]
    probe_results = []
    print("VJAW_HANDHELD_AXIS_PROBE_PLAN_STARTED")
    print("safety: offline planning only; no serial port or robot command is used")
    print(f"probe translation: {args.axis_distance_m * 100:.1f} cm; tool orientation: fixed at reference")
    selected_stages = tuple(
        item for item in PROBE_STAGES if args.only_stage is None or item[0] == args.only_stage)
    for name, begin, end, _ in selected_stages:
        measured_distance, timestamp, measured_relative = peak_relative_pose(
            tcp_rows, T_hand_start, name, begin, end)
        if measured_distance < 0.005:
            raise RuntimeError(f"{name} 手持位移仅 {measured_distance * 100:.1f}cm，方向不可靠")
        scaled_relative = np.eye(4)
        scaled_relative[:3, 3] = (measured_relative[:3, 3] /
                                  measured_distance * args.axis_distance_m)
        target = T_robot_start @ scaled_relative
        _, q_target, pos_m, rot_deg, margins = solve_target(
            backend, target, q_start, limits, rng)
        actual = backend.forward_kinematics(q_target)
        actual_delta_base = actual[:3, 3] - T_robot_start[:3, 3]
        step_deg = float(np.max(np.abs(q_target - q_start)))
        limiting_index = int(np.argmin(margins))
        accepted = (pos_m <= 0.003 and rot_deg <= 2.0 and float(margins.min()) >= 5.0)
        if not accepted:
            raise RuntimeError(
                f"{name} 2cm 探针不可安全验收：IK {pos_m * 1000:.1f}mm / {rot_deg:.1f}deg，"
                f"最小余量 {margins.min():.1f}deg")
        path.extend([q_target.tolist(), q_start.tolist()])
        probe_results.append({
            "label": name,
            "hand_peak_timestamp_s": timestamp,
            "measured_hand_relative_translation_in_start_tcp_m": measured_relative[:3, 3].tolist(),
            "measured_hand_peak_distance_m": measured_distance,
            "requested_tcp_delta_in_start_tcp_m": scaled_relative[:3, 3].tolist(),
            "expected_tcp_delta_in_robot_base_m": actual_delta_base.tolist(),
            "target_joint_model_deg": q_target.tolist(),
            "max_joint_delta_from_reference_deg": step_deg,
            "ik_position_error_mm": pos_m * 1000.0,
            "ik_rotation_error_deg": rot_deg,
            "minimum_joint_margin_deg": float(margins.min()),
            "closest_joint": JOINT_NAMES[limiting_index],
        })
        print(f"  {name}: hand peak={measured_distance * 100:.1f}cm; "
              f"robot target Δbase={np.round(actual_delta_base * 100, 2)}cm; "
              f"max Δjoint={step_deg:.2f}deg; margin={margins.min():.1f}deg")

    report = {
        "schema": "am_umi_vjaw_taskspace_retarget_plan_v1",
        "safety": (
            "offline candidate only; gripper is not represented. Execute only as an empty-workspace "
            "physical direction diagnostic with a visible TCP ruler/marker."
        ),
        "status": "candidate",
        "kind": "handheld_axis_probe_fixed_orientation",
        "dataset": str(trajectory_path),
        "episode": 0,
        "reference": str(reference_path),
        "reference_created_at": reference.get("created_at"),
        "tcp_frame": "right_tcp",
        "model_safe_limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "parameters": {
            "fps": 1.0,
            "axis_distance_m": args.axis_distance_m,
            "camera_tcp_geometry": str(geometry_path),
            "geometry_status": geometry.get("status"),
            "hand_start_window_s": [3.0, 7.5],
            "tool_orientation": "fixed_at_robot_reference",
            "stage_order": [item["label"] for item in probe_results],
        },
        "probe_results": probe_results,
        "position_error_mm": {"median": float(np.median([item["ik_position_error_mm"] for item in probe_results])),
                              "max": float(np.max([item["ik_position_error_mm"] for item in probe_results]))},
        "orientation_deviation_deg": {"median": 0.0, "max": 0.0},
        "minimum_joint_margin_deg": float(min(item["minimum_joint_margin_deg"] for item in probe_results)),
        "max_joint_step_deg": float(max(item["max_joint_delta_from_reference_deg"] for item in probe_results)),
        "joint_path_model_deg": path,
        "limitations": [
            "This plan validates direction visually or with an external TCP marker; it does not automatically measure real TCP position.",
            "Each translation is derived from the guided hand trajectory but reduced to a fixed 2 cm pulse.",
            "Do not infer contact-task accuracy from this no-contact diagnostic.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_HANDHELD_AXIS_PROBE_PLAN_OK")
    print("plan:", out)


if __name__ == "__main__":
    main()
