#!/usr/bin/env python3
"""Create an offline AM2Pro axis-test plan from a hand-held fixed-Tag probe.

The probe consists of camera translations while its orientation is held nearly
constant.  For a rigid camera/TCP fixture, those translation directions are
the same for the TCP, regardless of the still-unsettled camera-to-TCP rotation.

Safety: this script is planning only.  It never opens a serial port or sends
commands to the arm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS, URDF_PATH, load_joint_limits, pose_error,
)
from umi.common.cv_util import (  # noqa: E402
    detect_localize_aruco_tags, parse_aruco_config, parse_fisheye_intrinsics,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


JOINT_NAMES = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")
STAGES = {
    "forward": (5.0, 11.0),
    "up": (11.0, 17.0),
    "left": (17.0, 23.0),
}


def matrix_from_rvec_tvec(rvec, tvec):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()
    matrix[:3, 3] = np.asarray(tvec, dtype=float)
    return matrix


def read_trajectory(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["is_lost"].strip().lower() == "true":
                continue
            rows.append({key: float(row[key]) for key in ("timestamp", "x", "y", "z")})
    if not rows:
        raise ValueError("轨迹中没有可用的非丢失帧")
    return rows


def detect_reference_tag(reference_dir: Path, intrinsics_path: Path, aruco_path: Path, tag_id: int):
    image_path = reference_dir / "raw_camera.png"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取基准原始图像: {image_path}")
    intrinsics = parse_fisheye_intrinsics(json.loads(intrinsics_path.read_text(encoding="utf-8")))
    aruco = parse_aruco_config(yaml.safe_load(aruco_path.read_text(encoding="utf-8")))
    tags = detect_localize_aruco_tags(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), aruco["aruco_dict"],
        aruco["marker_size_map"], intrinsics)
    if tag_id not in tags:
        raise ValueError(f"V9 基准原始图像中未检测到世界 Tag {tag_id}")
    return {identifier: matrix_from_rvec_tvec(value["rvec"], value["tvec"])
            for identifier, value in tags.items()}


def solve_ik(backend, target, seeds, limits, steps=180, orientation_weight=0.35):
    best = None
    for seed in seeds:
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(steps):
            q = backend.inverse_kinematics(
                q, target, position_weight=1.0, orientation_weight=orientation_weight)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        pos_m, rot_deg = pose_error(target, actual)
        margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
        score = (pos_m / .005) ** 2 + (rot_deg / 3.0) ** 2 + max(0.0, 2.0 - margin) ** 2
        candidate = (score, q, pos_m, rot_deg, margin)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def main():
    parser = argparse.ArgumentParser(description="离线生成手持 TCP 轴向验证计划；不会控制机械臂。")
    parser.add_argument("--trajectory", required=True, help="固定 Tag 导出的 camera trajectory CSV")
    parser.add_argument("--start-reference", required=True, help="V9 reference.json")
    parser.add_argument("--hand-eye", required=True, help="新夹爪腕部相机手眼 JSON")
    parser.add_argument("--intrinsics", required=True, help="腕部相机内参 JSON")
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--world-tag-map", required=True)
    parser.add_argument("--world-tag-id", type=int, default=13)
    parser.add_argument("--axis-distance-m", type=float, default=.02)
    parser.add_argument("--wrist-flex-max-deg", type=float, default=85.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 < args.axis_distance_m <= .05:
        parser.error("axis-distance-m 必须在 (0, 0.05]；实体验证只允许最多 5 cm")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")

    trajectory = read_trajectory(Path(args.trajectory).expanduser().resolve())
    reference_path = Path(args.start_reference).expanduser().resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("tcp_frame") != "right_tcp":
        raise ValueError("基准不是当前新夹爪的 right_tcp 基准")
    q_ref = np.asarray(reference["robot_state"]["ActualQ"][:6], dtype=float)
    T_base_tcp = pose_to_mat(np.asarray(reference["robot_state"]["ActualTCPPose"], dtype=float))
    T_tcp_fixed_jaw = np.eye(4)
    T_tcp_fixed_jaw[2, 3] = -.236  # right_tcp is +236 mm along fixed-jaw z
    T_fixed_jaw_camera = np.asarray(
        json.loads(Path(args.hand_eye).expanduser().read_text(encoding="utf-8"))["tx_fixed_jaw2camera"], dtype=float)
    T_base_camera = T_base_tcp @ T_tcp_fixed_jaw @ T_fixed_jaw_camera
    reference_tags = detect_reference_tag(
        reference_path.parent, Path(args.intrinsics).expanduser(),
        Path(args.aruco_yaml).expanduser(), args.world_tag_id)
    T_camera_world = reference_tags[args.world_tag_id]
    T_base_world = T_base_camera @ T_camera_world

    world_map = json.loads(Path(args.world_tag_map).expanduser().read_text(encoding="utf-8"))
    tag_validation = None
    for raw_id, data in world_map.get("tags", {}).items():
        identifier = int(raw_id)
        if identifier == args.world_tag_id or identifier not in reference_tags:
            continue
        predicted = T_base_world @ np.asarray(data["T_world_tag"], dtype=float)
        measured = T_base_camera @ reference_tags[identifier]
        position_m, rotation_deg = pose_error(predicted, measured)
        tag_validation = {"tag_id": identifier, "position_error_mm": position_m * 1000,
                          "rotation_error_deg": rotation_deg}
        break

    baseline = np.median(np.asarray([[row["x"], row["y"], row["z"]]
                                     for row in trajectory if 1.0 <= row["timestamp"] <= 4.0]), axis=0)
    limits = load_joint_limits()
    limits[3, 1] = min(limits[3, 1], args.wrist_flex_max_deg)
    backend = create_kinematics_backend("ros2_dh", str(URDF_PATH), URDF_JOINTS, "right_tcp")
    rng = np.random.default_rng(20260910)
    result = {}
    print("HANDHELD_AXIS_PROBE_PLAN_STARTED")
    print("safety: offline only; no serial port or robot command will be used")
    for name, (start_s, end_s) in STAGES.items():
        samples = np.asarray([[row["x"], row["y"], row["z"]]
                              for row in trajectory if start_s <= row["timestamp"] <= end_s])
        if not len(samples):
            raise ValueError(f"{name} 阶段没有可用相机轨迹")
        deltas_world = samples - baseline
        measured_delta = deltas_world[np.argmax(np.linalg.norm(deltas_world, axis=1))]
        measured_norm = float(np.linalg.norm(measured_delta))
        if measured_norm < .01:
            raise ValueError(f"{name} 阶段位移仅 {measured_norm * 1000:.1f} mm，无法确定轴方向")
        direction_world = measured_delta / measured_norm
        delta_base = T_base_world[:3, :3] @ direction_world * args.axis_distance_m
        target = T_base_tcp.copy()
        target[:3, 3] += delta_base
        seeds = [q_ref]
        for j2, j3 in ((15., 15.), (30., 30.), (-15., 15.)):
            seed = q_ref.copy(); seed[1] += j2; seed[2] += j3
            seeds.append(np.clip(seed, limits[:, 0], limits[:, 1]))
        seeds.extend(rng.uniform(limits[:, 0], limits[:, 1], size=(12, 6)))
        _, q, position_m, rotation_deg, margin = solve_ik(backend, target, seeds, limits)
        accepted = position_m <= .005 and rotation_deg <= 3.0 and margin >= 2.0
        result[name] = {
            "recorded_peak_camera_translation_world_m": measured_delta.tolist(),
            "recorded_peak_distance_m": measured_norm,
            "normalized_world_direction": direction_world.tolist(),
            "planned_tcp_delta_base_m": delta_base.tolist(),
            "planned_distance_m": args.axis_distance_m,
            "target_tcp_pose_base": target[:3, 3].tolist() + Rotation.from_matrix(target[:3, :3]).as_rotvec().tolist(),
            "planned_joints_deg": q.tolist(),
            "ik_position_error_mm": position_m * 1000,
            "ik_rotation_error_deg": rotation_deg,
            "minimum_joint_margin_deg": margin,
            "accepted_for_low_speed_physical_test": accepted,
        }
        print(f"  {name}: observed={measured_norm * 1000:.1f} mm; "
              f"planned base delta={np.array2string(delta_base * 1000, precision=1)} mm; "
              f"IK={position_m * 1000:.2f} mm, margin={margin:.2f} deg; "
              f"{'PASS' if accepted else 'REJECT'}")
    report = {
        "schema": "am_umi_handheld_axis_probe_plan_v1",
        "safety": "offline only; no serial port or robot command was used",
        "purpose": "Validate camera/world translation directions against low-speed robot TCP movements.",
        "reference": str(reference_path), "trajectory": str(Path(args.trajectory).expanduser().resolve()),
        "world_tag_id": args.world_tag_id, "axis_distance_m": args.axis_distance_m,
        "joints_limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "tag_map_crosscheck": tag_validation,
        "axes": result,
        "accepted_axes": [name for name, value in result.items() if value["accepted_for_low_speed_physical_test"]],
        "limitations": [
            "This validates only translation axes while the hand-held camera orientation was approximately constant.",
            "It does not validate the camera-to-TCP rotation for hand-held pitch/roll motions.",
            "A PASS is a candidate for a supervised 2 cm physical dry-run, not permission for full replay.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("HANDHELD_AXIS_PROBE_PLAN_OK")
    print("accepted axes:", ", ".join(report["accepted_axes"]) or "none")
    print("report:", out)


if __name__ == "__main__":
    main()
