#!/usr/bin/env python3
"""Rank coarse camera-to-TCP rotation candidates without commanding a robot.

The fixed-point pivot calibration determines the TCP *translation* accurately,
but cannot determine its orientation.  This tool preserves that translation
and tries the 24 right-handed signed camera-axis permutations as a first
coarse search.  Every candidate is evaluated through the same body-relative,
hardware-bounded AM2Pro IK path used by the offline replay gate.

It is diagnostic only: it opens no serial port and sends no robot command.
The winning candidates still require a short, slow physical validation before
being adopted as a camera-to-TCP calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS,
    URDF_PATH,
    load_joint_limits,
    load_reference,
    pose_error,
    solve_target,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


AXIS_NAMES = ("cam_x", "cam_y", "cam_z")


def signed_axis_rotations():
    """Yield all 24 proper signed-permutation rotation matrices."""
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            matrix = np.zeros((3, 3), dtype=float)
            # Column j is TCP axis j expressed in the camera frame.
            for tcp_axis, camera_axis in enumerate(permutation):
                matrix[camera_axis, tcp_axis] = signs[tcp_axis]
            if np.linalg.det(matrix) > 0.5:
                label = ", ".join(
                    f"tcp_{'xyz'[j]}={'+' if signs[j] > 0 else '-'}{AXIS_NAMES[permutation[j]]}"
                    for j in range(3))
                yield label, matrix


def pair_frame_rotation(detection_path: Path, left_tag_id: int, right_tag_id: int) -> np.ndarray:
    """Estimate a stable gripper frame from the two finger ArUco poses.

    Pair x points from right-tag centre to left-tag centre. Pair z is the
    averaged tag-plane normal, and pair y completes the right-handed frame.
    The remaining mapping from these physical pair axes to TCP axes is still
    searched explicitly by the caller.
    """
    with detection_path.open("rb") as file:
        frames = pickle.load(file)
    rotations = []
    for frame in frames:
        tags = frame.get("tag_dict", {})
        if left_tag_id not in tags or right_tag_id not in tags:
            continue
        left, right = tags[left_tag_id], tags[right_tag_id]
        p_left = np.asarray(left["tvec"], dtype=float)
        p_right = np.asarray(right["tvec"], dtype=float)
        pair_x = p_left - p_right
        norm = np.linalg.norm(pair_x)
        if norm < 1e-6:
            continue
        pair_x /= norm
        normal = (Rotation.from_rotvec(np.asarray(left["rvec"], dtype=float)).as_matrix()[:, 2] +
                  Rotation.from_rotvec(np.asarray(right["rvec"], dtype=float)).as_matrix()[:, 2])
        normal -= pair_x * float(pair_x @ normal)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        pair_z = normal / norm
        pair_y = np.cross(pair_z, pair_x)
        pair_y /= np.linalg.norm(pair_y)
        rotations.append(np.column_stack((pair_x, pair_y, pair_z)))
    if len(rotations) < 20:
        raise ValueError("Too few frames with both finger tags to estimate pair frame")
    return Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()


def load_camera_poses(path: Path) -> np.ndarray:
    table = pd.read_csv(path)
    required = {"is_lost", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"trajectory CSV missing columns: {missing}")
    lost = table["is_lost"].astype(str).str.lower().isin(["true", "1"])
    if bool(lost.any()):
        raise ValueError(f"trajectory contains {int(lost.sum())} lost frames")
    poses = np.tile(np.eye(4), (len(table), 1, 1))
    poses[:, :3, :3] = Rotation.from_quat(
        table[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)).as_matrix()
    poses[:, :3, 3] = table[["x", "y", "z"]].to_numpy(float)
    return poses


def evaluate(label, rotation, camera_poses, translation, backend, q_start, t_start,
             limits, args, sample_indices):
    tx_cam_tcp = np.eye(4)
    tx_cam_tcp[:3, :3] = rotation
    tx_cam_tcp[:3, 3] = translation
    tcp = camera_poses[sample_indices] @ tx_cam_tcp
    rel0_inv = np.linalg.inv(tcp[0])
    q_prev = q_start.copy()
    rng = np.random.default_rng(20260909)
    pos_errors, rot_errors, margins, j2_at_lower = [], [], [], 0
    for pose in tcp:
        target = t_start @ (rel0_inv @ pose)
        q_prev, pos_error, rot_error = solve_target(
            backend, target, q_prev, limits, args, rng)
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)
        margin_each = np.minimum(q_prev - limits[:, 0], limits[:, 1] - q_prev)
        margins.append(float(np.min(margin_each)))
        j2_at_lower += int(q_prev[1] <= limits[1, 0] + 1.0)
    pos_errors = np.asarray(pos_errors)
    rot_errors = np.asarray(rot_errors)
    margins = np.asarray(margins)
    bad = ((pos_errors > args.max_position_error_m) |
           (rot_errors > args.max_rotation_error_deg))
    return {
        "candidate_index": None,
        "label": label,
        "rotation_matrix_camera_tcp": rotation.tolist(),
        "sampled_frames": int(len(sample_indices)),
        "bad_ik_frames": int(bad.sum()),
        "bad_ik_ratio": float(bad.mean()),
        "j2_near_lower_frames": int(j2_at_lower),
        "j2_near_lower_ratio": float(j2_at_lower / len(sample_indices)),
        "position_error_mm": {
            "median": float(np.median(pos_errors) * 1000),
            "max": float(np.max(pos_errors) * 1000),
        },
        "rotation_error_deg": {
            "median": float(np.median(rot_errors)),
            "max": float(np.max(rot_errors)),
        },
        "min_joint_margin_deg": float(np.min(margins)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="离线搜索手持相机→TCP 的粗旋转候选；不会连接或控制机械臂。")
    parser.add_argument("--trajectory", required=True,
                        help="固定 Tag 相机轨迹 CSV，例如 camera_trajectory_multi_tag_refined.csv")
    parser.add_argument("--geometry", required=True,
                        help="现有、已接受的 camera-to-TCP JSON；只使用其平移")
    parser.add_argument("--start-reference", required=True,
                        help="AM2Pro right_tcp 的 reference.json")
    parser.add_argument("--out", required=True, help="新的 JSON 报告路径，不能已存在")
    parser.add_argument("--frame-stride", type=int, default=30,
                        help="每隔多少原始帧评估一次；默认约每秒一帧")
    parser.add_argument("--ik-steps", type=int, default=100)
    parser.add_argument("--random-restarts", type=int, default=4)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--max-position-error-m", type=float, default=0.010)
    parser.add_argument("--max-rotation-error-deg", type=float, default=8.0)
    parser.add_argument("--candidate-indices", default=None,
                        help="只评估逗号分隔的候选编号（1--24），用于粗筛后的复核")
    parser.add_argument("--finger-tag-detection", default=None,
                        help="可选 tag_detection.pkl；提供后以双指 Tag 实测坐标系作为旋转搜索基底")
    parser.add_argument("--left-finger-tag-id", type=int, default=0)
    parser.add_argument("--right-finger-tag-id", type=int, default=1)
    args = parser.parse_args()
    if args.frame_stride <= 0 or args.ik_steps <= 0 or args.random_restarts < 0:
        parser.error("frame-stride、ik-steps 必须为正，random-restarts 不能为负")
    try:
        selected_indices = (None if args.candidate_indices is None else {
            int(item.strip()) for item in args.candidate_indices.split(",") if item.strip()})
    except ValueError as exc:
        parser.error("--candidate-indices 必须是类似 1,5 的整数列表")
    if selected_indices is not None and (not selected_indices or
                                         any(item < 1 or item > 24 for item in selected_indices)):
        parser.error("--candidate-indices 只能包含 1--24")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing report: {out}")
    geometry = json.loads(Path(args.geometry).expanduser().read_text(encoding="utf-8"))
    if geometry.get("schema") != "am_umi_camera_tcp_geometry_v1":
        parser.error("geometry schema is not am_umi_camera_tcp_geometry_v1")
    pose = np.asarray(geometry.get("pose_cam_tcp"), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        parser.error("geometry pose_cam_tcp must contain six finite values")
    camera_poses = load_camera_poses(Path(args.trajectory).expanduser().resolve())
    base_rotation = np.eye(3)
    base_description = "camera axes"
    if args.finger_tag_detection is not None:
        base_rotation = pair_frame_rotation(
            Path(args.finger_tag_detection).expanduser().resolve(),
            args.left_finger_tag_id, args.right_finger_tag_id)
        base_description = "measured two-finger Tag pair axes"
    sample_indices = np.arange(0, len(camera_poses), args.frame_stride, dtype=int)
    if sample_indices[-1] != len(camera_poses) - 1:
        sample_indices = np.r_[sample_indices, len(camera_poses) - 1]
    q_start, pose_start, _ = load_reference(Path(args.start_reference).expanduser().resolve())
    t_start = pose_to_mat(pose_start)
    limits = load_joint_limits()
    backend = create_kinematics_backend("ros2_dh", str(URDF_PATH), URDF_JOINTS, "right_tcp")
    solver_args = SimpleNamespace(
        ik_steps=args.ik_steps,
        random_restarts=args.random_restarts,
        orientation_weight=args.orientation_weight,
        max_position_error_m=args.max_position_error_m,
        max_rotation_error_deg=args.max_rotation_error_deg,
    )
    print("TCP_ROTATION_CANDIDATE_SEARCH_STARTED")
    print(f"camera frames: {len(camera_poses)}; sampled: {len(sample_indices)}")
    print(f"hardware-safe J2 lower limit: {limits[1, 0]:.2f} deg")
    print("rotation base:", base_description)
    results = []
    for index, (label, axis_rotation) in enumerate(signed_axis_rotations(), start=1):
        if selected_indices is not None and index not in selected_indices:
            continue
        rotation = base_rotation @ axis_rotation
        result = evaluate(label, rotation, camera_poses, pose[:3], backend, q_start,
                          t_start, limits, solver_args, sample_indices)
        result["candidate_index"] = index
        results.append(result)
        print(f"  {index:02d}/24 bad={result['bad_ik_frames']:2d} "
              f"J2lower={result['j2_near_lower_frames']:2d} {label}", flush=True)
    results.sort(key=lambda item: (
        item["bad_ik_frames"], item["j2_near_lower_frames"],
        item["position_error_mm"]["median"], item["rotation_error_deg"]["median"],
        -item["min_joint_margin_deg"]))
    report = {
        "schema": "am_umi_handheld_tcp_rotation_candidate_search_v1",
        "safety": "offline only; no serial port or robot command was used",
        "trajectory": str(Path(args.trajectory).expanduser().resolve()),
        "geometry_translation_source": str(Path(args.geometry).expanduser().resolve()),
        "reference": str(Path(args.start_reference).expanduser().resolve()),
        "sample_indices": sample_indices.tolist(),
        "evaluated_candidate_indices": sorted(selected_indices) if selected_indices is not None else list(range(1, 25)),
        "hardware_safe_limits_deg": limits.tolist(),
        "rotation_base": base_description,
        "rotation_base_matrix_camera_pair": base_rotation.tolist(),
        "ranking_rule": ["fewest bad IK frames", "fewest J2-lower frames",
                         "lower median position/rotation residual", "larger joint margin"],
        "candidates_ranked": results,
        "limitations": [
            "This ranks only coarse right-handed camera-axis permutations.",
            "It does not prove physical collision safety or replace a measured rotation calibration.",
            "Adopt no candidate without a short low-speed physical validation.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TCP_ROTATION_CANDIDATE_SEARCH_OK")
    for rank, item in enumerate(results[:5], start=1):
        print(f"rank {rank} (candidate {item['candidate_index']}): bad={item['bad_ik_frames']}/{item['sampled_frames']} "
              f"J2lower={item['j2_near_lower_frames']} "
              f"pos_med={item['position_error_mm']['median']:.2f}mm "
              f"rot_med={item['rotation_error_deg']['median']:.2f}deg "
              f"margin={item['min_joint_margin_deg']:.2f}deg")
        print("  ", item["label"])
    print("report:", out)


if __name__ == "__main__":
    main()
