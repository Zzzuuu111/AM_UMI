#!/usr/bin/env python3
"""Fit a V-jaw grasp-center point from static grasps of one vertical cylinder.

Unlike a conventional point-pivot calibration, a vertical cylinder supplies a
fixed *axis*, not a fixed point: when a V-jaw approaches at different tilts,
the useful grasp center may slide along that axis.  This program therefore
fits a point fixed in ``right_Fixed_Jaw`` whose world XY projection stays on
one unknown, fixed vertical cylinder axis.  The axis height is deliberately
unconstrained.

The input is produced by ``record_am2pro_hand_eye.py``.  Tag observations and
the already validated fixed-jaw-to-camera hand-eye transform are used only
offline; no robot or serial port is opened.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def solve_axis(samples: list[dict], sample_ids: list[int]) -> tuple[np.ndarray, int, np.ndarray]:
    """Solve [point_in_fixed_jaw_xyz, cylinder_axis_world_xy]."""
    blocks, rhs = [], []
    for sample_id in sample_ids:
        world_from_fixed_jaw = samples[sample_id]["world_from_fixed_jaw"]
        # R_WF * p_F + t_WF has the same XY coordinates for every grasp.
        blocks.append(np.c_[world_from_fixed_jaw[:2, :3], -np.eye(2)])
        rhs.append(-world_from_fixed_jaw[:2, 3])
    solution, _, rank, singular_values = np.linalg.lstsq(
        np.vstack(blocks), np.hstack(rhs), rcond=None)
    return solution, int(rank), singular_values


def residuals_mm(samples: list[dict], solution: np.ndarray) -> np.ndarray:
    point_fixed_jaw = solution[:3]
    axis_xy = solution[3:]
    errors = []
    for sample in samples:
        point_world = (sample["world_from_fixed_jaw"][:3, :3] @ point_fixed_jaw
                       + sample["world_from_fixed_jaw"][:3, 3])
        errors.append(float(np.linalg.norm(point_world[:2] - axis_xy) * 1000.0))
    return np.asarray(errors, dtype=float)


def servo_to_width_m(value: float, robot: dict | None) -> float | None:
    if robot is None or not np.isfinite(value):
        return None
    table = np.asarray(robot.get("gripper_width_servo_table", []), dtype=float)
    if table.ndim == 2 and table.shape[1] == 2 and len(table) >= 2:
        order = np.argsort(table[:, 1])
        return float(np.interp(value, table[order, 1], table[order, 0]))
    closed = robot.get("gripper_servo_closed")
    opened = robot.get("gripper_servo_open")
    minimum = robot.get("gripper_width_min")
    maximum = robot.get("gripper_width_max")
    if None in (closed, opened, minimum, maximum) or abs(float(opened) - float(closed)) < 1e-9:
        return None
    fraction = (value - float(closed)) / (float(opened) - float(closed))
    return float(np.clip(float(minimum) + fraction * (float(maximum) - float(minimum)),
                         float(minimum), float(maximum)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线拟合竖直圆柱轴线上的 V 型夹爪抓取中心；不会连接机器人。")
    parser.add_argument(
        "--input", required=True, nargs="+", metavar="PKL",
        help=("一个或多个 record_am2pro_hand_eye.py 的 .pkl；多文件仅可用于"
              "同一根在整个续拍期间未移动的圆柱"),
    )
    parser.add_argument("--hand-eye", required=True, help="已验证的 tx_fixed_jaw2camera JSON")
    parser.add_argument("--output", required=True, help="新 JSON；拒绝覆盖")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--robot-config", default=None,
                        help="可选；仅用于把保存的 J7 编码器值报告为毫米")
    parser.add_argument("--inlier-threshold-mm", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=6)
    args = parser.parse_args()

    if args.inlier_threshold_mm <= 0 or args.min_inliers < 3:
        parser.error("--inlier-threshold-mm 必须为正，--min-inliers 至少为 3")
    resolve = lambda path: (ROOT / path).resolve() if not Path(path).expanduser().is_absolute() else Path(path).expanduser().resolve()
    input_paths = [resolve(path) for path in args.input]
    hand_eye_path, output_path = map(resolve, (args.hand_eye, args.output))
    if output_path.exists():
        parser.error(f"拒绝覆盖已有结果: {output_path}")

    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        parser.error("找不到输入采样文件: " + ", ".join(missing_inputs))
    raw_samples = []
    for input_path in input_paths:
        loaded = pickle.load(input_path.open("rb"))
        if not isinstance(loaded, list):
            parser.error(f"输入不是样本列表: {input_path}")
        raw_samples.extend(loaded)
    if len(raw_samples) < 3:
        parser.error("至少需要 3 张含明显不同姿态的样本")
    hand_eye = json.loads(hand_eye_path.read_text(encoding="utf-8"))
    if "tx_fixed_jaw2camera" not in hand_eye:
        parser.error("hand-eye 文件缺少 tx_fixed_jaw2camera")
    fixed_jaw_from_camera = np.asarray(hand_eye["tx_fixed_jaw2camera"], dtype=float)
    if fixed_jaw_from_camera.shape != (4, 4):
        parser.error("tx_fixed_jaw2camera 必须是 4x4 矩阵")

    aruco = parse_aruco_config(yaml.safe_load(resolve(args.aruco_yaml).read_text(encoding="utf-8")))
    raw_intrinsics = parse_fisheye_intrinsics(json.loads(resolve(args.intrinsics).read_text(encoding="utf-8")))
    robot = None
    if args.robot_config:
        robot = yaml.safe_load(resolve(args.robot_config).read_text(encoding="utf-8"))["robots"][0]

    samples = []
    missing_tag_indices = []
    for raw_index, sample in enumerate(raw_samples, start=1):
        image = sample.get("img")
        if image is None:
            missing_tag_indices.append(raw_index)
            continue
        intrinsics = convert_fisheye_intrinsics_resolution(
            raw_intrinsics, target_resolution=(image.shape[1], image.shape[0]))
        tags = detect_localize_aruco_tags(image, aruco["aruco_dict"],
                                          aruco["marker_size_map"], intrinsics)
        if args.tag_id not in tags:
            missing_tag_indices.append(raw_index)
            continue
        tag = tags[args.tag_id]
        camera_from_world = np.eye(4)
        camera_from_world[:3, :3] = cv2.Rodrigues(tag["rvec"])[0]
        camera_from_world[:3, 3] = tag["tvec"]
        fixed_jaw_from_world = fixed_jaw_from_camera @ camera_from_world
        samples.append({
            "raw_index_1based": raw_index,
            "world_from_fixed_jaw": np.linalg.inv(fixed_jaw_from_world),
            "gripper_position_range_0_100": float(sample.get(
                "gripper_position_range_0_100", np.nan)),
        })

    if len(samples) < 3:
        parser.error(f"只有 {len(samples)} 张检测到 Tag {args.tag_id}；至少需要 3 张")

    # RANSAC-style exhaustive minimal-set search.  Three poses give six XY
    # equations for five unknowns; final values are always re-fit using all
    # accepted inliers rather than retaining a minimal-set solution.
    best = None
    for seed in itertools.combinations(range(len(samples)), 3):
        solution, rank, singular_values = solve_axis(samples, list(seed))
        if rank < 5 or singular_values[-1] < 1e-5:
            continue
        errors = residuals_mm(samples, solution)
        inliers = np.flatnonzero(errors <= args.inlier_threshold_mm)
        score = (int(len(inliers)),
                 -float(np.mean(errors[inliers])) if len(inliers) else -float("inf"),
                 -float(np.max(errors[inliers])) if len(inliers) else -float("inf"))
        if best is None or score > best[0]:
            best = (score, seed, inliers)
    if best is None:
        parser.error("所有最小姿态组都退化；需要更多不同方向的夹取姿态")

    _, seed, inlier_ids = best
    solution, rank, singular_values = solve_axis(samples, inlier_ids.tolist())
    errors = residuals_mm(samples, solution)
    # One final inlier pass using the re-fitted candidate.
    inlier_ids = np.flatnonzero(errors <= args.inlier_threshold_mm)
    solution, rank, singular_values = solve_axis(samples, inlier_ids.tolist())
    errors = residuals_mm(samples, solution)
    inlier_ids = np.flatnonzero(errors <= args.inlier_threshold_mm)

    widths = np.asarray([sample["gripper_position_range_0_100"] for sample in samples], dtype=float)
    inlier_widths = widths[inlier_ids]
    width_m = [servo_to_width_m(value, robot) for value in inlier_widths]
    width_m = np.asarray([value for value in width_m if value is not None], dtype=float)
    inlier_errors = errors[inlier_ids]
    accepted = (rank == 5 and len(inlier_ids) >= args.min_inliers
                and float(np.mean(inlier_errors)) <= 3.0
                and float(np.max(inlier_errors)) <= args.inlier_threshold_mm)

    result = {
        "schema": "am_umi_vjaw_grasp_center_cylinder_axis_v1",
        "status": "candidate" if accepted else "rejected_insufficient_axis_consensus",
        "parent_frame": "right_Fixed_Jaw",
        "tcp_frame": "grasp_center_at_sampled_width",
        "definition": ("Point rigidly attached to the fixed jaw whose world XY projection lies on "
                       "the fitted vertical cylinder axis. Cylinder-axis height is intentionally free."),
        "grasp_center_translation_parent_m": solution[:3].tolist(),
        "fitted_cylinder_axis_world_xy_m": solution[3:].tolist(),
        # Keep the old singular key for consumers that expect it, while
        # recording every source used by a continuation fit.
        "input": str(input_paths[0]),
        "inputs": [str(path) for path in input_paths],
        "hand_eye": str(hand_eye_path),
        "parameters": {
            "tag_id": args.tag_id,
            "inlier_threshold_mm": args.inlier_threshold_mm,
            "min_inliers": args.min_inliers,
            "ransac_seed_sample_indices_1based": [samples[i]["raw_index_1based"] for i in seed],
        },
        "j7": {
            "inlier_servo_median_0_100": float(np.nanmedian(inlier_widths)),
            "inlier_servo_span_0_100": float(np.nanmax(inlier_widths) - np.nanmin(inlier_widths)),
            "inlier_width_m_median": float(np.median(width_m)) if len(width_m) else None,
            "inlier_width_m_span": float(np.max(width_m) - np.min(width_m)) if len(width_m) else None,
        },
        "diagnostics": {
            "visible_tag_samples": len(samples),
            "missing_tag_raw_indices_1based": missing_tag_indices,
            "linear_system_rank": rank,
            "singular_values": singular_values.tolist(),
            "inlier_raw_indices_1based": [samples[i]["raw_index_1based"] for i in inlier_ids],
            "inlier_count": int(len(inlier_ids)),
            "inlier_transverse_residual_mm_mean": float(np.mean(inlier_errors)),
            "inlier_transverse_residual_mm_max": float(np.max(inlier_errors)),
            "all_transverse_residual_mm": errors.tolist(),
        },
        "limitations": [
            "This calibrates a grasp-center point at the sampled J7 width, not a universal fixed right_tcp.",
            "The fitted axis assumes the cylinder is vertical and stationary during one input recording.",
            "Use at least two accepted widths and a third independent cylinder before fitting grasp_center(width).",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AM2PRO_GRASP_CENTER_CYLINDER_AXIS_CALIBRATION_" + ("CANDIDATE" if accepted else "REJECTED"))
    print("visible samples:", len(samples), "inliers:", len(inlier_ids), "rank:", rank)
    print("grasp_center_in_fixed_jaw_m:", np.round(solution[:3], 6).tolist())
    print("axis residual mm mean/max: {:.3f}/{:.3f}".format(
        float(np.mean(inlier_errors)), float(np.max(inlier_errors))))
    print("J7 inlier servo median/span: {:.3f}/{:.3f}".format(
        float(np.nanmedian(inlier_widths)), float(np.nanmax(inlier_widths) - np.nanmin(inlier_widths))))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
