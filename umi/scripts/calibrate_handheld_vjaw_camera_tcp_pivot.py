#!/usr/bin/env python3
"""Derive a V-jaw fixed-jaw TCP geometry candidate from a pivot recording.

The TCP is defined as a point rigidly attached to the fixed jaw.  Its
translation is solved by holding that point stationary on a table dot while
the hand-held rig rotates.  Its axes are intentionally defined as the axes of
the fixed-jaw ArUco Tag: this gives a measured, repeatable tool frame without
pretending that an arbitrary moving-jaw Tag or a zero rotation is the tool
orientation.

The result is deliberately a ``candidate``.  A separate pivot recording must
validate it before it is allowed into formal demo conversion.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import click
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_handheld_camera_tcp_pivot import solve_pivot, tag_to_camera_matrix  # noqa: E402


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="V 型夹爪 pivot 录制的 tag_detection.pkl")
@click.option("--output", required=True, type=click.Path(dir_okay=False),
              help="新的候选相机到固定爪 TCP 几何 JSON；不得已存在")
@click.option("--world-tag-id", default=13, show_default=True, type=int)
@click.option("--fixed-jaw-tag-id", default=0, show_default=True, type=int)
@click.option("--moving-jaw-tag-id", default=1, show_default=True, type=int)
@click.option("--inlier-threshold-mm", default=8.0, show_default=True, type=float)
@click.option("--min-inlier-ratio", default=0.70, show_default=True, type=float)
def main(input_path, output, world_tag_id, fixed_jaw_tag_id, moving_jaw_tag_id,
         inlier_threshold_mm, min_inlier_ratio):
    """Create a geometry candidate; no robot or IMU is accessed."""
    source = Path(input_path).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise click.ClickException(f"为保护已有结果，拒绝覆盖：{destination}")
    if world_tag_id in (fixed_jaw_tag_id, moving_jaw_tag_id) \
            or fixed_jaw_tag_id == moving_jaw_tag_id:
        raise click.ClickException("世界 Tag、固定爪 Tag、活动爪 Tag 必须互不相同")
    if inlier_threshold_mm <= 0 or not 0 < min_inlier_ratio <= 1:
        raise click.ClickException("无效的 inlier 参数")

    with source.open("rb") as file:
        detections = pickle.load(file)

    camera_poses = []
    fixed_tag_poses = []
    moving_pair_frames = 0
    for frame in detections:
        tags = frame.get("tag_dict", {})
        if world_tag_id not in tags or fixed_jaw_tag_id not in tags:
            continue
        # T_world_camera from the fixed table Tag, and T_camera_fixedTag from
        # the Tag on the rigid jaw.  Both are needed for a V-jaw tool frame.
        camera_poses.append(np.linalg.inv(tag_to_camera_matrix(tags[world_tag_id])))
        fixed_tag_poses.append(tag_to_camera_matrix(tags[fixed_jaw_tag_id]))
        if moving_jaw_tag_id in tags:
            moving_pair_frames += 1
    if len(camera_poses) < 20:
        raise click.ClickException(
            f"同时看到世界 Tag {world_tag_id} 与固定爪 Tag {fixed_jaw_tag_id} 的帧仅 "
            f"{len(camera_poses)}；至少需要 20 帧")

    try:
        p_camera_tcp, p_world_tcp, residuals, inliers, singular_values = solve_pivot(
            camera_poses, inlier_threshold_m=inlier_threshold_mm / 1000.0)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    inlier_ratio = float(inliers.mean())
    if inlier_ratio < min_inlier_ratio:
        raise click.ClickException(
            f"固定点模型失败：{inliers.sum()}/{len(inliers)} ({inlier_ratio:.1%}) 帧在 "
            f"{inlier_threshold_mm:.1f} mm 内。固定爪 TCP 可能在桌面上滑动。")

    # Tag 0 is rigid to the fixed jaw.  Use its measured axes as the TCP axes;
    # this is an explicit frame definition, not a guessed camera orientation.
    inlier_fixed = [pose for pose, use in zip(fixed_tag_poses, inliers) if use]
    fixed_rotations = Rotation.from_matrix(np.stack([pose[:3, :3] for pose in inlier_fixed]))
    rotation_camera_tcp = fixed_rotations.mean().as_matrix()
    tcp_in_fixed_samples = []
    point_camera_tcp = np.r_[p_camera_tcp, 1.0]
    for fixed_pose in inlier_fixed:
        tcp_in_fixed_samples.append((np.linalg.inv(fixed_pose) @ point_camera_tcp)[:3])
    tcp_in_fixed_samples = np.asarray(tcp_in_fixed_samples)
    tcp_in_fixed = np.median(tcp_in_fixed_samples, axis=0)
    fixed_translation_spread_mm = np.linalg.norm(
        tcp_in_fixed_samples - tcp_in_fixed, axis=1) * 1000.0
    rotation_deviation_deg = np.degrees(
        (fixed_rotations.inv() * Rotation.from_matrix(rotation_camera_tcp)).magnitude())

    pose_cam_tcp = np.r_[p_camera_tcp, Rotation.from_matrix(rotation_camera_tcp).as_rotvec()]
    result = {
        "schema": "am_umi_camera_tcp_geometry_v1",
        "status": "candidate",
        "transform_direction": "camera_to_tcp",
        "pose_cam_tcp": pose_cam_tcp.tolist(),
        "tcp_definition": "fixed_jaw_inner_front_tip",
        "tcp_axes_definition": f"axes of fixed_jaw_tag_{fixed_jaw_tag_id}",
        "translation_method": "fixed_world_tag_pivot",
        "rotation_method": "measured_fixed_jaw_tag_axes",
        "world_tag_id": int(world_tag_id),
        "fixed_jaw_tag_id": int(fixed_jaw_tag_id),
        "moving_jaw_tag_id": int(moving_jaw_tag_id),
        "input": str(source),
        "quality": {
            "world_and_fixed_jaw_tag_frames": int(len(camera_poses)),
            "fixed_and_moving_jaw_tag_frames": int(moving_pair_frames),
            "pivot_inlier_frames": int(inliers.sum()),
            "pivot_inlier_ratio": inlier_ratio,
            "pivot_residual_rmse_mm": float(np.sqrt(np.mean(residuals[inliers] ** 2)) * 1000.0),
            "pivot_residual_p95_mm": float(np.percentile(residuals[inliers], 95) * 1000.0),
            "fixed_tag_tcp_translation_spread_p95_mm": float(np.percentile(fixed_translation_spread_mm, 95)),
            "fixed_tag_rotation_deviation_p95_deg": float(np.percentile(rotation_deviation_deg, 95)),
            "linear_system_singular_values": singular_values.tolist(),
        },
        "next_required_step": (
            "Use an independent fixed-point pivot recording to validate translation before "
            "marking this geometry accepted for formal demo conversion."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V_JAW_CAMERA_TCP_PIVOT_CANDIDATE_OK")
    print("world_and_fixed_jaw_tag_frames:", len(camera_poses))
    print(f"pivot_inliers: {inliers.sum()}/{len(inliers)} ({inlier_ratio:.1%})")
    print("pose_cam_tcp:", np.array2string(pose_cam_tcp, precision=6))
    print("pivot_rmse_mm: {:.3f}; p95_mm: {:.3f}".format(
        result["quality"]["pivot_residual_rmse_mm"], result["quality"]["pivot_residual_p95_mm"]))
    print("fixed_tag_tcp_translation_spread_p95_mm: {:.3f}; rotation_p95_deg: {:.3f}".format(
        result["quality"]["fixed_tag_tcp_translation_spread_p95_mm"],
        result["quality"]["fixed_tag_rotation_deviation_p95_deg"]))
    print("saved:", destination)


if __name__ == "__main__":
    main()
