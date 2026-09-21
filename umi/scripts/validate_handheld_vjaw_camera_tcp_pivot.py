#!/usr/bin/env python3
"""Independently validate a V-jaw fixed-jaw TCP pivot candidate.

This script never refits the TCP.  It projects the candidate into the fixed
world Tag frame over a fresh pivot recording, checks that the point remains
stationary, and also checks that the fixed-jaw Tag remains rigid relative to
the camera.  On success it writes a new ``accepted`` geometry JSON; the input
candidate is never changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.calibrate_handheld_camera_tcp_pivot import tag_to_camera_matrix  # noqa: E402


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="独立 pivot 录制的 tag_detection.pkl")
@click.option("--geometry", required=True, type=click.Path(exists=True, dir_okay=False),
              help="calibrate_handheld_vjaw_camera_tcp_pivot.py 的 candidate JSON")
@click.option("--output", required=True, type=click.Path(dir_okay=False),
              help="新的 accepted 几何 JSON；不得已存在")
@click.option("--inlier-threshold-mm", default=8.0, show_default=True, type=float)
@click.option("--min-inlier-ratio", default=0.70, show_default=True, type=float)
@click.option("--fixed-tag-translation-p95-mm", default=3.0, show_default=True, type=float)
@click.option("--fixed-tag-rotation-p95-deg", default=1.5, show_default=True, type=float)
def main(input_path, geometry, output, inlier_threshold_mm, min_inlier_ratio,
         fixed_tag_translation_p95_mm, fixed_tag_rotation_p95_deg):
    source = Path(input_path).expanduser().resolve()
    geometry_path = Path(geometry).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise click.ClickException(f"为保护已有结果，拒绝覆盖：{destination}")
    if (inlier_threshold_mm <= 0 or not 0 < min_inlier_ratio <= 1
            or fixed_tag_translation_p95_mm <= 0 or fixed_tag_rotation_p95_deg <= 0):
        raise click.ClickException("所有阈值必须为正，且 min-inlier-ratio 位于 (0,1]")
    geometry_data = json.loads(geometry_path.read_text(encoding="utf-8"))
    if geometry_data.get("schema") != "am_umi_camera_tcp_geometry_v1":
        raise click.ClickException("未知的 geometry schema")
    if geometry_data.get("tcp_definition") != "fixed_jaw_inner_front_tip":
        raise click.ClickException("geometry 不是 V 型固定爪 TCP 候选")
    pose = np.asarray(geometry_data.get("pose_cam_tcp"), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise click.ClickException("geometry 的 pose_cam_tcp 无效")
    world_tag_id = int(geometry_data["world_tag_id"])
    fixed_tag_id = int(geometry_data["fixed_jaw_tag_id"])

    import pickle
    with source.open("rb") as file:
        detections = pickle.load(file)

    world_tcp, fixed_poses = [], []
    for frame in detections:
        tags = frame.get("tag_dict", {})
        if world_tag_id not in tags or fixed_tag_id not in tags:
            continue
        tx_world_camera = np.linalg.inv(tag_to_camera_matrix(tags[world_tag_id]))
        world_tcp.append(tx_world_camera[:3, :3] @ pose[:3] + tx_world_camera[:3, 3])
        fixed_poses.append(tag_to_camera_matrix(tags[fixed_tag_id]))
    world_tcp = np.asarray(world_tcp)
    if len(world_tcp) < 20:
        raise click.ClickException("同时看到世界 Tag 和固定爪 Tag 的帧少于 20")

    center = np.median(world_tcp, axis=0)
    residuals_mm = np.linalg.norm(world_tcp - center, axis=1) * 1000.0
    inliers = residuals_mm <= inlier_threshold_mm
    ratio = float(inliers.mean())
    fixed_trans = np.stack([tx[:3, 3] for tx in fixed_poses])
    fixed_trans_residuals_mm = np.linalg.norm(
        fixed_trans - np.median(fixed_trans, axis=0), axis=1) * 1000.0
    fixed_rots = Rotation.from_matrix(np.stack([tx[:3, :3] for tx in fixed_poses]))
    fixed_mean = fixed_rots.mean()
    fixed_rot_residuals_deg = np.degrees((fixed_rots.inv() * fixed_mean).magnitude())

    quality = {
        "validation_frames": int(len(world_tcp)),
        "pivot_inlier_frames": int(inliers.sum()),
        "pivot_inlier_ratio": ratio,
        "pivot_inlier_rmse_mm": float(np.sqrt(np.mean(residuals_mm[inliers] ** 2))) if inliers.any() else float("inf"),
        "pivot_all_frame_p95_mm": float(np.percentile(residuals_mm, 95)),
        "fixed_tag_translation_p95_mm": float(np.percentile(fixed_trans_residuals_mm, 95)),
        "fixed_tag_rotation_p95_deg": float(np.percentile(fixed_rot_residuals_deg, 95)),
    }
    checks = {
        "pivot_inlier_ratio": ratio >= min_inlier_ratio,
        "fixed_tag_translation": quality["fixed_tag_translation_p95_mm"] <= fixed_tag_translation_p95_mm,
        "fixed_tag_rotation": quality["fixed_tag_rotation_p95_deg"] <= fixed_tag_rotation_p95_deg,
    }
    accepted = all(checks.values())
    result = dict(geometry_data)
    result["status"] = "accepted" if accepted else "rejected"
    result["independent_validation"] = {
        "input": str(source),
        "world_tcp_median_m": center.tolist(),
        "thresholds": {
            "inlier_threshold_mm": inlier_threshold_mm,
            "min_inlier_ratio": min_inlier_ratio,
            "fixed_tag_translation_p95_mm": fixed_tag_translation_p95_mm,
            "fixed_tag_rotation_p95_deg": fixed_tag_rotation_p95_deg,
        },
        "quality": quality,
        "checks": checks,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V_JAW_CAMERA_TCP_PIVOT_VALIDATION_" + ("OK" if accepted else "REJECTED"))
    print("pivot_inliers: {}/{} ({:.1%}); p95_mm: {:.3f}".format(
        inliers.sum(), len(inliers), ratio, quality["pivot_all_frame_p95_mm"]))
    print("fixed_tag: translation_p95_mm={:.3f}; rotation_p95_deg={:.3f}".format(
        quality["fixed_tag_translation_p95_mm"], quality["fixed_tag_rotation_p95_deg"]))
    print("checks:", checks)
    print("saved:", destination)
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
