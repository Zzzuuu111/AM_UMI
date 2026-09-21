#!/usr/bin/env python3
"""Repair planar-ArUco pose branch flips for a one-moving-jaw V gripper.

Each square ArUco detection has two IPPE pose solutions.  The ordinary
single-marker API usually selects the lower reprojection-error solution, but
that solution may switch branch from one frame to the next.  For the two small
Tags mounted on the V jaw this looks like an impossible instantaneous jump in
their relative angle and consequently an incorrect full-open gripper label.

This offline utility re-solves the stored corners with IPPE_SQUARE and selects
the physically plausible, temporally continuous fixed/moving Tag pair.  It
only rewrites Tag IDs 0/1 in a *new* pkl; table Tags and camera trajectory
inputs remain untouched.  Frames with no plausible pair have those two tool
Tags removed so the downstream width conversion interpolates rather than
trusting a bad measurement.
"""

from __future__ import annotations

import argparse
import copy
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
    get_relative_tag_rotation_deg,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线修复 V 型夹爪双 ArUco 的平面 PnP 姿态分支跳变。")
    parser.add_argument("--input", required=True, help="原始 tag_detection.pkl")
    parser.add_argument("--output", required=True, help="新的修复后 tag_detection.pkl（必须不存在）")
    parser.add_argument("--intrinsics", required=True, help="手持相机 fisheye 内参 JSON")
    parser.add_argument("--aruco-yaml", required=True, help="ArUco marker_size_map YAML")
    parser.add_argument("--fixed-tag-id", type=int, default=0)
    parser.add_argument("--moving-tag-id", type=int, default=1)
    parser.add_argument("--image-width", type=int, default=1920,
                        help="生成 pkl 时的视频宽度，默认 1920")
    parser.add_argument("--image-height", type=int, default=1080,
                        help="生成 pkl 时的视频高度，默认 1080")
    parser.add_argument(
        "--max-relative-angle-deg", type=float, default=60.0,
        help="物理可接受的固定爪/活动爪相对角上限；超过者不作为候选，默认 60°")
    parser.add_argument(
        "--continuity-weight", type=float, default=1.0,
        help="相邻帧相对角连续性的权重，默认 1.0")
    return parser.parse_args()


def marker_object_points(size_m: float) -> np.ndarray:
    half = float(size_m) / 2.0
    return np.asarray([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float64)


def solve_ippe_candidates(tag: dict, marker_size_m: float,
                          K: np.ndarray, D: np.ndarray) -> list[dict]:
    """Return positive-depth IPPE square candidates from stored ArUco corners."""
    try:
        corners = np.asarray(tag["corners"], dtype=np.float64).reshape(1, 4, 2)
    except (KeyError, TypeError, ValueError):
        return []
    if not np.isfinite(corners).all():
        return []
    undistorted = cv2.fisheye.undistortPoints(corners, K, D, P=K).reshape(-1, 2)
    solved = cv2.solvePnPGeneric(
        marker_object_points(marker_size_m), undistorted, K, np.zeros((1, 5)),
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not solved or not bool(solved[0]):
        return []
    rvecs, tvecs = solved[1], solved[2]
    reprojection = solved[3] if len(solved) > 3 else None
    candidates = []
    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not np.isfinite(np.r_[rvec, tvec]).all() or tvec[2] <= 0:
            continue
        error = float(np.asarray(reprojection).reshape(-1)[index]) if reprojection is not None else 0.0
        candidates.append({"rvec": rvec, "tvec": tvec, "error": error, "index": index})
    return candidates


def relative_angle_deg(fixed: dict, moving: dict) -> float:
    fixed_rot, _ = cv2.Rodrigues(fixed["rvec"])
    moving_rot, _ = cv2.Rodrigues(moving["rvec"])
    cosine = float(np.clip((np.trace(fixed_rot.T @ moving_rot) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def select_pair(candidates: list[dict], previous_angle: float | None,
                continuity_weight: float) -> dict:
    errors = np.asarray([entry["error"] for entry in candidates], dtype=np.float64)
    error_span = max(float(errors.max() - errors.min()), 1e-9)

    def score(entry: dict) -> tuple[float, float, float]:
        normalised_error = (entry["error"] - float(errors.min())) / error_span
        continuity = 0.0 if previous_angle is None else abs(entry["angle_deg"] - previous_angle)
        # A one-degree change should decisively outweigh sub-pixel
        # reprojection differences: physical jaw motion is continuous.
        return (continuity_weight * continuity + 0.05 * normalised_error,
                entry["error"], entry["angle_deg"])

    return min(candidates, key=score)


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"为保护已有结果，拒绝覆盖：{output}")
    if (args.max_relative_angle_deg <= 0 or args.continuity_weight < 0
            or args.image_width <= 0 or args.image_height <= 0):
        raise SystemExit("角度上限和图像尺寸必须为正，--continuity-weight 不能为负")

    detections = pickle.load(source.open("rb"))
    intrinsics_data = json.loads(Path(args.intrinsics).expanduser().read_text(encoding="utf-8"))
    base_intrinsics = parse_fisheye_intrinsics(intrinsics_data)
    config = parse_aruco_config(yaml.safe_load(Path(args.aruco_yaml).expanduser().read_text(encoding="utf-8")))
    marker_sizes = config["marker_size_map"]
    intrinsics = convert_fisheye_intrinsics_resolution(
        base_intrinsics, (args.image_width, args.image_height))
    for tag_id in (args.fixed_tag_id, args.moving_tag_id):
        if tag_id not in marker_sizes or marker_sizes[tag_id] is None:
            raise SystemExit(f"ArUco 配置中没有 Tag {tag_id} 的尺寸")

    repaired = copy.deepcopy(detections)
    previous_angle: float | None = None
    pair_frames = repaired_branches = dropped_pair_frames = 0
    old_values: list[float] = []
    new_values: list[float] = []

    for original_frame, new_frame in zip(detections, repaired):
        original_tags = original_frame.get("tag_dict", {})
        tags = new_frame.get("tag_dict", {})
        if args.fixed_tag_id not in original_tags or args.moving_tag_id not in original_tags:
            previous_angle = None
            continue
        pair_frames += 1
        old = get_relative_tag_rotation_deg(original_tags, args.fixed_tag_id, args.moving_tag_id)
        if old is not None and np.isfinite(old):
            old_values.append(float(old))
        try:
            fixed_candidates = solve_ippe_candidates(
                original_tags[args.fixed_tag_id], marker_sizes[args.fixed_tag_id],
                intrinsics["K"], intrinsics["D"])
            moving_candidates = solve_ippe_candidates(
                original_tags[args.moving_tag_id], marker_sizes[args.moving_tag_id],
                intrinsics["K"], intrinsics["D"])
        except (cv2.error, KeyError, TypeError, ValueError):
            fixed_candidates, moving_candidates = [], []

        pairs = []
        for fixed in fixed_candidates:
            for moving in moving_candidates:
                angle = relative_angle_deg(fixed, moving)
                if not np.isfinite(angle) or angle > args.max_relative_angle_deg:
                    continue
                pairs.append({
                    "fixed": fixed, "moving": moving, "angle_deg": angle,
                    "error": float(fixed["error"] + moving["error"]),
                })
        if not pairs:
            # Do not feed an arbitrary flipped pose to the width conversion.
            # Table Tags remain intact for camera trajectory processing.
            tags.pop(args.fixed_tag_id, None)
            tags.pop(args.moving_tag_id, None)
            new_frame["vjaw_pose_branch_fix"] = {"status": "dropped_no_physical_pair"}
            dropped_pair_frames += 1
            previous_angle = None
            continue

        selected = select_pair(pairs, previous_angle, args.continuity_weight)
        fixed, moving = selected["fixed"], selected["moving"]
        tags[args.fixed_tag_id]["rvec"] = fixed["rvec"].copy()
        tags[args.fixed_tag_id]["tvec"] = fixed["tvec"].copy()
        tags[args.moving_tag_id]["rvec"] = moving["rvec"].copy()
        tags[args.moving_tag_id]["tvec"] = moving["tvec"].copy()
        changed = fixed["index"] != 0 or moving["index"] != 0
        repaired_branches += int(changed)
        previous_angle = float(selected["angle_deg"])
        new_values.append(previous_angle)
        new_frame["vjaw_pose_branch_fix"] = {
            "status": "selected",
            "fixed_candidate": int(fixed["index"]),
            "moving_candidate": int(moving["index"]),
            "relative_angle_deg": previous_angle,
            "combined_reprojection_error": float(selected["error"]),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
        pickle.dump(repaired, file, protocol=pickle.HIGHEST_PROTOCOL)

    print("VJAW_TAG_POSE_BRANCH_FIX_OK")
    print("frames:", len(repaired), "dual_tag_frames:", pair_frames)
    print("branch_reselected_frames:", repaired_branches,
          "dropped_no_physical_pair:", dropped_pair_frames)
    if old_values:
        print("raw_relative_angle_deg_before: min/max=%.3f/%.3f" % (min(old_values), max(old_values)))
    if new_values:
        print("raw_relative_angle_deg_after : min/max=%.3f/%.3f" % (min(new_values), max(new_values)))
    print("output:", output)


if __name__ == "__main__":
    main()
