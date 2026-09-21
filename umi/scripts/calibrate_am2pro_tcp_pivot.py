"""Solve the new AM2Pro gripper TCP from a fixed-point pivot recording.

During collection, the closed fingertip midpoint is kept on one physical point
on the table while the arm/camera rotates around it.  A fixed Tag 13 is visible
in every frame.  The already validated wrist-camera hand-eye transform turns
each visual Tag pose into a wrist-frame pose, so this calibration does not
depend on the robot FK translation model.

This program is offline only.  It does not open the robot or move anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import cv2
import numpy as np
import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)


def main():
    parser = argparse.ArgumentParser(description="Offline AM2Pro closed-tip TCP pivot calibration.")
    parser.add_argument("--input", required=True, help="Pivot samples from record_am2pro_hand_eye.py")
    parser.add_argument("--hand-eye", required=True, help="Validated tx_fixed_jaw2camera JSON")
    parser.add_argument("--output", required=True, help="New TCP result JSON; never overwritten")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    args = parser.parse_args()

    resolve = lambda value: (ROOT_DIR / value).resolve()
    input_path, hand_eye_path, output_path = resolve(args.input), resolve(args.hand_eye), resolve(args.output)
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    hand_eye = json.loads(hand_eye_path.read_text())
    fixed_jaw_from_camera = np.asarray(hand_eye["tx_fixed_jaw2camera"], dtype=float)
    samples = pickle.load(input_path.open("rb"))
    aruco = parse_aruco_config(yaml.safe_load(resolve(args.aruco_yaml).read_text()))
    raw_intr = parse_fisheye_intrinsics(json.loads(resolve(args.intrinsics).read_text()))

    wrist_from_world = []
    indices = []
    for index, sample in enumerate(samples):
        image = sample["img"]
        intr = convert_fisheye_intrinsics_resolution(
            raw_intr, target_resolution=(image.shape[1], image.shape[0]))
        tags = detect_localize_aruco_tags(
            image, aruco["aruco_dict"], aruco["marker_size_map"], intr)
        if args.tag_id not in tags:
            continue
        tag = tags[args.tag_id]
        camera_from_world = np.eye(4)
        camera_from_world[:3, :3] = cv2.Rodrigues(tag["rvec"])[0]
        camera_from_world[:3, 3] = tag["tvec"]
        wrist_from_world.append(fixed_jaw_from_camera @ camera_from_world)
        indices.append(index + 1)
    if len(wrist_from_world) < 5:
        parser.error(f"need at least five visible Tag {args.tag_id} samples, got {len(wrist_from_world)}")

    # p_wrist = R_wrist_world * p_world + t_wrist_world is invariant across
    # all poses.  Stack p_wrist - R_i p_world = t_i and solve both unknowns.
    blocks, rhs = [], []
    for transform in wrist_from_world:
        blocks.append(np.c_[np.eye(3), -transform[:3, :3]])
        rhs.append(transform[:3, 3])
    solution, _, rank, singular_values = np.linalg.lstsq(np.vstack(blocks), np.hstack(rhs), rcond=None)
    tcp_in_fixed_jaw = solution[:3]
    pivot_in_world = solution[3:]
    residuals_mm = []
    for transform in wrist_from_world:
        predicted = transform[:3, :3] @ pivot_in_world + transform[:3, 3]
        residuals_mm.append(float(np.linalg.norm(predicted - tcp_in_fixed_jaw) * 1000.0))
    mean_mm, max_mm = float(np.mean(residuals_mm)), float(np.max(residuals_mm))
    recommended = rank == 6 and mean_mm <= 3.0 and max_mm <= 8.0

    result = {
        "tcp_frame": "closed_fingertip_midpoint",
        "parent_frame": "right_Fixed_Jaw",
        "translation_m": tcp_in_fixed_jaw.tolist(),
        "orientation": "inherits right_Fixed_Jaw orientation; position-only pivot calibration",
        "diagnostics": {
            "tag_id": args.tag_id,
            "sample_indices_1based": indices,
            "samples": len(wrist_from_world),
            "linear_system_rank": int(rank),
            "singular_values": singular_values.tolist(),
            "mean_residual_mm": mean_mm,
            "max_residual_mm": max_mm,
            "per_sample_residual_mm": residuals_mm,
            "urdf_ik_use_recommendation": "YES" if recommended else "NO",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, output_path.open("w"), indent=2)
    print("AM2PRO_TCP_PIVOT_CALIBRATION_OK")
    print("samples:", len(wrist_from_world), "rank:", rank)
    print("tcp_translation_in_fixed_jaw_m:", np.round(tcp_in_fixed_jaw, 6).tolist())
    print(f"pivot_residual_mm: mean={mean_mm:.3f} max={max_mm:.3f}")
    print("urdf_ik_use_recommendation:", result["diagnostics"]["urdf_ik_use_recommendation"])
    print("saved:", output_path)


if __name__ == "__main__":
    main()
