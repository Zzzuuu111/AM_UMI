"""Calibrate an AM2Pro eye-in-hand camera from static Tag observations.

The AM2Pro collector stores the transform from arm base to ``right_Fixed_Jaw``.
For a fixed world Tag and camera mounted on that wrist frame, OpenCV's standard
``calibrateHandEye`` directly returns ``right_Fixed_Jaw -> camera``: the exact
fixed-joint transform needed by the URDF.

This program is offline only; it never opens a serial port or camera device.
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
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Offline AM2Pro eye-in-hand calibration.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    args = parser.parse_args()

    resolve = lambda value: (ROOT_DIR / value).resolve()
    input_path, output_path = resolve(args.input), resolve(args.output)
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    samples = pickle.load(input_path.open("rb"))
    aruco = parse_aruco_config(yaml.safe_load(resolve(args.aruco_yaml).read_text()))
    raw_intr = parse_fisheye_intrinsics(json.loads(resolve(args.intrinsics).read_text()))

    base_from_gripper, camera_from_tag, original_indices = [], [], []
    for index, sample in enumerate(samples):
        image = sample["img"]
        intr = convert_fisheye_intrinsics_resolution(
            raw_intr, target_resolution=(image.shape[1], image.shape[0]))
        tags = detect_localize_aruco_tags(
            image, aruco["aruco_dict"], aruco["marker_size_map"], intr)
        if args.tag_id not in tags:
            continue
        tag = tags[args.tag_id]
        cam_tag = np.eye(4)
        cam_tag[:3, :3] = cv2.Rodrigues(tag["rvec"])[0]
        cam_tag[:3, 3] = tag["tvec"]
        camera_from_tag.append(cam_tag)
        base_from_gripper.append(pose_to_mat(sample["tcp_pose"]))
        original_indices.append(index + 1)
    if len(base_from_gripper) < 5:
        parser.error(f"need at least five Tag {args.tag_id} samples, got {len(base_from_gripper)}")

    # OpenCV's parameters are ^bT_g and ^cT_tag; return is ^gT_c.
    rotation, translation = cv2.calibrateHandEye(
        R_gripper2base=[x[:3, :3] for x in base_from_gripper],
        t_gripper2base=[x[:3, 3] for x in base_from_gripper],
        R_target2cam=[x[:3, :3] for x in camera_from_tag],
        t_target2cam=[x[:3, 3] for x in camera_from_tag],
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    gripper_from_camera = np.eye(4)
    gripper_from_camera[:3, :3] = rotation
    gripper_from_camera[:3, 3] = translation.reshape(3)

    # Every static observation should reconstruct the same base_from_tag.
    base_from_tag = [b @ gripper_from_camera @ c
                     for b, c in zip(base_from_gripper, camera_from_tag)]
    reference = base_from_tag[0]
    residuals = []
    for transform in base_from_tag:
        delta = np.linalg.inv(reference) @ transform
        residuals.append({
            "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1000.0),
            "rotation_deg": float(np.linalg.norm(
                Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi),
        })
    mean_t = float(np.mean([x["translation_mm"] for x in residuals]))
    max_t = float(np.max([x["translation_mm"] for x in residuals]))
    mean_r = float(np.mean([x["rotation_deg"] for x in residuals]))
    max_r = float(np.max([x["rotation_deg"] for x in residuals]))
    accepted = mean_t <= 10.0 and max_t <= 20.0 and mean_r <= 2.0 and max_r <= 4.0

    report = {
        "tx_fixed_jaw2camera": gripper_from_camera.tolist(),
        "tx_camera2fixed_jaw": np.linalg.inv(gripper_from_camera).tolist(),
        "diagnostics": {
            "tag_id": args.tag_id,
            "sample_indices_1based": original_indices,
            "method": "cv2.CALIB_HAND_EYE_PARK",
            "static_tag_consistency_mean_translation_mm": mean_t,
            "static_tag_consistency_max_translation_mm": max_t,
            "static_tag_consistency_mean_rotation_deg": mean_r,
            "static_tag_consistency_max_rotation_deg": max_r,
            "urdf_use_recommendation": "YES" if accepted else "NO",
            "per_sample": [
                {"sample_index_1based": index, **residual}
                for index, residual in zip(original_indices, residuals)
            ],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, output_path.open("w"), indent=2)
    print("AM2PRO_HAND_EYE_CALIBRATION_OK")
    print("samples:", len(base_from_gripper))
    print(f"translation_mm: mean={mean_t:.2f} max={max_t:.2f}")
    print(f"rotation_deg: mean={mean_r:.2f} max={max_r:.2f}")
    print("urdf_use_recommendation:", report["diagnostics"]["urdf_use_recommendation"])
    print("saved:", output_path)


if __name__ == "__main__":
    main()
