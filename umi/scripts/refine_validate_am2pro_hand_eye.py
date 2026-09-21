"""Robustly refine and validate static AM2Pro wrist-camera hand-eye samples.

Unlike the legacy generic hand-eye helper, this tool writes per-sample
geometric residuals and never overwrites its input or the first-pass result.
It is entirely offline: no camera, serial port, or robot is opened.
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
from scipy.optimize import least_squares
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


def transform_from_vec(vector: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    out[:3, 3] = vector[:3]
    return out


def vec_from_transform(transform: np.ndarray) -> np.ndarray:
    return np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_rotvec()]


def main():
    parser = argparse.ArgumentParser(description="Offline robust AM2Pro hand-eye refinement.")
    parser.add_argument("--input", required=True, help="Sample .pkl from record_am2pro_hand_eye.py")
    parser.add_argument("--output", required=True, help="New output .json; never overwritten")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    args = parser.parse_args()

    def resolve(value):
        return (ROOT_DIR / value).resolve()

    input_path, output_path = resolve(args.input), resolve(args.output)
    intr_path, aruco_path = resolve(args.intrinsics), resolve(args.aruco_yaml)
    if output_path.exists():
        parser.error(f"refusing to overwrite existing output: {output_path}")
    samples = pickle.load(input_path.open("rb"))
    if len(samples) < 6:
        parser.error("at least six saved samples are required")

    aruco = parse_aruco_config(yaml.safe_load(aruco_path.read_text()))
    raw_intr = parse_fisheye_intrinsics(json.loads(intr_path.read_text()))
    observed, gripper_from_base, sample_indices = [], [], []
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
        observed.append(camera_from_world)
        # Collector stores base_from_fixed_jaw; OpenCV's convention below is
        # fixed_jaw_from_base.
        gripper_from_base.append(np.linalg.inv(pose_to_mat(sample["tcp_pose"])))
        sample_indices.append(index)

    if len(observed) < 6:
        parser.error(f"only {len(observed)} samples contain Tag {args.tag_id}")

    # C_T_W = C_T_G @ G_T_B @ B_T_W.  Solve C_T_G and B_T_W jointly.
    def residual(vector, indices):
        camera_from_gripper = transform_from_vec(vector[:6])
        base_from_world = transform_from_vec(vector[6:])
        result = []
        for i in indices:
            delta = np.linalg.inv(observed[i]) @ (
                camera_from_gripper @ gripper_from_base[i] @ base_from_world)
            # Normalize to comparable engineering units: 10 mm and 3 degrees.
            result.extend(delta[:3, 3] / 0.010)
            result.extend(Rotation.from_matrix(delta[:3, :3]).as_rotvec() / np.deg2rad(3.0))
        return np.asarray(result)

    # A generic OpenCV initialization is adequate; robust nonlinear refinement
    # then suppresses a hand-held capture taken before the pose was settled.
    r_wb, t_wb, r_gc, t_gc = cv2.calibrateRobotWorldHandEye(
        R_world2cam=[x[:3, :3] for x in observed],
        t_world2cam=[x[:3, 3] for x in observed],
        R_base2gripper=[x[:3, :3] for x in gripper_from_base],
        t_base2gripper=[x[:3, 3] for x in gripper_from_base],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
    )
    world_from_base = np.eye(4)
    world_from_base[:3, :3], world_from_base[:3, 3] = r_wb, t_wb.reshape(3)
    camera_from_gripper = np.eye(4)
    camera_from_gripper[:3, :3], camera_from_gripper[:3, 3] = r_gc, t_gc.reshape(3)
    start = np.r_[vec_from_transform(camera_from_gripper), vec_from_transform(np.linalg.inv(world_from_base))]

    all_ids = list(range(len(observed)))
    first = least_squares(lambda x: residual(x, all_ids), start, loss="huber", f_scale=1.0,
                          max_nfev=3000)

    def errors(vector):
        camera_from_gripper = transform_from_vec(vector[:6])
        base_from_world = transform_from_vec(vector[6:])
        result = []
        for image_index in all_ids:
            delta = np.linalg.inv(observed[image_index]) @ (
                camera_from_gripper @ gripper_from_base[image_index] @ base_from_world)
            result.append((
                float(np.linalg.norm(delta[:3, 3]) * 1000.0),
                float(np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec()) * 180.0 / np.pi),
            ))
        return result

    preliminary = errors(first.x)
    trans = np.array([x[0] for x in preliminary])
    rot = np.array([x[1] for x in preliminary])
    robust_limit = lambda values, minimum: max(
        minimum, float(np.median(values) + 3.0 * 1.4826 * np.median(np.abs(values - np.median(values)))))
    trans_limit, rot_limit = robust_limit(trans, 50.0), robust_limit(rot, 10.0)
    inlier_ids = [i for i, (t, r) in enumerate(preliminary) if t <= trans_limit and r <= rot_limit]
    if len(inlier_ids) < 6:
        inlier_ids = all_ids
    final = least_squares(lambda x: residual(x, inlier_ids), first.x, loss="huber", f_scale=1.0,
                          max_nfev=3000)
    final_errors = errors(final.x)

    inlier_errors = [final_errors[i] for i in inlier_ids]
    mean_t = float(np.mean([x[0] for x in inlier_errors]))
    max_t = float(np.max([x[0] for x in inlier_errors]))
    mean_r = float(np.mean([x[1] for x in inlier_errors]))
    max_r = float(np.max([x[1] for x in inlier_errors]))
    recommendation = (mean_t <= 20.0 and max_t <= 50.0 and mean_r <= 5.0 and max_r <= 10.0)

    camera_from_gripper = transform_from_vec(final.x[:6])
    base_from_world = transform_from_vec(final.x[6:])
    world_from_base = np.linalg.inv(base_from_world)
    report = {
        "tx_base2world": world_from_base.tolist(),
        "tx_gripper2camera": camera_from_gripper.tolist(),
        "diagnostics": {
            "tag_id": args.tag_id,
            "input_samples": len(samples),
            "tag_valid_samples": len(observed),
            "inlier_original_sample_indices_1based": [sample_indices[i] + 1 for i in inlier_ids],
            "outlier_original_sample_indices_1based": [sample_indices[i] + 1 for i in all_ids if i not in inlier_ids],
            "preliminary_inlier_threshold_translation_mm": trans_limit,
            "preliminary_inlier_threshold_rotation_deg": rot_limit,
            "inlier_mean_translation_mm": mean_t,
            "inlier_max_translation_mm": max_t,
            "inlier_mean_rotation_deg": mean_r,
            "inlier_max_rotation_deg": max_r,
            "per_original_sample": [
                {"sample_index_1based": sample_indices[i] + 1,
                 "translation_error_mm": final_errors[i][0],
                 "rotation_error_deg": final_errors[i][1],
                 "inlier": i in inlier_ids}
                for i in all_ids
            ],
            "urdf_use_recommendation": "YES" if recommendation else "NO",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, output_path.open("w"), indent=2)
    print("AM2PRO_HAND_EYE_ROBUST_VALIDATION_OK")
    print("valid_samples:", len(observed), "inliers:", len(inlier_ids))
    print("inliers (1-based):", report["diagnostics"]["inlier_original_sample_indices_1based"])
    print("outliers (1-based):", report["diagnostics"]["outlier_original_sample_indices_1based"])
    print(f"inlier_translation_mm: mean={mean_t:.2f} max={max_t:.2f}")
    print(f"inlier_rotation_deg: mean={mean_r:.2f} max={max_r:.2f}")
    print("urdf_use_recommendation:", report["diagnostics"]["urdf_use_recommendation"])
    print("saved:", output_path)


if __name__ == "__main__":
    main()
