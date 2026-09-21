"""Validate a tag-anchored visual SLAM trajectory against a fixed ArUco tag.

This is an offline diagnostic.  It never opens a live camera, serial port, or
robot controller.  The fixed known-size tag gives an independent metric camera
pose for every detected video frame.  We align that pose series to the SLAM
world once, then measure residual position/orientation drift.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def load_intrinsics(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))["intrinsics"]
    camera = np.array(
        [[data["focal_length"], 0.0, data["principal_pt_x"]],
         [0.0, data["focal_length"], data["principal_pt_y"]],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    distortion = np.array(
        [data[f"radial_distortion_{i}"] for i in range(1, 5)], dtype=float
    ).reshape(4, 1)
    return camera, distortion


def read_frame_times(path: pathlib.Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as file:
        return np.asarray([float(row["relative_s"]) for row in csv.DictReader(file)])


def read_trajectory(path: pathlib.Path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    lost = np.asarray([row["is_lost"].strip().lower() == "true" for row in rows])
    positions = np.asarray([[float(row[key]) for key in ("x", "y", "z")] for row in rows])
    quaternions = np.asarray(
        [[float(row[key]) for key in ("q_x", "q_y", "q_z", "q_w")] for row in rows]
    )
    return timestamps, lost, positions, quaternions


def detector_for(dictionary_name: str):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None, dictionary, params


def detect(detector, image, dictionary=None, params=None):
    if detector is not None:
        return detector.detectMarkers(image)
    return cv2.aruco.detectMarkers(image, dictionary, parameters=params)


def tag_camera_poses(video: pathlib.Path, frame_times: np.ndarray, tag_id: int,
                     tag_size_m: float, camera: np.ndarray, distortion: np.ndarray,
                     dictionary_name: str):
    detector = detector_for(dictionary_name)
    if isinstance(detector, tuple):
        detector, dictionary, params = detector
    else:
        dictionary = params = None
    half = tag_size_m / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]], dtype=float
    )
    frame_indices, positions, rotations = [], [], []
    print("TAG_TRAJECTORY_REFERENCE_EXTRACTION_STARTED", flush=True)
    with av.open(str(video)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index >= len(frame_times):
                break
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detect(detector, gray, dictionary, params)
            if ids is None:
                continue
            hits = np.flatnonzero(ids.reshape(-1) == tag_id)
            if len(hits) != 1:
                continue
            image_points = np.asarray(corners[int(hits[0])], dtype=float).reshape(4, 1, 2)
            undistorted = cv2.fisheye.undistortPoints(image_points, camera, distortion, P=camera)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, undistorted, camera, np.zeros((4, 1)),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok or not np.isfinite(rvec).all() or tvec[2, 0] <= 0:
                continue
            # solvePnP is Tag -> Camera.  We need Camera -> Tag/world.
            r_camera_tag = Rotation.from_rotvec(rvec.ravel()).as_matrix()
            r_tag_camera = r_camera_tag.T
            p_camera_tag = -r_tag_camera @ tvec.reshape(3)
            frame_indices.append(frame_index)
            positions.append(p_camera_tag)
            rotations.append(r_tag_camera)
            if len(frame_indices) % 750 == 0:
                print(f"已提取 {len(frame_indices)} 帧 Tag 相机位姿", flush=True)
    if len(frame_indices) < 100:
        raise RuntimeError(f"仅检测到 {len(frame_indices)} 帧 Tag {tag_id}，不足以验证轨迹")
    return np.asarray(frame_indices), np.asarray(positions), Rotation.from_matrix(np.asarray(rotations))


def rigid_alignment(reference: np.ndarray, estimate: np.ndarray):
    """Return R,t where estimate ~= reference @ R.T + t (row-vector form)."""
    reference_center = reference.mean(axis=0)
    estimate_center = estimate.mean(axis=0)
    a = reference - reference_center
    b = estimate - estimate_center
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = estimate_center - reference_center @ rotation.T
    return rotation, translation


def max_run(mask: np.ndarray) -> int:
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="用固定 ArUco Tag 验证视觉 SLAM 轨迹")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--trajectory", required=True,
                        help="session-dir 内的轨迹 CSV 文件名")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    session = pathlib.Path(args.session_dir).expanduser().resolve()
    trajectory_name = pathlib.Path(args.trajectory)
    if trajectory_name.name != args.trajectory:
        parser.error("--trajectory 必须是 session-dir 内的单个 CSV 文件名")
    trajectory_path = session / trajectory_name
    required = [session / "raw_video.mp4", session / "frame_timestamps.csv", trajectory_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("缺少文件：\n" + "\n".join(missing))
    out = pathlib.Path(args.out).expanduser().resolve() if args.out else session / (
        trajectory_path.stem + "_tag_validation.json"
    )
    if out.exists():
        raise SystemExit(f"为保护已有报告，拒绝覆盖：{out}")

    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text(encoding="utf-8"))
    sizes = config["marker_size_map"]
    tag_size_m = float(sizes.get(args.tag_id, sizes.get(str(args.tag_id), sizes["default"])))
    frame_times = read_frame_times(session / "frame_timestamps.csv")
    trajectory_t, lost, slam_positions, slam_quaternions = read_trajectory(trajectory_path)
    if len(trajectory_t) != len(frame_times):
        raise RuntimeError("轨迹行数与视频时间戳数量不一致，不能逐帧验证")

    tag_frames, tag_positions, tag_rotations = tag_camera_poses(
        session / "raw_video.mp4", frame_times, args.tag_id, tag_size_m,
        *load_intrinsics(pathlib.Path(args.intrinsics).expanduser().resolve()),
        config["aruco_dict"]["predefined"],
    )
    valid = ~lost[tag_frames]
    if valid.sum() < 100:
        raise RuntimeError("Tag 位姿与有效 SLAM 位姿的重叠帧不足 100")
    indices = tag_frames[valid]
    reference_positions = tag_positions[valid]
    estimated_positions = slam_positions[indices]
    slam_rotations = Rotation.from_quat(slam_quaternions[indices])

    r_world, translation = rigid_alignment(reference_positions, estimated_positions)
    aligned_positions = reference_positions @ r_world.T + translation
    position_error = np.linalg.norm(estimated_positions - aligned_positions, axis=1)
    # Also expose the best similarity scale: visual tag anchoring should be near 1.
    ref_centered = reference_positions - reference_positions.mean(axis=0)
    est_centered = estimated_positions - estimated_positions.mean(axis=0)
    rotated_reference = ref_centered @ r_world.T
    similarity_scale = float(np.sum(est_centered * rotated_reference) /
                             np.sum(rotated_reference * rotated_reference))

    tag_to_slam_rotations = slam_rotations * tag_rotations[valid].inv()
    r_world_orientation = tag_to_slam_rotations.mean()
    orientation_error_deg = np.rad2deg(
        (r_world_orientation.inv() * tag_to_slam_rotations).magnitude()
    )
    lost_run = max_run(lost)
    fps = len(frame_times) / max(frame_times[-1] - frame_times[0], 1e-9)
    pass_recommendation = bool(
        lost.mean() <= 0.01
        and np.quantile(position_error, 0.95) <= 0.10
        and np.quantile(orientation_error_deg, 0.95) <= 10.0
        and 0.85 <= similarity_scale <= 1.15
    )
    report = {
        "schema": "am_umi_tag_anchored_visual_trajectory_validation_v1",
        "session": str(session), "trajectory": str(trajectory_path),
        "tag_id": args.tag_id, "tag_size_m": tag_size_m,
        "trajectory_frame_count": int(len(lost)),
        "tag_reference_frame_count": int(len(tag_frames)),
        "matched_valid_frame_count": int(valid.sum()),
        "lost_ratio": float(lost.mean()),
        "longest_lost_s": float(lost_run / fps),
        "similarity_scale_slam_per_tag_meter": similarity_scale,
        "position_error_m": {
            "median": float(np.median(position_error)),
            "p95": float(np.quantile(position_error, 0.95)),
            "max": float(position_error.max()),
        },
        "orientation_error_deg": {
            "median": float(np.median(orientation_error_deg)),
            "p95": float(np.quantile(orientation_error_deg, 0.95)),
            "max": float(orientation_error_deg.max()),
        },
        "tag_world_to_slam_rotation": r_world.tolist(),
        "tag_world_to_slam_translation_m": translation.tolist(),
        "pass_recommendation": pass_recommendation,
        "note": "Tag pose is an independent fixed-world reference. Passing validates this recording's visual trajectory; it does not validate IMU/VIO fusion.",
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("TAG_ANCHORED_VISUAL_TRAJECTORY_VALIDATION_OK")
    print(f"matched_valid_frames: {valid.sum()} / {len(lost)}; lost_ratio: {lost.mean():.3%}")
    print(f"scale_slam_per_tag_meter: {similarity_scale:.4f}")
    print("position_error_m: median={:.4f}, p95={:.4f}, max={:.4f}".format(
        np.median(position_error), np.quantile(position_error, 0.95), position_error.max()))
    print("orientation_error_deg: median={:.3f}, p95={:.3f}, max={:.3f}".format(
        np.median(orientation_error_deg), np.quantile(orientation_error_deg, 0.95), orientation_error_deg.max()))
    print("pass_recommendation:", "YES" if pass_recommendation else "NO")
    print("report:", out)


if __name__ == "__main__":
    main()
