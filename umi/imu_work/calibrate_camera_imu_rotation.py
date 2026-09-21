"""Estimate hand-held camera<-IMU rotation and timing from a fixed ArUco tag.

The tag is fixed in the scene.  Its frame-to-frame pose gives camera angular
velocity; that signal is aligned to JY901B gyro measurements.  This estimates
only the rotational part of the camera--IMU extrinsic and a host-clock time
offset.  It is an offline operation and never opens a device or moves a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import load_preferred_frame_times


def load_frame_times(path: pathlib.Path) -> np.ndarray:
    times, _ = load_preferred_frame_times(path.parent)
    return times


def camera_matrix_from_json(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("intrinsic_type") != "FISHEYE":
        raise RuntimeError("当前求解器需要 FISHEYE 格式的相机内参 JSON")
    item = data["intrinsics"]
    focal = float(item["focal_length"])
    camera_matrix = np.array([
        [focal, 0.0, float(item["principal_pt_x"])],
        [0.0, focal, float(item["principal_pt_y"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    distortion = np.asarray([
        item["radial_distortion_1"], item["radial_distortion_2"],
        item["radial_distortion_3"], item["radial_distortion_4"],
    ], dtype=np.float64).reshape(4, 1)
    return camera_matrix, distortion


def tag_size_from_config(path: pathlib.Path, tag_id: int) -> float:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    sizes = config["marker_size_map"]
    size = sizes.get(tag_id, sizes.get(str(tag_id), sizes.get("default")))
    if size is None or float(size) <= 0:
        raise RuntimeError(f"ArUco 配置中找不到 Tag {tag_id} 的物理边长")
    return float(size)


def pose_samples(video_path: pathlib.Path, frame_times: np.ndarray,
                 tag_id: int, tag_size_m: float,
                 camera_matrix: np.ndarray, distortion: np.ndarray,
                 dictionary_name: str) -> tuple[np.ndarray, list[Rotation], int]:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                if hasattr(cv2.aruco, "ArucoDetector") else None)
    half = tag_size_m / 2.0
    object_points = np.asarray([
        [-half, +half, 0.0], [+half, +half, 0.0],
        [+half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=np.float64)
    zero_distortion = np.zeros((4, 1), dtype=np.float64)
    times: list[float] = []
    rotations: list[Rotation] = []
    decoded = 0

    print("TAG_POSE_EXTRACTION_STARTED", flush=True)
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if frame_index >= len(frame_times):
                raise RuntimeError("视频帧数超过 frame_timestamps.csv")
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, dictionary, parameters=parameters)
            if ids is None:
                continue
            matching = np.flatnonzero(ids.reshape(-1) == tag_id)
            if len(matching) != 1:
                continue
            image_points = np.asarray(corners[int(matching[0])], dtype=np.float64).reshape(4, 1, 2)
            undistorted = cv2.fisheye.undistortPoints(
                image_points, camera_matrix, distortion, P=camera_matrix)
            ok, rvec, tvec = cv2.solvePnP(
                object_points, undistorted, camera_matrix, zero_distortion,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok or not np.isfinite(rvec).all() or tvec[2, 0] <= 0:
                continue
            times.append(float(frame_times[frame_index]))
            rotations.append(Rotation.from_rotvec(rvec.reshape(3)))
            if len(times) % 500 == 0:
                print(f"已得到 Tag {tag_id} 位姿 {len(times)} 帧", flush=True)
    if decoded != len(frame_times):
        raise RuntimeError(f"视频 {decoded} 帧与时间戳 {len(frame_times)} 行不一致")
    return np.asarray(times), rotations, decoded


def camera_angular_velocity(times: np.ndarray, rotations: list[Rotation],
                            pair_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate angular velocity expressed in camera coordinates.

    ``R_ct`` maps the fixed tag frame to camera frame.  The relative rotation
    R_ct(t2) @ R_ct(t1).T is the inverse of camera body rotation, hence the
    leading minus sign below.
    """
    out_times = []
    out_omega = []
    for first in range(len(times) - 1):
        target = times[first] + pair_seconds
        second = int(np.searchsorted(times, target, side="left"))
        if second >= len(times):
            break
        delta_t = times[second] - times[first]
        if not (pair_seconds * 0.75 <= delta_t <= pair_seconds * 1.35):
            continue
        relative = rotations[second] * rotations[first].inv()
        omega = -relative.as_rotvec() / delta_t
        speed = float(np.linalg.norm(omega))
        if 0.05 <= speed <= 4.0:
            out_times.append((times[first] + times[second]) / 2.0)
            out_omega.append(omega)
    if len(out_times) < 80:
        raise RuntimeError(
            f"只有 {len(out_times)} 个可用相机角速度样本；请确认固定 Tag 在旋转时清晰可见")
    return np.asarray(out_times), np.asarray(out_omega)


def interpolate_gyro(times: np.ndarray, gyro_times: np.ndarray,
                     gyro_rad_s: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(times, gyro_times, gyro_rad_s[:, axis]) for axis in range(3)
    ])


def fit_rotation_and_offset(camera_times: np.ndarray, camera_omega: np.ndarray,
                            gyro_times: np.ndarray, gyro_rad_s: np.ndarray,
                            max_offset_s: float, offset_step_s: float):
    candidates = []
    offsets = np.arange(-max_offset_s, max_offset_s + offset_step_s / 2, offset_step_s)
    for offset in offsets:
        query_times = camera_times + offset
        valid = ((query_times >= gyro_times[0]) & (query_times <= gyro_times[-1]))
        if valid.sum() < 80:
            continue
        camera = camera_omega[valid]
        gyro = interpolate_gyro(query_times[valid], gyro_times, gyro_rad_s)
        # Emphasize intentional rotation and de-emphasize tiny pose noise.
        weights = np.clip(np.linalg.norm(camera, axis=1), 0.1, 2.0)
        transform, _ = Rotation.align_vectors(camera, gyro, weights=weights)
        residual = camera - transform.apply(gyro)
        rmse = float(np.sqrt(np.average(np.sum(residual ** 2, axis=1), weights=weights)))
        correlation = float(np.sum(weights * np.sum(camera * transform.apply(gyro), axis=1)) /
                            np.sqrt(np.sum(weights * np.sum(camera ** 2, axis=1)) *
                                    np.sum(weights * np.sum(gyro ** 2, axis=1))))
        candidates.append((rmse, -correlation, float(offset), transform, int(valid.sum()), correlation))
    if not candidates:
        raise RuntimeError("没有覆盖相机时间范围的 IMU 数据")
    best = min(candidates, key=lambda value: (value[0], value[1]))
    return best, candidates


def main():
    parser = argparse.ArgumentParser(description="离线估计手持相机—IMU 旋转外参和时间偏移")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--tag-id", type=int, default=13,
                        help="固定在桌面的 ArUco Tag 编号（不是夹爪上的 0、1）")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--pair-seconds", type=float, default=0.20,
                        help="相隔多久的 Tag 位姿用于计算一次相机角速度")
    parser.add_argument("--max-offset-seconds", type=float, default=0.25)
    parser.add_argument("--offset-step-seconds", type=float, default=0.002)
    parser.add_argument("--out", required=True,
                        help="新的 JSON 输出；为保护已有结果拒绝覆盖")
    args = parser.parse_args()
    if args.pair_seconds <= 0 or args.max_offset_seconds <= 0 or args.offset_step_seconds <= 0:
        parser.error("时间参数必须为正数")

    session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
    output_path = pathlib.Path(args.out).expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"为保护已有标定，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_times, frame_time_path = load_preferred_frame_times(session_dir)
    print("camera_frame_timestamp_source:", frame_time_path)
    imu = np.load(session_dir / "imu_raw.npz")
    required = {"t_gyro_monotonic_s", "gyro_raw", "gyro_scale_deg_s_per_lsb", "recording_start_monotonic_ns"}
    missing = required - set(imu.files)
    if missing:
        raise RuntimeError(f"imu_raw.npz 缺少字段：{sorted(missing)}")
    bias_path = pathlib.Path(args.gyro_bias).expanduser().resolve()
    bias_deg_s = np.asarray(json.loads(bias_path.read_text(encoding="utf-8"))["bias_deg_s"], dtype=np.float64)
    if bias_deg_s.shape != (3,):
        raise RuntimeError("gyro bias 文件的 bias_deg_s 格式应为三个数")
    gyro_times = np.asarray(imu["t_gyro_monotonic_s"], dtype=np.float64)
    recording_start_s = float(imu["recording_start_monotonic_ns"]) / 1e9
    gyro_times = gyro_times - recording_start_s
    gyro_deg_s = np.asarray(imu["gyro_raw"], dtype=np.float64) * float(imu["gyro_scale_deg_s_per_lsb"])
    gyro_rad_s = np.deg2rad(gyro_deg_s - bias_deg_s)
    order = np.argsort(gyro_times)
    gyro_times, gyro_rad_s = gyro_times[order], gyro_rad_s[order]

    config_path = pathlib.Path(args.aruco_config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dictionary_name = config["aruco_dict"]["predefined"]
    camera_matrix, distortion = camera_matrix_from_json(pathlib.Path(args.intrinsics).expanduser().resolve())
    tag_size_m = tag_size_from_config(config_path, args.tag_id)
    tag_times, tag_rotations, decoded = pose_samples(
        session_dir / "raw_video.mp4", frame_times, args.tag_id, tag_size_m,
        camera_matrix, distortion, dictionary_name)
    camera_times, camera_omega = camera_angular_velocity(
        tag_times, tag_rotations, args.pair_seconds)
    print(f"TAG_POSE_EXTRACTION_OK: {len(tag_times)} Tag 位姿帧；{len(camera_times)} 个相机角速度样本")
    print("ROTATION_TIME_ALIGNMENT_STARTED", flush=True)
    best, candidates = fit_rotation_and_offset(
        camera_times, camera_omega, gyro_times, gyro_rad_s,
        args.max_offset_seconds, args.offset_step_seconds)
    rmse, _, offset_s, rotation_ci, count, correlation = best
    matrix_ci = rotation_ci.as_matrix()
    result = {
        "schema": "am_umi_camera_imu_rotation_time_v1",
        "session_dir": str(session_dir),
        "fixed_tag_id": args.tag_id,
        "fixed_tag_size_m": tag_size_m,
        "intrinsics_source": str(pathlib.Path(args.intrinsics).expanduser().resolve()),
        "gyro_bias_source": str(bias_path),
        "camera_convention": "OpenCV: +X image right, +Y image down, +Z forward through lens",
        "rotation_name": "R_camera_imu",
        "rotation_definition": "omega_camera_rad_s ~= R_camera_imu @ omega_imu_rad_s",
        "R_camera_imu": matrix_ci.tolist(),
        "euler_xyz_deg": rotation_ci.as_euler("xyz", degrees=True).tolist(),
        "imu_time_offset_for_camera_s": offset_s,
        "time_offset_definition": "for camera host time t, compare IMU at t + imu_time_offset_for_camera_s",
        "pair_seconds": args.pair_seconds,
        "tag_pose_frames": len(tag_times),
        "camera_angular_velocity_samples": count,
        "weighted_angular_velocity_rmse_rad_s": rmse,
        "weighted_vector_correlation": correlation,
        "offset_search": {
            "min_s": -args.max_offset_seconds, "max_s": args.max_offset_seconds,
            "step_s": args.offset_step_seconds,
            "best_five": [
                {"rmse_rad_s": row[0], "correlation": row[5], "offset_s": row[2]}
                for row in sorted(candidates, key=lambda value: (value[0], value[1]))[:5]
            ],
        },
        "note": "Preliminary rotational extrinsic/time estimate from one fixed Tag. Validate on an independent session before using it for formal SLAM.",
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CAMERA_IMU_ROTATION_TIME_CALIBRATION_OK")
    print(f"R_camera_imu euler_xyz_deg: {result['euler_xyz_deg']}")
    print(f"imu_time_offset_for_camera_s: {offset_s:+.4f}")
    print(f"weighted_vector_correlation: {correlation:.4f}")
    print(f"weighted_angular_velocity_rmse_rad_s: {rmse:.4f}")
    print("saved:", output_path)


if __name__ == "__main__":
    main()
