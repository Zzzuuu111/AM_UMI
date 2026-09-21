"""Validate a saved camera--IMU rotation/time calibration on another session."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import load_preferred_frame_times

from calibrate_camera_imu_rotation import (
    camera_angular_velocity,
    camera_matrix_from_json,
    interpolate_gyro,
    load_frame_times,
    pose_samples,
    tag_size_from_config,
)


def main():
    parser = argparse.ArgumentParser(description="用独立录制验证相机—IMU 旋转/时间标定")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--pair-seconds", type=float, default=0.20)
    parser.add_argument("--report-name", default="camera_imu_rotation_validation.json")
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
    calibration_path = pathlib.Path(args.calibration).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = np.asarray(calibration["R_camera_imu"], dtype=np.float64)
    if matrix.shape != (3, 3):
        raise RuntimeError("标定文件 R_camera_imu 必须是 3×3 矩阵")
    rotation_ci = Rotation.from_matrix(matrix)
    offset_s = float(calibration["imu_time_offset_for_camera_s"])

    bias = np.asarray(json.loads(pathlib.Path(args.gyro_bias).expanduser().read_text(
        encoding="utf-8"))["bias_deg_s"], dtype=np.float64)
    frame_times, frame_time_path = load_preferred_frame_times(session_dir)
    print("camera_frame_timestamp_source:", frame_time_path)
    imu = np.load(session_dir / "imu_raw.npz")
    gyro_times = np.asarray(imu["t_gyro_monotonic_s"], dtype=np.float64)
    gyro_times -= float(imu["recording_start_monotonic_ns"]) / 1e9
    gyro = np.asarray(imu["gyro_raw"], dtype=np.float64)
    gyro *= float(imu["gyro_scale_deg_s_per_lsb"])
    gyro = np.deg2rad(gyro - bias)
    order = np.argsort(gyro_times)
    gyro_times, gyro = gyro_times[order], gyro[order]

    config_path = pathlib.Path(args.aruco_config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    camera_matrix, distortion = camera_matrix_from_json(
        pathlib.Path(args.intrinsics).expanduser().resolve())
    tag_times, tag_rotations, _ = pose_samples(
        session_dir / "raw_video.mp4", frame_times, args.tag_id,
        tag_size_from_config(config_path, args.tag_id), camera_matrix, distortion,
        config["aruco_dict"]["predefined"])
    camera_times, camera_omega = camera_angular_velocity(
        tag_times, tag_rotations, args.pair_seconds)
    query_times = camera_times + offset_s
    valid = ((query_times >= gyro_times[0]) & (query_times <= gyro_times[-1]))
    if valid.sum() < 80:
        raise RuntimeError("独立会话中可比较的相机/IMU 样本不足")
    camera_omega = camera_omega[valid]
    imu_in_camera = rotation_ci.apply(interpolate_gyro(query_times[valid], gyro_times, gyro))
    weights = np.clip(np.linalg.norm(camera_omega, axis=1), 0.1, 2.0)
    residual = camera_omega - imu_in_camera
    rmse = float(np.sqrt(np.average(np.sum(residual ** 2, axis=1), weights=weights)))
    correlation = float(np.sum(weights * np.sum(camera_omega * imu_in_camera, axis=1)) /
                        np.sqrt(np.sum(weights * np.sum(camera_omega ** 2, axis=1)) *
                                np.sum(weights * np.sum(imu_in_camera ** 2, axis=1))))
    report = {
        "schema": "am_umi_camera_imu_rotation_validation_v1",
        "session_dir": str(session_dir),
        "calibration": str(calibration_path),
        "fixed_tag_id": args.tag_id,
        "camera_frame_timestamp_source": str(frame_time_path),
        "applied_imu_time_offset_for_camera_s": offset_s,
        "tag_pose_frames": len(tag_times),
        "camera_angular_velocity_samples": int(valid.sum()),
        "weighted_vector_correlation": correlation,
        "weighted_angular_velocity_rmse_rad_s": rmse,
        "pass_recommendation": bool(correlation >= 0.85 and rmse <= 0.20),
        "thresholds": {"min_correlation": 0.85, "max_rmse_rad_s": 0.20},
    }
    report_path = session_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CAMERA_IMU_ROTATION_VALIDATION_OK")
    print("session:", session_dir)
    print("tag_pose_frames:", len(tag_times))
    print("camera_angular_velocity_samples:", int(valid.sum()))
    print(f"weighted_vector_correlation: {correlation:.4f}")
    print(f"weighted_angular_velocity_rmse_rad_s: {rmse:.4f}")
    print("pass_recommendation:", "YES" if report["pass_recommendation"] else "NO")
    print("report:", report_path)


if __name__ == "__main__":
    main()
