#!/usr/bin/env python3
"""Convert one AM_UMI hand-held fixed-Tag demo into a policy-ready zarr.

This is deliberately a small, single-camera conversion path for the current
EMEET hand-held rig.  It does *not* use IMU/VIO.  The TCP trajectory comes
from the direct multi-world-Tag camera trajectory plus the measured
camera-to-TCP geometry.  The RGB array uses exactly the same crop/mask path as
the AM2Pro policy reference, rather than resizing the entire fisheye frame.

The command is offline only: it never opens the serial port and never sends a
robot command.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from umi.common.cv_util import (  # noqa: E402
    get_dual_tag_gripper_width,
    get_relative_tag_rotation_deg,
)
from umi.common.pose_util import mat_to_pose, pose_to_mat  # noqa: E402
from umi.common.view_reference import preprocess_uvc_bgr  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将固定世界 Tag 的手持演示离线转为 AM_UMI zarr；绝不控制机械臂。")
    parser.add_argument("--session-dir", required=True, help="录制目录")
    parser.add_argument("--output", required=True, help="新的输出 .zarr 目录（必须不存在）")
    parser.add_argument("--trajectory-name", default="camera_trajectory_multi_tag_robust.csv")
    parser.add_argument("--tag-detection-name", default="tag_detection.pkl",
                        help="会话目录内用于夹爪宽度的 Tag 检测 pkl；默认 tag_detection.pkl")
    parser.add_argument("--camera-tcp-geometry", required=True,
                        help="含 accepted pose_cam_tcp 的相机到 TCP 几何 JSON")
    parser.add_argument("--allow-provisional-camera-tcp-geometry", action="store_true",
                        help="仅用于低速诊断：允许 status=candidate 的未正式验证几何")
    parser.add_argument("--gripper-range", required=True,
                        help="双指 ArUco 开合范围 JSON")
    parser.add_argument("--reference", required=True,
                        help="right_tcp 基准目录中的 reference.json，用它读取策略裁剪")
    parser.add_argument("--gripper-max-m", type=float, default=0.074,
                        help="新夹爪的物理最大开口，默认 0.074 m")
    parser.add_argument("--gripper-smooth-window-frames", type=int, default=1,
                        help=("夹爪视觉宽度的离线稳健平滑窗口；1=关闭。"
                              "V 型夹爪正式处理建议 9（约 0.3 秒）。"))
    parser.add_argument("--nominal-tag-depth-m", type=float, default=None,
                        help="覆盖 gripper-range 记录的双指 Tag 标称深度")
    return parser.parse_args()


def load_transform(path: Path, allow_provisional: bool = False) -> tuple[np.ndarray, str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "am_umi_camera_tcp_geometry_v1":
        raise ValueError(f"未知相机-TCP 几何格式：{path}")
    status = data.get("status")
    if status not in (None, "accepted"):
        if status != "candidate" or not allow_provisional:
            raise ValueError(f"相机-TCP 几何未通过验证：{path}")
    pose = np.asarray(data.get("pose_cam_tcp"), dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose_cam_tcp 必须是 6 个有限数值")
    return pose_to_mat(pose), status


def load_crop(reference: Path) -> dict:
    data = json.loads(reference.read_text(encoding="utf-8"))
    crop = data.get("camera_config", {}).get("crop")
    if not isinstance(crop, dict) or "center" not in crop or "size" not in crop:
        raise ValueError("reference.json 缺少 camera_config/crop")
    if len(crop["center"]) != 2 or int(crop["size"]) <= 0:
        raise ValueError("reference.json 的 crop 无效")
    return {"center": [int(x) for x in crop["center"]], "size": int(crop["size"])}


def smooth_gripper_widths(widths: np.ndarray, window_frames: int,
                          max_width_m: float) -> np.ndarray:
    """Suppress per-frame visual width noise without inventing overshoot.

    A centered median removes isolated PnP/tag jitter; the following box
    average makes the remaining command path differentiable enough for the
    single moving jaw.  This is offline label processing, so centering the
    window does not add deployment latency.  Widths remain bounded by their
    physical range.
    """
    values = np.asarray(widths, dtype=np.float64).reshape(-1)
    if window_frames <= 1:
        return values.astype(np.float32)
    if window_frames % 2 == 0:
        raise ValueError("--gripper-smooth-window-frames 必须为奇数，或设为 1 关闭")
    filtered = median_filter(values, size=window_frames, mode="nearest")
    filtered = uniform_filter1d(filtered, size=window_frames, mode="nearest")
    return np.clip(filtered, 0.0, max_width_m).astype(np.float32)


def add_width_smoothing_report(report: dict, raw: np.ndarray, filtered: np.ndarray,
                               window_frames: int) -> dict:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    filtered = np.asarray(filtered, dtype=np.float64).reshape(-1)
    report["width_smoothing"] = {
        "method": "median_then_box_average" if window_frames > 1 else "none",
        "window_frames": int(window_frames),
        "raw_width_range_m": [float(raw.min()), float(raw.max())],
        "smoothed_width_range_m": [float(filtered.min()), float(filtered.max())],
        "raw_total_variation_m": float(np.abs(np.diff(raw)).sum()),
        "smoothed_total_variation_m": float(np.abs(np.diff(filtered)).sum()),
    }
    return report


def interpolate_widths(detections: list, gripper_range: dict,
                       max_width_m: float, nominal_depth: float | None,
                       smooth_window_frames: int) -> tuple[np.ndarray, dict]:
    measurement = gripper_range.get("width_measurement", "dual_tag_abs_camera_x_m")
    if measurement == "relative_tag_rotation_deg":
        fixed_id = int(gripper_range["fixed_jaw_tag_id"])
        moving_id = int(gripper_range["moving_jaw_tag_id"])
        raw_endpoints = np.asarray(gripper_range.get("raw_closed_open"), dtype=np.float64)
        if raw_endpoints.shape != (2,) or abs(raw_endpoints[1] - raw_endpoints[0]) < 1e-6:
            raise ValueError("V 型夹爪标定的 raw_closed_open 无效")
        sample_indices, raw_values = [], []
        for index, frame in enumerate(detections):
            raw = get_relative_tag_rotation_deg(frame["tag_dict"], fixed_id, moving_id)
            if raw is not None and np.isfinite(raw):
                sample_indices.append(index)
                raw_values.append(float(raw))
        if len(sample_indices) < 2:
            raise ValueError("有效固定爪/活动爪双 Tag 帧不足 2，不能恢复 V 型夹爪开合")
        frames = np.arange(len(detections), dtype=np.float64)
        raw_full = np.interp(frames, np.asarray(sample_indices), np.asarray(raw_values))
        unclipped = ((raw_full - raw_endpoints[0]) /
                     (raw_endpoints[1] - raw_endpoints[0]) * max_width_m)
        raw_widths = np.clip(unclipped, 0.0, max_width_m)
        widths = smooth_gripper_widths(raw_widths, smooth_window_frames, max_width_m)
        report = {
            "measurement": measurement,
            "fixed_jaw_tag_id": fixed_id,
            "moving_jaw_tag_id": moving_id,
            "dual_tag_frames": int(len(sample_indices)),
            "dual_tag_ratio": float(len(sample_indices) / len(detections)),
            "raw_measurement_range_deg": [float(np.min(raw_values)), float(np.max(raw_values))],
            "raw_calibration_endpoints": raw_endpoints.tolist(),
            "physical_gripper_range_m": [0.0, float(max_width_m)],
            "endpoint_clipped_frames": int(np.count_nonzero(np.abs(unclipped - raw_widths) > 1e-8)),
        }
        return widths, add_width_smoothing_report(
            report, raw_widths, widths, smooth_window_frames)
    if measurement != "dual_tag_abs_camera_x_m":
        raise ValueError(f"不支持的 gripper-range width_measurement: {measurement}")
    left_id = int(gripper_range.get("left_finger_tag_id", 0))
    right_id = int(gripper_range.get("right_finger_tag_id", 1))
    raw_endpoints = np.asarray(gripper_range.get("aruco_measured_width"), dtype=np.float64)
    if raw_endpoints.shape != (2,) or raw_endpoints[1] <= raw_endpoints[0]:
        raise ValueError("gripper-range 的 aruco_measured_width 无效")
    if max_width_m <= 0:
        raise ValueError("--gripper-max-m 必须为正")
    if nominal_depth is None:
        nominal_depth = float(gripper_range.get("nominal_tag_depth_m", 0.082))

    sample_indices, raw_values = [], []
    for index, frame in enumerate(detections):
        raw = get_dual_tag_gripper_width(
            frame["tag_dict"], left_id, right_id, nominal_z=nominal_depth)
        if raw is not None and np.isfinite(raw):
            sample_indices.append(index)
            raw_values.append(float(raw))
    if len(sample_indices) < 2:
        raise ValueError("有效双指 Tag 帧不足 2，不能恢复夹爪开合")

    frames = np.arange(len(detections), dtype=np.float64)
    raw_full = np.interp(frames, np.asarray(sample_indices), np.asarray(raw_values))
    unclipped = ((raw_full - raw_endpoints[0]) /
                 (raw_endpoints[1] - raw_endpoints[0]) * max_width_m)
    raw_widths = np.clip(unclipped, 0.0, max_width_m)
    widths = smooth_gripper_widths(raw_widths, smooth_window_frames, max_width_m)
    report = {
        "left_finger_tag_id": left_id,
        "right_finger_tag_id": right_id,
        "dual_tag_frames": int(len(sample_indices)),
        "dual_tag_ratio": float(len(sample_indices) / len(detections)),
        "raw_measurement_range_m": [float(np.min(raw_values)), float(np.max(raw_values))],
        "raw_calibration_endpoints_m": raw_endpoints.tolist(),
        "physical_gripper_range_m": [0.0, float(max_width_m)],
        "endpoint_clipped_frames": int(np.count_nonzero(np.abs(unclipped - raw_widths) > 1e-8)),
    }
    return widths, add_width_smoothing_report(
        report, raw_widths, widths, smooth_window_frames)


def main() -> None:
    args = parse_args()
    session = Path(args.session_dir).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"为保护已有输出，拒绝覆盖：{output}")
    trajectory_path = session / args.trajectory_name
    video_path = session / "raw_video.mp4"
    detection_path = session / args.tag_detection_name
    for path in (trajectory_path, video_path, detection_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tx_cam_tcp, geometry_status = load_transform(
        Path(args.camera_tcp_geometry).expanduser().resolve(),
        allow_provisional=args.allow_provisional_camera_tcp_geometry)
    crop = load_crop(Path(args.reference).expanduser().resolve())
    gripper_range = json.loads(Path(args.gripper_range).expanduser().read_text(encoding="utf-8"))
    detections = pickle.load(detection_path.open("rb"))
    trajectory = pd.read_csv(trajectory_path)
    required_columns = {"is_lost", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"}
    missing = required_columns.difference(trajectory.columns)
    if missing:
        raise ValueError(f"轨迹 CSV 缺少字段：{sorted(missing)}")
    if len(trajectory) != len(detections):
        raise ValueError(f"轨迹 {len(trajectory)} 帧与 Tag 检测 {len(detections)} 帧不一致")
    lost = trajectory["is_lost"].astype(str).str.lower().isin(["true", "1"])
    if bool(lost.any()):
        raise ValueError(f"轨迹含 {int(lost.sum())} 帧丢失；请先修复/分段，不生成混入丢失帧的 zarr")

    camera_pos = trajectory[["x", "y", "z"]].to_numpy(dtype=np.float64)
    camera_rot = Rotation.from_quat(trajectory[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=np.float64))
    tx_world_camera = np.tile(np.eye(4, dtype=np.float64), (len(trajectory), 1, 1))
    tx_world_camera[:, :3, :3] = camera_rot.as_matrix()
    tx_world_camera[:, :3, 3] = camera_pos
    tcp_pose = mat_to_pose(tx_world_camera @ tx_cam_tcp).astype(np.float32)
    widths, width_report = interpolate_widths(
        detections, gripper_range, args.gripper_max_m, args.nominal_tag_depth_m,
        args.gripper_smooth_window_frames)

    images = []
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            bgr = frame.to_ndarray(format="bgr24")
            images.append(preprocess_uvc_bgr(bgr, camera_crop=crop))
    images = np.asarray(images, dtype=np.uint8)
    if len(images) != len(tcp_pose):
        raise ValueError(f"视频解码 {len(images)} 帧与轨迹 {len(tcp_pose)} 帧不一致")

    action = np.concatenate([tcp_pose[:, :3], tcp_pose[:, 3:], widths[:, None]], axis=1)
    buffer = ReplayBuffer.create_empty_numpy()
    buffer.add_episode({
        "action": action.astype(np.float32),
        "camera0_rgb": images,
        "robot0_eef_pos": tcp_pose[:, :3],
        "robot0_eef_rot_axis_angle": tcp_pose[:, 3:],
        "robot0_gripper_width": widths[:, None],
        "robot0_demo_start_pose": np.repeat(tcp_pose[:1], len(tcp_pose), axis=0),
        "robot0_demo_end_pose": np.repeat(tcp_pose[-1:], len(tcp_pose), axis=0),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer.save_to_path(str(output), compressors="disk")

    timestamps = trajectory.get("timestamp", pd.Series(np.arange(len(trajectory)) / 30.0)).to_numpy(dtype=float)
    dt = np.diff(timestamps)
    report = {
        "schema": "am_umi_handheld_fixed_tag_zarr_conversion_v1",
        "safety": "offline only; no serial port or robot command was used",
        "session": str(session),
        "output": str(output),
        "trajectory": str(trajectory_path),
        "tag_detection": str(detection_path),
        "camera_tcp_geometry": str(Path(args.camera_tcp_geometry).expanduser().resolve()),
        "camera_tcp_geometry_status": geometry_status,
        "demonstrated_tcp_definition": json.loads(
            Path(args.camera_tcp_geometry).expanduser().read_text(encoding="utf-8")
        ).get("tcp_definition"),
        "reference": str(Path(args.reference).expanduser().resolve()),
        "policy_crop": crop,
        "frames": int(len(tcp_pose)),
        "duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "median_fps": float(1.0 / np.median(dt)) if len(dt) and np.median(dt) > 0 else None,
        "tcp_path_length_m": float(np.linalg.norm(np.diff(tcp_pose[:, :3], axis=0), axis=1).sum()),
        "gripper": width_report,
        "image_shape": list(images.shape),
        "limitations": [
            "轨迹来自固定世界 Tag 的直接视觉位姿，不使用 IMU/VIO。",
            "这是离线运动学候选；仍须经过 IK 筛选和后续低速实物 replay 才能保留为训练数据。",
            *( ["本次转换使用了未正式验证的 camera-to-TCP 旋转候选；只可用于诊断。"]
               if geometry_status == "candidate" else []),
        ],
    }
    report_path = output / "handheld_fixed_tag_conversion.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("HANDHELD_FIXED_TAG_ZARR_CONVERSION_OK")
    print("output:", output)
    print("frames:", len(tcp_pose), "tcp_path_length_m:", f"{report['tcp_path_length_m']:.4f}")
    print("dual_finger_tag_ratio:", f"{width_report['dual_tag_ratio']:.3%}",
          "endpoint_clipped_frames:", width_report["endpoint_clipped_frames"])
    print("policy_image_shape:", tuple(images.shape[1:]))
    print("report:", report_path)


if __name__ == "__main__":
    main()
