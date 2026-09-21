#!/usr/bin/env python3
"""Read-only live validation of a V-jaw encoder-to-URDF candidate.

It compares the camera pose predicted from current servo readbacks against the
pose observed from one fixed world ArUco Tag.  No motor register is written:
the program never enables torque and never sends a joint/gripper command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402
from scripts.fit_am2pro_vjaw_joint_model import JOINT_NAMES, root_transform  # noqa: E402
from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


ARM_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex",
             "wrist_flex", "wrist_yaw", "wrist_roll")
MOTORS = {
    "shoulder_pan": Motor(1, "sts3250", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3095", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3095", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3250", MotorNormMode.DEGREES),
    "wrist_yaw": Motor(5, "sts3250", MotorNormMode.DEGREES),
    "wrist_roll": Motor(6, "sts3250", MotorNormMode.DEGREES),
    "gripper": Motor(7, "sts3250", MotorNormMode.RANGE_0_100),
}


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def pose_error(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    delta = np.linalg.inv(observed) @ predicted
    return (float(np.linalg.norm(delta[:3, 3]) * 1000.0),
            float(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--intrinsics", default="calibration/robot_wrist_camera/intrinsics/emeet_wrist_1920x1080_30fps_fisheye_v1.json")
    parser.add_argument("--aruco-yaml", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--camera-device", default="/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("--samples 至少为 5")

    candidate = json.loads(resolve(args.candidate).read_text(encoding="utf-8"))
    if candidate.get("status") != "candidate":
        parser.error("candidate 状态不是 candidate")
    best = candidate.get("best_candidate", {})
    sign = np.asarray(best.get("sign"), dtype=np.float64)
    offset = np.asarray(best.get("offset_deg"), dtype=np.float64)
    base_from_world = np.asarray(best.get("base_from_world"), dtype=np.float64)
    if sign.shape != (6,) or offset.shape != (6,) or base_from_world.shape != (4, 4):
        parser.error("candidate 缺少完整 sign / offset_deg / base_from_world")

    config = yaml.safe_load(resolve(args.robot_config).read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser().resolve()
    if robot.get("ik_backend") != "placo" or not urdf_path.is_file():
        parser.error("robot-config 必须引用存在的官方 URDF 且 ik_backend=placo")
    tcp_from_camera = np.linalg.inv(root_transform(urdf_path, "right_tcp")) @ root_transform(urdf_path, "right_camera")
    backend = create_kinematics_backend("placo", str(urdf_path), JOINT_NAMES, "right_tcp")

    aruco = parse_aruco_config(yaml.safe_load(resolve(args.aruco_yaml).read_text(encoding="utf-8")))
    raw_intrinsics = parse_fisheye_intrinsics(json.loads(resolve(args.intrinsics).read_text(encoding="utf-8")))
    camera_path = os.path.realpath(args.camera_device)
    bus = FeetechMotorsBus(port=args.port, motors=MOTORS)
    camera = None
    errors, q_records = [], []
    try:
        bus.connect()
        bus.calibration = bus.read_calibration()
        camera = av.open(camera_path, format="v4l2", options={
            "input_format": "mjpeg", "video_size": "1920x1080", "framerate": "30"})
        frames = camera.decode(video=0)
        print("VJAW_LIVE_MODEL_CHECK_STARTED")
        print("READ_ONLY: 不启用扭矩、不写电机、不发送任何运动命令。")
        print("保持机械臂静止，Tag %d 清晰可见；正在采样…" % args.tag_id)
        attempts = 0
        while len(errors) < args.samples and attempts < args.samples * 5:
            attempts += 1
            image = next(frames).to_ndarray(format="rgb24")
            intrinsics = convert_fisheye_intrinsics_resolution(
                raw_intrinsics, target_resolution=(image.shape[1], image.shape[0]))
            tags = detect_localize_aruco_tags(
                image, aruco["aruco_dict"], aruco["marker_size_map"], intrinsics)
            if args.tag_id not in tags:
                continue
            positions = bus.sync_read("Present_Position", ARM_NAMES)
            raw_q = np.asarray([float(positions[name]) for name in ARM_NAMES])
            q_model = sign * raw_q + offset
            base_from_camera = backend.forward_kinematics(q_model) @ tcp_from_camera
            predicted = np.linalg.inv(base_from_camera) @ base_from_world
            tag = tags[args.tag_id]
            observed = np.eye(4, dtype=np.float64)
            observed[:3, :3] = cv2.Rodrigues(tag["rvec"])[0]
            observed[:3, 3] = tag["tvec"]
            errors.append(pose_error(observed, predicted))
            q_records.append(raw_q)
        if len(errors) < args.samples:
            raise RuntimeError(f"仅得到 {len(errors)}/{args.samples} 张有效 Tag {args.tag_id} 图像")
        values = np.asarray(errors)
        drift = np.ptp(np.asarray(q_records), axis=0)
        print("VJAW_LIVE_MODEL_CHECK_OK")
        print("samples:", len(values), "joint_drift_deg:", np.round(drift, 3).tolist())
        print("translation_mm: median={:.2f}; p95={:.2f}; max={:.2f}".format(
            np.median(values[:, 0]), np.percentile(values[:, 0], 95), values[:, 0].max()))
        print("rotation_deg: median={:.2f}; p95={:.2f}; max={:.2f}".format(
            np.median(values[:, 1]), np.percentile(values[:, 1], 95), values[:, 1].max()))
        print("RECOMMENDATION:", "PASS" if (np.percentile(values[:, 0], 95) <= 25.0 and
                                               np.percentile(values[:, 1], 95) <= 4.0) else "RECHECK")
    finally:
        if camera is not None:
            camera.close()
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
