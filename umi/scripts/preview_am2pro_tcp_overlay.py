#!/usr/bin/env python3
"""Read-only, live wrist-camera overlay for AM2Pro TCP-frame validation.

The overlay projects the active AM_UMI ``right_tcp`` and the provisional
fixed-jaw inner-front-tip point into the *physical wrist-camera image*.  It
does not instantiate the controller and never enables torque or writes any
motor register.  The only robot-bus operation is repeated Present_Position
readback.

This is intentionally a calibration diagnostic, not a robot-control tool:
the green point should visually sit on the intended fixed-jaw contact point.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lerobot.motors import Motor, MotorNormMode  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus  # noqa: E402
from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    parse_fisheye_intrinsics,
)
from umi.real_world.am2pro_joint_mapping import encoder_to_model, mapping_from_config  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402
from scripts.record_am2pro_hand_eye import HandEyeWebPreview, next_rgb_frame  # noqa: E402


MOTOR_SPECS = (
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
)
ARM_NAMES = [item[0] for item in MOTOR_SPECS[:6]]
URDF_JOINT_NAMES = (
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
)


def resolve(root: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def transform_from_pose(pose: list[float]) -> np.ndarray:
    """Build T_parent_child from [x,y,z,rx,ry,rz] (metres / rotvec)."""
    if len(pose) != 6:
        raise ValueError("TCP transform pose_parent_child must contain 6 values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def fisheye_pixel(point_camera: np.ndarray, intr: dict) -> tuple[int, int] | None:
    """Project a camera-frame point; return None if it is behind/outside image."""
    point_camera = np.asarray(point_camera, dtype=np.float64).reshape(3)
    width, height = map(int, intr["DIM"])
    if point_camera[2] <= 1e-5:
        return None
    projected, _ = cv2.fisheye.projectPoints(
        point_camera.reshape(1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)),
        intr["K"], intr["D"])
    x, y = projected.reshape(2)
    if not np.isfinite(x + y) or x < 0 or x >= width or y < 0 or y >= height:
        return None
    return int(round(x)), int(round(y))


def mark(image: np.ndarray, pixel: tuple[int, int] | None, color, label: str) -> bool:
    if pixel is None:
        return False
    x, y = pixel
    cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS,
                   markerSize=25, thickness=3, line_type=cv2.LINE_AA)
    cv2.circle(image, (x, y), 8, color, 2, cv2.LINE_AA)
    cv2.putText(image, label, (x + 13, y - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, label, (x + 13, y - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, color, 1, cv2.LINE_AA)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default="example/eval_robots_config_vjaw_ros2.yaml")
    parser.add_argument("--camera-device",
                        default="/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0")
    parser.add_argument("--intrinsics", default=(
        "calibration/robot_wrist_camera/intrinsics/"
        "emeet_wrist_1920x1080_30fps_fisheye_v1.json"))
    parser.add_argument("--hand-eye", default=(
        "calibration/robot_wrist_camera/hand_eye/new_gripper_camera_v4_hand_eye.json"))
    parser.add_argument("--right-tcp-to-fixed-tip", default=(
        "calibration/robot_wrist_camera/tcp/"
        "vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json"))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--web-port", type=int, default=8770)
    parser.add_argument("--read-hz", type=float, default=10.0)
    args = parser.parse_args()

    if args.read_hz <= 0:
        parser.error("--read-hz must be positive")
    config_path = resolve(ROOT_DIR, args.robot_config)
    config = yaml.safe_load(config_path.read_text())
    robot = config["robots"][0]
    intr = convert_fisheye_intrinsics_resolution(
        parse_fisheye_intrinsics(json.loads(resolve(ROOT_DIR, args.intrinsics).read_text())),
        target_resolution=(args.width, args.height))
    hand_eye = json.loads(resolve(ROOT_DIR, args.hand_eye).read_text())
    # The calibration matrix maps a homogeneous point fixed_jaw -> camera.
    T_camera_fixed_jaw = np.asarray(hand_eye["tx_fixed_jaw2camera"], dtype=np.float64)
    bridge = json.loads(resolve(ROOT_DIR, args.right_tcp_to_fixed_tip).read_text())
    if bridge.get("parent_frame") != "right_tcp":
        parser.error("TCP bridge parent must be right_tcp")
    T_right_tcp_fixed_tip = transform_from_pose(bridge["pose_parent_child"])

    urdf_path = resolve(ROOT_DIR, robot["urdf_path"])
    common = dict(
        name=robot.get("ik_backend", "placo"), urdf_path=str(urdf_path),
        joint_names=URDF_JOINT_NAMES)
    fk_tcp = create_kinematics_backend(target_frame_name="right_tcp", **common)
    fk_fixed_jaw = create_kinematics_backend(target_frame_name="right_Fixed_Jaw", **common)

    motors = {name: Motor(mid, model, norm) for name, mid, model, norm in MOTOR_SPECS}
    bus = FeetechMotorsBus(port=robot["robot_usb_port"], motors=motors)
    camera = None
    preview = None
    try:
        # Explicitly read-only: no configure_motors, torque calls or register writes.
        bus.connect()
        bus.calibration = bus.read_calibration()
        camera = av.open(os.path.realpath(args.camera_device), format="v4l2", options={
            "input_format": "mjpeg", "video_size": f"{args.width}x{args.height}",
            "framerate": str(args.fps),
        })
        frames = camera.decode(video=0)
        _ = next_rgb_frame(frames)
        preview = HandEyeWebPreview(args.web_port, allowed_commands=set())
        print("READ_ONLY_TCP_OVERLAY_READY")
        print("  browser:", preview.url)
        print("  orange: active AM_UMI right_tcp; green: fixed-jaw inner-front-tip candidate")
        print("  safety: Present_Position readback + camera read only; no torque/goal/gripper write")
        period = 1.0 / args.read_hz
        last_read = 0.0
        state = None
        while True:
            now = time.monotonic()
            if state is None or now - last_read >= period:
                positions = bus.sync_read("Present_Position", list(motors))
                q_encoder = np.array([float(positions[name]) for name in ARM_NAMES])
                q_model = encoder_to_model(q_encoder, mapping_from_config(robot, ROOT_DIR))
                T_base_tcp = fk_tcp.forward_kinematics(q_model)
                T_base_fixed_jaw = fk_fixed_jaw.forward_kinematics(q_model)
                # This derives the active model's exact TCP relation to the
                # fixed jaw rather than hard-coding the 236-mm CAD offset.
                T_fixed_jaw_tcp = np.linalg.inv(T_base_fixed_jaw) @ T_base_tcp
                T_camera_tcp = T_camera_fixed_jaw @ T_fixed_jaw_tcp
                T_camera_tip = T_camera_tcp @ T_right_tcp_fixed_tip
                state = (q_model, T_camera_tcp[:3, 3], T_camera_tip[:3, 3])
                last_read = now
            raw_rgb = next_rgb_frame(frames)
            image = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            q_model, p_tcp, p_tip = state
            tcp_visible = mark(image, fisheye_pixel(p_tcp, intr), (0, 140, 255), "right_tcp")
            tip_visible = mark(image, fisheye_pixel(p_tip, intr), (120, 235, 130), "fixed_tip candidate")
            panel = image[10:121, 10:760]
            dark = panel.copy()
            dark[:] = (18, 18, 18)
            cv2.addWeighted(dark, 0.78, panel, 0.22, 0, panel)
            text1 = "READ-ONLY overlay: orange=right_tcp, green=fixed_tip candidate"
            text2 = "visible: tcp=%s tip=%s | q model deg: %s" % (
                "yes" if tcp_visible else "outside", "yes" if tip_visible else "outside",
                np.array2string(q_model, precision=1, suppress_small=True))
            cv2.putText(image, text1, (25, 46), cv2.FONT_HERSHEY_SIMPLEX, .62,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, text2, (25, 91), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (235, 235, 235), 1, cv2.LINE_AA)
            preview.publish(image)
            preview.set_status("只读实时投影；Ctrl+C 在终端退出。候选点必须与实际固定爪接触位置对齐才可接受。")
    except KeyboardInterrupt:
        print("READ_ONLY_TCP_OVERLAY_STOPPED")
    finally:
        if preview is not None:
            preview.stop()
        if camera is not None:
            camera.close()
        bus.disconnect()


if __name__ == "__main__":
    main()
