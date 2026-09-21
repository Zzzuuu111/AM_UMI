"""Keyboard-only, torque-held AM2Pro hand-eye calibration collector.

The arm begins torque-free. Press C once to hold the current encoder pose
without a stale-goal jump. While held, use a dedicated key pair for each
joint to make small position steps. Press P to save a camera/encoder sample
and C again to release torque. No Cartesian IK or replay command is used.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time

import av
import cv2
import numpy as np
import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import mat_to_pose
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend
from record_am2pro_hand_eye import (
    ARM_NAMES,
    MOTOR_SPECS,
    URDF_JOINT_NAMES,
    HandEyeWebPreview,
    next_rgb_frame,
)

ALL_NAMES = [item[0] for item in MOTOR_SPECS]
# Conservatively verified operating bounds for the two load-sensitive joints.
# These are software guards for keyboard increments only; they do not alter
# calibration or servo EEPROM limits.
KEYBOARD_GUARDS_DEG = {
    "shoulder_lift": (-98.0, 9.22),
    "wrist_flex": (-94.31, 85.0),
}

# Each pair is intentionally separate: no prior "select joint" key is needed.
# Signs are motor-coordinate signs, not a promise of a visual world direction.
JOINT_KEY_COMMANDS = {
    "z": (0, -1.0), "x": (0, +1.0),  # J1 shoulder_pan
    "w": (1, +1.0), "s": (1, -1.0),  # J2 shoulder_lift
    "r": (2, +1.0), "f": (2, -1.0),  # J3 elbow_flex
    "t": (3, +1.0), "g": (3, -1.0),  # J4 wrist_flex
    "y": (4, +1.0), "h": (4, -1.0),  # J5 wrist_yaw
    "u": (5, +1.0), "j": (5, -1.0),  # J6 wrist_roll
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new .pkl file; never overwritten")
    parser.add_argument("--resume", action="store_true",
                        help="append to an existing compatible sample .pkl instead of overwriting it")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--camera-device", default="/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--intrinsics", default="calibration/robot_wrist_camera/intrinsics/emeet_wrist_1920x1080_30fps_fisheye_v1.json")
    parser.add_argument("--aruco-yaml", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--robot-config", default=None,
                        help="可选机器人配置；新 V 型夹爪应传 eval_robots_config_vjaw_ros2.yaml")
    parser.add_argument("--urdf-path", default=None,
                        help="覆盖 robot-config 的 URDF 路径")
    parser.add_argument("--ik-backend", choices=("ros2_dh", "placo"), default=None,
                        help="覆盖 robot-config 的 IK/FK 后端；未指定时保持旧路径 ros2_dh")
    parser.add_argument("--tcp-frame", default="right_Fixed_Jaw",
                        help="保存样本使用的 URDF 坐标系；新 V 型夹爪使用 right_tcp")
    parser.add_argument("--web-preview", action="store_true")
    parser.add_argument("--web-preview-port", type=int, default=8766)
    parser.add_argument("--capture-settle-frames", type=int, default=45,
                        help="fresh frames to drain after settling (default: 45, about 1.5 s)")
    parser.add_argument("--settle-timeout-s", type=float, default=10.0,
                        help="maximum wait after P for actual joints to settle (default: 10)")
    parser.add_argument("--settle-window-s", type=float, default=1.5,
                        help="duration over which joints must remain still (default: 1.5)")
    parser.add_argument("--settle-max-drift-deg", type=float, default=0.15,
                        help="maximum total joint drift over the settle window (default: 0.15)")
    parser.add_argument("--joint-step-deg", type=float, default=2.0,
                        help="W/S adjustment, in (0, 2] degrees; default 2")
    parser.add_argument("--enable-torque-on-capture", action="store_true",
                        help="required acknowledgement: C will lock/release torque")
    args = parser.parse_args()
    if not args.enable_torque_on_capture:
        parser.error("add --enable-torque-on-capture: C will lock/release the arm")
    if (args.capture_settle_frames < 1 or not 0 < args.joint_step_deg <= 2
            or args.settle_timeout_s <= 0 or args.settle_window_s <= 0
            or args.settle_max_drift_deg <= 0):
        parser.error("invalid settle or keyboard-step parameter")

    out_path = (ROOT_DIR / args.output).resolve()
    if out_path.exists() and not args.resume:
        parser.error(f"refusing to overwrite existing output: {out_path}")
    if args.resume and not out_path.is_file():
        parser.error(f"--resume requires an existing sample file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    intr_path = (ROOT_DIR / args.intrinsics).resolve()
    aruco_path = (ROOT_DIR / args.aruco_yaml).resolve()
    if not intr_path.is_file() or not aruco_path.is_file():
        parser.error("intrinsics or ArUco configuration file does not exist")

    robot_cfg = {}
    if args.robot_config is not None:
        config_path = (ROOT_DIR / args.robot_config).resolve()
        if not config_path.is_file():
            parser.error(f"robot-config 不存在: {config_path}")
        config = yaml.safe_load(config_path.read_text())
        robots = config.get("robots", []) if isinstance(config, dict) else []
        if not robots:
            parser.error("robot-config 缺少 robots[0]")
        robot_cfg = robots[0]
    backend_name = args.ik_backend or robot_cfg.get("ik_backend", "ros2_dh")
    configured_urdf = args.urdf_path or robot_cfg.get(
        "urdf_path", ROOT_DIR / "alohamini2pro_right_arm_kinematics.urdf")
    urdf_path = pathlib.Path(configured_urdf).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT_DIR / urdf_path).resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF 不存在: {urdf_path}")
    joint_flip_names = set(robot_cfg.get("joint_flip", ["wrist_roll"]))
    invalid_flip = joint_flip_names.difference(ARM_NAMES)
    if invalid_flip:
        parser.error(f"robot-config 中不是手臂关节的 joint_flip: {sorted(invalid_flip)}")

    aruco_cfg = parse_aruco_config(yaml.safe_load(aruco_path.read_text()))
    raw_intr = parse_fisheye_intrinsics(json.loads(intr_path.read_text()))
    intr = convert_fisheye_intrinsics_resolution(raw_intr, target_resolution=(args.width, args.height))
    motors = {name: Motor(mid, model, norm) for name, mid, model, norm in MOTOR_SPECS}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    kin = create_kinematics_backend(
        backend_name, urdf_path=str(urdf_path), joint_names=URDF_JOINT_NAMES,
        target_frame_name=args.tcp_frame)
    camera = preview = None
    if args.resume:
        with out_path.open("rb") as file:
            samples = pickle.load(file)
        if not isinstance(samples, list) or any("img" not in sample or "joint_deg" not in sample for sample in samples):
            parser.error(f"--resume input is not a compatible hand-eye sample list: {out_path}")
        print(f"RESUMING_HAND_EYE_COLLECTION: existing samples={len(samples)}", flush=True)
    else:
        samples: list[dict] = []
    locked = False
    goal = None

    def read_arm_pose():
        positions = bus.sync_read("Present_Position", ARM_NAMES)
        q = np.asarray([float(positions[name]) for name in ARM_NAMES], dtype=float)
        q_model = q.copy()
        for index, name in enumerate(ARM_NAMES):
            if name in joint_flip_names:
                q_model[index] *= -1.0
        return q, mat_to_pose(kin.forward_kinematics(q_model))

    def lock_or_release():
        nonlocal locked, goal
        if locked:
            bus.disable_torque()
            locked = False
            print("TORQUE_RELEASED：扭矩已解除。", flush=True)
            return
        pos = bus.sync_read("Present_Position", ALL_NAMES)
        goal = {name: float(pos[name]) for name in ALL_NAMES}
        # Write goals before enabling torque: avoids a stale previous command.
        bus.sync_write("Goal_Position", goal)
        bus.enable_torque()
        locked = True
        print("TORQUE_LOCKED：已锁住当前姿态。J1=Z/X，J2=W/S，J3=R/F，J4=T/G，J5=Y/H，J6=U/J；P 拍照，C 释放。", flush=True)

    def step_joint(index, sign):
        if not locked or goal is None:
            print("请先按 C 锁住机械臂；扭矩关闭时不允许键盘运动。", flush=True)
            return
        name = ARM_NAMES[index]
        proposed = goal[name] + sign * args.joint_step_deg
        if name in KEYBOARD_GUARDS_DEG:
            low, high = KEYBOARD_GUARDS_DEG[name]
            # If an arm starts outside this conservative range, let it move
            # inward but never issue a command farther into the unsafe side.
            if (sign < 0 and proposed < low) or (sign > 0 and proposed > high):
                print(f"安全拦截：{name} 的键盘目标必须保持在 [{low:+.2f}, {high:+.2f}]° 内。",
                      flush=True)
                return
        goal[name] = proposed
        bus.sync_write("Goal_Position", {name: goal[name]})
        print(f"J{index + 1} {name} target={goal[name]:+.2f} deg", flush=True)

    def capture(frames):
        if not locked:
            print("请先按 C 锁住机械臂，再按 P 拍照。", flush=True)
            return
        print("P 已收到：等待机械臂真正停稳后才拍照…", flush=True)
        deadline = time.monotonic() + args.settle_timeout_s
        stable_start = None
        stable_anchor_q = None
        stable_q = None
        while time.monotonic() < deadline:
            q_now, _ = read_arm_pose()
            target_error = float(np.max(np.abs(q_now - np.asarray([goal[name] for name in ARM_NAMES]))))
            # Compare against the start of the entire stillness window, not
            # merely the previous 100-ms sample. This rejects slow creep.
            if (stable_anchor_q is None
                    or float(np.max(np.abs(q_now - stable_anchor_q))) > args.settle_max_drift_deg):
                stable_anchor_q = q_now
                stable_start = time.monotonic()
            # Hand-eye calibration pairs the image with the *read-back*
            # encoder pose.  A heavy tool may settle with a static load error
            # from its last commanded target; that is still a valid pair.
            # Therefore require stillness, not target convergence.
            if time.monotonic() - stable_start >= args.settle_window_s:
                stable_q = q_now
                break
            time.sleep(0.10)
        else:
            print("本次不拍：机械臂在 %.1f 秒内仍在变化。等待后再按 P。" % args.settle_timeout_s,
                  flush=True)
            return
        last = None
        for _ in range(args.capture_settle_frames):
            last = next(frames)
        image = last.to_ndarray(format="rgb24")
        q, tcp = read_arm_pose()
        capture_drift = float(np.max(np.abs(q - stable_q)))
        if capture_drift > args.settle_max_drift_deg:
            print("本次不拍：取图期间关节又变化 %.2f°。等待后再按 P。" % capture_drift, flush=True)
            return
        tags = detect_localize_aruco_tags(
            img=image, aruco_dict=aruco_cfg["aruco_dict"],
            marker_size_map=aruco_cfg["marker_size_map"], fisheye_intr_dict=intr)
        if args.tag_id not in tags:
            print(f"未检测到 Tag {args.tag_id}；本次不保存。", flush=True)
            return
        samples.append({
            "img": image, "tcp_pose": tcp, "joint_deg": q,
            "capture_monotonic_s": time.monotonic(), "tag_id": args.tag_id,
            "capture_method": "keyboard_joint_steps_torque_held",
            "lock_goal_joint_deg": [goal[name] for name in ARM_NAMES],
            "tcp_frame": args.tcp_frame,
            "urdf_path": str(urdf_path),
            "ik_backend": backend_name,
            "joint_flip": sorted(joint_flip_names),
        })
        with out_path.open("wb") as file:
            pickle.dump(samples, file)
        print("joint_deg:", np.round(q, 3).tolist(), flush=True)
        print("capture_settle: static; target_error=%.2f° (仅记录) max_capture_drift=%.2f°" % (target_error, capture_drift), flush=True)
        print("fixed_jaw_pose_base:", np.round(tcp, 6).tolist(), flush=True)
        print(f"HAND_EYE_SAMPLE_SAVED: {len(samples)}; file: {out_path}", flush=True)

    try:
        bus.connect()
        bus.calibration = bus.read_calibration()
        # Mirrors controller configuration while torque is off. It never sends a pose.
        bus.disable_torque()
        bus.configure_motors()
        for name in ALL_NAMES:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", name, 80 if name == "wrist_flex" else 24)
            bus.write("I_Coefficient", name, 0)
            bus.write("D_Coefficient", name, 32)
        bus.disable_torque()

        camera = av.open(os.path.realpath(args.camera_device), format="v4l2", options={
            "input_format": "mjpeg", "video_size": f"{args.width}x{args.height}", "framerate": str(args.fps)})
        frames = camera.decode(video=0)
        _ = next_rgb_frame(frames)
        if args.web_preview:
            preview = HandEyeWebPreview(args.web_preview_port, allowed_commands={
                "c", "p", "d", "q", *JOINT_KEY_COMMANDS.keys()})
            print("WEB_PREVIEW_READY:", preview.url, flush=True)

        print("KEYBOARD_HOLD_HAND_EYE_COLLECTOR_READY", flush=True)
        print(f"模型：backend={backend_name}; tcp_frame={args.tcp_frame}; urdf={urdf_path}", flush=True)
        print("初始为扭矩关闭。确认机械臂已在安全位置后按 C 锁住；之后不需要手动移动。", flush=True)
        print("锁住后：J1=Z/X，J2=W/S，J3=R/F，J4=T/G，J5=Y/H，J6=U/J；每次 %.1f°；P=拍照；C=释放。" % args.joint_step_deg, flush=True)

        while True:
            if preview is None:
                try:
                    command = input("keyboard_hold_hand_eye> ").strip().lower()
                except EOFError:
                    command = "q"
            else:
                image = next_rgb_frame(frames)
                tags = detect_localize_aruco_tags(
                    img=image, aruco_dict=aruco_cfg["aruco_dict"],
                    marker_size_map=aruco_cfg["marker_size_map"], fisheye_intr_dict=intr)
                tag_ok = args.tag_id in tags
                display = image[..., ::-1].copy()
                state = "已锁住" if locked else "扭矩关闭"
                status = (f"{state} | Tag {args.tag_id}: {'OK' if tag_ok else 'MISSING'} | "
                          f"J1 Z/X J2 W/S J3 R/F J4 T/G J5 Y/H J6 U/J | P C | saved {len(samples)}")
                cv2.putText(display, status, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (0, 220, 0) if tag_ok else (0, 0, 255), 2, cv2.LINE_AA)
                preview.publish(display)
                preview.set_status(status)
                command = preview.get_command()
                if command is None:
                    continue

            if command == "q":
                break
            if command == "c":
                lock_or_release()
            elif command == "p":
                capture(frames)
            elif command in JOINT_KEY_COMMANDS:
                index, sign = JOINT_KEY_COMMANDS[command]
                step_joint(index, sign)
            elif command == "d":
                if samples:
                    samples.pop()
                    with out_path.open("wb") as file:
                        pickle.dump(samples, file)
                    print(f"删除成功；当前有效样本数：{len(samples)}", flush=True)
            else:
                print("可用键：C P D Q；J1=Z/X，J2=W/S，J3=R/F，J4=T/G，J5=Y/H，J6=U/J。", flush=True)

        with out_path.open("wb") as file:
            pickle.dump(samples, file)
        print("KEYBOARD_HOLD_HAND_EYE_COLLECTION_SAVED", flush=True)
        print("samples:", len(samples), flush=True)
        if samples:
            saved_q = np.asarray([sample["joint_deg"] for sample in samples], dtype=float)
            spans = np.ptp(saved_q, axis=0)
            print("JOINT_COVERAGE_DEG:", np.round(spans, 2).tolist(), flush=True)
            weak = [f"J{index + 1} {name}={spans[index]:.2f}°"
                    for index, name in enumerate(ARM_NAMES) if spans[index] < 12.0]
            if weak:
                print("COVERAGE_WARNING: 以下关节变化不足 12°，手眼/模型对齐不可判定："
                      + "; ".join(weak), flush=True)
            else:
                print("JOINT_COVERAGE_OK: 六个关节均已变化至少 12°。", flush=True)
        print("output:", out_path, flush=True)
    finally:
        # Ctrl+C and errors leave the heavy tool released.
        try:
            if bus.is_connected:
                bus.disable_torque()
        finally:
            if camera is not None:
                camera.close()
            if preview is not None:
                preview.stop()
            if bus.is_connected:
                bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
