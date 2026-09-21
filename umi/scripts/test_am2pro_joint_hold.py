#!/usr/bin/env python3
"""Low-speed single-joint tracking diagnostic for the AM2Pro.

This is deliberately not a replay tool.  It holds the other five arm joints
at their readback positions, moves one selected joint slowly, samples the
actual joint angle, then returns to the initial six-joint pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.real_world.am2pro_interpolation_controller import AM2ProInterpolationController
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    mapping_from_config,
    model_safe_limits_from_config,
)


NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
URDF_NAMES = ("right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
              "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll")


def joint_limits_deg(urdf_path: Path, model_safe_limits):
    root = ET.parse(urdf_path).getroot()
    nodes = {node.get("name"): node for node in root.findall("joint")}
    limits = np.asarray([
        (math.degrees(float(nodes[name].find("limit").get("lower"))),
         math.degrees(float(nodes[name].find("limit").get("upper"))))
        for name in URDF_NAMES
    ], dtype=np.float64)
    for index, name in enumerate(NAMES):
        if name not in model_safe_limits:
            continue
        limits[index] = model_safe_limits[name]
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("URDF 与编码器安全范围相交后为空")
    return limits


def main():
    parser = argparse.ArgumentParser(description="AM2Pro 单关节保持诊断；会实际低速移动指定关节")
    parser.add_argument("--robot-config", default="example/eval_robots_config.yaml")
    parser.add_argument("--joint", choices=NAMES, required=True)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-deg", type=float,
                              help="指定该关节的绝对目标角度")
    target_group.add_argument("--delta-deg", type=float,
                              help="启动后读取当前角度，再加上此相对角度作为目标")
    parser.add_argument("--duration-s", type=float, default=4.0)
    parser.add_argument("--hold-s", type=float, default=5.0)
    parser.add_argument("--sample-hz", type=float, default=4.0)
    parser.add_argument("--preposition-q-deg", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help=("可选：先将六轴预摆至此模型角度，再只测试指定关节；"
                              "用于复现某个完整姿态下的单关节回调。会在结束时回到启动姿态"))
    parser.add_argument("--preposition-duration-s", type=float, default=30.0,
                        help="指定 --preposition-q-deg 时的预摆时长（默认 30 秒）")
    parser.add_argument("--preposition-tolerance-deg", type=float, default=1.5,
                        help="预摆结束前要求的最大六轴误差（默认 1.5 度）")
    parser.add_argument("--telemetry", action="store_true",
                        help="在服务器终端输出当前/负载/温度原始寄存器值")
    parser.add_argument("--temporary-p-coefficient", type=int, default=None,
                        help="仅本次测试临时覆盖该关节 P（0–254）；退出时自动恢复 EEPROM 原值")
    args = parser.parse_args()
    if (args.duration_s <= 0 or args.hold_s <= 0 or args.sample_hz <= 0 or
            args.preposition_duration_s <= 0 or args.preposition_tolerance_deg <= 0):
        parser.error("时长、采样率与预摆容差必须为正")
    if (args.temporary_p_coefficient is not None and
            not 0 <= args.temporary_p_coefficient <= 254):
        parser.error("temporary-p-coefficient 必须在 0–254")

    config = yaml.safe_load(Path(args.robot_config).read_text())
    rc = config["robots"][0]
    index = NAMES.index(args.joint)
    configured_urdf = Path(rc.get("urdf_path", ROOT / "alohamini2pro_right_arm_kinematics.urdf"))
    if not configured_urdf.is_absolute():
        configured_urdf = (ROOT / configured_urdf).resolve()
    if not configured_urdf.is_file():
        raise FileNotFoundError(f"URDF 不存在: {configured_urdf}")
    mapping = mapping_from_config(rc, ROOT)
    model_safe_limits = model_safe_limits_from_config(rc, mapping)
    all_joint_limits = joint_limits_deg(configured_urdf, model_safe_limits)
    lower_limit, upper_limit = all_joint_limits[index]

    temporary_p_overrides = (
        {args.joint: args.temporary_p_coefficient}
        if args.temporary_p_coefficient is not None else None)
    if temporary_p_overrides:
        print(f"TEMPORARY_P_TEST: {args.joint} P={args.temporary_p_coefficient}; "
              "退出后控制器会自动恢复此电机原有 EEPROM P 值。")

    with AM2ProInterpolationController(
        robot_usb_port=rc.get("robot_usb_port", "/dev/ttyACM0"),
        frequency=50,
        receive_latency=rc.get("robot_obs_latency", 0.005),
        server_python=rc.get("robot_python"),
        ik_backend=rc.get("ik_backend", "ros2_dh"),
        urdf_path=str(configured_urdf),
        flip_joints=rc.get("joint_flip", []),
        joint_model_candidate=rc.get("joint_model_candidate"),
        gripper_width_min=rc.get("gripper_width_min", 0.0),
        gripper_width_max=rc.get("gripper_width_max", 0.074),
        gripper_servo_closed=rc.get("gripper_servo_closed", 5.3),
        gripper_servo_open=rc.get("gripper_servo_open", 93.6),
        diagnostic_motor=args.joint if args.telemetry else None,
        diagnostic_interval_s=1.0 / args.sample_hz,
        position_p_coefficients=rc.get("position_p_coefficients"),
        position_p_overrides=temporary_p_overrides,
        verbose=True,
    ) as robot:
        deadline = time.monotonic() + 5.0
        state = robot.get_state()
        while state is None and time.monotonic() < deadline:
            time.sleep(0.05)
            state = robot.get_state()
        if state is None:
            raise RuntimeError("控制器没有返回状态")
        start_q = np.asarray(state["ActualQ"][:6], dtype=np.float64)
        test_base_q = start_q.copy()
        preposition_q = None
        if args.preposition_q_deg is not None:
            preposition_q = np.asarray(args.preposition_q_deg, dtype=np.float64)
            invalid = np.flatnonzero((preposition_q < all_joint_limits[:, 0]) |
                                     (preposition_q > all_joint_limits[:, 1]))
            if len(invalid):
                labels = ", ".join(
                    f"{NAMES[i]}={preposition_q[i]:+.2f}° outside "
                    f"[{all_joint_limits[i, 0]:+.2f}, {all_joint_limits[i, 1]:+.2f}]°"
                    for i in invalid)
                raise RuntimeError(f"拒绝越界预摆：{labels}")
            print("FULL_POSE_PREPOSITION_STARTED")
            print("  initial_q_deg:", np.round(start_q, 2))
            print("  target_q_deg :", np.round(preposition_q, 2))
            print(f"  duration={args.preposition_duration_s:.1f}s; "
                  f"tolerance={args.preposition_tolerance_deg:.2f}°")
            robot.servoJ(preposition_q, duration=args.preposition_duration_s)
            deadline = time.monotonic() + args.preposition_duration_s + 8.0
            latest = state
            while time.monotonic() < deadline:
                time.sleep(0.05)
                observed = robot.get_state()
                if observed is None:
                    continue
                latest = observed
                q = np.asarray(observed["ActualQ"][:6], dtype=np.float64)
                error = q - preposition_q
                if (time.monotonic() >= deadline - 8.0 and
                        np.max(np.abs(error)) <= args.preposition_tolerance_deg):
                    break
            actual_preposition = np.asarray(latest["ActualQ"][:6], dtype=np.float64)
            preposition_error = actual_preposition - preposition_q
            print("  readback_q_deg:", np.round(actual_preposition, 2))
            print("  error_deg     :", np.round(preposition_error, 2))
            if np.max(np.abs(preposition_error)) > args.preposition_tolerance_deg:
                print("Preposition failed; returning to initial joints ...")
                robot.servoJ(start_q, duration=3.0)
                time.sleep(3.2)
                raise RuntimeError(
                    "预摆未收敛：最大误差 "
                    f"{np.max(np.abs(preposition_error)):.2f}° > "
                    f"{args.preposition_tolerance_deg:.2f}°")
            print("FULL_POSE_PREPOSITION_REACHED")
            # Use the requested pose, rather than a readback with normal encoder
            # quantisation, as the other-five-joint hold target.  This gives a
            # reproducible six-axis context for the following one-joint test.
            test_base_q = preposition_q.copy()
        target_q = test_base_q.copy()
        if args.target_deg is not None:
            target_q[index] = args.target_deg
            target_description = f"absolute target {args.target_deg:+.2f} deg"
        else:
            target_q[index] = test_base_q[index] + args.delta_deg
            target_description = (f"test-base {test_base_q[index]:+.2f} + "
                                  f"delta {args.delta_deg:+.2f} deg")
        if not lower_limit <= target_q[index] <= upper_limit:
            raise RuntimeError(
                f"拒绝越界测试：{args.joint} 目标 {target_q[index]:+.2f}°，"
                f"URDF 范围 [{lower_limit:+.2f}, {upper_limit:+.2f}]°")
        print("SINGLE_JOINT_HOLD_TEST_STARTED")
        print("joint:", args.joint, "index:", index + 1)
        print(f"joint_limit_deg: [{lower_limit:+.2f}, {upper_limit:+.2f}]")
        print("target mode:", target_description)
        print("start_q_deg:", np.round(start_q, 2))
        if preposition_q is not None:
            print("test_base_q_deg:", np.round(test_base_q, 2))
        print("target_q_deg:", np.round(target_q, 2))
        print("This sends no gripper command. Ctrl+C stops and returns to start joints.")
        robot.servoJ(target_q, duration=args.duration_s)
        t0 = time.monotonic()
        next_sample = t0
        try:
            while time.monotonic() - t0 <= args.duration_s + args.hold_s:
                now = time.monotonic()
                if now >= next_sample:
                    observed = robot.get_state()
                    if observed is not None:
                        q = np.asarray(observed["ActualQ"][:6], dtype=np.float64)
                        error = q[index] - target_q[index]
                        phase = "MOVE" if now - t0 < args.duration_s else "HOLD"
                        print(f"t={now-t0:5.2f}s {phase} {args.joint}: "
                              f"target={target_q[index]:+.2f} actual={q[index]:+.2f} "
                              f"error={error:+.2f} deg", flush=True)
                    next_sample += 1.0 / args.sample_hz
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("SINGLE_JOINT_HOLD_TEST_INTERRUPTED")
        finally:
            print("Returning to initial joints ...")
            robot.servoJ(start_q, duration=3.0)
            time.sleep(3.2)
            print("Returned.")


if __name__ == "__main__":
    main()
