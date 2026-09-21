#!/usr/bin/env python3
"""Execute exactly one supervised, low-speed axis test from an offline plan.

This is deliberately narrower than replay: it prepositions the arm at the
saved reference, moves to one pre-approved 2 cm Cartesian test target, holds,
and returns.  It never commands the gripper.  ``--execute`` is mandatory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.real_world.am2pro_interpolation_controller import AM2ProInterpolationController  # noqa: E402


def get_state_or_raise(robot):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = robot.get_state()
        if state is not None:
            return state
        time.sleep(.05)
    raise RuntimeError("控制器未返回关节状态")


def move_and_verify(robot, target, duration, tolerance, label):
    print(f"{label}：机械臂将在 {duration:.1f} 秒内缓慢移动")
    print("  目标关节角:", np.round(target, 2))
    robot.servoJ(target, duration=duration)
    end = time.monotonic() + duration
    latest = get_state_or_raise(robot)
    while time.monotonic() < end:
        time.sleep(.1)
        candidate = robot.get_state()
        if candidate is not None:
            latest = candidate
    actual = np.asarray(latest["ActualQ"][:6], dtype=float)
    error = actual - target
    print("  实际关节角:", np.round(actual, 2))
    print("  关节误差  :", np.round(error, 2), "度")
    if np.max(np.abs(error)) > tolerance:
        raise RuntimeError(
            f"{label} 未达到允许误差：最大 {np.max(np.abs(error)):.2f}° > {tolerance:.2f}°")
    return actual


def main():
    parser = argparse.ArgumentParser(description="AM2Pro 单轴向 2 cm 实体测试；会实际移动机械臂。")
    parser.add_argument("--plan", required=True, help="plan_am2pro_handheld_axis_probe.py 的 JSON 输出")
    parser.add_argument("--axis", choices=("forward", "up", "left"), required=True)
    parser.add_argument("--robot-config", default="example/eval_robots_config.yaml")
    parser.add_argument("--reference-duration-s", type=float, default=15.0)
    parser.add_argument("--axis-duration-s", type=float, default=8.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    parser.add_argument("--return-duration-s", type=float, default=10.0)
    parser.add_argument("--prepare-wait-s", type=float, default=8.0,
                        help="到达基准位后，实体动作开始前的观察准备时间")
    parser.add_argument("--tolerance-deg", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true",
                        help="确认实际控制机器人；省略此参数仅显示计划")
    args = parser.parse_args()
    if min(args.reference_duration_s, args.axis_duration_s, args.hold_s, args.prepare_wait_s,
           args.return_duration_s, args.tolerance_deg) <= 0:
        parser.error("所有时间和 tolerance 必须为正")
    plan = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
    axis = plan.get("axes", {}).get(args.axis)
    if not axis:
        parser.error(f"计划中没有 {args.axis} 轴")
    if not axis.get("accepted_for_low_speed_physical_test"):
        parser.error(f"计划已拒绝 {args.axis} 轴；不允许实体执行")
    reference_path = Path(plan["reference"])
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    q_reference = np.asarray(reference["robot_state"]["ActualQ"][:6], dtype=float)
    q_target = np.asarray(axis["planned_joints_deg"], dtype=float)
    print("AM2PRO_AXIS_PROBE_TEST")
    print("测试方向:", args.axis, "；TCP 基座位移 [m]:", axis["planned_tcp_delta_base_m"])
    print("安全说明：不会发送夹爪开合命令。清空工作区，并让手保持在急停/断电位置附近。")
    print("动作流程：回到基准位 -> %.1f cm 轴向移动 -> 停住 -> 回到基准位。" %
          (float(axis.get("planned_distance_m", 0.02)) * 100.0))
    if not args.execute:
        print("仅预览：加上 --execute 才会控制机械臂。")
        print("基准关节角:", np.round(q_reference, 2))
        print("轴向目标关节角:", np.round(q_target, 2))
        return
    config = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    robot_config = config["robots"][0]
    with AM2ProInterpolationController(
        robot_usb_port=robot_config.get("robot_usb_port", "/dev/ttyACM0"),
        frequency=50, receive_latency=robot_config.get("robot_obs_latency", .005),
        server_python=robot_config.get("robot_python"), ik_backend=robot_config.get("ik_backend", "ros2_dh"),
        flip_joints=robot_config.get("joint_flip", []),
        gripper_width_min=robot_config.get("gripper_width_min", 0.0),
        gripper_width_max=robot_config.get("gripper_width_max", .074),
        gripper_servo_closed=robot_config.get("gripper_servo_closed", 5.3),
        gripper_servo_open=robot_config.get("gripper_servo_open", 93.6), verbose=True,
    ) as robot:
        try:
            move_and_verify(robot, q_reference, args.reference_duration_s, args.tolerance_deg,
                            "正在回到 V9 基准位")
            print(f"准备阶段：已到基准位；请观察机械臂，{args.prepare_wait_s:.1f} 秒后才开始轴向移动。")
            time.sleep(args.prepare_wait_s)
            move_and_verify(robot, q_target, args.axis_duration_s, args.tolerance_deg,
                            "正在执行轴向目标")
            print(f"观察阶段：保持 {args.hold_s:.1f} 秒；确认实际方向是否符合“{args.axis}”。")
            time.sleep(args.hold_s)
        except KeyboardInterrupt:
            print("AXIS PROBE INTERRUPTED")
        finally:
            print("正在安全返回基准位 ...")
            robot.servoJ(q_reference, duration=args.return_duration_s)
            time.sleep(args.return_duration_s + .2)
            print("轴向测试完成；机械臂已返回基准位。")


if __name__ == "__main__":
    main()
