#!/usr/bin/env python3
"""Supervised three-axis AM2Pro test based on an offline-approved plan.

Sequence: reference -> forward -> reference (back) -> up -> reference (down)
-> left -> reference (right).  There is no gripper command.  A robot command
is impossible unless ``--execute`` is explicitly supplied.
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


def get_state(robot):
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        state = robot.get_state()
        if state is not None:
            return state
        time.sleep(.05)
    raise RuntimeError("控制器未返回关节状态")


def move(robot, target, duration, tolerance, label):
    print(f"[机械臂动作] {label}：{duration:.1f} 秒", flush=True)
    print("  target joints:", np.round(target, 2), flush=True)
    robot.servoJ(target, duration=duration)
    end = time.monotonic() + duration
    latest = get_state(robot)
    while time.monotonic() < end:
        time.sleep(.1)
        observed = robot.get_state()
        if observed is not None:
            latest = observed
    actual = np.asarray(latest["ActualQ"][:6], dtype=float)
    error = actual - target
    print("  actual joints:", np.round(actual, 2), flush=True)
    print("  joint error :", np.round(error, 2), "deg", flush=True)
    if np.max(np.abs(error)) > tolerance:
        raise RuntimeError(f"{label} 未收敛：最大关节误差 {np.max(np.abs(error)):.2f}° > {tolerance:.2f}°")


def main():
    parser = argparse.ArgumentParser(description="AM2Pro 前后/上下/左右实体轴向测试；会实际移动机械臂。")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--robot-config", default="example/eval_robots_config.yaml")
    parser.add_argument("--reference-duration-s", type=float, default=15.0)
    parser.add_argument("--prepare-wait-s", type=float, default=8.0)
    parser.add_argument("--axis-duration-s", type=float, default=8.0)
    parser.add_argument("--hold-s", type=float, default=1.5)
    parser.add_argument("--between-axis-wait-s", type=float, default=2.0)
    parser.add_argument("--tolerance-deg", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if min(args.reference_duration_s, args.prepare_wait_s, args.axis_duration_s,
           args.hold_s, args.between_axis_wait_s, args.tolerance_deg) <= 0:
        parser.error("所有时间和 tolerance 必须为正")
    plan = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
    axes = plan.get("axes", {})
    required = ("forward", "up", "left")
    missing = [name for name in required if not axes.get(name, {}).get("accepted_for_low_speed_physical_test")]
    if missing:
        parser.error("计划未批准这些轴，拒绝执行: " + ", ".join(missing))
    reference = json.loads(Path(plan["reference"]).read_text(encoding="utf-8"))
    q_reference = np.asarray(reference["robot_state"]["ActualQ"][:6], dtype=float)
    print("AM2PRO_AXIS_PROBE_SEQUENCE")
    print("This is a physical robot test; no gripper command will be sent.")
    print("动作总览：")
    print("  1) 回到 V9 基准位；随后等待 %.0f 秒。" % args.prepare_wait_s)
    print("  2) 前方 %.0f cm，再回基准位（这段回程就是后方）。" % (plan["axis_distance_m"] * 100))
    print("  3) 上方 %.0f cm，再回基准位（这段回程就是下方）。" % (plan["axis_distance_m"] * 100))
    print("  4) 左方 %.0f cm，再回基准位（这段回程就是右方）。" % (plan["axis_distance_m"] * 100))
    print("请清空工作区、不要放物体；随时 Ctrl+C，程序会尝试回到基准位。")
    if not args.execute:
        print("DRY_RUN_ONLY: add --execute to allow physical motion.")
        return
    config = yaml.safe_load(Path(args.robot_config).read_text(encoding="utf-8"))
    config = config["robots"][0]
    with AM2ProInterpolationController(
        robot_usb_port=config.get("robot_usb_port", "/dev/ttyACM0"), frequency=50,
        receive_latency=config.get("robot_obs_latency", .005), server_python=config.get("robot_python"),
        ik_backend=config.get("ik_backend", "ros2_dh"), flip_joints=config.get("joint_flip", []),
        gripper_width_min=config.get("gripper_width_min", 0), gripper_width_max=config.get("gripper_width_max", .074),
        gripper_servo_closed=config.get("gripper_servo_closed", 5.3),
        gripper_servo_open=config.get("gripper_servo_open", 93.6), verbose=True,
    ) as robot:
        try:
            move(robot, q_reference, args.reference_duration_s, args.tolerance_deg, "回到 V9 基准位")
            print(f"[准备] 请观察机械臂；{args.prepare_wait_s:.0f} 秒后开始前方动作。", flush=True)
            time.sleep(args.prepare_wait_s)
            for name, outward, return_name in (
                ("forward", "向前", "向后回到基准位"),
                ("up", "向上", "向下回到基准位"),
                ("left", "向左", "向右回到基准位"),
            ):
                target = np.asarray(axes[name]["planned_joints_deg"], dtype=float)
                move(robot, target, args.axis_duration_s, args.tolerance_deg, outward)
                print(f"[观察] {outward}到位，保持 {args.hold_s:.1f} 秒。", flush=True)
                time.sleep(args.hold_s)
                move(robot, q_reference, args.axis_duration_s, args.tolerance_deg, return_name)
                if name != "left":
                    print(f"[准备] {args.between_axis_wait_s:.1f} 秒后进行下一轴。", flush=True)
                    time.sleep(args.between_axis_wait_s)
        except KeyboardInterrupt:
            print("AXIS_SEQUENCE_INTERRUPTED")
        finally:
            print("[安全] 最终回到 V9 基准位 ...", flush=True)
            robot.servoJ(q_reference, duration=args.reference_duration_s)
            time.sleep(args.reference_duration_s + .2)
            print("AXIS_SEQUENCE_FINISHED", flush=True)


if __name__ == "__main__":
    main()
