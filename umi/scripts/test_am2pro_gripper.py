"""
AM2Pro gripper test: open/close via schedule_gripper.

Run in the UMI env:
    conda activate umi
    python scripts/test_am2pro_gripper.py \
        --robot-config example/eval_robots_config_vjaw_ros2.yaml

SAFETY: make sure nothing (including fingers!) is between the gripper jaws.
"""
import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from umi.real_world.am2pro_interpolation_controller import (
    AM2ProInterpolationController,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", default="example/eval_robots_config_vjaw_ros2.yaml")
    parser.add_argument("--port", default=None, help="覆盖 robot-config 的串口")
    parser.add_argument("--open-width-m", type=float, default=0.067)
    parser.add_argument("--closed-width-m", type=float, default=0.009)
    args = parser.parse_args()

    robot = yaml.safe_load(Path(args.robot_config).expanduser().read_text(encoding="utf-8"))["robots"][0]
    if args.open_width_m < 0 or args.closed_width_m < 0:
        parser.error("开口宽度不能为负")
    width_max = float(robot.get("gripper_width_max", 0.074))
    if args.open_width_m > width_max or args.closed_width_m > width_max:
        parser.error(f"测试宽度不能超过配置 gripper_width_max={width_max:.3f}m")

    ctrl = AM2ProInterpolationController(
        robot_usb_port=args.port or robot.get("robot_usb_port", "/dev/ttyACM0"),
        server_python=robot.get("robot_python"),
        ik_backend=robot.get("ik_backend", "ros2_dh"),
        urdf_path=robot.get("urdf_path"),
        joint_model_candidate=robot.get("joint_model_candidate"),
        gripper_width_min=robot.get("gripper_width_min", 0.0),
        gripper_width_max=width_max,
        gripper_servo_closed=robot.get("gripper_servo_closed", 5.3),
        gripper_servo_open=robot.get("gripper_servo_open", 93.6),
        gripper_width_servo_table=robot.get("gripper_width_servo_table"),
        position_p_coefficients=robot.get("position_p_coefficients"),
        verbose=True, launch_timeout=30.0)

    print("[1] starting controller ...")
    try:
        ctrl.start(wait=True)
    except Exception as e:
        print("    START FAILED:", e)
        return 1

    try:
        time.sleep(1.0)
        st = ctrl.get_state()
        print("[2] current gripper width:", round(float(st["gripper_position"]), 4), "m")

        print(f"[3] opening to {args.open_width_m:.3f} m ...")
        ctrl.schedule_gripper(args.open_width_m, time.time() + 0.1)
        time.sleep(2.0)
        st = ctrl.get_state()
        print("    width now:", round(float(st["gripper_position"]), 4), "m")

        print(f"[4] closing to {args.closed_width_m:.3f} m ...")
        ctrl.schedule_gripper(args.closed_width_m, time.time() + 0.1)
        time.sleep(2.0)
        st = ctrl.get_state()
        print("    width now:", round(float(st["gripper_position"]), 4), "m")
    finally:
        print("[5] stopping controller ...")
        ctrl.stop()
    print("DONE")


if __name__ == "__main__":
    main()
