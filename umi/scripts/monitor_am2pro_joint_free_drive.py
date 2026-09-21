#!/usr/bin/env python3
"""Continuously read one AM2Pro joint with torque disabled (no motor writes)."""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


SPECS = (
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
)


def main():
    parser = argparse.ArgumentParser(description="AM2Pro 无扭矩单关节实时读数")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--joint", default="wrist_flex", choices=[item[0] for item in SPECS])
    parser.add_argument("--hz", type=float, default=5.0)
    args = parser.parse_args()
    if args.hz <= 0:
        parser.error("hz 必须为正")
    motors = {name: Motor(mid, model, norm) for name, mid, model, norm in SPECS}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    try:
        bus.connect()
        bus.calibration = bus.read_calibration()
        print("FREE_DRIVE_JOINT_MONITOR_READY")
        print("No torque is enabled and no command is sent. Ctrl+C exits.")
        while True:
            value = float(bus.read("Present_Position", args.joint))
            print(f"{args.joint}: {value:+.2f} deg", flush=True)
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        print("FREE_DRIVE_JOINT_MONITOR_STOPPED")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
