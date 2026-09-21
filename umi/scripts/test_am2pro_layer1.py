"""
AM2Pro Layer-1 hardware verification (READ-ONLY).

Connects the Feetech motor bus, reads EEPROM calibration and present positions
of all 7 motors. Writes NOTHING to the motors (torque stays disabled).

Run in the lerobot_alohamini environment:
    /home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python \
        scripts/test_am2pro_layer1.py --port /dev/ttyUSB0
"""
import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Same spec as umi/real_world/am2pro_controller_server.py
_MOTOR_SPEC = [
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="serial device, e.g. /dev/ttyUSB0")
    args = parser.parse_args()

    motors = {n: Motor(i, m, nm) for n, i, m, nm in _MOTOR_SPEC}
    bus = FeetechMotorsBus(port=args.port, motors=motors)

    print(f"[1] connecting bus at {args.port} ...")
    bus.connect()
    print("    bus connected OK")

    print("[2] reading calibration from motor EEPROM ...")
    try:
        calib = bus.read_calibration()
        bus.calibration = calib
        print("    calibration:", calib)
    except Exception as e:
        print("    calibration read FAILED:", e)
        calib = None

    print("[3] reading present positions ...")
    try:
        names = [n for n, _, _, _ in _MOTOR_SPEC]
        pos = bus.sync_read("Present_Position", names)
        for n in names:
            print(f"    {n:14s} -> {pos[n]:.2f}")
    except Exception as e:
        print("    position read FAILED:", e)

    print("[4] disconnecting ...")
    try:
        bus.disconnect()
    except Exception as e:
        print("    disconnect warning:", e)
    print("DONE")


if __name__ == "__main__":
    main()
