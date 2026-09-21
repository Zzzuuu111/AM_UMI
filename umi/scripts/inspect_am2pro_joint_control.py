#!/usr/bin/env python3
"""Read AM2Pro Feetech joint control registers without writing anything."""

import argparse

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
FIELDS = (
    "Torque_Enable", "Operating_Mode", "P_Coefficient", "I_Coefficient",
    "D_Coefficient", "Max_Torque_Limit", "Torque_Limit", "Protection_Current",
    "Overload_Torque", "Min_Position_Limit", "Max_Position_Limit",
    "Homing_Offset", "Present_Position", "Present_Load", "Present_Current",
    "Present_Voltage", "Present_Temperature", "Status", "Moving",
)


def main():
    parser = argparse.ArgumentParser(description="只读检查 AM2Pro 单关节 PID/扭矩/状态寄存器")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--joint", default="wrist_flex", choices=[item[0] for item in SPECS])
    args = parser.parse_args()
    motors = {name: Motor(mid, model, norm) for name, mid, model, norm in SPECS}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    try:
        bus.connect()
        # EEPROM reads only: assigning the returned calibration object locally
        # allows us to report the registered range in normalized degrees.  It
        # does not write a register, enable torque, or configure a motor.
        calibration = bus.read_calibration()
        bus.calibration = calibration
        print("AM2PRO_JOINT_CONTROL_READ_ONLY")
        print("joint:", args.joint, "port:", args.port)
        for field in FIELDS:
            try:
                value = bus.read(field, args.joint, normalize=False)
                print(f"{field}: {value}")
            except Exception as exc:
                print(f"{field}: READ_FAILED ({exc})")
        item = calibration[args.joint]
        print("EEPROM_calibration_ticks:",
              f"min={item.range_min} max={item.range_max} homing_offset={item.homing_offset}")
        try:
            normalized = bus.sync_read("Present_Position", [args.joint])[args.joint]
            # Use the bus's existing read-only normalization implementation to
            # express its EEPROM endpoints in exactly the same degrees.
            low, high = bus._normalize({item.id: item.range_min})[item.id], \
                bus._normalize({item.id: item.range_max})[item.id]
            print(f"normalized_position_deg: {float(normalized):.3f}")
            print(f"EEPROM_range_deg: [{min(low, high):.3f}, {max(low, high):.3f}]")
        except Exception as exc:
            print(f"NORMALIZED_RANGE_READ_FAILED: {exc}")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
