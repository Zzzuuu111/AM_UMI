"""Read AM2Pro servo positions without enabling torque or commanding motion."""

import argparse

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


MOTOR_SPECS = (
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()

    motors = {
        name: Motor(motor_id, model, norm_mode)
        for name, motor_id, model, norm_mode in MOTOR_SPECS
    }
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    try:
        bus.connect()
        # These are EEPROM / present-position reads only.  Do not call
        # configure_motors(), torque_enabled(), write(), or sync_write().
        bus.calibration = bus.read_calibration()
        positions = bus.sync_read("Present_Position", list(motors))
        print("AM2PRO_READ_STATE_OK")
        print("port:", args.port)
        for name in motors:
            print(f"{name}: {float(positions[name]):.3f}")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
