"""
Gripper servo jog: find the correct open/closed servo values (0-100 scale).

The gripper servo runs in RANGE_0_100 mode where 0-100 = 0-300 deg of
physical horn rotation. The current config maps width [0, 0.09] m onto a
~96-unit span (~287 deg) - far more rotation than the ~20 mm of jaw travel
actually needs. Jog the servo and note the values for:

    closed: jaws just touching (0 mm gap)
    open:   jaws at the max gap your task needs (~20 mm)

Then update gripper_servo_closed / gripper_servo_open in
example/eval_robots_config.yaml.

Usage (lerobot_alohamini env):
    python scripts/gripper_jog.py --port /dev/ttyACM0 --read
    python scripts/gripper_jog.py --port /dev/ttyACM0 --value 80
"""
import argparse
import sys
import time

REPO_ROOT = __file__.rsplit('/', 2)[0]
sys.path.insert(0, REPO_ROOT)

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from umi.real_world.am2pro_controller_server import _MOTOR_SPEC, GRIPPER_MOTOR_NAME


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--value", type=float, default=None, help="target 0-100")
    parser.add_argument("--read", action="store_true", help="print current value only")
    args = parser.parse_args()

    motors = {name: Motor(mid, model, mode) for name, mid, model, mode in _MOTOR_SPEC}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()
    bus.calibration = bus.read_calibration()
    with bus.torque_disabled():
        bus.configure_motors()
        for name, mid, model, mode in _MOTOR_SPEC:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

    try:
        cur = float(bus.read("Present_Position", GRIPPER_MOTOR_NAME))
        print(f"gripper current = {cur:.2f} (0-100 scale, 0-100 = 0-300 deg physical)")
        if args.read or args.value is None:
            return
        target = max(0.0, min(100.0, args.value))
        bus.write("Goal_Position", GRIPPER_MOTOR_NAME, target)
        for _ in range(10):
            time.sleep(0.2)
            p = float(bus.read("Present_Position", GRIPPER_MOTOR_NAME))
            print(f"  -> {p:.2f}", end="\r")
        print()
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
