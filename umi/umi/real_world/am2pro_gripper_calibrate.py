"""
One-time gripper calibration for the AM2Pro (AlohaMini 2/2 Pro) arm.

Runs under the lerobot environment (Python 3.12), NOT the UMI env, because it
imports lerobot's FeetechMotorsBus. It drives ONLY the gripper servo
(motor ID 7, sts3250, RANGE_0_100) so the arm joints are left alone.

You physically move the gripper to two reference poses and this script records
the corresponding servo values, producing the four mapping parameters the
controller needs:

    gripper_width_min     = 0.0            (fully closed -> jaw gap 0 m, fixed)
    gripper_width_max     = <measured>     (fully open jaw gap, meters)
    gripper_servo_closed  = <servo value>  (0-100 at fully closed)
    gripper_servo_open    = <servo value>  (0-100 at fully open)

Run it with the lerobot python:
    /home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python \
        umi/real_world/am2pro_gripper_calibrate.py --port /dev/ttyACM0
"""

import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

GRIPPER_NAME = "gripper"
GRIPPER_ID = 7
GRIPPER_MODEL = "sts3250"


def read_pos(bus):
    """Current gripper servo position, normalized to 0-100."""
    return float(bus.read("Present_Position", GRIPPER_NAME))


def write_goal(bus, value):
    value = min(100.0, max(0.0, float(value)))
    bus.write("Goal_Position", GRIPPER_NAME, value)
    return value


def print_status(servo_closed, servo_open, width_max):
    def fmt(x):
        return "?" if x is None else f"{x:.1f}"
    print("  --------------------------------------------")
    print(f"  servo_closed (c) = {fmt(servo_closed)}   -> width 0.0 m")
    print(f"  servo_open   (o) = {fmt(servo_open)}   -> width {width_max:.4f} m")
    print("  --------------------------------------------")


def print_help():
    print(
        "\nCommands:\n"
        "  <value>       move gripper to servo value 0-100 (e.g. `50`)\n"
        "  + / -         nudge +1 / -1\n"
        "  step <N>      set nudge step size (default 1)\n"
        "  c             record current position as CLOSED (width = 0)\n"
        "  o             record current position as OPEN\n"
        "  w <meters>    set measured max jaw gap (e.g. `w 0.085`)\n"
        "  p             print current status\n"
        "  q             quit and print the mapping parameters\n"
    )


def main():
    parser = argparse.ArgumentParser(description="AM2Pro gripper calibration")
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB serial port")
    args = parser.parse_args()

    motor = Motor(GRIPPER_ID, GRIPPER_MODEL, MotorNormMode.RANGE_0_100)
    bus = FeetechMotorsBus(port=args.port, motors={GRIPPER_NAME: motor})
    bus.connect()
    bus.calibration = bus.read_calibration()

    # Torque-enable the gripper only, POSITION mode, low torque limits.
    with bus.torque_disabled():
        bus.configure_motors()
        bus.write("Operating_Mode", GRIPPER_NAME, OperatingMode.POSITION.value)
        bus.write("Max_Torque_Limit", GRIPPER_NAME, 500)
        bus.write("Protection_Current", GRIPPER_NAME, 250)
        bus.write("Overload_Torque", GRIPPER_NAME, 25)

    servo_closed = None
    servo_open = None
    width_max = 0.09  # placeholder; overwrite with `w <meters>`
    step = 1.0

    print(f"[calib] gripper ready on {args.port}, current = {read_pos(bus):.1f}")
    print_help()

    try:
        while True:
            raw = input("calib> ").strip().lower()
            if not raw:
                continue
            parts = raw.split()

            if raw == "q":
                break
            elif raw == "c":
                servo_closed = read_pos(bus)
                print(f"  recorded CLOSED servo = {servo_closed:.1f}")
            elif raw == "o":
                servo_open = read_pos(bus)
                print(f"  recorded OPEN servo = {servo_open:.1f}")
            elif raw == "p":
                print_status(servo_closed, servo_open, width_max)
            elif raw == "h":
                print_help()
            elif raw == "+":
                write_goal(bus, read_pos(bus) + step)
                print(f"  -> {read_pos(bus):.1f}")
            elif raw == "-":
                write_goal(bus, read_pos(bus) - step)
                print(f"  -> {read_pos(bus):.1f}")
            elif parts[0] == "step" and len(parts) == 2:
                step = float(parts[1])
                print(f"  nudge step = {step}")
            elif parts[0] == "w" and len(parts) == 2:
                width_max = float(parts[1])
                print(f"  max width = {width_max:.4f} m")
            else:
                # numeric move
                try:
                    target = float(raw)
                except ValueError:
                    print("  unknown command (h for help)")
                    continue
                write_goal(bus, target)
                print(f"  -> {read_pos(bus):.1f}")

    except (KeyboardInterrupt, EOFError):
        print()

    bus.disconnect()

    print("\n===== calibration result =====")
    if servo_closed is None or servo_open is None:
        print("WARNING: you did not record both `c` (closed) and `o` (open).")
        print("Re-run and record both before using these parameters.")
    print("gripper_width_min    = 0.0")
    print(f"gripper_width_max    = {width_max:.4f}")
    print(f"gripper_servo_closed = {servo_closed if servo_closed is None else round(servo_closed, 1)}")
    print(f"gripper_servo_open   = {servo_open if servo_open is None else round(servo_open, 1)}")
    print("==============================")
    print("Paste these into eval_robots_config.yaml under the am2pro robot entry,")
    print("or pass them as the controller CLI args (--gripper-*).")


if __name__ == "__main__":
    sys.exit(main())
