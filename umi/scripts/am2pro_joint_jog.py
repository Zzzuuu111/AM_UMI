"""
Single-joint jog test: verify each servo's PHYSICAL direction against the model.

A reversed servo cannot be detected by pose readback (FK is model-space
self-consistent), so we drive ONE joint at a time and ask the human to watch
which way the arm physically moves.

Usage (lerobot_alohamini env):
    conda activate lerobot_alohamini
    python scripts/am2pro_joint_jog.py --port /dev/ttyACM0 --joint shoulder_lift --delta 10

For each joint: read current angle -> write goal +delta deg -> wait -> read back
-> restore original. delta=10 keeps the motion small. Ctrl+C restores the
original goal before exiting.
"""
import argparse
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = __file__.rsplit('/', 2)[0]
sys.path.insert(0, REPO_ROOT)

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from umi.real_world.am2pro_controller_server import (
    _MOTOR_SPEC, MOTOR_ARM_NAMES, URDF_PATH)

JOINT_INDEX = {name: i for i, name in enumerate(MOTOR_ARM_NAMES)}


def load_limits():
    root = ET.parse(URDF_PATH).getroot()
    out = {}
    for j in root.findall('joint'):
        lim = j.find('limit')
        name = j.get('name')
        if lim is None or name is None:
            continue
        # URDF uses the right_ prefix (right_shoulder_pan, ...);
        # wrist_yaw additionally has a _joint suffix in the URDF
        for motor_name in MOTOR_ARM_NAMES:
            if name in (f"right_{motor_name}", f"right_{motor_name}_joint"):
                out[motor_name] = (float(lim.get('lower')), float(lim.get('upper')))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--joint", required=True, choices=MOTOR_ARM_NAMES)
    parser.add_argument("--delta", type=float, default=10.0, help="degrees")
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()

    limits = load_limits()
    lo_deg, hi_deg = np.degrees(limits[args.joint])

    motors = {name: Motor(mid, model, mode) for name, mid, model, mode in _MOTOR_SPEC}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()
    bus.calibration = bus.read_calibration()
    with bus.torque_disabled():
        bus.configure_motors()
        for name in MOTOR_ARM_NAMES:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

    restore = None
    try:
        # ---- model prediction for this joint (user-view description) ----
        try:
            from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend
            urdf_joints = [
                f"right_{n}" if n != "wrist_yaw" else "right_wrist_yaw_joint"
                for n in MOTOR_ARM_NAMES]
            bk = create_kinematics_backend('ros2_dh', URDF_PATH, urdf_joints, 'right_Fixed_Jaw')
            all_q = np.array([float(bus.read("Present_Position", n)) for n in MOTOR_ARM_NAMES])
            J = bk.ik.jacobian(np.deg2rad(all_q))
            axis = J[3:, JOINT_INDEX[args.joint]]          # rotation axis in right_Base
            T = bk.forward_kinematics(all_q)
            hand_f = T[:3, :3] @ np.array([0, 0, 1.0])      # hand pointing dir
            # physical: X=左, Y=后, Z=上; user stands at -Y looking toward +Y
            omega = axis / max(np.linalg.norm(axis), 1e-9)
            df = np.cross(omega, hand_f)
            screen = []
            if abs(df[0]) > 0.3 * np.linalg.norm(df):
                screen.append('左' if df[0] > 0 else '右')
            if abs(df[2]) > 0.3 * np.linalg.norm(df):
                screen.append('上' if df[2] > 0 else '下')
            vis_rot = -omega[1]
            if abs(omega[1]) > 0.5:
                screen.append('从你视角顺时针' if vis_rot < 0 else '从你视角逆时针')
            elif abs(omega[0]) > 0.5 or abs(omega[2]) > 0.5:
                screen.append('以左右/上下为轴的翻转')
            print(f"MODEL PREDICT(+{args.delta}deg): {'/'.join(screen) if screen else '几乎不动'}")
        except Exception as e:
            print(f"(model prediction unavailable: {e})")

        cur = float(bus.read("Present_Position", args.joint))
        goal = cur + args.delta
        clipped = max(lo_deg, min(hi_deg, goal))
        if abs(clipped - goal) > 0.5:
            print(f"WARNING: goal {goal:.1f} clipped to limit {clipped:.1f}")
            goal = clipped
        print(f"[{args.joint}] current={cur:.2f} deg  ->  goal={goal:.2f} deg "
              f"(delta={goal-cur:+.1f})  [limits {lo_deg:.0f}, {hi_deg:.0f}]")
        print("WATCH the arm: which way does it physically move? (Ctrl+C = restore)")

        bus.write("Goal_Position", args.joint, goal)
        restore = cur
        for i in range(int(args.duration * 5)):
            time.sleep(0.2)
            p = float(bus.read("Present_Position", args.joint))
            print(f"  t={0.2*(i+1):.1f}s  actual={p:.2f} deg", end="\r")
        print()
        p = float(bus.read("Present_Position", args.joint))
        print(f"final readback: {p:.2f} deg (delta {p-cur:+.1f})")
    finally:
        if restore is not None:
            bus.write("Goal_Position", args.joint, restore)
            time.sleep(0.5)
            print(f"restored goal to {restore:.2f} deg")
        bus.disconnect()


if __name__ == "__main__":
    main()
