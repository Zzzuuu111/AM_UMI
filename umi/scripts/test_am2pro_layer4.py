"""
AM2Pro Layer-4 test: IK closed loop (ARM WILL MOVE A LITTLE).

Staged test:
  1. Start the controller (spawns the lerobot-env server; torque ON, holds pose)
  2. Read current TCP pose (FK from actual joint angles)
  3. Move to current pose + small offset (default: +2 cm in Z, i.e. upward)
  4. Read back achieved pose and report error vs target (mm)
  5. Move back to the original pose
  6. Stop the controller

Run in the UMI env:
    conda activate umi
    python scripts/test_am2pro_layer4.py --port /dev/ttyACM0

SAFETY: keep the area around the arm clear; the move is small (~2 cm).
Ctrl+C stops the script; the controller's stop() also disables motion.
"""
import argparse
import sys
import os
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from umi.real_world.am2pro_interpolation_controller import (
    AM2ProInterpolationController,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--dx", type=float, default=0.0)
    parser.add_argument("--dy", type=float, default=0.0)
    parser.add_argument("--dz", type=float, default=0.02, help="Z offset, +up (m)")
    parser.add_argument("--flip-joints", default="wrist_roll",
                        help="comma-separated reversed joints (verified: wrist_roll)")
    parser.add_argument("--server-python",
                        default="/home/zzzjh/anaconda3/envs/AM_UMI/bin/python",
                        help="Python interpreter used for the AM2Pro controller server")
    args = parser.parse_args()

    ctrl = AM2ProInterpolationController(
        robot_usb_port=args.port, verbose=True, launch_timeout=30.0,
        flip_joints=[n.strip() for n in args.flip_joints.split(",") if n.strip()],
        server_python=args.server_python)

    print("[1] starting controller (spawns server, torque ON, holds pose) ...")
    try:
        ctrl.start(wait=True)
        print("    server ready OK")
    except Exception as e:
        print("    START FAILED:", e)
        return 1

    try:
        time.sleep(1.5)
        st = ctrl.get_state()
        if st is None:
            print("    no state received from server")
            return 1
        pose0 = st["ActualTCPPose"].astype(float)
        print("[2] current pose (x,y,z,rx,ry,rz):", np.round(pose0, 4))
        print("    gripper width:", round(float(st["gripper_position"]), 4), "m")

        target = pose0.copy()
        target[0] += args.dx
        target[1] += args.dy
        target[2] += args.dz
        print(f"[3] moving to pose + ({args.dx}, {args.dy}, {args.dz}) m over 3s ...")
        ctrl.servoL(target, duration=3.0)
        time.sleep(4.5)

        st = ctrl.get_state()
        pose1 = st["ActualTCPPose"].astype(float)
        err_mm = float(np.linalg.norm(pose1[:3] - target[:3])) * 1000.0
        print("[4] achieved pose:", np.round(pose1, 4))
        print(f"    position error vs target: {err_mm:.1f} mm")

        print("[5] moving back to original pose over 3s ...")
        ctrl.servoL(pose0, duration=3.0)
        time.sleep(4.5)
        print("    done moving back")
    finally:
        print("[6] stopping controller ...")
        ctrl.stop()
    print("DONE")


if __name__ == "__main__":
    main()
