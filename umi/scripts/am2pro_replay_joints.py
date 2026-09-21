"""
Joint-space replay: solve the ENTIRE demo trajectory to joint angles OFFLINE
(multi-seed IK, as many iterations as needed), then stream the joint goals
directly to the servos. Completely bypasses online IK convergence limits.

Usage (lerobot_alohamini env, drives the bus directly - no IK server):
    python scripts/am2pro_replay_joints.py \
        --dataset imu_work/demo_batch3_dataset.zarr --episode 0 \
        --fps 30 --speed 0.5 --scale 0.7

Pipeline per frame:
    target = T_cur @ (T_0^-1 @ T_i)          # body-relative, scaled
    q_i    = offline IK (warm-start from previous frame + multi-seed retry)
    stream q_i + gripper to the servos

Ctrl+C stops. Requires the arm posed at a reachable start configuration
(see printed joints; wrist visibly bent, avoid elbow fully straight).
"""
import argparse
import sys
import time

import numpy as np
import zarr

REPO_ROOT = __file__.rsplit('/', 2)[0]
sys.path.insert(0, REPO_ROOT)

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402
from lerobot.motors import Motor  # noqa: E402
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode  # noqa: E402
from umi.real_world.am2pro_controller_server import (  # noqa: E402
    _MOTOR_SPEC, MOTOR_ARM_NAMES, GRIPPER_MOTOR_NAME, URDF_PATH)

register_codecs()
URDF_JOINTS = ['right_shoulder_pan', 'right_shoulder_lift', 'right_elbow_flex',
               'right_wrist_flex', 'right_wrist_yaw_joint', 'right_wrist_roll']
LIMITS = [(-2.24, 2.13), (-3.30, 0.16), (-0.07, 3.22),
          (-1.65, 1.48), (-1.25, 1.19), (-3.14, 3.14)]


def solve_frame(bk, T_t, q_prev, rng):
    """Warm-start IK; on poor convergence, retry with random seeds."""
    best = None
    seeds = [q_prev] + [np.degrees([rng.uniform(*l) for l in LIMITS]) for _ in range(3)]
    for q_seed in seeds:
        q = np.deg2rad(q_seed).copy()
        for _ in range(60):
            q, _ = bk.ik.step(q, T_t, dt=0.02, position_gain=40, orientation_gain=20,
                              max_joint_velocity=3.0, max_joint_step=0.2)
        e = np.linalg.norm(bk.forward_kinematics(np.rad2deg(q))[:3, 3] - T_t[:3, 3])
        if best is None or e < best[0]:
            best = (e, np.rad2deg(q))
        if best[0] < 0.005:      # 5mm good enough, stop early
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--scale", type=float, default=0.7)
    ap.add_argument("--gripper-servo-closed", type=float, default=97.1)
    ap.add_argument("--gripper-servo-open", type=float, default=1.4)
    ap.add_argument("--gripper-width-min", type=float, default=0.0)
    ap.add_argument("--gripper-width-max", type=float, default=0.09)
    args = ap.parse_args()

    bk = create_kinematics_backend('ros2_dh', URDF_PATH, URDF_JOINTS, 'right_Fixed_Jaw')

    store = zarr.ZipStore(args.dataset, mode='r')
    rb = ReplayBuffer.copy_from_store(store, zarr.MemoryStore())
    ep = rb.get_episode(args.episode)
    pos, rot = ep['robot0_eef_pos'], ep['robot0_eef_rot_axis_angle']
    width = ep['robot0_gripper_width'].reshape(-1)
    demo = np.stack([pose_to_mat(np.concatenate([pos[i], rot[i]])) for i in range(len(pos))])
    n = len(pos)

    # ---- connect bus and read the current pose ----
    motors = {name: Motor(mid, model, mode) for name, mid, model, mode in _MOTOR_SPEC}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()
    bus.calibration = bus.read_calibration()
    with bus.torque_disabled():
        bus.configure_motors()
        for name in MOTOR_ARM_NAMES:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
    cur_q = np.array([float(bus.read("Present_Position", n)) for n in MOTOR_ARM_NAMES])
    T_cur = bk.forward_kinematics(cur_q)
    print(f"start joints: {np.round(cur_q,1)}")

    # ---- offline joint-space solve of the whole trajectory ----
    print(f"offline IK solving {n} frames ...")
    T0_inv = np.linalg.inv(demo[0])
    rng = np.random.default_rng(0)
    q_traj = np.zeros((n, 6))
    q_prev = cur_q.copy()
    errs = []
    t0 = time.time()
    for i in range(n):
        M = T0_inv @ demo[i]
        M = M.copy()
        M[:3, 3] *= args.scale
        T_t = T_cur @ M
        e, q_i = solve_frame(bk, T_t, q_prev, rng)
        q_traj[i] = q_i
        q_prev = q_i
        errs.append(e)
    errs = np.array(errs)
    print(f"solved in {time.time()-t0:.0f}s | IK 误差: 中位={np.median(errs)*1000:.1f}mm "
          f"最大={errs.max()*1000:.1f}mm | >20mm占比={(errs>0.02).mean()*100:.0f}%")
    if np.median(errs) > 0.02:
        print("WARNING: many frames unreachable at this scale; try --scale lower "
              "or a different start pose.")

    # ---- stream joint goals ----
    print("REPLAYING joint trajectory (Ctrl+C to stop) ...")
    dt = 1.0 / args.fps
    try:
        for i in range(n):
            goals = {name: float(q_traj[i][j]) for j, name in enumerate(MOTOR_ARM_NAMES)}
            w = float(np.clip(width[i], args.gripper_width_min, args.gripper_width_max))
            ratio = (w - args.gripper_width_min) / (args.gripper_width_max - args.gripper_width_min)
            goals[GRIPPER_MOTOR_NAME] = args.gripper_servo_closed + ratio * (
                args.gripper_servo_open - args.gripper_servo_closed)
            bus.sync_write("Goal_Position", goals)
            time.sleep(dt / args.speed)
    except KeyboardInterrupt:
        print("\nreplay interrupted")
    finally:
        bus.disconnect()
    print("done")


if __name__ == "__main__":
    main()
