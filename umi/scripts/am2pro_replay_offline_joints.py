#!/usr/bin/env python3
"""Safely replay a handheld demo as an offline-solved AM2Pro joint trajectory.

Unlike ``am2pro_replay_episode.py``, this tool does *not* run a single online
Cartesian IK iteration for every live control tick.  It first solves and gates
the complete body-relative trajectory offline, then sends model-space joint
goals through the existing AM2Pro controller server.

It always moves to the supplied right_tcp reference in the same torque-enabled
controller session before replay.  No raw motor-bus writes occur here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
import zarr
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from umi.common.pose_util import mat_to_pose, pose_to_mat  # noqa: E402
from umi.real_world.am2pro_interpolation_controller import (  # noqa: E402
    AM2ProInterpolationController,
)
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    MOTOR_ARM_NAMES,
    mapping_from_config,
    model_safe_limits_from_config,
)


URDF_PATH = ROOT / "alohamini2pro_right_arm_kinematics.urdf"
URDF_JOINTS = (
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
)
JOINT_LABELS = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")

# Measured from the current AM2Pro servo calibration: J2's hardware
# Min_Position_Limit maps to about -99.96 deg, whereas the generic CAD URDF
# declares -188.80 deg.  Keep one degree of clearance until the physical arm
# is deliberately re-calibrated and this value is revalidated.
CURRENT_HARDWARE_LIMITS_DEG = {
    "J2 lift": (-98.0, None),
}


def load_reference(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("tcp_frame") != "right_tcp":
        raise ValueError("reference 必须是新夹爪 right_tcp 基准")
    state = data.get("robot_state", {})
    q = np.asarray(state.get("ActualQ", [])[:6], dtype=np.float64)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("reference.json 缺少有效 robot_state/ActualQ")
    return q


def joint_limits_deg(urdf_path: Path, model_safe_limits=None):
    root = ET.parse(urdf_path).getroot()
    nodes = {node.get("name"): node for node in root.findall("joint")}
    limits = np.asarray([
        (math.degrees(float(nodes[name].find("limit").get("lower"))),
         math.degrees(float(nodes[name].find("limit").get("upper"))))
        for name in URDF_JOINTS
    ], dtype=np.float64)
    for index, label in enumerate(JOINT_LABELS):
        lower, upper = CURRENT_HARDWARE_LIMITS_DEG.get(label, (None, None))
        if lower is not None:
            limits[index, 0] = max(limits[index, 0], lower)
        if upper is not None:
            limits[index, 1] = min(limits[index, 1], upper)
    for name, interval in (model_safe_limits or {}).items():
        if name not in MOTOR_ARM_NAMES:
            raise ValueError(f"model safety limits 包含未知关节: {name}")
        index = MOTOR_ARM_NAMES.index(name)
        lower, upper = interval
        limits[index] = (lower, upper)
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("URDF 与编码器安全范围相交后为空")
    return limits


def pose_error(target, actual):
    pos = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    delta = Rotation.from_matrix(target[:3, :3]).inv() * Rotation.from_matrix(actual[:3, :3])
    return pos, float(math.degrees(delta.magnitude()))


def solve_target(backend, target, seed, limits, steps, orientation_weight, rng):
    """Warm-started complete IK solve with bounded restarts when required."""
    seeds = [seed]
    # Random alternatives are only a fallback; preserving the warm-started
    # branch keeps the joint path continuous.
    for _ in range(2):
        seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))
    best = None
    for initial in seeds:
        q = initial.copy()
        for _ in range(steps):
            q = backend.inverse_kinematics(
                q, target, position_weight=1.0,
                orientation_weight=orientation_weight)
            # Enforce the calibrated hardware envelope during, not merely
            # after, solving.  This lets the remaining joints compensate for
            # a constrained J2 whenever a feasible branch exists.
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        pos_err, rot_err = pose_error(target, actual)
        score = pos_err / 0.002 + rot_err / 2.0
        if best is None or score < best[0]:
            best = (score, q, pos_err, rot_err)
        if pos_err <= 0.002 and rot_err <= 2.0:
            break
    return best[1], best[2], best[3]


def load_demo(path: Path, episode: int):
    register_codecs()
    if not path.is_dir():
        raise ValueError("当前离线关节 replay 只接受目录形式的 zarr 数据集")
    replay = ReplayBuffer.create_from_path(str(path), mode="r")
    if episode < 0 or episode >= replay.n_episodes:
        raise ValueError(f"dataset 只有 {replay.n_episodes} 个 episode，无法选择 {episode}")
    ep = replay.get_episode(episode)
    pos = np.asarray(ep["robot0_eef_pos"], dtype=np.float64)
    rot = np.asarray(ep["robot0_eef_rot_axis_angle"], dtype=np.float64)
    if pos.shape != rot.shape or pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("robot0_eef_pos / robot0_eef_rot_axis_angle 形状无效")
    return np.stack([pose_to_mat(np.r_[p, r]) for p, r in zip(pos, rot)])


def preposition(robot, target_q, duration, speed_cap, tolerance):
    state = robot.get_state()
    if state is None:
        raise RuntimeError("控制器未返回关节状态")
    current = np.asarray(state["ActualQ"][:6], dtype=np.float64)
    duration = max(float(duration), float(np.max(np.abs(target_q - current))) / speed_cap)
    print(f"REFERENCE PREPOSITION: {duration:.1f}s, speed cap {speed_cap:.1f} deg/s")
    print("  current:", np.round(current, 1))
    print("  target :", np.round(target_q, 1))
    robot.servoJ(target_q, duration=duration)
    done_at = time.monotonic() + duration
    deadline = done_at + 5.0
    latest = state
    while time.monotonic() < deadline:
        time.sleep(0.1)
        candidate = robot.get_state()
        if candidate is not None:
            latest = candidate
            error = np.asarray(latest["ActualQ"][:6], dtype=np.float64) - target_q
            if time.monotonic() >= done_at and np.max(np.abs(error)) <= tolerance:
                break
    actual = np.asarray(latest["ActualQ"][:6], dtype=np.float64)
    error = actual - target_q
    print("  readback:", np.round(actual, 1))
    print("  error   :", np.round(error, 2), "deg (actual - target)")
    if np.max(np.abs(error)) > tolerance:
        raise RuntimeError(
            f"基准复位未收敛：最大误差 {np.max(np.abs(error)):.2f}° > {tolerance:.2f}°；未开始 replay")
    print("REFERENCE PREPOSITION reached.")
    return latest


def main():
    parser = argparse.ArgumentParser(description="AM2Pro 离线完整 IK 关节回放；会实际驱动机器人")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", default="example/eval_robots_config.yaml")
    parser.add_argument("--urdf-path", default=None,
                        help="覆盖 robot-config 的 urdf_path；新 V 型夹爪使用 ROS2 kinematic URDF")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.7,
                        help="执行速度倍率；0.7 用于平滑关节加速度，1.0 为原始时间")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ik-steps", type=int, default=80)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--max-position-error-mm", type=float, default=2.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=2.0)
    parser.add_argument("--joint-margin-deg", type=float, default=5.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=220.0)
    parser.add_argument("--max-joint-accel-deg-s2", type=float, default=8000.0)
    parser.add_argument("--max-tracking-error-deg", type=float, default=5.0,
                        help="实物任一关节相对离线目标的最大允许误差；超出即中止并返回")
    parser.add_argument("--execution-mode", choices=("timed", "settled"), default="settled",
                        help="timed=按 demo 时间连续发送；settled=每个关节目标实际到位后才发下一点")
    parser.add_argument("--waypoint-stride", type=int, default=1,
                        help="settled 模式每隔多少源帧取一个关节目标；1 表示不丢任何动作点")
    parser.add_argument("--settle-tolerance-deg", type=float, default=0.8,
                        help="settled 模式允许发下一点前，所有关节必须达到的误差")
    parser.add_argument("--settle-max-joint-speed-deg-s", type=float, default=10.0,
                        help="settled 模式每个关节目标的最大平均速度")
    parser.add_argument("--waypoint-timeout-s", type=float, default=3.0,
                        help="settled 模式单个关节目标的最大等待时间")
    parser.add_argument("--reference-duration", type=float, default=18.0)
    parser.add_argument("--reference-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--reference-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--return", dest="do_return", action="store_true", default=True)
    parser.add_argument("--no-return", dest="do_return", action="store_false")
    args = parser.parse_args()
    if (args.fps <= 0 or args.speed <= 0 or args.scale <= 0 or
            args.waypoint_stride <= 0 or args.settle_tolerance_deg <= 0 or
            args.settle_max_joint_speed_deg_s <= 0 or args.waypoint_timeout_s <= 0):
        parser.error("fps、speed、scale 必须为正")

    config = yaml.safe_load(Path(args.robot_config).read_text())
    robot_cfg = config["robots"][0]
    configured_urdf = args.urdf_path or robot_cfg.get("urdf_path", str(URDF_PATH))
    urdf_path = Path(configured_urdf).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF 不存在: {urdf_path}")
    dataset = Path(args.dataset).expanduser().resolve()
    reference_q = load_reference(Path(args.start_reference).expanduser().resolve())
    poses = load_demo(dataset, args.episode)
    if args.max_frames is not None:
        poses = poses[:max(1, min(len(poses), args.max_frames))]
    mapping = mapping_from_config(robot_cfg, ROOT)
    model_safe_limits = model_safe_limits_from_config(robot_cfg, mapping)
    limits = joint_limits_deg(urdf_path, model_safe_limits=model_safe_limits)
    print("active hardware-safe limits:", {
        label: {"lower_deg": float(limits[i, 0]), "upper_deg": float(limits[i, 1])}
        for i, label in enumerate(JOINT_LABELS)
        if label in CURRENT_HARDWARE_LIMITS_DEG or MOTOR_ARM_NAMES[i] in model_safe_limits
    })
    backend_name = robot_cfg.get("ik_backend", "ros2_dh")
    backend = create_kinematics_backend(backend_name, str(urdf_path), URDF_JOINTS, "right_tcp")
    print(f"kinematics: backend={backend_name}; urdf={urdf_path}")

    print("OFFLINE_JOINT_REPLAY_PREPARE")
    print(f"frames: {len(poses)}; source duration: {(len(poses)-1)/args.fps:.2f}s; "
          f"execution duration: {(len(poses)-1)/args.fps/args.speed:.2f}s")
    print("No robot command has been sent during this offline preparation.")

    with AM2ProInterpolationController(
        robot_usb_port=robot_cfg.get("robot_usb_port", "/dev/ttyACM0"),
        frequency=50,
        receive_latency=robot_cfg.get("robot_obs_latency", 0.005),
        server_python=robot_cfg.get("robot_python"),
        ik_backend=robot_cfg.get("ik_backend", "ros2_dh"),
        urdf_path=str(urdf_path),
        flip_joints=robot_cfg.get("joint_flip", []),
        joint_model_candidate=robot_cfg.get("joint_model_candidate"),
        gripper_width_min=robot_cfg.get("gripper_width_min", 0.0),
        gripper_width_max=robot_cfg.get("gripper_width_max", 0.074),
        gripper_servo_closed=robot_cfg.get("gripper_servo_closed", 5.3),
        gripper_servo_open=robot_cfg.get("gripper_servo_open", 93.6),
        verbose=True,
    ) as robot:
        deadline = time.monotonic() + 5.0
        while robot.get_state() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        state = preposition(robot, reference_q, args.reference_duration,
                             args.reference_speed_deg_s, args.reference_tolerance_deg)
        start_q = np.asarray(state["ActualQ"][:6], dtype=np.float64)
        start_t = pose_to_mat(np.asarray(state["ActualTCPPose"], dtype=np.float64))

        # Solve after the verified actual start state; this preserves the
        # demonstrated body-relative motion even if readback is a fraction of
        # a degree away from the saved V9 state.
        rel0_inv = np.linalg.inv(poses[0])
        q_prev = start_q.copy()
        q_traj, pos_errors, rot_errors = [], [], []
        rng = np.random.default_rng(0)
        print("OFFLINE_IK_SOLVING: no replay command is being sent yet ...")
        for i, pose in enumerate(poses):
            relative = rel0_inv @ pose
            relative = relative.copy()
            relative[:3, 3] *= args.scale
            target = start_t @ relative
            q_prev, pos_err, rot_err = solve_target(
                backend, target, q_prev, limits, args.ik_steps,
                args.orientation_weight, rng)
            q_traj.append(q_prev)
            pos_errors.append(pos_err)
            rot_errors.append(rot_err)
        q_traj = np.asarray(q_traj)
        pos_errors = np.asarray(pos_errors)
        rot_errors = np.asarray(rot_errors)
        margins = np.minimum(q_traj - limits[:, 0], limits[:, 1] - q_traj)
        period = 1.0 / args.fps / args.speed
        velocity = np.abs(np.diff(q_traj, axis=0)) / period
        acceleration = np.abs(np.diff(q_traj, n=2, axis=0)) / (period * period)
        max_pos_mm = float(np.max(pos_errors) * 1000)
        max_rot_deg = float(np.max(rot_errors))
        min_margin = float(np.min(margins))
        max_vel = float(np.max(velocity, initial=0.0))
        max_acc = float(np.max(acceleration, initial=0.0))
        print("OFFLINE_IK_RESULT")
        print(f"  max_tcp_position_error_mm: {max_pos_mm:.3f}")
        print(f"  max_tcp_rotation_error_deg: {max_rot_deg:.3f}")
        print(f"  min_joint_margin_deg: {min_margin:.3f}")
        print(f"  max_joint_speed_deg_s: {max_vel:.1f}")
        print(f"  max_joint_accel_deg_s2: {max_acc:.1f}")
        if (max_pos_mm > args.max_position_error_mm or
                max_rot_deg > args.max_rotation_error_deg or
                min_margin < args.joint_margin_deg or
                max_vel > args.max_joint_speed_deg_s or
                max_acc > args.max_joint_accel_deg_s2):
            raise RuntimeError("离线关节轨迹未通过安全门；未开始 replay")

        print("OFFLINE_JOINT_REPLAYING: gripper is held at its current opening.")
        if args.execution_mode == "settled":
            print("  mode: state-driven / no demo clock; each target waits for "
                  f"all joints within {args.settle_tolerance_deg:.2f} deg")
            indices = list(range(0, len(q_traj), args.waypoint_stride))
            if indices[-1] != len(q_traj) - 1:
                indices.append(len(q_traj) - 1)
        else:
            print("  mode: timed / follows the demo clock")
            indices = list(range(len(q_traj)))
        try:
            if args.execution_mode == "settled":
                # Do not continuously replace a still-active servoJ command.
                # The next waypoint is issued only after physical readback has
                # reached the previous target.  This deliberately makes the
                # replay duration scene/mechanics dependent rather than tied
                # to the hand-held video's timestamps.
                for step, i in enumerate(indices):
                    q = q_traj[i]
                    before = robot.get_state()
                    if before is None:
                        raise RuntimeError("执行时控制器状态丢失")
                    before_q = np.asarray(before["ActualQ"][:6], dtype=np.float64)
                    max_delta = float(np.max(np.abs(q - before_q)))
                    duration = max(0.12, max_delta / args.settle_max_joint_speed_deg_s)
                    robot.servoJ(q, duration=duration)
                    deadline = time.monotonic() + max(args.waypoint_timeout_s, duration + 0.5)
                    latest = before
                    while time.monotonic() < deadline:
                        time.sleep(0.02)
                        candidate = robot.get_state()
                        if candidate is None:
                            continue
                        latest = candidate
                        actual = np.asarray(latest["ActualQ"][:6], dtype=np.float64)
                        error = actual - q
                        if np.max(np.abs(error)) <= args.settle_tolerance_deg:
                            break
                    actual = np.asarray(latest["ActualQ"][:6], dtype=np.float64)
                    error = actual - q
                    max_error = float(np.max(np.abs(error)))
                    if max_error > args.settle_tolerance_deg:
                        worst = int(np.argmax(np.abs(error)))
                        print("    target:", np.round(q, 1))
                        print("    actual:", np.round(actual, 1))
                        print("    error :", np.round(error, 2), "deg (actual - target)")
                        raise RuntimeError(
                            f"状态驱动 replay 超时：{JOINT_LABELS[worst]} 仍差 "
                            f"{error[worst]:+.2f}°；未发送后续目标")
                    if step % max(1, int(round(args.fps / 5))) == 0 or step == len(indices) - 1:
                        print(f"  waypoint {step + 1}/{len(indices)} (source frame {i}): "
                              f"settled max error={max_error:.2f} deg", flush=True)
            else:
                next_time = time.monotonic()
                for i, q in enumerate(q_traj):
                    robot.servoJ(q, duration=period)
                    # Check every commanded sample, not merely once per second:
                    # a failed servo must not receive another second of targets.
                    observed = robot.get_state()
                    if observed is not None:
                        actual = np.asarray(observed["ActualQ"][:6], dtype=np.float64)
                        error = actual - q
                        max_error = float(np.max(np.abs(error)))
                        if max_error > args.max_tracking_error_deg:
                            worst = int(np.argmax(np.abs(error)))
                            print("    target:", np.round(q, 1))
                            print("    actual:", np.round(actual, 1))
                            print("    error :", np.round(error, 2), "deg (actual - target)")
                            raise RuntimeError(
                                f"实物跟踪保护中止：{JOINT_LABELS[worst]} 误差 "
                                f"{error[worst]:+.2f}°，超过 "
                                f"{args.max_tracking_error_deg:.2f}°")
                        if i % max(1, int(round(args.fps))) == 0:
                            print(f"  t={i/args.fps:.1f}s max_joint_tracking_error="
                                  f"{max_error:.2f} deg", flush=True)
                            print("    target:", np.round(q, 1))
                            print("    actual:", np.round(actual, 1))
                            print("    error :", np.round(error, 2), "deg (actual - target)")
                    next_time += period
                    sleep_time = next_time - time.monotonic()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("OFFLINE_JOINT_REPLAY_INTERRUPTED")
        except RuntimeError as exc:
            print(f"OFFLINE_JOINT_REPLAY_ABORTED: {exc}")
        finally:
            if args.do_return:
                print("Returning to verified start joints ...")
                robot.servoJ(start_q, duration=3.0)
                time.sleep(3.2)
                print("Returned.")


if __name__ == "__main__":
    main()
