#!/usr/bin/env python3
"""Offline arm-replay gate for AM_UMI hand-held demonstration datasets.

For every episode, this tool treats the recorded motion as body-relative,
exactly like ``am2pro_replay_episode.py``:

    T_target[i] = T_robot_start @ inv(T_demo[0]) @ T_demo[i]

It then solves the whole trajectory with the *current* AM2Pro URDF and the
new ``right_tcp`` frame.  No serial port is opened and no robot command is
ever sent.  A rejected episode should not be used for AM2Pro training until it
has been re-recorded or deliberately scaled/edited.

This is a kinematic/dynamic screening gate, not a collision certificate:
the present URDF has a measured TCP but not a complete collision model of the
new long gripper, and it cannot see obstacles in the real scene.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
from umi.common.pose_util import pose_to_mat  # noqa: E402
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

# The active J2 servo calibration has a lower commandable limit of about
# -99.96 deg.  The generic CAD URDF is wider (-188.80 deg), so use a one-degree
# clearance until a deliberate physical recalibration proves otherwise.
CURRENT_HARDWARE_LIMITS_DEG = {
    1: (-98.0, None),  # index 1 = shoulder_lift / J2
}


def open_replay_buffer(path: Path):
    if path.is_dir():
        return ReplayBuffer.create_from_path(str(path), mode="r"), None
    store = zarr.ZipStore(str(path), mode="r")
    return ReplayBuffer.copy_from_store(store, zarr.MemoryStore()), store


def load_reference(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    state = data.get("robot_state")
    if not state or "ActualTCPPose" not in state or "ActualQ" not in state:
        raise ValueError("reference.json 缺少 robot_state/ActualTCPPose/ActualQ")
    frame = data.get("tcp_frame")
    if frame != "right_tcp":
        raise ValueError(
            f"reference tcp_frame={frame!r}; 此筛选器只接受新夹爪 right_tcp 基准。")
    q = np.asarray(state["ActualQ"], dtype=np.float64)
    if q.shape[0] < 6 or not np.all(np.isfinite(q[:6])):
        raise ValueError("reference ActualQ 无效")
    pose = np.asarray(state["ActualTCPPose"], dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("reference ActualTCPPose 无效")
    return q[:6], pose, data


def load_joint_limits(urdf_path: Path, model_safe_limits=None):
    root = ET.parse(urdf_path).getroot()
    nodes = {joint.get("name"): joint for joint in root.findall("joint")}
    limits = np.asarray([
        (math.degrees(float(nodes[name].find("limit").get("lower"))),
         math.degrees(float(nodes[name].find("limit").get("upper"))))
        for name in URDF_JOINTS
    ], dtype=np.float64)
    # Legacy configuration recorded J2's safety envelope directly in model
    # coordinates.
    for index, (lower, upper) in CURRENT_HARDWARE_LIMITS_DEG.items():
        if lower is not None:
            limits[index, 0] = max(limits[index, 0], lower)
        if upper is not None:
            limits[index, 1] = min(limits[index, 1], upper)
    # Explicit physical-model bounds may deliberately replace a CAD limit
    # after EEPROM evidence and a conservative margin have been recorded in
    # the robot config. They are validated by model_safe_limits_from_config.
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
    position_m = float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
    delta = Rotation.from_matrix(target[:3, :3]).inv() * Rotation.from_matrix(actual[:3, :3])
    return position_m, float(math.degrees(delta.magnitude()))


def compact_frame_ranges(indices: np.ndarray) -> list[list[int]]:
    """Convert sorted frame indices to compact inclusive [start, end] ranges."""
    values = np.asarray(indices, dtype=int).reshape(-1)
    if not len(values):
        return []
    ranges = []
    start = previous = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == previous + 1:
            previous = value
            continue
        ranges.append([start, previous])
        start = previous = value
    ranges.append([start, previous])
    return ranges


def solve_target(backend, target, seed_deg, limits_deg, args, rng):
    """Warm-start DLS, then use bounded random restarts only if needed."""
    seeds = [np.asarray(seed_deg, dtype=np.float64)]
    for _ in range(args.random_restarts):
        seeds.append(rng.uniform(limits_deg[:, 0], limits_deg[:, 1]))
    best_q, best_pos, best_rot = None, math.inf, math.inf
    for initial in seeds:
        q = initial.copy()
        for _ in range(args.ik_steps):
            q = backend.inverse_kinematics(
                q, target, position_weight=1.0,
                orientation_weight=args.orientation_weight)
            # The generic CAD model has wider ranges than this calibrated arm.
            # Project every iteration so the solver can use J3/J4/J5 to
            # compensate instead of returning an impossible J2 solution.
            q = np.clip(q, limits_deg[:, 0], limits_deg[:, 1])
        actual = backend.forward_kinematics(q)
        pos_error, rot_error = pose_error(target, actual)
        score = pos_error / args.max_position_error_m + rot_error / args.max_rotation_error_deg
        best_score = (best_pos / args.max_position_error_m +
                      best_rot / args.max_rotation_error_deg)
        if score < best_score:
            best_q, best_pos, best_rot = q, pos_error, rot_error
        if pos_error <= args.max_position_error_m and rot_error <= args.max_rotation_error_deg:
            break
    return best_q, best_pos, best_rot


def episode_pose_matrices(ep):
    if "robot0_eef_pos" not in ep or "robot0_eef_rot_axis_angle" not in ep:
        raise ValueError("episode 缺少 robot0_eef_pos 或 robot0_eef_rot_axis_angle")
    pos = np.asarray(ep["robot0_eef_pos"], dtype=np.float64)
    rot = np.asarray(ep["robot0_eef_rot_axis_angle"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or rot.shape != pos.shape:
        raise ValueError("episode 末端位姿数组形状无效")
    if len(pos) < 2 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
        raise ValueError("episode 末端位姿为空或含 NaN/Inf")
    return np.stack([pose_to_mat(np.r_[p, r]) for p, r in zip(pos, rot)])


def filter_episode(episode_id, ep, backend, q_start, t_start, limits, args):
    poses = episode_pose_matrices(ep)
    width = np.asarray(ep.get("robot0_gripper_width", []), dtype=np.float64).reshape(-1)
    reasons = []
    # zarr stores the converted labels as float32.  A correctly clamped
    # 0.074 m endpoint may consequently read back as 0.07400000095; that is
    # numerical representation, not a physically over-open gripper.
    width_epsilon_m = 1e-6
    if len(width) != len(poses):
        reasons.append("gripper_length_mismatch")
    elif not np.all(np.isfinite(width)):
        reasons.append("gripper_non_finite")
    elif np.any((width < args.gripper_min_m - width_epsilon_m) |
                 (width > args.gripper_max_m + width_epsilon_m)):
        reasons.append("gripper_out_of_range")

    rel0_inv = np.linalg.inv(poses[0])
    rng = np.random.default_rng(episode_id)
    q_prev = q_start.copy()
    solved = []
    pos_errors, rot_errors = [], []
    for pose in poses:
        relative = rel0_inv @ pose
        relative = relative.copy()
        relative[:3, 3] *= args.translation_scale
        target = t_start @ relative
        q_prev, pos_error, rot_error = solve_target(
            backend, target, q_prev, limits, args, rng)
        solved.append(q_prev)
        pos_errors.append(pos_error)
        rot_errors.append(rot_error)

    q = np.asarray(solved)
    pos_errors = np.asarray(pos_errors)
    rot_errors = np.asarray(rot_errors)
    bad_ik = ((pos_errors > args.max_position_error_m) |
              (rot_errors > args.max_rotation_error_deg))
    if bad_ik.any():
        reasons.append("ik_residual")
    edge_distance = np.minimum(q - limits[:, 0], limits[:, 1] - q)
    near_limit = edge_distance < args.joint_limit_margin_deg
    if near_limit.any():
        reasons.append("joint_limit_margin")
    velocity = np.abs(np.diff(q, axis=0)) * args.fps
    acceleration = np.abs(np.diff(q, n=2, axis=0)) * args.fps * args.fps
    if velocity.size and velocity.max() > args.max_joint_speed_deg_s:
        reasons.append("joint_speed")
    if acceleration.size and acceleration.max() > args.max_joint_accel_deg_s2:
        reasons.append("joint_acceleration")

    return {
        "episode": int(episode_id),
        "frames": int(len(poses)),
        "duration_s": float((len(poses) - 1) / args.fps),
        "accepted": not reasons,
        "reasons": reasons,
        "ik_position_error_mm": {
            "median": float(np.median(pos_errors) * 1000),
            "max": float(np.max(pos_errors) * 1000),
            "bad_frames": int(bad_ik.sum()),
        },
        "ik_rotation_error_deg": {
            "median": float(np.median(rot_errors)),
            "max": float(np.max(rot_errors)),
        },
        "bad_ik_frame_ranges": compact_frame_ranges(np.flatnonzero(bad_ik)),
        "joint_limit_min_margin_deg": float(np.min(edge_distance)),
        "joint_limit_min_margin_each_deg": {
            name: float(np.min(edge_distance[:, index]))
            for index, name in enumerate(("J1 pan", "J2 lift", "J3 elbow",
                                          "J4 flex", "J5 yaw", "J6 roll"))
        },
        "near_limit_frame_ranges_by_joint": {
            name: compact_frame_ranges(np.flatnonzero(near_limit[:, index]))
            for index, name in enumerate(("J1 pan", "J2 lift", "J3 elbow",
                                          "J4 flex", "J5 yaw", "J6 roll"))
        },
        "solved_joint_range_deg": {
            name: [float(q[:, index].min()), float(q[:, index].max())]
            for index, name in enumerate(("J1 pan", "J2 lift", "J3 elbow",
                                          "J4 flex", "J5 yaw", "J6 roll"))
        },
        "joint_speed_max_deg_s": float(velocity.max(initial=0.0)),
        "joint_acceleration_max_deg_s2": float(acceleration.max(initial=0.0)),
        "gripper_width_range_m": ([float(width.min()), float(width.max())]
                                  if len(width) else None),
    }


def main():
    parser = argparse.ArgumentParser(
        description="离线筛选可由 AM2Pro 新夹爪复现的 UMI episode；绝不连接机械臂。")
    parser.add_argument("--dataset", required=True, help="已转换的 UMI zarr 目录或 ZipStore 文件")
    parser.add_argument("--start-reference", required=True,
                        help="新夹爪 right_tcp 的 reference.json，例如 follow_umi_task_start_v3_tag_visible/reference.json")
    parser.add_argument("--robot-config", default="example/eval_robots_config.yaml",
                        help="读取 ik_backend 与 urdf_path；不会连接机械臂")
    parser.add_argument("--urdf-path", default=None,
                        help="覆盖 robot-config 的 urdf_path")
    parser.add_argument("--report", default=None, help="JSON 报告路径")
    parser.add_argument("--accepted-out", default=None, help="只含可接受 episode 编号的文本文件")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--translation-scale", type=float, default=1.0,
                        help="相对平移缩放；1.0 表示不缩放、严格筛选")
    parser.add_argument("--ik-steps", type=int, default=60)
    parser.add_argument("--random-restarts", type=int, default=2)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--max-position-error-m", type=float, default=0.010)
    parser.add_argument("--max-rotation-error-deg", type=float, default=8.0)
    parser.add_argument("--joint-limit-margin-deg", type=float, default=5.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=100.0)
    parser.add_argument("--max-joint-accel-deg-s2", type=float, default=800.0)
    parser.add_argument("--gripper-min-m", type=float, default=0.0)
    parser.add_argument("--gripper-max-m", type=float, default=0.074)
    args = parser.parse_args()
    if (args.fps <= 0 or args.translation_scale <= 0 or args.ik_steps <= 0 or
            args.random_restarts < 0 or args.orientation_weight <= 0):
        parser.error("fps、scale、ik-steps、orientation-weight 必须为正；random-restarts 不能为负")

    config = yaml.safe_load(Path(args.robot_config).read_text())
    robot_cfg = config["robots"][0]
    configured_urdf = args.urdf_path or robot_cfg.get("urdf_path", str(URDF_PATH))
    urdf_path = Path(configured_urdf).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF 不存在: {urdf_path}")
    dataset = Path(args.dataset).expanduser().resolve()
    reference_path = Path(args.start_reference).expanduser().resolve()
    q_start, pose_start, reference = load_reference(reference_path)
    t_start = pose_to_mat(pose_start)
    mapping = mapping_from_config(robot_cfg, ROOT)
    model_safe_limits = model_safe_limits_from_config(robot_cfg, mapping)
    limits = load_joint_limits(urdf_path, model_safe_limits=model_safe_limits)
    print(f"model-safe J2 lift range: [{limits[1, 0]:.2f}, {limits[1, 1]:.2f}] deg")
    print("joint mapping:", mapping["source"])
    backend_name = robot_cfg.get("ik_backend", "ros2_dh")
    backend = create_kinematics_backend(backend_name, str(urdf_path), URDF_JOINTS, "right_tcp")
    print(f"kinematics: backend={backend_name}; urdf={urdf_path}")
    register_codecs()
    replay, store = open_replay_buffer(dataset)
    try:
        episodes = []
        for episode_id in range(replay.n_episodes):
            try:
                result = filter_episode(
                    episode_id, replay.get_episode(episode_id), backend,
                    q_start, t_start, limits, args)
            except Exception as exc:  # retain a report for malformed episodes
                result = {"episode": int(episode_id), "accepted": False,
                          "reasons": ["read_or_solver_error"], "error": str(exc)}
            episodes.append(result)
            verdict = "ACCEPT" if result["accepted"] else "REJECT"
            print(f"episode {episode_id}: {verdict}"
                  + ("" if result["accepted"] else f" ({', '.join(result['reasons'])})"),
                  flush=True)
    finally:
        if store is not None:
            store.close()

    accepted = [item["episode"] for item in episodes if item["accepted"]]
    rejected = [item["episode"] for item in episodes if not item["accepted"]]
    report = {
        "schema": "am_umi_am2pro_arm_replay_filter_v1",
        "safety": "offline only; no serial port or robot command is used",
        "dataset": str(dataset),
        "reference": str(reference_path),
        "reference_created_at": reference.get("created_at"),
        "tcp_frame": "right_tcp",
        "parameters": {key: value for key, value in vars(args).items()
                       if key not in {"dataset", "start_reference", "report", "accepted_out"}},
        "accepted_episodes": accepted,
        "rejected_episodes": rejected,
        "episodes": episodes,
        "limitations": [
            "筛选验证 IK、关节限位余量、理论关节速度/加速度和夹爪开度。",
            "它不等于实物碰撞检测；新长夹爪缺少完整碰撞模型，首条通过数据仍须做低速受限物理 dry-run。",
        ],
    }
    report_path = (Path(args.report).expanduser().resolve() if args.report else
                   dataset / "am2pro_arm_replay_filter.json" if dataset.is_dir() else
                   dataset.with_name(dataset.name + ".am2pro_arm_replay_filter.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    accepted_path = (Path(args.accepted_out).expanduser().resolve() if args.accepted_out else
                     report_path.with_name("accepted_episodes.txt"))
    accepted_path.write_text("\n".join(map(str, accepted)) + ("\n" if accepted else ""), encoding="utf-8")
    print("AM2PRO_ARM_REPLAY_FILTER_OK")
    print(f"accepted: {len(accepted)}; rejected: {len(rejected)}")
    print("report:", report_path)
    print("accepted episode ids:", accepted_path)


if __name__ == "__main__":
    main()
