#!/usr/bin/env python3
"""Offline multi-branch IK planner for a hand-held UMI replay episode.

Unlike a one-step, warm-started IK loop, this keeps a beam of candidate joint
solutions at every sampled TCP pose and chooses the lowest-cost *whole-path*
branch.  It is intended to discover alternatives such as a shoulder-forward /
elbow-compensated posture before any physical replay is attempted.

Safety: this program never imports a robot controller, opens a serial port, or
sends a command to hardware.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS, URDF_PATH, episode_pose_matrices, load_joint_limits,
    load_reference, open_replay_buffer, pose_error,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


JOINT_NAMES = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")


@dataclass
class Node:
    q: np.ndarray
    cost: float
    parent: int
    pos_error_m: float
    rot_error_deg: float
    min_margin_deg: float


def solve_one(backend, target, seed, limits, steps, orientation_weight):
    q = np.asarray(seed, dtype=float).copy()
    for _ in range(steps):
        q = backend.inverse_kinematics(
            q, target, position_weight=1.0, orientation_weight=orientation_weight)
        q = np.clip(q, limits[:, 0], limits[:, 1])
    actual = backend.forward_kinematics(q)
    pos_error_m, rot_error_deg = pose_error(target, actual)
    margin = float(np.min(np.minimum(q - limits[:, 0], limits[:, 1] - q)))
    return q, pos_error_m, rot_error_deg, margin


def distinct_append(candidates, item, threshold_deg=2.0):
    """Keep distinct joint-space branches; retain the cheaper duplicate."""
    for index, old in enumerate(candidates):
        if np.max(np.abs(old.q - item.q)) < threshold_deg:
            if item.cost < old.cost:
                candidates[index] = item
            return
    candidates.append(item)


def make_seeds(parent_q, q_start, limits, random_restarts, rng):
    seeds = [parent_q]
    # Two deliberate shoulder/elbow alternatives make the desired "reach
    # forward by unfolding" branch visible even if the last local solution was
    # pinned against J2's calibrated lower boundary.
    for shoulder_delta, elbow_delta in ((25.0, 20.0), (50.0, 40.0), (-20.0, 20.0)):
        candidate = parent_q.copy()
        candidate[1] += shoulder_delta
        candidate[2] += elbow_delta
        seeds.append(np.clip(candidate, limits[:, 0], limits[:, 1]))
    if not np.array_equal(parent_q, q_start):
        seeds.append(q_start.copy())
    for _ in range(random_restarts):
        seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))
    return seeds


def incremental_cost(q, parent_q, pos_error_m, rot_error_deg, margin_deg,
                     position_tolerance_m, rotation_tolerance_deg, safe_margin_deg):
    fit = (pos_error_m / position_tolerance_m) ** 2 + (rot_error_deg / rotation_tolerance_deg) ** 2
    # Strongly prefer a feasible pose, but retain imperfect branches in the
    # beam so the report can identify an actually unreachable segment.
    bad_penalty = 30.0 if (pos_error_m > position_tolerance_m or
                           rot_error_deg > rotation_tolerance_deg) else 0.0
    margin_penalty = max(0.0, safe_margin_deg - margin_deg) ** 2 * 2.0
    smooth_penalty = float(np.mean(((q - parent_q) / 15.0) ** 2))
    return fit + bad_penalty + margin_penalty + smooth_penalty


def compact_ranges(indices):
    if not indices:
        return []
    result = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value == previous + 1:
            previous = value
            continue
        result.append([start, previous])
        start = previous = value
    result.append([start, previous])
    return result


def main():
    parser = argparse.ArgumentParser(
        description="离线多分支 IK 轨迹规划；不会连接或控制机械臂。")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--out", required=True, help="新 JSON 路径；拒绝覆盖")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=15,
                        help="每隔多少源帧规划一个 waypoint")
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--random-restarts", type=int, default=3)
    parser.add_argument("--ik-steps", type=int, default=80)
    parser.add_argument("--orientation-weight", type=float, default=0.35)
    parser.add_argument("--position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--safe-margin-deg", type=float, default=1.5)
    parser.add_argument("--wrist-flex-max-deg", type=float, default=85.0,
                        help="新重夹爪的保守 J4 上限；避免靠近 +92.5° 物理端点")
    args = parser.parse_args()
    if (args.frame_stride <= 0 or args.beam_width <= 0 or args.random_restarts < 0 or
            args.ik_steps <= 0 or args.position_tolerance_mm <= 0 or
            args.rotation_tolerance_deg <= 0 or args.safe_margin_deg < 0):
        parser.error("stride、beam、IK 阶数和容差参数无效")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")
    q_start, pose_start, reference = load_reference(Path(args.start_reference).expanduser().resolve())
    t_start = pose_to_mat(pose_start)
    limits = load_joint_limits()
    limits[3, 1] = min(limits[3, 1], args.wrist_flex_max_deg)
    if np.any(q_start < limits[:, 0]) or np.any(q_start > limits[:, 1]):
        parser.error("保存的起始关节位姿不在当前安全限位内")
    replay, store = open_replay_buffer(Path(args.dataset).expanduser().resolve())
    try:
        if args.episode < 0 or args.episode >= replay.n_episodes:
            parser.error(f"episode 必须在 [0, {replay.n_episodes - 1}]")
        poses = episode_pose_matrices(replay.get_episode(args.episode))
    finally:
        if store is not None:
            store.close()
    source_frames = np.arange(0, len(poses), args.frame_stride, dtype=int)
    if source_frames[-1] != len(poses) - 1:
        source_frames = np.r_[source_frames, len(poses) - 1]
    pose_samples = poses[source_frames]
    rel0_inv = np.linalg.inv(pose_samples[0])
    backend = create_kinematics_backend("ros2_dh", str(URDF_PATH), URDF_JOINTS, "right_tcp")
    rng = np.random.default_rng(20260909)
    layers: list[list[Node]] = []
    print("MULTI_BRANCH_IK_PLANNING_STARTED")
    print("safety: offline only; no robot command will be sent")
    print(f"source frames: {len(poses)}; waypoints: {len(source_frames)}; beam width: {args.beam_width}")
    print(f"J2 safe range: [{limits[1, 0]:.2f}, {limits[1, 1]:.2f}] deg; J4 max: {limits[3, 1]:.2f} deg")
    for waypoint, pose in enumerate(pose_samples):
        target = t_start @ (rel0_inv @ pose)
        parents = layers[-1] if layers else [Node(q_start, 0.0, -1, 0.0, 0.0, math.inf)]
        candidates: list[Node] = []
        for parent_index, parent in enumerate(parents):
            for seed in make_seeds(parent.q, q_start, limits, args.random_restarts, rng):
                q, pos_error_m, rot_error_deg, margin = solve_one(
                    backend, target, seed, limits, args.ik_steps, args.orientation_weight)
                cost = parent.cost + incremental_cost(
                    q, parent.q, pos_error_m, rot_error_deg, margin,
                    args.position_tolerance_mm / 1000.0, args.rotation_tolerance_deg,
                    args.safe_margin_deg)
                distinct_append(candidates, Node(q, cost, parent_index, pos_error_m,
                                                  rot_error_deg, margin))
        candidates.sort(key=lambda item: item.cost)
        layers.append(candidates[:args.beam_width])
        best = layers[-1][0]
        if waypoint % 10 == 0 or waypoint == len(pose_samples) - 1:
            print(f"  waypoint {waypoint + 1}/{len(source_frames)} frame={source_frames[waypoint]} "
                  f"branches={len(layers[-1])} pos={best.pos_error_m * 1000:.2f}mm "
                  f"rot={best.rot_error_deg:.2f}deg margin={best.min_margin_deg:.2f}deg", flush=True)
    # Backtrack the best full path.  Its parent index is relative to the
    # preceding beam layer, which is stable because each layer was sorted once.
    path = []
    index = 0
    for layer_index in range(len(layers) - 1, -1, -1):
        node = layers[layer_index][index]
        path.append(node)
        index = node.parent
    path.reverse()
    pos_error_m = np.asarray([node.pos_error_m for node in path])
    rot_error_deg = np.asarray([node.rot_error_deg for node in path])
    margin_deg = np.asarray([node.min_margin_deg for node in path])
    bad_waypoints = np.flatnonzero((pos_error_m > args.position_tolerance_mm / 1000.0) |
                                   (rot_error_deg > args.rotation_tolerance_deg)).tolist()
    near_waypoints = np.flatnonzero(margin_deg < args.safe_margin_deg).tolist()
    report = {
        "schema": "am_umi_am2pro_multi_branch_ik_plan_v1",
        "safety": "offline only; no serial port or robot command was used",
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "episode": args.episode,
        "reference": str(Path(args.start_reference).expanduser().resolve()),
        "reference_created_at": reference.get("created_at"),
        "source_frame_indices": source_frames.tolist(),
        "limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "parameters": vars(args),
        "accepted": not bad_waypoints and not near_waypoints,
        "bad_waypoint_indices": bad_waypoints,
        "bad_source_frame_ranges": compact_ranges([int(source_frames[i]) for i in bad_waypoints]),
        "near_limit_waypoint_indices": near_waypoints,
        "near_limit_source_frame_ranges": compact_ranges([int(source_frames[i]) for i in near_waypoints]),
        "position_error_mm": {"median": float(np.median(pos_error_m) * 1000),
                              "max": float(np.max(pos_error_m) * 1000)},
        "rotation_error_deg": {"median": float(np.median(rot_error_deg)),
                               "max": float(np.max(rot_error_deg))},
        "min_joint_margin_deg": float(np.min(margin_deg)),
        "planned_joint_targets_deg": [node.q.tolist() for node in path],
        "limitations": [
            "The plan is kinematic only; it has no complete collision model for the new gripper.",
            "A planned path must still pass a low-speed physical dry-run before training use.",
            "This plan is sparse at frame-stride and must be interpolated/revalidated before replay.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("MULTI_BRANCH_IK_PLANNING_" + ("ACCEPT" if report["accepted"] else "REJECT"))
    print(f"bad waypoints: {len(bad_waypoints)}/{len(path)}; near-limit waypoints: {len(near_waypoints)}/{len(path)}")
    print(f"position error median/max: {report['position_error_mm']['median']:.3f}/"
          f"{report['position_error_mm']['max']:.3f} mm")
    print(f"minimum joint margin: {report['min_joint_margin_deg']:.3f} deg")
    print("report:", out)


if __name__ == "__main__":
    main()
