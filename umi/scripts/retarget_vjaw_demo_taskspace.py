#!/usr/bin/env python3
"""Plan a safe robot-joint retargeting of a visual-only V-jaw demonstration.

This is deliberately an *offline planning* step.  It preserves the demonstrated
TCP position path while treating transport-phase orientation as soft, so a
physically reachable pick/place task is not rejected merely because the human
wrist used a different redundant posture.  It never opens a serial port.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS, compact_frame_ranges, episode_pose_matrices, load_joint_limits,
    load_reference, open_replay_buffer, pose_error,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    mapping_from_config, model_safe_limits_from_config,
)
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


JOINT_NAMES = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")


def blended_orientation(start_rotation: np.ndarray, demonstrated_rotation: np.ndarray,
                        alpha: float) -> np.ndarray:
    """Interpolate from the safe start attitude to the demonstrated attitude."""
    delta = (Rotation.from_matrix(start_rotation).inv()
             * Rotation.from_matrix(demonstrated_rotation))
    return (Rotation.from_matrix(start_rotation)
            * Rotation.from_rotvec(delta.as_rotvec() * alpha)).as_matrix()


def joint_limit_cost(margins: np.ndarray, args) -> float:
    """Return a smooth preference for clearance above the hard safety margin.

    The hard margin still decides safety.  This optional term matters earlier:
    it lets a trajectory-level beam keep an otherwise slightly less-smooth
    shoulder/elbow branch alive before its wrist is forced against a limit.
    It is deliberately opt-in so established plans retain their behaviour.
    """
    if args.joint_limit_barrier_weight <= 0:
        return 0.0
    clearance = np.maximum(0.0, np.asarray(margins) - args.joint_margin_deg)
    scale = args.joint_limit_barrier_scale_deg
    return float(args.joint_limit_barrier_weight
                 * np.mean((scale / (clearance + scale)) ** 2))


def local_axis_vector(axis_name: str) -> np.ndarray:
    return {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}[axis_name]


def target_with_free_tool_twist(demonstrated_target: np.ndarray,
                                axis_name: str, twist_deg: float) -> np.ndarray:
    """Keep one demonstrated tool axis while freeing rotation around it.

    Post-multiplication rotates in TCP-local coordinates, so the selected
    local axis remains identical in the robot/world frame.  This is a 5D
    task: position (3D) plus approach-axis direction (2D).
    """
    target = demonstrated_target.copy()
    target[:3, :3] = (demonstrated_target[:3, :3]
                       @ Rotation.from_rotvec(
                           local_axis_vector(axis_name) * np.deg2rad(twist_deg)
                       ).as_matrix())
    return target


def approach_axis_error_deg(demonstrated_target: np.ndarray, actual: np.ndarray,
                            axis_name: str) -> float:
    axis = local_axis_vector(axis_name)
    desired = demonstrated_target[:3, :3] @ axis
    observed = actual[:3, :3] @ axis
    cosine = float(np.clip(np.dot(desired, observed), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def load_right_tcp_to_source_tcp(path: Path, allow_provisional: bool) -> tuple[np.ndarray, dict]:
    """Load a fixed transform from robot ``right_tcp`` to a demonstrated TCP.

    The source zarr may describe a task point rigidly attached to the fixed
    jaw, while the robot kinematics backend still solves only the URDF
    ``right_tcp`` frame. Keeping this transform explicit avoids silently
    treating two different physical points as the same TCP.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "am_umi_tcp_frame_transform_v1":
        raise ValueError(f"未知 TCP frame transform 格式：{path}")
    if data.get("parent_frame") != "right_tcp":
        raise ValueError("TCP frame transform 的 parent_frame 必须是 right_tcp")
    if not isinstance(data.get("child_frame"), str) or not data["child_frame"]:
        raise ValueError("TCP frame transform 缺少 child_frame")
    status = data.get("status")
    if status not in (None, "accepted"):
        if status != "candidate" or not allow_provisional:
            raise ValueError(
                "TCP frame transform 尚未验收；仅低速诊断可加 "
                "--allow-provisional-tcp-frame-transform"
            )
    pose = np.asarray(data.get("pose_parent_child"), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("pose_parent_child 必须是 6 个有限数值")
    return pose_to_mat(pose), data


def wrap_twist_deg(twist_deg: float) -> float:
    return float((twist_deg + 180.0) % 360.0 - 180.0)


def solve_soft_target_candidates(backend, target, q_previous, q_start, limits, rng, args,
                                 max_candidates=1):
    """Return distinct IK branches for one soft task-space target.

    A single seed normally converges to the locally closest posture.  That is
    useful for greedy planning, but it hides valid shoulder/elbow alternatives
    from a lookahead planner.  Keep a small number of materially different
    local optima so the beam can choose continuity over the immediately
    cheapest posture.
    """
    seeds = [q_previous, q_start]
    # Explicit shoulder/elbow alternatives help avoid the branch that folds
    # J2/J3 merely to copy a human wrist rotation exactly.
    seed_offsets = [(0.0, 20.0, 20.0, 0.0, 0.0, 0.0),
                    (0.0, -20.0, 20.0, 0.0, 0.0, 0.0),
                    (0.0, 30.0, -20.0, 0.0, 0.0, 0.0)]
    if args.expanded_branch_seeds:
        # The default three shoulder/elbow perturbations are fast but can all
        # converge to the same wrist-limited local optimum.  These seeds are
        # not commands; they are only offline IK starting points representing
        # materially different base/shoulder/elbow/wrist branches.
        seed_offsets.extend([
            (25.0, 20.0, 20.0, -20.0, 0.0, 0.0),
            (-25.0, -20.0, 20.0, 20.0, 0.0, 0.0),
            (25.0, 30.0, -20.0, -25.0, 20.0, 0.0),
            (-25.0, -30.0, 20.0, 25.0, -20.0, 0.0),
            (0.0, 20.0, -25.0, 25.0, 25.0, -25.0),
            (0.0, -20.0, 25.0, -25.0, -25.0, 25.0),
        ])
    for offset in seed_offsets:
        seed = q_previous.copy()
        seed += np.asarray(offset)
        seeds.append(np.clip(seed, limits[:, 0], limits[:, 1]))
    for _ in range(args.random_restarts):
        seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))

    candidates = []
    for seed in seeds:
        q = np.asarray(seed, dtype=float).copy()
        for _ in range(args.ik_steps):
            q = backend.inverse_kinematics(
                q, target, position_weight=1.0,
                orientation_weight=args.orientation_weight)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        position_m, rotation_deg = pose_error(target, actual)
        margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        smooth = float(np.mean(((q - q_previous) / args.smooth_joint_scale_deg) ** 2))
        # Position is the task invariant; orientation is permitted to differ
        # during transport up to the declared soft tolerance.
        score = (position_m / args.position_tolerance_m) ** 2 * 50.0
        # ``position_only`` is deliberately for free-space transport.  It
        # must not secretly prefer a particular tool attitude through the
        # candidate score after inverse kinematics has been asked to ignore
        # attitude; that would recreate the same artificial wrist-limit
        # failure under a different name.
        if args.orientation_weight > 0:
            score += (rotation_deg / args.orientation_tolerance_deg) ** 2
        score += max(0.0, args.joint_margin_deg - float(margins.min())) ** 2 * 4.0
        score += joint_limit_cost(margins, args)
        score += smooth * args.smoothness_weight
        candidates.append((score, q, position_m, rotation_deg, margins))

    # Multiple random seeds often converge to numerically identical poses.
    # Deduplicate only those near-identical solutions; a different elbow or
    # shoulder branch remains available even if it has a slightly higher cost.
    unique = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if any(float(np.max(np.abs(candidate[1] - kept[1]))) < 0.5 for kept in unique):
            continue
        unique.append(candidate)
        if len(unique) >= max_candidates:
            break
    return unique


def solve_soft_target(backend, target, q_previous, q_start, limits, rng, args):
    """Find the locally best soft IK pose (legacy greedy-planning interface)."""
    return solve_soft_target_candidates(
        backend, target, q_previous, q_start, limits, rng, args, max_candidates=1)[0]


def solve_constrained_lookahead(backend, demonstrated_targets, q_start, limits, rng, args):
    """Keep multiple continuous IK hypotheses and choose after seeing the full path.

    This is a small beam search, not a per-frame greedy solve.  A low-alpha
    hypothesis is retained alongside the current high-alpha hypothesis, so the
    planner can begin a safe attitude retreat before a high-alpha branch runs
    into a joint limit.  Each beam edge must satisfy the position, margin and
    maximum joint-step gates.
    """
    width = args.lookahead_beam_width
    joint_branches = args.lookahead_joint_branches
    start_rotation = demonstrated_targets[0][:3, :3]
    position_only = args.orientation_mode == "position_only"
    initial_alphas = (np.array([0.0]) if position_only
                      else np.linspace(1.0, 0.0, width))
    states = [
        {"q": q_start.copy(), "alpha": float(alpha), "cost": (1.0 - alpha) ** 2,
         "node": None}
        for alpha in initial_alphas
    ]
    for frame_index, demonstrated_target in enumerate(demonstrated_targets):
        children = []
        for state in states:
            # Keep conservative alternatives alive for the whole horizon.  A
            # branch may retain or reduce attitude following, but does not
            # greedily recover it before future feasibility is known.
            alpha_options = ({0.0} if position_only else {
                float(np.clip(state["alpha"] - args.constrained_orientation_alpha_step, 0.0, 1.0)),
                float(state["alpha"]),
            })
            for alpha in sorted(alpha_options, reverse=True):
                target = demonstrated_target.copy()
                target[:3, :3] = blended_orientation(
                    start_rotation, demonstrated_target[:3, :3], alpha)
                for _local_cost, q, position_m, _rotation_deg, margins in solve_soft_target_candidates(
                        backend, target, state["q"], q_start, limits, rng, args,
                        max_candidates=joint_branches):
                    step = float(np.max(np.abs(q - state["q"])))
                    if (position_m > args.position_tolerance_m or
                            float(margins.min()) < args.joint_margin_deg or
                            (frame_index and step > args.constrained_max_joint_step_deg)):
                        continue
                    _, demonstrated_error_deg = pose_error(
                        demonstrated_target, backend.forward_kinematics(q))
                    cost = state["cost"] + (1.0 - alpha) ** 2
                    cost += (step / args.smooth_joint_scale_deg) ** 2 * args.smoothness_weight
                    cost += joint_limit_cost(margins, args)
                    node = (state["node"], q.copy(), alpha, position_m,
                            demonstrated_error_deg, margins.copy())
                    children.append({"q": q, "alpha": alpha, "cost": cost, "node": node})
        if not children:
            raise RuntimeError(f"lookahead IK 在第 {frame_index} 帧没有连续安全分支")

        # Preserve both attitude and joint-branch diversity.  The old planner
        # kept at most one state per alpha bin, so a distinct shoulder/elbow
        # solution was discarded immediately whenever it had the same alpha.
        # Split the finite beam between coarse alpha bins and local IK branches.
        alpha_bins = max(2, int(math.ceil(width / joint_branches)))
        bins = {}
        for child in sorted(children, key=lambda item: item["cost"]):
            bin_index = min(alpha_bins - 1, int(round(child["alpha"] * (alpha_bins - 1))))
            bucket = bins.setdefault(bin_index, [])
            if any(float(np.max(np.abs(child["q"] - kept["q"]))) < 0.5 for kept in bucket):
                continue
            if len(bucket) < joint_branches:
                bucket.append(child)
        states = [item for bucket in bins.values() for item in bucket]
        if len(states) < width:
            for child in sorted(children, key=lambda item: item["cost"]):
                if any(float(np.max(np.abs(child["q"] - kept["q"]))) < 0.5 for kept in states):
                    continue
                states.append(child)
                if len(states) >= width:
                    break
        states = sorted(states, key=lambda item: item["cost"])[:width]
        if frame_index and frame_index % 150 == 0:
            print(f"  lookahead frame {frame_index}/{len(demonstrated_targets)}: "
                  f"beam={len(states)} alpha=[{min(s['alpha'] for s in states):.2f},"
                  f"{max(s['alpha'] for s in states):.2f}]", flush=True)

    node = min(states, key=lambda item: item["cost"])["node"]
    path, alphas, position_errors, orientation_errors, margins = [], [], [], [], []
    while node is not None:
        parent, q, alpha, position_m, orientation_deg, margin = node
        path.append(q.tolist())
        alphas.append(float(alpha))
        position_errors.append(float(position_m))
        orientation_errors.append(float(orientation_deg))
        margins.append(margin.tolist())
        node = parent
    return (list(reversed(path)), list(reversed(alphas)), list(reversed(position_errors)),
            list(reversed(orientation_errors)), list(reversed(margins)))


def solve_approach_axis_lookahead(backend, demonstrated_targets, q_start, limits, rng, args):
    """Plan a continuous 5D path with free rotation about one TCP-local axis.

    The free twist is represented by a small, rate-limited set of target
    rotations.  Each sampled target is solved with the regular full-pose IK,
    while the beam chooses the twist and robot branch that remain smooth and
    away from limits.  This avoids pretending that a named wrist motor alone
    is redundant: all six joints may participate in the compensation.
    """
    width = args.lookahead_beam_width
    joint_branches = args.lookahead_joint_branches
    twist_step = args.free_tool_twist_step_deg
    states = [{"q": q_start.copy(), "twist_deg": 0.0, "cost": 0.0, "node": None}]
    for frame_index, demonstrated_target in enumerate(demonstrated_targets):
        children = []
        for state in states:
            twist_options = {
                wrap_twist_deg(state["twist_deg"] + delta)
                for delta in (-twist_step, 0.0, twist_step)
            }
            for twist_deg in sorted(twist_options):
                target = target_with_free_tool_twist(
                    demonstrated_target, args.free_tool_axis, twist_deg)
                for _local_cost, q, position_m, _rotation_deg, margins in solve_soft_target_candidates(
                        backend, target, state["q"], q_start, limits, rng, args,
                        max_candidates=joint_branches):
                    step = float(np.max(np.abs(q - state["q"])))
                    if (position_m > args.position_tolerance_m or
                            float(margins.min()) < args.joint_margin_deg or
                            (frame_index and step > args.constrained_max_joint_step_deg)):
                        continue
                    actual = backend.forward_kinematics(q)
                    _, demonstrated_error_deg = pose_error(demonstrated_target, actual)
                    axis_error_deg = approach_axis_error_deg(
                        demonstrated_target, actual, args.free_tool_axis)
                    if axis_error_deg > args.approach_axis_tolerance_deg:
                        continue
                    twist_change = abs(wrap_twist_deg(twist_deg - state["twist_deg"]))
                    cost = state["cost"]
                    cost += (step / args.smooth_joint_scale_deg) ** 2 * args.smoothness_weight
                    cost += joint_limit_cost(margins, args)
                    cost += (twist_change / max(twist_step, 1e-6)) ** 2 * args.free_tool_twist_weight
                    node = (state["node"], q.copy(), twist_deg, position_m,
                            demonstrated_error_deg, axis_error_deg, margins.copy())
                    children.append({"q": q, "twist_deg": twist_deg,
                                     "cost": cost, "node": node})
        if not children:
            raise RuntimeError(f"5D approach-axis IK 在第 {frame_index} 帧没有连续安全分支")

        # Preserve alternatives across the whole free-twist range, rather
        # than retaining only the locally cheapest wrist posture.
        twist_bins = max(3, int(math.ceil(width / joint_branches)))
        buckets = {}
        for child in sorted(children, key=lambda item: item["cost"]):
            fraction = (child["twist_deg"] + 180.0) / 360.0
            bin_index = min(twist_bins - 1, int(fraction * twist_bins))
            bucket = buckets.setdefault(bin_index, [])
            if any(float(np.max(np.abs(child["q"] - kept["q"]))) < 0.5 for kept in bucket):
                continue
            if len(bucket) < joint_branches:
                bucket.append(child)
        states = [item for bucket in buckets.values() for item in bucket]
        if len(states) < width:
            for child in sorted(children, key=lambda item: item["cost"]):
                if any(float(np.max(np.abs(child["q"] - kept["q"]))) < 0.5 for kept in states):
                    continue
                states.append(child)
                if len(states) >= width:
                    break
        states = sorted(states, key=lambda item: item["cost"])[:width]
        if frame_index and frame_index % 150 == 0:
            print(f"  5D lookahead frame {frame_index}/{len(demonstrated_targets)}: "
                  f"beam={len(states)} twist=[{min(s['twist_deg'] for s in states):.1f},"
                  f"{max(s['twist_deg'] for s in states):.1f}] deg", flush=True)

    node = min(states, key=lambda item: item["cost"])["node"]
    path, twists, position_errors, orientation_errors, axis_errors, margins = [], [], [], [], [], []
    while node is not None:
        parent, q, twist_deg, position_m, orientation_deg, axis_error_deg, margin = node
        path.append(q.tolist())
        twists.append(float(twist_deg))
        position_errors.append(float(position_m))
        orientation_errors.append(float(orientation_deg))
        axis_errors.append(float(axis_error_deg))
        margins.append(margin.tolist())
        node = parent
    return (list(reversed(path)), list(reversed(twists)), list(reversed(position_errors)),
            list(reversed(orientation_errors)), list(reversed(axis_errors)),
            list(reversed(margins)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线重定向 V 型夹爪手持任务路径；不会连接、通电或控制机械臂。")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--out", required=True, help="输出 JSON；拒绝覆盖")
    parser.add_argument("--right-tcp-to-source-tcp",
                        help=("可选的 right_tcp→示范 TCP 固定变换 JSON。示范若以固定爪"
                              "尖端等 contact_tcp 标注，IK 前会显式换回 right_tcp。"))
    parser.add_argument("--allow-provisional-tcp-frame-transform", action="store_true",
                        help="仅低速诊断：允许 status=candidate 的 TCP frame transform")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=25.0,
                        help="搬运段允许的 TCP 朝向偏差；不是实体安全范围")
    parser.add_argument("--orientation-weight", type=float, default=0.08,
                        help="IK 对示范朝向的权重；小于 1 表示优先位置")
    parser.add_argument("--orientation-mode", choices=("demonstrated", "start_fixed", "constrained", "constrained_lookahead", "approach_axis_lookahead", "position_only"),
                        default="demonstrated",
                        help=("demonstrated=跟随手持朝向；start_fixed=保持机器人起始工具朝向；"
                              "constrained=尽量跟随，接近不可达/限位时平滑回退；"
                              "approach_axis_lookahead=保持夹爪接近轴、释放绕该轴滚转的5D搜索；"
                              "position_only=仅约束TCP位置，供无接触搬运段诊断"))
    parser.add_argument("--lookahead-beam-width", type=int, default=6,
                        help="constrained_lookahead 保留的并行 IK 分支数")
    parser.add_argument("--lookahead-joint-branches", type=int, default=2,
                        help="每个朝向候选保留的不同肩/肘 IK 分支数")
    parser.add_argument("--constrained-orientation-alpha-step", type=float, default=0.125,
                        help="constrained 模式每次回退的朝向跟随比例")
    parser.add_argument("--constrained-orientation-recovery-per-frame", type=float, default=0.05,
                        help="constrained 模式每帧最多恢复的朝向跟随比例")
    parser.add_argument("--constrained-max-joint-step-deg", type=float, default=3.0,
                        help="constrained 模式相邻源帧任一关节允许的最大变化；超过即拒绝该 IK 分支")
    parser.add_argument("--joint-margin-deg", type=float, default=5.0)
    parser.add_argument("--joint-limit-barrier-weight", type=float, default=0.0,
                        help="可选：规划时偏好远离关节限位；0 保持旧行为")
    parser.add_argument("--joint-limit-barrier-scale-deg", type=float, default=20.0,
                        help="限位势垒开始明显起作用的额外余量")
    parser.add_argument("--expanded-branch-seeds", action="store_true",
                        help="离线 IK 额外尝试不同底座/肩肘/手腕初值分支")
    parser.add_argument("--free-tool-axis", choices=("x", "y", "z"), default="x",
                        help="5D 模式中保持的 TCP 局部接近轴")
    parser.add_argument("--free-tool-twist-step-deg", type=float, default=10.0,
                        help="5D 模式每源帧允许改变的自由工具滚转采样间隔")
    parser.add_argument("--free-tool-twist-weight", type=float, default=0.03,
                        help="5D 模式抑制自由工具滚转来回切换的代价")
    parser.add_argument("--approach-axis-tolerance-deg", type=float, default=3.0,
                        help="5D 模式中夹爪接近轴允许的方向误差")
    parser.add_argument("--ik-steps", type=int, default=140)
    parser.add_argument("--random-restarts", type=int, default=8)
    parser.add_argument("--smooth-joint-scale-deg", type=float, default=15.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    args = parser.parse_args()
    if (args.fps <= 0 or args.translation_scale <= 0 or args.position_tolerance_mm <= 0 or
            args.orientation_tolerance_deg <= 0 or args.orientation_weight < 0 or
            args.joint_margin_deg < 0 or args.joint_limit_barrier_weight < 0 or
            args.joint_limit_barrier_scale_deg <= 0 or args.ik_steps <= 0 or args.random_restarts < 0 or
            args.free_tool_twist_step_deg <= 0 or args.free_tool_twist_weight < 0 or
            args.approach_axis_tolerance_deg <= 0 or
            not 0 < args.constrained_orientation_alpha_step <= 1 or
            not 0 <= args.constrained_orientation_recovery_per_frame <= 1 or
            args.constrained_max_joint_step_deg <= 0 or args.lookahead_beam_width < 2 or
            args.lookahead_joint_branches < 1 or
            args.lookahead_joint_branches > args.lookahead_beam_width):
        parser.error("参数必须为正；joint-margin 与 random-restarts 允许为零")
    if args.orientation_mode == "position_only":
        # Make the mode self-contained: callers do not have to remember a
        # second --orientation-weight 0 flag, and the saved plan records the
        # effective setting.
        args.orientation_weight = 0.0
    elif args.orientation_mode == "approach_axis_lookahead":
        # Each sampled free-twist target is a full pose supplied to the
        # numerical backend.  A substantial attitude weight is therefore
        # necessary to make its retained local axis a real 5D constraint,
        # rather than merely a weak preference that collapses all twists into
        # the same position-only branch.
        args.orientation_weight = max(args.orientation_weight, 1.0)
    # Keep the command-line unit human-friendly while the solver uses metres.
    args.position_tolerance_m = args.position_tolerance_mm / 1000.0
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")

    config = yaml.safe_load(Path(args.robot_config).expanduser().read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    mapping = mapping_from_config(robot, ROOT)
    limits = load_joint_limits(urdf_path, model_safe_limits_from_config(robot, mapping))
    backend = create_kinematics_backend(robot.get("ik_backend", "placo"), str(urdf_path),
                                        URDF_JOINTS, "right_tcp")
    q_start, pose_start, reference = load_reference(Path(args.start_reference).expanduser().resolve())
    T_robot_start = pose_to_mat(pose_start)
    tx_right_source = np.eye(4, dtype=float)
    source_tcp_frame = "right_tcp"
    source_tcp_transform_status = "implicit_identity"
    source_tcp_transform_path = None
    if args.right_tcp_to_source_tcp:
        source_tcp_transform_path = Path(args.right_tcp_to_source_tcp).expanduser().resolve()
        tx_right_source, source_transform = load_right_tcp_to_source_tcp(
            source_tcp_transform_path, args.allow_provisional_tcp_frame_transform)
        source_tcp_frame = source_transform["child_frame"]
        source_tcp_transform_status = source_transform.get("status")
    replay, store = open_replay_buffer(Path(args.dataset).expanduser().resolve())
    try:
        if args.episode < 0 or args.episode >= replay.n_episodes:
            parser.error(f"episode 必须在 [0, {replay.n_episodes - 1}]")
        poses = episode_pose_matrices(replay.get_episode(args.episode))
    finally:
        if store is not None:
            store.close()

    relative_start_inverse = np.linalg.inv(poses[0])
    q_previous = q_start.copy()
    rng = np.random.default_rng(20260911)
    joint_path, position_errors, orientation_errors, margin_path = [], [], [], []
    orientation_following_alpha = []
    continuity_violation = []
    previous_alpha = 1.0
    print("VJAW_TASKSPACE_RETARGET_STARTED")
    print("safety: offline only; no serial port or robot command is used")
    print(f"frames: {len(poses)}; position tolerance: {args.position_tolerance_mm:.1f} mm; "
          f"soft orientation tolerance: {args.orientation_tolerance_deg:.1f} deg")
    demonstrated_targets = []
    T_robot_source_start = T_robot_start @ tx_right_source
    for pose in poses:
        relative = relative_start_inverse @ pose
        relative = relative.copy()
        relative[:3, 3] *= args.translation_scale
        # ``relative`` is expressed in the source TCP frame. First apply it
        # to the corresponding source point on the robot, then convert the
        # desired source pose back to the URDF right_tcp target used by IK.
        demonstrated_source_target = T_robot_source_start @ relative
        demonstrated_targets.append(demonstrated_source_target @ np.linalg.inv(tx_right_source))
    lookahead_result = None
    approach_axis_result = None
    if args.orientation_mode in ("constrained_lookahead", "position_only"):
        lookahead_result = solve_constrained_lookahead(
            backend, demonstrated_targets, q_start, limits, rng, args)
    elif args.orientation_mode == "approach_axis_lookahead":
        approach_axis_result = solve_approach_axis_lookahead(
            backend, demonstrated_targets, q_start, limits, rng, args)
    for index, pose in enumerate(poses):
        demonstrated_target = demonstrated_targets[index]
        target = demonstrated_target.copy()
        if args.orientation_mode == "start_fixed":
            # Keep the physical tool's known-safe start attitude during the
            # transport path.  This is deliberately explicit: it is a
            # task-space retargeting candidate, not claimed as a 6D clone.
            target[:3, :3] = T_robot_start[:3, :3]
            alpha = 0.0
            _, q_previous, position_m, _target_rotation_deg, margins = solve_soft_target(
                backend, target, q_previous, q_start, limits, rng, args)
        elif args.orientation_mode == "demonstrated":
            alpha = 1.0
            _, q_previous, position_m, _target_rotation_deg, margins = solve_soft_target(
                backend, target, q_previous, q_start, limits, rng, args)
        elif args.orientation_mode == "constrained":
            # First try the full hand orientation.  If it cannot meet both the
            # position and joint-margin requirements, lower only the attitude
            # target while preserving the demonstrated TCP position.  Recovery
            # is rate-limited, preventing a one-frame wrist snap back to alpha=1.
            maximum_alpha = min(1.0, previous_alpha + args.constrained_orientation_recovery_per_frame)
            alpha_values = np.arange(maximum_alpha, -1e-9,
                                     -args.constrained_orientation_alpha_step)
            if alpha_values[-1] > 1e-9:
                alpha_values = np.append(alpha_values, 0.0)
            selected = None
            best_safe = None
            for candidate_alpha in alpha_values:
                candidate_target = demonstrated_target.copy()
                candidate_target[:3, :3] = blended_orientation(
                    T_robot_start[:3, :3], demonstrated_target[:3, :3], float(candidate_alpha))
                candidate = solve_soft_target(backend, candidate_target, q_previous, q_start,
                                              limits, rng, args)
                _, candidate_q, candidate_position_m, _candidate_rotation_deg, candidate_margins = candidate
                safe = (candidate_position_m <= args.position_tolerance_m and
                        float(candidate_margins.min()) >= args.joint_margin_deg)
                step = float(np.max(np.abs(candidate_q - q_previous)))
                if safe and (best_safe is None or step < best_safe[0]):
                    best_safe = (step, float(candidate_alpha), candidate_q,
                                 candidate_position_m, candidate_margins)
                if safe and step <= args.constrained_max_joint_step_deg:
                    selected = float(candidate_alpha), candidate_q, candidate_position_m, candidate_margins
                    break
            if selected is None:
                # Do not silently accept a discontinuous branch.  Preserve the
                # closest otherwise-safe result for an evidence report, then
                # mark this frame as rejected below.
                if best_safe is not None:
                    _step, alpha, q_previous, position_m, margins = best_safe
                else:
                    candidate_target = demonstrated_target.copy()
                    candidate_target[:3, :3] = T_robot_start[:3, :3]
                    _, q_previous, position_m, _candidate_rotation_deg, margins = solve_soft_target(
                        backend, candidate_target, q_previous, q_start, limits, rng, args)
                    alpha = 0.0
            else:
                alpha, q_previous, position_m, margins = selected
            previous_alpha = alpha
        elif args.orientation_mode in ("constrained_lookahead", "position_only"):
            (lookahead_path, lookahead_alphas, lookahead_position_errors,
             _lookahead_orientation_errors, lookahead_margins) = lookahead_result
            q_previous = np.asarray(lookahead_path[index], dtype=float)
            alpha = lookahead_alphas[index]
            position_m = lookahead_position_errors[index]
            margins = np.asarray(lookahead_margins[index], dtype=float)
            previous_alpha = alpha
        else:
            (axis_path, axis_twists, axis_position_errors,
             _axis_orientation_errors, _axis_errors, axis_margins) = approach_axis_result
            q_previous = np.asarray(axis_path[index], dtype=float)
            # Keep the existing field numeric and make the actual free-twist
            # trajectory explicit in the report below.
            alpha = 1.0
            position_m = axis_position_errors[index]
            margins = np.asarray(axis_margins[index], dtype=float)
            previous_alpha = alpha
        # Record orientation deviation with respect to the original hand
        # demonstration even when a task-space mode intentionally changes it.
        _, rotation_deg = pose_error(demonstrated_target, backend.forward_kinematics(q_previous))
        step_from_previous = (float(np.max(np.abs(q_previous - np.asarray(joint_path[-1]))))
                              if joint_path else 0.0)
        joint_path.append(q_previous.tolist())
        position_errors.append(position_m)
        orientation_errors.append(rotation_deg)
        margin_path.append(margins.tolist())
        orientation_following_alpha.append(alpha)
        continuity_violation.append(step_from_previous)
        if index and index % 150 == 0:
            print(f"  frame {index}/{len(poses)}: position={position_m * 1000:.1f} mm; "
                  f"orientation={rotation_deg:.1f} deg; margin={min(margins):.1f} deg", flush=True)

    q_path = np.asarray(joint_path)
    position_errors = np.asarray(position_errors)
    orientation_errors = np.asarray(orientation_errors)
    margin_path = np.asarray(margin_path)
    approach_axis_errors = np.asarray([
        approach_axis_error_deg(demonstrated_targets[index], backend.forward_kinematics(q_path[index]),
                                  args.free_tool_axis)
        for index in range(len(q_path))
    ])
    bad_position = position_errors > args.position_tolerance_mm / 1000.0
    # In position-only transport, orientation deviation is diagnostic data,
    # not a rejection condition.  Contact/key frames need a separate
    # task-specific orientation check before any gripper command is enabled.
    if args.orientation_mode == "position_only":
        bad_orientation = np.zeros_like(orientation_errors, dtype=bool)
    elif args.orientation_mode == "approach_axis_lookahead":
        bad_orientation = approach_axis_errors > args.approach_axis_tolerance_deg
    else:
        bad_orientation = orientation_errors > args.orientation_tolerance_deg
    bad_margin = margin_path.min(axis=1) < args.joint_margin_deg
    continuity_violation = np.asarray(continuity_violation)
    bad_continuity = continuity_violation > args.constrained_max_joint_step_deg
    velocity = np.abs(np.diff(q_path, axis=0)) * args.fps
    report = {
        "schema": "am_umi_vjaw_taskspace_retarget_plan_v1",
        "safety": "offline plan only; no serial port or robot command was used",
        "status": "candidate" if not (bad_position.any() or bad_orientation.any() or bad_margin.any() or bad_continuity.any()) else "rejected",
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "episode": args.episode,
        "reference": str(Path(args.start_reference).expanduser().resolve()),
        "reference_created_at": reference.get("created_at"),
        "tcp_frame": "right_tcp",
        "demonstrated_tcp_frame": source_tcp_frame,
        "right_tcp_to_demonstrated_tcp": {
            "path": str(source_tcp_transform_path) if source_tcp_transform_path else None,
            "status": source_tcp_transform_status,
            "pose_parent_child": (None if source_tcp_transform_path is None
                                  else np.r_[tx_right_source[:3, 3],
                                             Rotation.from_matrix(tx_right_source[:3, :3]).as_rotvec()].tolist()),
        },
        "model_safe_limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "parameters": vars(args),
        "position_error_mm": {"median": float(np.median(position_errors) * 1000),
                              "max": float(np.max(position_errors) * 1000)},
        "orientation_deviation_deg": {"median": float(np.median(orientation_errors)),
                                        "max": float(np.max(orientation_errors))},
        "approach_axis": (args.free_tool_axis
                          if args.orientation_mode == "approach_axis_lookahead" else None),
        "approach_axis_deviation_deg": {"median": float(np.median(approach_axis_errors)),
                                          "max": float(np.max(approach_axis_errors))},
        "minimum_joint_margin_deg": float(margin_path.min()),
        "bad_position_frame_ranges": compact_frame_ranges(np.flatnonzero(bad_position)),
        "bad_orientation_frame_ranges": compact_frame_ranges(np.flatnonzero(bad_orientation)),
        "near_limit_frame_ranges": compact_frame_ranges(np.flatnonzero(bad_margin)),
        "branch_discontinuity_frame_ranges": compact_frame_ranges(np.flatnonzero(bad_continuity)),
        "max_joint_step_deg": float(continuity_violation.max(initial=0.0)),
        "orientation_following_alpha": orientation_following_alpha,
        "free_tool_twist_deg": (approach_axis_result[1]
                                if args.orientation_mode == "approach_axis_lookahead" else None),
        "reduced_orientation_frame_ranges": compact_frame_ranges(
            np.flatnonzero(np.asarray(orientation_following_alpha) < 1.0 - 1e-9)),
        "joint_path_model_deg": joint_path,
        "joint_margin_each_deg": margin_path.tolist(),
        "max_raw_joint_step_speed_deg_s": float(velocity.max(initial=0.0)),
        "limitations": [
            "This retargets pose labels only; it does not yet create a training zarr or execute a robot replay.",
            "Orientation tolerance is appropriate for transport only. Grasp/contact frames need a task-specific final-pose check.",
            "A candidate must be retimed and pass an empty-workspace physical dry-run before training use.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_TASKSPACE_RETARGET_" + report["status"].upper())
    print(f"position mm median/max: {report['position_error_mm']['median']:.3f}/"
          f"{report['position_error_mm']['max']:.3f}")
    print(f"orientation deg median/max: {report['orientation_deviation_deg']['median']:.3f}/"
          f"{report['orientation_deviation_deg']['max']:.3f}")
    print(f"minimum joint margin: {report['minimum_joint_margin_deg']:.3f} deg")
    print("report:", out)


if __name__ == "__main__":
    main()
