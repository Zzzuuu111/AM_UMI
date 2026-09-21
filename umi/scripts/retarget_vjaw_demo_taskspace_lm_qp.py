#!/usr/bin/env python3
"""Offline LM/DLS + box-QP IK experiment for the current AM2Pro TCP model.

This is intentionally separate from the replay stack.  It uses the active
Placo FK as its sole geometry source, estimates a numerical Jacobian from that
FK, and solves each damped IK step as a bounded least-squares QP.  It never
opens a serial port or controls the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import lsq_linear
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


def numerical_jacobian(backend, q_rad: np.ndarray, epsilon_rad: float = 1e-5) -> np.ndarray:
    """6x6 TCP Jacobian in metres/rad and radians/rad from active Placo FK."""
    q_deg = np.rad2deg(q_rad)
    current = backend.forward_kinematics(q_deg)
    jacobian = np.empty((6, 6), dtype=float)
    for index in range(6):
        shifted = q_rad.copy()
        shifted[index] += epsilon_rad
        moved = backend.forward_kinematics(np.rad2deg(shifted))
        jacobian[:3, index] = (moved[:3, 3] - current[:3, 3]) / epsilon_rad
        jacobian[3:, index] = Rotation.from_matrix(
            moved[:3, :3] @ current[:3, :3].T).as_rotvec() / epsilon_rad
    return jacobian


def limit_barrier(margins_rad: np.ndarray, margin_rad: float, scale_rad: float) -> float:
    clearance = np.maximum(0.0, margins_rad - margin_rad)
    return float(np.mean((scale_rad / (clearance + scale_rad)) ** 2))


def local_axis(axis_name: str) -> np.ndarray:
    return {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}[axis_name]


def axis_error_rad(target: np.ndarray, actual: np.ndarray, axis_name: str) -> float:
    desired = target[:3, :3] @ local_axis(axis_name)
    observed = actual[:3, :3] @ local_axis(axis_name)
    return float(np.arccos(np.clip(np.dot(desired, observed), -1.0, 1.0)))


def orthogonal_basis(axis: np.ndarray) -> np.ndarray:
    """Return two orthonormal rows spanning the plane normal to ``axis``."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return np.vstack((first, second))


class LmBoxQpIk:
    """LM/DLS IK where each linearized step is a box-constrained QP."""

    def __init__(self, backend, lower_rad, upper_rad, args):
        self.backend = backend
        self.lower = lower_rad
        self.upper = upper_rad
        self.args = args
        self.margin = np.deg2rad(args.joint_margin_deg)
        self.safe_lower = lower_rad + self.margin
        self.safe_upper = upper_rad - self.margin
        self.center = 0.5 * (self.safe_lower + self.safe_upper)

    def solve(self, target: np.ndarray, seed_rad: np.ndarray, task_mode: str,
              axis_strength: float = 1.0):
        q = np.clip(np.asarray(seed_rad, dtype=float), self.safe_lower, self.safe_upper)
        max_step = np.deg2rad(self.args.qp_max_step_deg)
        last_smin = 0.0
        last_damping = 0.0
        for _ in range(self.args.ik_iterations):
            actual = self.backend.forward_kinematics(np.rad2deg(q))
            position_error = target[:3, 3] - actual[:3, 3]
            orientation_error = Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            task_axis_error = axis_error_rad(target, actual, self.args.free_tool_axis)
            axis_hard = task_mode == "axis" and axis_strength >= 1.0 - 1e-9
            orientation_ok = (
                np.linalg.norm(orientation_error) <= np.deg2rad(self.args.orientation_tolerance_deg)
                if task_mode == "full" else
                task_axis_error <= np.deg2rad(self.args.approach_axis_tolerance_deg)
                if axis_hard else True
            )
            if (np.linalg.norm(position_error) <= self.args.position_tolerance_mm / 1000.0
                    and orientation_ok):
                break
            jacobian = numerical_jacobian(self.backend, q)
            position_jacobian = self.args.position_task_weight * jacobian[:3]
            position_rhs = self.args.position_task_weight * position_error
            if task_mode == "axis":
                desired_axis = target[:3, :3] @ local_axis(self.args.free_tool_axis)
                basis = orthogonal_basis(desired_axis)
                axis_weight = self.args.orientation_task_weight * axis_strength
                weighted_jacobian = np.vstack((position_jacobian,
                                                axis_weight * basis @ jacobian[3:]))
                weighted_error = np.concatenate((position_rhs,
                                                  axis_weight * basis @ orientation_error))
            else:
                orientation_weight = (self.args.orientation_task_weight if task_mode == "full"
                                      else self.args.transport_orientation_weight)
                weighted_jacobian = np.vstack((position_jacobian, orientation_weight * jacobian[3:]))
                weighted_error = np.concatenate((position_rhs, orientation_weight * orientation_error))
            singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
            last_smin = float(singular_values[-1])
            ratio = max(0.0, self.args.singularity_threshold - last_smin) / self.args.singularity_threshold
            last_damping = self.args.base_damping * (1.0 + self.args.singularity_damping_gain * ratio * ratio)

            # QP: min ||W J dq - W e||² + λ||dq||²
            #         + wc ||dq - k(q_center-q)||²
            # subject to joint-limit and per-iteration step bounds.
            center_delta = self.args.centering_gain * (self.center - q)
            matrix = np.vstack((
                weighted_jacobian,
                np.sqrt(last_damping) * np.eye(6),
                np.sqrt(self.args.centering_weight) * np.eye(6),
            ))
            rhs = np.concatenate((weighted_error, np.zeros(6),
                                  np.sqrt(self.args.centering_weight) * center_delta))
            lower = np.maximum(self.safe_lower - q, -max_step)
            upper = np.minimum(self.safe_upper - q, max_step)
            result = lsq_linear(matrix, rhs, bounds=(lower, upper),
                                method="trf", tol=1e-8, lsmr_tol="auto")
            if not result.success or not np.all(np.isfinite(result.x)):
                break
            delta = result.x
            if float(np.max(np.abs(delta))) < 1e-8:
                break
            q = np.clip(q + delta, self.safe_lower, self.safe_upper)

        actual = self.backend.forward_kinematics(np.rad2deg(q))
        position_m, orientation_deg = pose_error(target, actual)
        margins = np.minimum(q - self.lower, self.upper - q)
        return (q, position_m, orientation_deg,
                np.rad2deg(axis_error_rad(target, actual, self.args.free_tool_axis)),
                margins, last_smin, last_damping)


def candidate_seeds(q_previous: np.ndarray, q_start: np.ndarray, lower, upper, rng, restarts: int):
    seeds = [q_previous, q_start]
    offsets_deg = (
        (0, 20, 20, 0, 0, 0), (0, -20, 20, 0, 0, 0),
        (25, 20, 20, -20, 0, 0), (-25, -20, 20, 20, 0, 0),
        (20, 30, -20, -25, 20, 0), (-20, -30, 20, 25, -20, 0),
    )
    for offset in offsets_deg:
        seeds.append(np.clip(q_previous + np.deg2rad(offset), lower, upper))
    for _ in range(restarts):
        seeds.append(rng.uniform(lower, upper))
    return seeds


def solve_candidates(solver, target, q_previous, q_start, rng, args, count, task_mode,
                     axis_strength: float = 1.0):
    candidates = []
    for seed in candidate_seeds(q_previous, q_start, solver.safe_lower, solver.safe_upper,
                                rng, args.random_restarts):
        # A trajectory beam may already contain a valid pose for this target.
        # Preserve that exact pose as a candidate before an LM step applies a
        # centering/null-space preference and accidentally walks away from it.
        seed_actual = solver.backend.forward_kinematics(np.rad2deg(seed))
        seed_position_m, seed_orientation_deg = pose_error(target, seed_actual)
        seed_axis_deg = np.rad2deg(axis_error_rad(target, seed_actual, args.free_tool_axis))
        seed_margins = np.minimum(seed - solver.lower, solver.upper - seed)
        task_orientation_error = (seed_orientation_deg if task_mode == "full" else
                                  seed_axis_deg * axis_strength if task_mode == "axis" else 0.0)
        seed_smooth = float(np.mean(
            ((seed - q_previous) / np.deg2rad(args.smooth_joint_scale_deg)) ** 2))
        candidates.append((
            (seed_position_m / (args.position_tolerance_mm / 1000.0)) ** 2 * 100.0
            + (task_orientation_error / args.orientation_tolerance_deg) ** 2 * 10.0
            + seed_smooth * args.smoothness_weight,
            seed.copy(), seed_position_m, seed_orientation_deg, seed_axis_deg,
            seed_margins, 0.0, 0.0,
        ))
        q, position_m, orientation_deg, axis_deg, margins, smin, damping = solver.solve(
            target, seed, task_mode, axis_strength)
        smooth = float(np.mean(((q - q_previous) / np.deg2rad(args.smooth_joint_scale_deg)) ** 2))
        score = (position_m / (args.position_tolerance_mm / 1000.0)) ** 2 * 100.0
        task_orientation_error = (orientation_deg if task_mode == "full" else
                                  axis_deg * axis_strength if task_mode == "axis" else 0.0)
        score += (task_orientation_error / args.orientation_tolerance_deg) ** 2 * 10.0
        score += smooth * args.smoothness_weight
        score += args.limit_barrier_weight * limit_barrier(
            margins, solver.margin, np.deg2rad(args.limit_barrier_scale_deg))
        candidates.append((score, q, position_m, orientation_deg, axis_deg, margins, smin, damping))
    unique = []
    for candidate in sorted(candidates, key=lambda item: item[0]):
        if any(float(np.max(np.abs(candidate[1] - kept[1]))) < np.deg2rad(0.5) for kept in unique):
            continue
        unique.append(candidate)
        if len(unique) >= count:
            break
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Placo-FK LM/DLS + box-QP retarget experiment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--source-start-frame", type=int, default=0,
                        help="仅诊断该源帧及之后的窗口；目标仍相对整段首帧计算")
    parser.add_argument("--source-end-frame", type=int, default=None,
                        help="仅诊断到该源帧（含）；默认整段末帧")
    parser.add_argument("--initial-q-deg", nargs=6, type=float, default=None,
                        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="窗口诊断的已验证关节起点；默认 reference 关节姿态")
    parser.add_argument("--position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--approach-axis-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--free-tool-axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--axis-constraint-range", action="append", nargs=2, type=int,
                        metavar=("START", "END"), default=[],
                        help="在该源帧闭区间启用5D接近轴约束；可重复指定")
    parser.add_argument("--axis-pre-ramp-frames", type=int, default=30,
                        help="每个5D接触段前逐帧收紧接近轴约束的源帧数；0 表示立即切换")
    parser.add_argument("--joint-margin-deg", type=float, default=5.0)
    parser.add_argument("--ik-iterations", type=int, default=100)
    parser.add_argument("--random-restarts", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--joint-branches", type=int, default=2)
    parser.add_argument("--qp-max-step-deg", type=float, default=2.0)
    parser.add_argument("--source-max-joint-step-deg", type=float, default=3.0)
    parser.add_argument("--position-task-weight", type=float, default=40.0)
    parser.add_argument("--orientation-task-weight", type=float, default=7.0)
    parser.add_argument("--transport-orientation-weight", type=float, default=0.5,
                        help="非接触运输段的弱完整朝向代价；0 表示纯位置")
    parser.add_argument("--base-damping", type=float, default=0.02)
    parser.add_argument("--singularity-threshold", type=float, default=0.08)
    parser.add_argument("--singularity-damping-gain", type=float, default=6.0)
    parser.add_argument("--centering-weight", type=float, default=0.02)
    parser.add_argument("--centering-gain", type=float, default=0.02)
    parser.add_argument("--limit-barrier-weight", type=float, default=0.5)
    parser.add_argument("--limit-barrier-scale-deg", type=float, default=20.0)
    parser.add_argument("--smooth-joint-scale-deg", type=float, default=15.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    args = parser.parse_args()
    if (args.frame_stride < 1 or args.source_start_frame < 0 or
            args.position_tolerance_mm <= 0 or
            args.orientation_tolerance_deg <= 0 or args.joint_margin_deg < 0 or
            args.approach_axis_tolerance_deg <= 0 or
            args.axis_pre_ramp_frames < 0 or
            args.ik_iterations < 1 or args.random_restarts < 0 or args.beam_width < 1 or
            args.joint_branches < 1 or args.joint_branches > args.beam_width or
            args.qp_max_step_deg <= 0 or args.source_max_joint_step_deg <= 0 or
            args.position_task_weight <= 0 or args.orientation_task_weight <= 0 or
            args.transport_orientation_weight < 0 or
            args.base_damping <= 0 or args.singularity_threshold <= 0 or
            args.singularity_damping_gain < 0 or args.centering_weight < 0 or
            args.centering_gain < 0 or args.limit_barrier_weight < 0 or
            args.limit_barrier_scale_deg <= 0 or args.smooth_joint_scale_deg <= 0 or
            args.smoothness_weight < 0):
        parser.error("invalid non-positive planner parameter")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")

    config = yaml.safe_load(Path(args.robot_config).expanduser().read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf = Path(robot["urdf_path"]).expanduser()
    if not urdf.is_absolute():
        urdf = (ROOT / urdf).resolve()
    mapping = mapping_from_config(robot, ROOT)
    limits_deg = load_joint_limits(urdf, model_safe_limits_from_config(robot, mapping))
    lower, upper = np.deg2rad(limits_deg[:, 0]), np.deg2rad(limits_deg[:, 1])
    backend = create_kinematics_backend("placo", str(urdf), URDF_JOINTS, "right_tcp")
    q_start_deg, pose_start, reference = load_reference(Path(args.start_reference).expanduser().resolve())
    q_start = (np.deg2rad(q_start_deg) if args.initial_q_deg is None
               else np.deg2rad(np.asarray(args.initial_q_deg, dtype=float)))
    q_start = np.clip(q_start, lower + np.deg2rad(args.joint_margin_deg),
                      upper - np.deg2rad(args.joint_margin_deg))
    T_robot_start = pose_to_mat(pose_start)
    replay, store = open_replay_buffer(Path(args.dataset).expanduser().resolve())
    try:
        poses = episode_pose_matrices(replay.get_episode(args.episode))
    finally:
        if store is not None:
            store.close()
    source_end = len(poses) - 1 if args.source_end_frame is None else args.source_end_frame
    if source_end < args.source_start_frame or source_end >= len(poses):
        parser.error(f"source frame window must lie in [0, {len(poses) - 1}]")
    source_indices = list(range(args.source_start_frame, source_end + 1, args.frame_stride))
    if source_indices[-1] != source_end:
        source_indices.append(source_end)
    for start, end in args.axis_constraint_range:
        if start < 0 or end < start or end >= len(poses):
            parser.error(f"axis-constraint range must lie in [0, {len(poses) - 1}]")

    def task_mode_for(source_frame: int) -> tuple[str, float]:
        if not args.axis_constraint_range:
            return "full", 1.0
        strength = 0.0
        for start, end in args.axis_constraint_range:
            if start <= source_frame <= end:
                strength = 1.0
                break
            if args.axis_pre_ramp_frames:
                ramp_start = max(0, start - args.axis_pre_ramp_frames)
                if ramp_start <= source_frame < start:
                    progress = (source_frame - ramp_start + 1) / (start - ramp_start + 1)
                    strength = max(strength, progress)
        return ("axis", strength) if strength > 0.0 else ("transport", 0.0)
    relative_start_inverse = np.linalg.inv(poses[0])
    targets = [T_robot_start @ (relative_start_inverse @ poses[index]) for index in source_indices]
    solver = LmBoxQpIk(backend, lower, upper, args)
    rng = np.random.default_rng(20260915)
    states = [{"q": q_start.copy(), "cost": 0.0, "node": None}]
    failure_frame = None
    planned_task_modes, planned_axis_strengths = [], []
    for local_index, target in enumerate(targets):
        task_mode, axis_strength = task_mode_for(source_indices[local_index])
        children = []
        for state in states:
            for _local_cost, q, pos_m, ori_deg, axis_deg, margins, smin, damping in solve_candidates(
                    solver, target, state["q"], q_start, rng, args,
                    max(args.joint_branches * 6, 12),
                    task_mode, axis_strength):
                step_deg = float(np.max(np.abs(np.rad2deg(q - state["q"]))))
                task_orientation_error = (ori_deg if task_mode == "full" else
                                          axis_deg if task_mode == "axis" and axis_strength >= 1.0 - 1e-9
                                          else 0.0)
                if (pos_m > args.position_tolerance_mm / 1000.0 or
                        task_orientation_error > (args.orientation_tolerance_deg if task_mode == "full"
                                                  else args.approach_axis_tolerance_deg) or
                        float(margins.min()) < solver.margin - 1e-9 or
                        (local_index and step_deg > args.source_max_joint_step_deg)):
                    continue
                cost = state["cost"] + (step_deg / args.smooth_joint_scale_deg) ** 2 * args.smoothness_weight
                cost += args.limit_barrier_weight * limit_barrier(
                    margins, solver.margin, np.deg2rad(args.limit_barrier_scale_deg))
                node = (state["node"], q.copy(), task_mode, axis_strength, pos_m, ori_deg, axis_deg,
                        margins.copy(), smin, damping)
                children.append({"q": q, "cost": cost, "node": node})
        if not children:
            failure_frame = source_indices[local_index]
            break
        retained = []
        for child in sorted(children, key=lambda item: item["cost"]):
            if any(float(np.max(np.abs(child["q"] - kept["q"]))) < np.deg2rad(0.5) for kept in retained):
                continue
            retained.append(child)
            if len(retained) >= args.beam_width:
                break
        states = retained
        planned_task_modes.append(task_mode)
        planned_axis_strengths.append(axis_strength)
        if local_index and local_index % 100 == 0:
            print(f"  LM_QP frame {source_indices[local_index]}/{len(poses)}: beam={len(states)}", flush=True)

    # Preserve the best fully safe prefix even on rejection.  It allows a
    # subsequent dense local diagnostic to start from a known configuration
    # rather than incorrectly reusing the episode's initial posture.
    if states:
        node = min(states, key=lambda item: item["cost"])["node"]
        path, task_modes, axis_strengths, positions, orientations, axis_errors, margins, smins, dampings = [], [], [], [], [], [], [], [], []
        while node is not None:
            parent, q, task_mode, axis_strength, pos_m, ori_deg, axis_deg, margin, smin, damping = node
            path.append(np.rad2deg(q).tolist())
            task_modes.append(task_mode)
            axis_strengths.append(axis_strength)
            positions.append(pos_m * 1000.0)
            orientations.append(ori_deg)
            axis_errors.append(axis_deg)
            margins.append(np.rad2deg(margin).tolist())
            smins.append(smin)
            dampings.append(damping)
            node = parent
        path, task_modes, axis_strengths, positions, orientations, axis_errors, margins, smins, dampings = (
            list(reversed(path)), list(reversed(task_modes)), list(reversed(axis_strengths)), list(reversed(positions)),
            list(reversed(orientations)), list(reversed(axis_errors)), list(reversed(margins)),
            list(reversed(smins)), list(reversed(dampings)))
        status = "candidate" if failure_frame is None else "rejected"
    else:
        path, task_modes, axis_strengths, positions, orientations, axis_errors, margins, smins, dampings = [], [], [], [], [], [], [], [], []
        status = "rejected"
    report = {
        "schema": "am_umi_vjaw_lm_qp_ik_probe_v1",
        "safety": "offline diagnostic only; no robot connection or command was used",
        "status": status,
        "failure_source_frame": failure_frame,
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "reference": str(Path(args.start_reference).expanduser().resolve()),
        "reference_created_at": reference.get("created_at"),
        "tcp_frame": "right_tcp",
        "fk_backend": "placo (finite-difference Jacobian)",
        "solver": "iterative LM/DLS with scipy bounded least-squares QP, multi-start and trajectory beam",
        "task_mode_per_solved_frame": task_modes,
        "axis_strength_per_solved_frame": axis_strengths,
        "axis_constraint_ranges": args.axis_constraint_range,
        "axis_pre_ramp_frames": args.axis_pre_ramp_frames,
        "free_tool_axis": args.free_tool_axis,
        "parameters": vars(args),
        "requested_source_frame_indices": source_indices,
        "solved_source_frame_indices": source_indices[:len(path)],
        "initial_q_model_deg": np.rad2deg(q_start).tolist(),
        "joint_path_model_deg": path,
        "position_error_mm": ({"median": float(np.median(positions)), "max": float(np.max(positions))}
                              if positions else None),
        "orientation_error_deg": ({"median": float(np.median(orientations)), "max": float(np.max(orientations))}
                                  if orientations else None),
        "approach_axis_error_deg": ({"median": float(np.median(axis_errors)), "max": float(np.max(axis_errors))}
                                    if axis_errors else None),
        "minimum_joint_margin_deg": (float(np.min(margins)) if margins else None),
        "minimum_singular_value": (float(np.min(smins)) if smins else None),
        "maximum_adaptive_damping": (float(np.max(dampings)) if dampings else None),
        "joint_margin_each_deg": margins,
        "limitations": [
            "Full 6D target only; no collision/object-contact model is included.",
            "A rejected full-pose result does not prove no analytic branch exists, but a candidate still requires retiming and empty-workspace physical validation.",
            "This probe does not replace the verified Placo replay planner.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_LM_QP_IK_" + status.upper())
    if failure_frame is not None:
        print(f"failure source frame: {failure_frame}/{len(poses) - 1}")
    else:
        print(f"position mm median/max: {report['position_error_mm']['median']:.3f}/{report['position_error_mm']['max']:.3f}")
        print(f"orientation deg median/max: {report['orientation_error_deg']['median']:.3f}/{report['orientation_error_deg']['max']:.3f}")
        print(f"minimum joint margin: {report['minimum_joint_margin_deg']:.3f} deg")
        print(f"minimum singular value: {report['minimum_singular_value']:.6f}")
    print("report:", out)


if __name__ == "__main__":
    main()
