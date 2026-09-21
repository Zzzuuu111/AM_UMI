#!/usr/bin/env python3
"""Create an offline fixed-jaw-tip pivot probe for the real AM2Pro arm.

At every probe endpoint the *modelled* ``fixed_jaw_inner_front_tip`` remains
at its start position.  Only the right_tcp attitude changes.  A correct
right_tcp->tip transform therefore keeps the physical fixed-jaw inner-front
tip on a paper cross; a wrong transform makes it leave the cross.

The program is offline only.  It produces a normal candidate joint plan that
must be explicitly executed later by the supervised replay runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import URDF_JOINTS, load_joint_limits, pose_error
from scripts.retarget_vjaw_demo_taskspace import JOINT_NAMES, load_right_tcp_to_source_tcp
from umi.common.pose_util import pose_to_mat
from umi.real_world.am2pro_joint_mapping import mapping_from_config, model_safe_limits_from_config
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend


AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def load_reference(path: Path) -> tuple[np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("tcp_frame") != "right_tcp":
        raise ValueError("reference 必须是 right_tcp 基准")
    q = np.asarray(data.get("robot_state", {}).get("ActualQ", [])[:6], dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("reference 缺少有效 ActualQ")
    return q, data


def solve_target(backend, target: np.ndarray, seed: np.ndarray, limits: np.ndarray,
                 *, steps: int, restarts: int, rng: np.random.Generator):
    seeds = [seed]
    for _ in range(restarts):
        seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))
    best = None
    for initial in seeds:
        q = np.asarray(initial, dtype=float).copy()
        for _ in range(steps):
            q = backend.inverse_kinematics(q, target, position_weight=1.0,
                                           orientation_weight=0.35)
            q = np.clip(q, limits[:, 0], limits[:, 1])
        actual = backend.forward_kinematics(q)
        position_m, rotation_deg = pose_error(target, actual)
        margins = np.minimum(q - limits[:, 0], limits[:, 1] - q)
        score = position_m / 0.003 + rotation_deg / 2.0 + max(0.0, 10.0 - margins.min())
        candidate = (score, q, position_m, rotation_deg, margins)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线生成 fixed-tip 定点旋转探针；不会连接、通电或控制机器人。")
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--right-tcp-to-fixed-tip", required=True)
    parser.add_argument("--allow-provisional-tcp-frame-transform", action="store_true")
    parser.add_argument("--axes", default="y",
                        help="right_tcp 局部轴序列，例如 y 或 y,z；每轴依次 +tilt/-tilt")
    parser.add_argument("--tilt-deg", type=float, default=10.0)
    parser.add_argument("--ik-steps", type=int, default=100)
    parser.add_argument("--random-restarts", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 3.0 <= args.tilt_deg <= 15.0:
        parser.error("--tilt-deg 必须在 3 到 15 度；这是低速无接触尺量测试")
    axes = [item.strip() for item in args.axes.split(",") if item.strip()]
    if not axes or any(item not in AXES for item in axes):
        parser.error("--axes 只能是以逗号分隔的 x、y、z")
    if args.ik_steps <= 0 or args.random_restarts < 0:
        parser.error("ik-steps 必须为正，random-restarts 不能为负")
    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"拒绝覆盖已有计划: {out}")

    reference_path = Path(args.start_reference).expanduser().resolve()
    config_path = Path(args.robot_config).expanduser().resolve()
    transform_path = Path(args.right_tcp_to_fixed_tip).expanduser().resolve()
    q_start, reference = load_reference(reference_path)
    T_right_tip, transform = load_right_tcp_to_source_tcp(
        transform_path, args.allow_provisional_tcp_frame_transform)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    mapping = mapping_from_config(robot, ROOT)
    limits = load_joint_limits(urdf_path, model_safe_limits_from_config(robot, mapping))
    backend = create_kinematics_backend(robot.get("ik_backend", "placo"), str(urdf_path),
                                        URDF_JOINTS, "right_tcp")

    # Deliberately base the invariant point on FK(q_start), not a saved camera
    # pose.  This makes the probe test the physical/model rigid offset without
    # folding an unrelated global hand-eye/start-pose error into the result.
    T_right_start = backend.forward_kinematics(q_start)
    tip_position = (T_right_start @ T_right_tip)[:3, 3].copy()
    right_translation_to_tip = T_right_tip[:3, 3].copy()
    q_previous = q_start.copy()
    path = [q_start.tolist()]
    results = []
    rng = np.random.default_rng(20260916)

    print("VJAW_FIXED_TIP_PIVOT_PROBE_PLAN_STARTED")
    print("safety: offline planning only; no serial port or robot command is used")
    print("fixed point: modelled fixed_jaw_inner_front_tip at reference")
    print(f"axes: {axes}; tilt: +/-{args.tilt_deg:.1f} deg")
    for axis_name in axes:
        for direction in (1.0, -1.0):
            target = np.eye(4)
            target[:3, :3] = (T_right_start[:3, :3] @ Rotation.from_rotvec(
                AXES[axis_name] * np.deg2rad(direction * args.tilt_deg)).as_matrix())
            target[:3, 3] = tip_position - target[:3, :3] @ right_translation_to_tip
            _, q, pos_m, rot_deg, margins = solve_target(
                backend, target, q_previous, limits, steps=args.ik_steps,
                restarts=args.random_restarts, rng=rng)
            if pos_m > 0.003 or rot_deg > 2.0 or margins.min() < 10.0:
                worst = JOINT_NAMES[int(np.argmin(margins))]
                raise RuntimeError(
                    f"{axis_name}{'+' if direction > 0 else '-'} 探针不满足安全门："
                    f"IK {pos_m * 1000:.2f}mm/{rot_deg:.2f}deg，"
                    f"余量 {margins.min():.2f}deg ({worst})")
            actual = backend.forward_kinematics(q)
            model_tip = actual @ T_right_tip
            point_error_mm = float(np.linalg.norm(model_tip[:3, 3] - tip_position) * 1000)
            label = f"local_{axis_name}_{'plus' if direction > 0 else 'minus'}"
            path.extend([q.tolist(), q_start.tolist()])
            results.append({
                "label": label,
                "right_tcp_rotation_delta_deg": float(direction * args.tilt_deg),
                "target_joint_model_deg": q.tolist(),
                "ik_position_error_mm": float(pos_m * 1000),
                "ik_rotation_error_deg": float(rot_deg),
                "model_fixed_tip_point_error_mm": point_error_mm,
                "minimum_joint_margin_deg": float(margins.min()),
                "closest_joint": JOINT_NAMES[int(np.argmin(margins))],
            })
            q_previous = q_start.copy()
            print(f"  {label}: tip model error={point_error_mm:.3f}mm; "
                  f"margin={margins.min():.1f}deg; max dq={np.max(np.abs(q - q_start)):.1f}deg")

    report = {
        "schema": "am_umi_vjaw_taskspace_retarget_plan_v1",
        "status": "candidate",
        "kind": "fixed_tip_real_arm_pivot_probe",
        "safety": ("offline-generated low-speed empty-workspace diagnostic only; place a paper cross "
                   "under the real fixed-jaw inner-front tip and supervise execution"),
        "reference": str(reference_path),
        "reference_created_at": reference.get("created_at"),
        "tcp_frame": "right_tcp",
        "demonstrated_tcp_frame": transform["child_frame"],
        "right_tcp_to_demonstrated_tcp": {
            "path": str(transform_path), "status": transform.get("status"),
            "pose_parent_child": np.r_[T_right_tip[:3, 3],
                                       Rotation.from_matrix(T_right_tip[:3, :3]).as_rotvec()].tolist(),
        },
        "parameters": {"fps": 1.0, "axes": axes, "tilt_deg": args.tilt_deg,
                       "ik_steps": args.ik_steps, "random_restarts": args.random_restarts},
        "model_safe_limits_deg": {name: limits[i].tolist() for i, name in enumerate(JOINT_NAMES)},
        "probe_results": results,
        "position_error_mm": {"median": float(np.median([x["ik_position_error_mm"] for x in results])),
                              "max": float(np.max([x["ik_position_error_mm"] for x in results]))},
        "orientation_deviation_deg": {"median": 0.0, "max": 0.0},
        "minimum_joint_margin_deg": float(min(x["minimum_joint_margin_deg"] for x in results)),
        "max_joint_step_deg": float(max(np.max(np.abs(np.asarray(x["target_joint_model_deg"]) - q_start))
                                        for x in results)),
        "joint_path_model_deg": path,
        "limitations": [
            "A pass only establishes that the model plan is safe; use a physical paper cross/ruler to judge the real tip drift.",
            "This relative pivot test does not replace an independent global hand-eye or workspace calibration.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_FIXED_TIP_PIVOT_PROBE_PLAN_OK")
    print("plan:", out)


if __name__ == "__main__":
    main()
