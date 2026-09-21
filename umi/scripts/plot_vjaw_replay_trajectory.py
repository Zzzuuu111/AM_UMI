#!/usr/bin/env python3
"""Overlay a hand-held fixed-tip demo, planned FK, and adaptive replay FK.

The optional replay trace contains motor readbacks, not an external motion
capture system.  Its red curve therefore diagnoses joint tracking/model error;
it cannot by itself prove a real physical TCP calibration error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import URDF_JOINTS, episode_pose_matrices, open_replay_buffer
from scripts.retarget_vjaw_demo_taskspace import load_right_tcp_to_source_tcp
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend

# Keep these colours identical in the 3D and per-axis views.
# Blue=recorded demo, orange=offline IK plan, light green=robot readback FK.
DEMO_COLOR = "#1f77b4"
PLAN_COLOR = "#ff7f0e"
REPLAY_COLOR = "#98df8a"


def relative_points(poses: list[np.ndarray] | np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    values = list(poses)
    if not values:
        return np.empty((0, 3), dtype=float)
    inverse = np.linalg.inv(values[0] if reference is None else reference)
    return np.asarray([(inverse @ pose)[:3, 3] for pose in values], dtype=float)


def equal_axes(axis, arrays: list[np.ndarray]) -> None:
    points = np.vstack([item for item in arrays if len(item)])
    center = points.mean(axis=0)
    half_range = max(np.ptp(points, axis=0).max() / 2, 0.01) * 1.10
    axis.set_xlim(center[0] - half_range, center[0] + half_range)
    axis.set_ylim(center[1] - half_range, center[1] + half_range)
    axis.set_zlim(center[2] - half_range, center[2] + half_range)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制手持 demo、离线计划与 replay 读回 FK 的相对 3D 轨迹。")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--right-tcp-to-fixed-tip", required=True)
    parser.add_argument("--allow-provisional-tcp-frame-transform", action="store_true")
    parser.add_argument("--trace", default=None, help="可选 adaptive replay trace JSON")
    parser.add_argument("--anchor-demo-and-plan-at-actual-start", action="store_true",
                        help=("有 --trace 时，将 demo 与 IK 计划整体平移到 replay 的首个实际读回位置；"
                              "黑点仍表示计划/基准起点，用于显示 preposition 起始偏差"))
    parser.add_argument("--out", required=True, help="输出 PNG；拒绝覆盖")
    parser.add_argument("--report", default=None, help="可选 JSON 数值报告")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"拒绝覆盖已有图片: {out}")
    plan = json.loads(Path(args.plan).expanduser().read_text(encoding="utf-8"))
    q_plan = np.asarray(plan.get("joint_path_model_deg", []), dtype=float)
    if q_plan.ndim != 2 or q_plan.shape[1] != 6 or not len(q_plan):
        parser.error("计划缺少 joint_path_model_deg")
    transform, _ = load_right_tcp_to_source_tcp(
        Path(args.right_tcp_to_fixed_tip).expanduser().resolve(),
        args.allow_provisional_tcp_frame_transform)
    config = yaml.safe_load(Path(args.robot_config).expanduser().read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf = Path(robot["urdf_path"]).expanduser()
    if not urdf.is_absolute():
        urdf = (ROOT / urdf).resolve()
    backend = create_kinematics_backend(robot.get("ik_backend", "placo"), str(urdf),
                                        URDF_JOINTS, "right_tcp")

    replay, store = open_replay_buffer(Path(args.dataset).expanduser().resolve())
    try:
        episode = replay.get_episode(int(plan.get("episode", 0)))
        hand_poses = episode_pose_matrices(episode)
        source_gripper_widths_m = np.asarray(
            episode.get("robot0_gripper_width", []), dtype=float).reshape(-1)
    finally:
        if store is not None:
            store.close()
    plan_poses = [backend.forward_kinematics(q) @ transform for q in q_plan]
    hand_xyz = relative_points(hand_poses)
    plan_xyz = relative_points(plan_poses)
    if len(source_gripper_widths_m) < len(plan_xyz) or not np.isfinite(source_gripper_widths_m).all():
        source_gripper_widths_m = np.empty((0,), dtype=float)
    else:
        source_gripper_widths_m = source_gripper_widths_m[:len(plan_xyz)]

    actual_xyz = np.empty((0, 3), dtype=float)
    actual_progress = np.empty((0,), dtype=float)
    j7_progress = np.empty((0,), dtype=float)
    j7_demo_m = np.empty((0,), dtype=float)
    j7_command_m = np.empty((0,), dtype=float)
    j7_actual_m = np.empty((0,), dtype=float)
    actual_errors_mm = np.empty((0,), dtype=float)
    trace_status = None
    if args.trace:
        trace = json.loads(Path(args.trace).expanduser().read_text(encoding="utf-8"))
        if trace.get("schema") != "am_umi_vjaw_adaptive_replay_trace_v1":
            parser.error("不是受支持的 adaptive replay trace")
        samples = trace.get("samples", [])
        actual_poses = []
        errors = []
        progress = []
        for sample in samples:
            q = np.asarray(sample.get("actual_q_deg", []), dtype=float)
            anchor_q = np.asarray(sample.get("anchor_target_q_deg", []), dtype=float)
            phase = float(sample.get("phase", np.nan))
            if (q.shape != (6,) or anchor_q.shape != (6,) or not np.isfinite(q).all() or
                    not np.isfinite(anchor_q).all() or not np.isfinite(phase)):
                continue
            pose = backend.forward_kinematics(q) @ transform
            anchor_pose = backend.forward_kinematics(anchor_q) @ transform
            actual_poses.append(pose)
            # anchor_index belongs to the dense, retimed smooth path (often
            # thousands of samples), whereas plan_poses is the source-frame
            # plan (often hundreds).  Compare with the recorded dense anchor
            # target, never use the dense index to index the source plan.
            errors.append(float(np.linalg.norm(pose[:3, 3] - anchor_pose[:3, 3]) * 1000.0))
            progress.append(phase)
        # Use planned start as the common origin: an initial readback offset is
        # intentionally visible rather than silently aligned away.
        actual_xyz = relative_points(actual_poses, reference=plan_poses[0])
        actual_progress = np.asarray(progress, dtype=float)
        if len(actual_progress):
            actual_progress /= max(float(np.max(actual_progress)), 1.0)
        actual_errors_mm = np.asarray(errors, dtype=float)
        trace_status = trace.get("status")

        # New traces retain the source-frame position and J7 command/readback.
        # Keep them on exactly the same adaptive path-progress axis as XYZ.
        j7_phase, j7_source_frame, j7_command, j7_actual = [], [], [], []
        for sample in samples:
            try:
                phase = float(sample.get("phase", np.nan))
                source_frame = float(sample.get("source_frame_progress", np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite([phase, source_frame]).all():
                continue
            j7_phase.append(phase)
            j7_source_frame.append(source_frame)
            command = sample.get("gripper_command_width_m")
            actual_width = sample.get("actual_gripper_width_m")
            j7_command.append(float(command) if command is not None else np.nan)
            j7_actual.append(float(actual_width) if actual_width is not None else np.nan)
        if j7_phase and len(source_gripper_widths_m):
            j7_progress = np.asarray(j7_phase, dtype=float)
            j7_progress /= max(float(np.max(j7_progress)), 1.0)
            j7_demo_m = np.interp(np.clip(j7_source_frame, 0.0, len(source_gripper_widths_m) - 1.0),
                                  np.arange(len(source_gripper_widths_m), dtype=float),
                                  source_gripper_widths_m)
            j7_command_m = np.asarray(j7_command, dtype=float)
            j7_actual_m = np.asarray(j7_actual, dtype=float)

    if args.anchor_demo_and_plan_at_actual_start and not len(actual_xyz):
        parser.error("--anchor-demo-and-plan-at-actual-start 需要有效的 --trace")

    # Default is a pure relative-motion comparison: demo and plan start at
    # zero.  The alternate mode is for replay diagnostics: use the first
    # robot readback as the common start of demo+plan+actual, while retaining
    # a black point at the saved plan/reference origin.  The vector between
    # the black point and the common start is then the modelled preposition
    # offset rather than being silently hidden.
    start_shift = actual_xyz[0] if args.anchor_demo_and_plan_at_actual_start else np.zeros(3)
    hand_display_xyz = hand_xyz + start_shift
    plan_display_xyz = plan_xyz + start_shift
    actual_display_xyz = actual_xyz
    hand_label = ("hand-held fixed-tip (anchored at actual start)"
                  if args.anchor_demo_and_plan_at_actual_start else "hand-held fixed-tip (relative)")
    plan_label = ("offline plan FK (anchored at actual start)"
                  if args.anchor_demo_and_plan_at_actual_start else "offline plan FK (relative)")

    # The 3D view answers "where does it go?".  The right-side component
    # plots answer the more diagnostic question "which axis diverges, and
    # from which point in the motion?"  Every curve is expressed relative to
    # the planned start, so all three components visibly start at zero.
    fig = plt.figure(figsize=(16, 10.5), constrained_layout=True)
    layout = fig.add_gridspec(4, 2, width_ratios=(1.15, 1.0), wspace=.30, hspace=.38)
    axis = fig.add_subplot(layout[:, 0], projection="3d")
    axis.plot(hand_display_xyz[:, 0], hand_display_xyz[:, 1], hand_display_xyz[:, 2], color=DEMO_COLOR, lw=2,
              label=hand_label)
    axis.plot(plan_display_xyz[:, 0], plan_display_xyz[:, 1], plan_display_xyz[:, 2], color=PLAN_COLOR, lw=2,
              label=plan_label)
    if len(actual_display_xyz):
        axis.plot(actual_display_xyz[:, 0], actual_display_xyz[:, 1], actual_display_xyz[:, 2], color=REPLAY_COLOR, lw=2,
                  alpha=.9, label="replay readback FK")
    axis.scatter([0], [0], [0], color="black", s=36, label="planned/reference start")
    axis.set_xlabel("reference-tip X (m)")
    axis.set_ylabel("reference-tip Y (m)")
    axis.set_zlabel("reference-tip Z (m)")
    axis.set_title("3D trajectory overlay (fixed-tip, relative to planned/reference start)")
    equal_axes(axis, [hand_display_xyz, plan_display_xyz, actual_display_xyz])
    axis.legend(loc="best")

    hand_progress = np.linspace(0.0, 1.0, len(hand_xyz))
    plan_progress = np.linspace(0.0, 1.0, len(plan_xyz))
    colors = (DEMO_COLOR, PLAN_COLOR, REPLAY_COLOR)
    components = ("X", "Y", "Z")
    for component_index, component_name in enumerate(components):
        component_axis = fig.add_subplot(layout[component_index, 1])
        component_axis.plot(hand_progress, hand_display_xyz[:, component_index], color=colors[0], lw=2,
                            label="hand-held fixed-tip")
        component_axis.plot(plan_progress, plan_display_xyz[:, component_index], color=colors[1], lw=2,
                            label="offline plan FK")
        if len(actual_display_xyz):
            component_axis.plot(actual_progress, actual_display_xyz[:, component_index], color=colors[2],
                                lw=1.35, alpha=.9, label="replay readback FK")
        component_axis.scatter([0], [0], c="black", marker="o", s=25, zorder=5,
                               label="planned/reference start (0 m)")
        component_axis.axhline(0.0, color="0.55", lw=.7, ls="--")
        component_axis.grid(alpha=.28)
        component_axis.set_ylabel(f"{component_name} relative to reference (m)")
        component_axis.set_xlim(0.0, 1.0)
        if component_index == 0:
            component_axis.set_title("Per-axis displacement comparison")
            component_axis.legend(loc="best", fontsize=8)
    gripper_axis = fig.add_subplot(layout[3, 1])
    if len(j7_progress):
        gripper_axis.plot(j7_progress, j7_demo_m * 1000.0, color=DEMO_COLOR, lw=2,
                          label="hand-held width label", zorder=2)
        if np.isfinite(j7_command_m).any():
            gripper_axis.plot(j7_progress, j7_command_m * 1000.0, color=PLAN_COLOR, lw=2.0,
                              ls="--", label="replay J7 command", zorder=4)
        if np.isfinite(j7_actual_m).any():
            gripper_axis.plot(j7_progress, j7_actual_m * 1000.0, color=REPLAY_COLOR, lw=2.0,
                              ls=":", label="AM2Pro J7 readback", zorder=5)
        gripper_axis.legend(loc="best", fontsize=8)
    elif len(source_gripper_widths_m):
        gripper_axis.plot(np.linspace(0.0, 1.0, len(source_gripper_widths_m)),
                          source_gripper_widths_m * 1000.0, color=DEMO_COLOR, lw=2,
                          label="hand-held width label")
        gripper_axis.text(.5, .5, "This replay trace has no J7 command/readback fields.\n"
                                   "Blue is source label only.",
                          ha="center", va="center", transform=gripper_axis.transAxes, fontsize=8)
        gripper_axis.legend(loc="best", fontsize=8)
    else:
        gripper_axis.text(.5, .5, "No valid source J7 width labels.", ha="center", va="center",
                          transform=gripper_axis.transAxes, fontsize=9)
    gripper_axis.set_title("J7 opening comparison")
    gripper_axis.set_ylabel("opening width (mm)")
    gripper_axis.set_xlabel("normalized motion progress (0 = start, 1 = end)")
    gripper_axis.set_xlim(0.0, 1.0)
    gripper_axis.grid(alpha=.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    report = {
        "schema": "am_umi_vjaw_trajectory_comparison_v1",
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "plan": str(Path(args.plan).expanduser().resolve()),
        "trace": str(Path(args.trace).expanduser().resolve()) if args.trace else None,
        "trace_status": trace_status,
        "hand_frames": int(len(hand_xyz)), "plan_frames": int(len(plan_xyz)),
        "replay_readback_samples": int(len(actual_xyz)),
        "j7_trace_samples": int(len(j7_progress)),
        "demo_and_plan_anchored_at_actual_start": bool(args.anchor_demo_and_plan_at_actual_start),
        "reference_to_first_replay_readback_model_offset_mm": (
            {"xyz": (actual_xyz[0] * 1000.0).tolist(),
             "norm": float(np.linalg.norm(actual_xyz[0]) * 1000.0)}
            if len(actual_xyz) else None),
        "replay_model_position_error_mm": (
            {"median": float(np.median(actual_errors_mm)), "max": float(np.max(actual_errors_mm))}
            if len(actual_errors_mm) else None),
        "limitations": [
            "All curves use their start-tip frame; this compares relative motions, not a shared global tabletop map.",
            "Replay red curve is FK from motor readback, so external physical TCP/camera error is not directly measured.",
        ],
    }
    report_path = (Path(args.report).expanduser().resolve() if args.report else
                   out.with_suffix(".json"))
    if report_path.exists():
        parser.error(f"拒绝覆盖已有报告: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_TRAJECTORY_COMPARISON_OK")
    print("plot:", out)
    print("report:", report_path)
    if report["replay_model_position_error_mm"]:
        info = report["replay_model_position_error_mm"]
        print(f"replay readback-vs-plan median/max: {info['median']:.2f}/{info['max']:.2f} mm")


if __name__ == "__main__":
    main()
