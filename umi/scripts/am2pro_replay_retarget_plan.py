#!/usr/bin/env python3
"""Safely execute an already-gated V-jaw task-space retargeting plan.

This is deliberately separate from the legacy 6D demonstration replay.  The
plan has already selected a robot-feasible redundant posture offline; this
runner only validates that plan, prepositions to its saved reference, then
executes sparse state-driven waypoints.  Arm-only is the default.  An explicit
opt-in can replay conservative binary J7 open/close events derived from
recorded gripper labels; those provisional labels are never treated as an
exact continuous physical aperture.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.real_world.am2pro_interpolation_controller import (  # noqa: E402
    AM2ProInterpolationController,
)
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    MOTOR_ARM_NAMES, mapping_from_config, model_safe_limits_from_config,
)
from scripts.filter_am2pro_replayability import open_replay_buffer  # noqa: E402


URDF_JOINTS = (
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
)
JOINT_LABELS = ("J1 pan", "J2 lift", "J3 elbow", "J4 flex", "J5 yaw", "J6 roll")


def load_reference(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("tcp_frame") != "right_tcp":
        raise ValueError("reference 必须是 right_tcp 基准")
    q = np.asarray(data.get("robot_state", {}).get("ActualQ", [])[:6], dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError("reference.json 缺少有效的 robot_state/ActualQ")
    return q


def joint_limits(urdf_path: Path, model_safe_limits: dict[str, tuple[float, float]]) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    nodes = {node.get("name"): node for node in root.findall("joint")}
    limits = np.asarray([
        (math.degrees(float(nodes[name].find("limit").get("lower"))),
         math.degrees(float(nodes[name].find("limit").get("upper"))))
        for name in URDF_JOINTS
    ], dtype=float)
    for index, name in enumerate(MOTOR_ARM_NAMES):
        if name in model_safe_limits:
            limits[index] = model_safe_limits[name]
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("URDF 与模型安全范围相交后为空")
    return limits


def preposition(robot, target_q: np.ndarray, duration: float, speed_cap: float,
                tolerance: float) -> np.ndarray:
    state = robot.get_state()
    if state is None:
        raise RuntimeError("控制器未返回关节状态")
    current = np.asarray(state["ActualQ"][:6], dtype=float)
    duration = max(duration, float(np.max(np.abs(target_q - current))) / speed_cap)
    print(f"REFERENCE PREPOSITION: {duration:.1f}s, speed cap {speed_cap:.1f} deg/s")
    print("  current:", np.round(current, 1))
    print("  target :", np.round(target_q, 1))
    robot.servoJ(target_q, duration=duration)
    latest = state
    finished_at = time.monotonic() + duration
    deadline = finished_at + 6.0
    while time.monotonic() < deadline:
        time.sleep(0.05)
        state = robot.get_state()
        if state is None:
            continue
        latest = state
        error = np.asarray(state["ActualQ"][:6], dtype=float) - target_q
        if time.monotonic() >= finished_at and np.max(np.abs(error)) <= tolerance:
            break
    actual = np.asarray(latest["ActualQ"][:6], dtype=float)
    error = actual - target_q
    print("  readback:", np.round(actual, 1))
    print("  error   :", np.round(error, 2), "deg (actual - target)")
    if np.max(np.abs(error)) > tolerance:
        raise RuntimeError(f"基准复位未收敛：最大误差 {np.max(np.abs(error)):.2f}° > {tolerance:.2f}°")
    print("REFERENCE PREPOSITION reached.")
    return actual


def wait_waypoint(robot, target_q: np.ndarray, speed: float, tolerance: float,
                  timeout: float) -> float:
    state = robot.get_state()
    if state is None:
        raise RuntimeError("执行时控制器状态丢失")
    current = np.asarray(state["ActualQ"][:6], dtype=float)
    duration = max(0.20, float(np.max(np.abs(target_q - current))) / speed)
    robot.servoJ(target_q, duration=duration)
    latest = state
    deadline = time.monotonic() + max(timeout, duration + 1.0)
    while time.monotonic() < deadline:
        time.sleep(0.02)
        state = robot.get_state()
        if state is None:
            continue
        latest = state
        actual = np.asarray(state["ActualQ"][:6], dtype=float)
        error = actual - target_q
        if np.max(np.abs(error)) <= tolerance:
            return float(np.max(np.abs(error)))
    actual = np.asarray(latest["ActualQ"][:6], dtype=float)
    error = actual - target_q
    worst = int(np.argmax(np.abs(error)))
    print("    target:", np.round(target_q, 1))
    print("    actual:", np.round(actual, 1))
    print("    error :", np.round(error, 2), "deg (actual - target)")
    raise RuntimeError(f"状态驱动回放超时：{JOINT_LABELS[worst]} 仍差 {error[worst]:+.2f}°")


def build_smooth_trajectory(path: np.ndarray, source_fps: float, sample_hz: float,
                            max_speed: float, max_acceleration: float):
    """Retiming + shape-preserving interpolation of an already safe joint plan.

    PCHIP is used instead of a cubic spline because it does not overshoot each
    joint's samples.  A global time stretch is then applied until sampled
    velocity and acceleration are within the requested conservative caps.
    """
    if len(path) < 2:
        return np.array([0.0]), path.copy(), 0.0, 0.0, np.array([0.0])
    delta = np.abs(np.diff(path, axis=0))
    source_dt = 1.0 / source_fps
    segment_dt = np.maximum(source_dt, np.max(delta, axis=1) / max_speed)
    knots = np.r_[0.0, np.cumsum(segment_dt)]

    def sample(knots_local):
        duration = float(knots_local[-1])
        times = np.arange(0.0, duration, 1.0 / sample_hz)
        times = np.r_[times, duration] if not np.isclose(times[-1], duration) else times
        trajectory = np.column_stack([
            PchipInterpolator(knots_local, path[:, joint])(times)
            for joint in range(6)
        ])
        velocity = np.diff(trajectory, axis=0) * sample_hz
        acceleration = np.diff(trajectory, n=2, axis=0) * sample_hz * sample_hz
        return times, trajectory, float(np.max(np.abs(velocity), initial=0.0)), \
            float(np.max(np.abs(acceleration), initial=0.0))

    # The nonuniform knots already satisfy the nominal velocity bound on each
    # raw segment.  PCHIP may slightly increase derivative peaks, so check the
    # final 50 Hz samples and stretch all time monotonically if needed.
    for _ in range(8):
        times, trajectory, actual_speed, actual_acceleration = sample(knots)
        stretch = max(1.0, actual_speed / max_speed,
                      math.sqrt(actual_acceleration / max_acceleration))
        if stretch <= 1.001:
            return times, trajectory, actual_speed, actual_acceleration, knots
        knots *= stretch * 1.01
    return times, trajectory, actual_speed, actual_acceleration, knots


def load_source_gripper_widths(plan: dict, count: int) -> np.ndarray:
    """Read the source demonstration's per-frame physical-width labels."""
    dataset = Path(plan.get("dataset", "")).expanduser()
    if not dataset.exists():
        raise ValueError(f"计划所指 dataset 不存在，无法读取夹爪标签: {dataset}")
    replay, store = open_replay_buffer(dataset)
    try:
        episode = replay.get_episode(int(plan.get("episode", 0)))
        widths = np.asarray(episode.get("robot0_gripper_width", []), dtype=float).reshape(-1)
    finally:
        if store is not None:
            store.close()
    if len(widths) < count or not np.all(np.isfinite(widths[:count])):
        raise ValueError("dataset 缺少与关节计划等长且有效的 robot0_gripper_width 标签")
    return widths[:count]


def load_binary_gripper_events(plan: dict, count: int, *, open_threshold: float,
                               closed_threshold: float, min_stable_frames: int,
                               open_command: float, closed_command: float):
    """Convert recorded widths into debounced open/closed source-frame events.

    The visual V-jaw label is only reliable enough for a state transition: its
    physical width is nonlinear and older recordings can be endpoint-clipped.
    A deadband plus debounce means transient tag noise cannot actuate J7.
    ``None`` before the first stable state deliberately means "do not move J7".
    """
    widths = load_source_gripper_widths(plan, count)

    events = []
    active_state = None
    candidate_state = None
    candidate_count = 0
    for frame, width in enumerate(widths[:count]):
        observed = "open" if width >= open_threshold else (
            "closed" if width <= closed_threshold else None)
        if observed is None:
            candidate_state = None
            candidate_count = 0
            continue
        if observed == candidate_state:
            candidate_count += 1
        else:
            candidate_state = observed
            candidate_count = 1
        if candidate_count >= min_stable_frames and observed != active_state:
            active_state = observed
            command = open_command if observed == "open" else closed_command
            events.append((frame, command, observed, float(width)))
    return events


def map_gripper_events_to_smooth(events, knots: np.ndarray, smooth_times: np.ndarray):
    """Map source-frame events to dense-path indices after arm retiming."""
    mapped = []
    source_indices = np.arange(len(knots), dtype=float)
    for source_frame, command, state, label_width in events:
        event_time = float(np.interp(source_frame, source_indices, knots))
        sample_index = int(np.searchsorted(smooth_times, event_time, side="left"))
        sample_index = min(sample_index, len(smooth_times) - 1)
        mapped.append((sample_index, command, state, source_frame, event_time, label_width))
    return mapped


def map_continuous_gripper_widths_to_smooth(widths: np.ndarray, knots: np.ndarray,
                                            smooth_times: np.ndarray) -> np.ndarray:
    """Linearly resample source widths onto the retimed arm-path clock.

    Every source width is preserved at its corresponding path knot.  Linear
    interpolation is intentionally used instead of a spline, so noisy labels
    cannot overshoot either physical opening endpoint.  The adaptive executor
    sends these targets according to *path* progress, not wall clock time.
    """
    if len(widths) != len(knots):
        raise ValueError("夹爪标签长度与关节路径长度不一致")
    return np.interp(smooth_times, np.asarray(knots, dtype=float), widths).astype(float)


def execute_smooth(robot, trajectory: np.ndarray, sample_hz: float,
                   max_tracking_error: float) -> None:
    """Stream retimed joint targets at a fixed rate; abort on excessive lag."""
    period = 1.0 / sample_hz
    next_time = time.monotonic()
    for index, target_q in enumerate(trajectory):
        robot.servoJStream(target_q)
        state = robot.get_state()
        if state is not None:
            actual = np.asarray(state["ActualQ"][:6], dtype=float)
            error = actual - target_q
            largest = float(np.max(np.abs(error)))
            if largest > max_tracking_error:
                worst = int(np.argmax(np.abs(error)))
                print("    target:", np.round(target_q, 1))
                print("    actual:", np.round(actual, 1))
                print("    error :", np.round(error, 2), "deg (actual - target)")
                raise RuntimeError(
                    f"连续回放跟踪保护中止：{JOINT_LABELS[worst]} 误差 "
                    f"{error[worst]:+.2f}° > {max_tracking_error:.2f}°")
            if index % max(1, int(sample_hz)) == 0:
                print(f"  t={index / sample_hz:.1f}s tracking max error={largest:.2f} deg", flush=True)
        next_time += period
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


def execute_queue(robot, trajectory: np.ndarray, sample_hz: float,
                  advance_tolerance: float, max_tracking_error: float,
                  progress_timeout: float, chunk_samples: int,
                  command_speed: float, min_command_duration: float,
                  reversal_settle_after: float,
                  reversal_settle_duration: float) -> None:
    """Run a monotonic FIFO of small joint-space chunks without a clock deadline.

    A chunk endpoint is popped only when the physical readback is close to it.
    This preserves path order (unlike choosing the globally nearest point,
    which can jump across a self-crossing path), but lets the arm take as long
    as it needs for each local segment.  We deliberately use one ordinary
    servoJ per small chunk rather than replacing a 20 ms servoJ on every tick:
    the latter restarts the server's finite-duration interpolation repeatedly.

    A Feetech position loop can need a deliberate, slow re-approach when a
    loaded joint reverses by only a small amount.  If a dense path reverses a
    joint inside a chunk and the ordinary short command has not settled, send
    exactly one long servoJ to that same endpoint.  This is intentionally a
    recovery for reversals only; normal chunks retain their short duration.
    """
    _ = sample_hz  # kept in the signature to document the dense source path.
    index = 0
    active_index = None
    active_target = None
    active_started_at = None
    active_reversal_settle_sent = False
    started_at = time.monotonic()
    last_progress = started_at
    last_print_second = -1
    last_wait_report_second = -1
    while index < len(trajectory):
        if active_index is None:
            active_index = min(index + chunk_samples - 1, len(trajectory) - 1)
            active_target = trajectory[active_index]
            state = robot.get_state()
            if state is None:
                raise RuntimeError("执行时控制器状态丢失")
            current = np.asarray(state["ActualQ"][:6], dtype=float)
            duration = max(min_command_duration,
                           float(np.max(np.abs(active_target - current))) / command_speed)
            robot.servoJ(active_target, duration=duration)
            active_started_at = time.monotonic()
            active_reversal_settle_sent = False
            last_wait_report_second = -1
            print(f"  FIFO_CHUNK {index}..{active_index}/{len(trajectory) - 1}; "
                  f"command duration={duration:.2f}s", flush=True)
        state = robot.get_state()
        if state is not None:
            actual = np.asarray(state["ActualQ"][:6], dtype=float)
            error = actual - active_target
            largest = float(np.max(np.abs(error)))
            if largest > max_tracking_error:
                worst = int(np.argmax(np.abs(error)))
                print("    target:", np.round(active_target, 1))
                print("    actual:", np.round(actual, 1))
                print("    error :", np.round(error, 2), "deg (actual - target)")
                raise RuntimeError(
                    f"FIFO 跟踪保护中止：{JOINT_LABELS[worst]} 误差 "
                    f"{error[worst]:+.2f}° > {max_tracking_error:.2f}°")
            if largest <= advance_tolerance:
                index = active_index + 1
                active_index = None
                active_target = None
                active_started_at = None
                active_reversal_settle_sent = False
                last_progress = time.monotonic()
            elapsed_second = int(time.monotonic() - started_at)
            if elapsed_second != last_print_second:
                last_print_second = elapsed_second
                print(f"  FIFO {index}/{len(trajectory)}; current target error="
                      f"{largest:.2f} deg", flush=True)
            if active_index is not None and largest > advance_tolerance:
                now = time.monotonic()
                if (not active_reversal_settle_sent and
                        now - active_started_at >= reversal_settle_after and
                        index > 0):
                    # A physical joint can still be catching up for several
                    # chunks after the planned curve has turned.  Search the
                    # preceding two seconds of dense (50 Hz) samples instead
                    # of only the immediately preceding sample.  At least
                    # 0.01 deg avoids treating encoder quantisation as a
                    # reversal.
                    chunk_delta = active_target - trajectory[index]
                    reversed_axes = np.zeros(6, dtype=bool)
                    history_start = max(0, index - int(round(2.0 * sample_hz)))
                    for axis in range(6):
                        if abs(chunk_delta[axis]) < 0.01:
                            continue
                        history_delta = np.diff(
                            trajectory[history_start:index + 1, axis])
                        meaningful = history_delta[np.abs(history_delta) >= 0.01]
                        reversed_axes[axis] = bool(
                            len(meaningful) and
                            np.any(meaningful * chunk_delta[axis] < 0.0))
                    if np.any(reversed_axes):
                        labels = ", ".join(
                            JOINT_LABELS[i] for i in np.flatnonzero(reversed_axes))
                        robot.servoJ(active_target, duration=reversal_settle_duration)
                        active_started_at = now
                        active_reversal_settle_sent = True
                        last_wait_report_second = -1
                        print(f"  FIFO_REVERSAL_SETTLE endpoint "
                              f"{active_index}/{len(trajectory) - 1}; "
                              f"axes={labels}; duration={reversal_settle_duration:.1f}s",
                              flush=True)
                wait_second = int(time.monotonic() - last_progress)
                if wait_second != last_wait_report_second:
                    last_wait_report_second = wait_second
                    worst = int(np.argmax(np.abs(error)))
                    print(f"    FIFO_WAITING chunk endpoint {active_index}/{len(trajectory) - 1}: "
                          f"{JOINT_LABELS[worst]} target={active_target[worst]:+.2f} "
                          f"actual={actual[worst]:+.2f} error={error[worst]:+.2f} deg",
                          flush=True)
        active_timeout = progress_timeout
        if active_reversal_settle_sent:
            # The physical J2 diagnostic needed about 11.3 seconds to enter
            # the 1.5-degree tolerance after a 10-second slow re-approach.
            active_timeout = max(active_timeout, reversal_settle_duration + 4.0)
        if active_started_at is not None and time.monotonic() - active_started_at > active_timeout:
            raise RuntimeError(
                f"FIFO 无进度超过 {active_timeout:.1f}s；当前块终点 "
                f"{active_index}/{len(trajectory) - 1} 未到位")
        time.sleep(0.02)


def execute_adaptive(robot, trajectory: np.ndarray, sample_hz: float,
                     lead_seconds: float, slow_error: float, hold_error: float,
                     min_speed_scale: float, progress_timeout: float,
                     max_tracking_error: float, final_tolerance: float,
                     gripper_events=(), gripper_width_trajectory=None,
                     gripper_command_hz: float = 10.0,
                     gripper_deadband_m: float = 0.001,
                     source_frame_progress_trajectory=None,
                     trace_samples: list[dict] | None = None,
                     trace_started_at: float | None = None) -> None:
    """Continuously stream a path while stretching its clock from readback error.

    Unlike queue, this never deliberately ends a 0.3-second command at each
    FIFO boundary.  The physical arm is sent a small fixed look-ahead target at
    the controller frequency.  The *path clock* advances at 1x when the arm is
    close to the current path phase, slows continuously as it falls behind, and
    pauses before a large lag becomes unsafe.  No future point is skipped.
    """
    if hold_error <= slow_error:
        raise ValueError("adaptive hold error 必须大于 slow error")
    if gripper_width_trajectory is not None:
        gripper_width_trajectory = np.asarray(gripper_width_trajectory, dtype=float)
        if (gripper_width_trajectory.shape != (len(trajectory),) or
                not np.all(np.isfinite(gripper_width_trajectory))):
            raise ValueError("连续 J7 轨迹形状无效")
    if source_frame_progress_trajectory is not None:
        source_frame_progress_trajectory = np.asarray(source_frame_progress_trajectory, dtype=float)
        if (source_frame_progress_trajectory.shape != (len(trajectory),) or
                not np.all(np.isfinite(source_frame_progress_trajectory))):
            raise ValueError("源帧进度轨迹形状无效")
    period = 1.0 / sample_hz
    lead_samples = max(1, int(round(lead_seconds * sample_hz)))
    phase = 0.0
    final_index = len(trajectory) - 1
    last_phase_progress = time.monotonic()
    last_print_second = -1
    next_time = time.monotonic()
    gripper_event_index = 0
    gripper_last_command_index = -1
    gripper_last_command_width = None
    gripper_last_report_second = -1
    gripper_stride_samples = max(1, int(round(sample_hz / gripper_command_hz)))
    print(f"ADAPTIVE_STREAM start; lead={lead_samples} samples ({lead_seconds:.2f}s), "
          f"slow/hold={slow_error:.2f}/{hold_error:.2f} deg", flush=True)
    while True:
        state = robot.get_state()
        if state is None:
            raise RuntimeError("执行时控制器状态丢失")
        actual = np.asarray(state["ActualQ"][:6], dtype=float)
        raw_gripper_width = float(state.get("gripper_position", np.nan))
        actual_gripper_width_m = raw_gripper_width if np.isfinite(raw_gripper_width) else None
        anchor_index = min(int(math.floor(phase)), final_index)
        command_index = min(anchor_index + lead_samples, final_index)
        source_frame_progress = (float(source_frame_progress_trajectory[anchor_index])
                                 if source_frame_progress_trajectory is not None else None)
        while (gripper_event_index < len(gripper_events) and
               anchor_index >= gripper_events[gripper_event_index][0]):
            _, width, label, source_frame, _, label_width = gripper_events[gripper_event_index]
            robot.schedule_gripper(width, time.time() + 0.05)
            gripper_last_command_width = float(width)
            print(f"  J7_EVENT source_frame={source_frame}; state={label}; "
                  f"label_width={label_width * 1000:.1f}mm -> command={width * 1000:.1f}mm",
                  flush=True)
            gripper_event_index += 1
        if gripper_width_trajectory is not None:
            width = float(gripper_width_trajectory[anchor_index])
            command_due = (gripper_last_command_index < 0 or
                           anchor_index - gripper_last_command_index >= gripper_stride_samples or
                           anchor_index >= final_index)
            command_changed = (gripper_last_command_width is None or
                               abs(width - gripper_last_command_width) >= gripper_deadband_m or
                               anchor_index >= final_index)
            if command_due and command_changed:
                robot.schedule_gripper(width, time.time() + 0.05)
                gripper_last_command_index = anchor_index
                gripper_last_command_width = width
            gripper_second = int(anchor_index / sample_hz)
            if gripper_second != gripper_last_report_second:
                gripper_last_report_second = gripper_second
                print(f"  J7_CONTINUOUS path={anchor_index}/{final_index}; "
                      f"label/command={width * 1000:.1f}mm", flush=True)
        anchor_target = trajectory[anchor_index]
        command_target = trajectory[command_index]
        anchor_error = float(np.max(np.abs(actual - anchor_target)))
        command_error = float(np.max(np.abs(actual - command_target)))
        if command_error > max_tracking_error:
            worst = int(np.argmax(np.abs(actual - command_target)))
            print("    target:", np.round(command_target, 1))
            print("    actual:", np.round(actual, 1))
            print("    error :", np.round(actual - command_target, 2), "deg (actual - target)")
            raise RuntimeError(
                f"自适应连续回放跟踪保护中止：{JOINT_LABELS[worst]} 误差 "
                f"{actual[worst] - command_target[worst]:+.2f}° > {max_tracking_error:.2f}°")
        robot.servoJStream(command_target)

        if anchor_index >= final_index:
            if command_error <= final_tolerance:
                if trace_samples is not None:
                    trace_samples.append({
                        "elapsed_s": float(time.monotonic() - (trace_started_at or time.monotonic())),
                        "phase": float(phase), "anchor_index": anchor_index,
                        "command_index": command_index, "speed_scale": 0.0,
                        "source_frame_progress": source_frame_progress,
                        "actual_q_deg": actual.tolist(),
                        "anchor_target_q_deg": anchor_target.tolist(),
                        "command_target_q_deg": command_target.tolist(),
                        "anchor_error_deg": anchor_error, "command_error_deg": command_error,
                        "actual_gripper_width_m": actual_gripper_width_m,
                        "gripper_command_width_m": (float(gripper_last_command_width)
                                                    if gripper_last_command_width is not None else None),
                    })
                print(f"  ADAPTIVE final reached; error={command_error:.2f} deg", flush=True)
                return
            speed_scale = 0.0
        elif anchor_error <= slow_error:
            speed_scale = 1.0
        elif anchor_error < hold_error:
            ratio = (hold_error - anchor_error) / (hold_error - slow_error)
            speed_scale = min_speed_scale + (1.0 - min_speed_scale) * ratio
        else:
            speed_scale = 0.0

        if trace_samples is not None:
            trace_samples.append({
                "elapsed_s": float(time.monotonic() - (trace_started_at or time.monotonic())),
                "phase": float(phase), "anchor_index": anchor_index,
                "command_index": command_index, "speed_scale": float(speed_scale),
                "source_frame_progress": source_frame_progress,
                "actual_q_deg": actual.tolist(),
                "anchor_target_q_deg": anchor_target.tolist(),
                "command_target_q_deg": command_target.tolist(),
                "anchor_error_deg": anchor_error, "command_error_deg": command_error,
                "actual_gripper_width_m": actual_gripper_width_m,
                "gripper_command_width_m": (float(gripper_last_command_width)
                                            if gripper_last_command_width is not None else None),
            })

        if speed_scale > 0.0 and anchor_index < final_index:
            old_phase = phase
            phase = min(float(final_index), phase + speed_scale)
            if phase > old_phase:
                last_phase_progress = time.monotonic()
        elif time.monotonic() - last_phase_progress > progress_timeout:
            worst = int(np.argmax(np.abs(actual - anchor_target)))
            raise RuntimeError(
                f"自适应路径相位无进度超过 {progress_timeout:.1f}s；"
                f"当前 {anchor_index}/{final_index}，{JOINT_LABELS[worst]} "
                f"误差 {actual[worst] - anchor_target[worst]:+.2f}°")

        elapsed_second = int(time.monotonic() - next_time + period)
        # Use path phase rather than wall time in the status line: it makes
        # adaptive slowdown visible without producing 50 Hz log noise.
        phase_second = int(anchor_index / sample_hz)
        if phase_second != last_print_second:
            last_print_second = phase_second
            print(f"  ADAPTIVE path={anchor_index}/{final_index}; scale={speed_scale:.2f}; "
                  f"anchor/command error={anchor_error:.2f}/{command_error:.2f} deg", flush=True)
        next_time += period
        sleep_time = next_time - time.monotonic()
        if sleep_time > 0:
            time.sleep(sleep_time)


def generate_replay_plots(*, plan: dict, plan_path: Path, robot_config: Path,
                          trace_path: Path) -> None:
    """Generate readback comparison plots after the controller has stopped.

    Plotting is deliberately run after the robot context closes and the optional
    return-to-reference move completes.  A plotting failure is diagnostic only:
    it must not keep torque enabled or change the replay outcome.
    """
    transform_info = plan.get("right_tcp_to_demonstrated_tcp", {})
    transform_path = Path(transform_info.get("path", "")).expanduser()
    dataset_path = Path(plan.get("dataset", "")).expanduser()
    if not transform_path.is_file() or not dataset_path.exists():
        print("REPLAY_PLOT_SKIPPED: plan is missing its dataset or TCP transform", flush=True)
        return

    stem = trace_path.with_suffix("")
    trajectory_plot = stem.with_name(stem.name + ".trajectory.png")
    trajectory_report = stem.with_name(stem.name + ".trajectory.json")
    commands = [[
        sys.executable, str(ROOT / "scripts" / "plot_vjaw_replay_trajectory.py"),
        "--dataset", str(dataset_path),
        "--plan", str(plan_path),
        "--robot-config", str(robot_config),
        "--right-tcp-to-fixed-tip", str(transform_path),
        "--trace", str(trace_path),
        "--out", str(trajectory_plot),
        "--report", str(trajectory_report),
    ]]
    if transform_info.get("status") != "accepted":
        commands[0].append("--allow-provisional-tcp-frame-transform")

    for command in commands:
        output_path = command[command.index("--out") + 1]
        print("REPLAY_PLOT_GENERATING:", output_path, flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"REPLAY_PLOT_FAILED: exit={result.returncode}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="执行已验收的 V 型夹爪任务空间重定向计划；会实际驱动机械臂")
    parser.add_argument("--joint-plan", required=True)
    parser.add_argument("--start-reference", required=True)
    parser.add_argument("--robot-config", default="example/eval_robots_config_vjaw_ros2.yaml")
    parser.add_argument("--waypoint-stride", type=int, default=30,
                        help="每隔多少源帧执行一个目标；默认 30 即约每秒一个")
    parser.add_argument("--max-source-frames", type=int, default=60,
                        help="默认只执行前 60 个源帧；配合 --full-plan 才允许整段")
    parser.add_argument("--full-plan", action="store_true", help="明确允许执行整条候选计划")
    parser.add_argument("--execution-mode", choices=("settled", "smooth", "queue", "adaptive"), default="settled",
                        help=("settled=停稳走点；smooth=按时钟连续；queue=实物到点才出队的 FIFO；"
                              "adaptive=根据实物误差连续拉伸路径时钟"))
    parser.add_argument("--source-fps", type=float, default=None,
                        help="候选计划的源帧率；默认读取计划中的 fps")
    parser.add_argument("--smooth-sample-hz", type=float, default=50.0)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=8.0,
                        help="smooth 模式的保守关节速度上限")
    parser.add_argument("--max-joint-accel-deg-s2", type=float, default=50.0,
                        help="smooth 模式的保守关节加速度上限")
    parser.add_argument("--max-tracking-error-deg", type=float, default=5.0,
                        help="smooth 模式任一关节相对流式目标的最大允许误差")
    parser.add_argument("--queue-advance-tolerance-deg", type=float, default=1.5,
                        help="queue 模式中实物距当前点不超过此值才允许出队")
    parser.add_argument("--queue-progress-timeout-s", type=float, default=12.0,
                        help="queue 模式连续没有任何点出队的最长等待时间")
    parser.add_argument("--queue-chunk-samples", type=int, default=10,
                        help="queue 模式每个 FIFO 小块包含的密集源样本数")
    parser.add_argument("--queue-min-command-duration-s", type=float, default=0.30,
                        help="queue 模式每个小块的最短 servoJ 时长")
    parser.add_argument("--queue-reversal-settle-after-s", type=float, default=1.0,
                        help="小块未收敛且检测到关节反向后，多久补发一次慢速 settle（默认 1 秒）")
    parser.add_argument("--queue-reversal-settle-duration-s", type=float, default=10.0,
                        help="反向微调补发 servoJ 的时长（默认 10 秒；仅反向块触发）")
    parser.add_argument("--adaptive-lead-s", type=float, default=0.20,
                        help="adaptive 模式连续发送的前视时长")
    parser.add_argument("--adaptive-slow-error-deg", type=float, default=1.0,
                        help="adaptive 模式超过该锚点误差即开始连续减速")
    parser.add_argument("--adaptive-hold-error-deg", type=float, default=2.0,
                        help="adaptive 模式超过该锚点误差即暂停路径相位")
    parser.add_argument("--adaptive-min-speed-scale", type=float, default=0.20,
                        help="adaptive 减速区间的最小路径时间倍率")
    parser.add_argument("--adaptive-progress-timeout-s", type=float, default=12.0,
                        help="adaptive 路径相位暂停的最长允许时间")
    parser.add_argument("--enable-gripper", action="store_true",
                        help=("显式启用 J7 标签回放；仅 adaptive 模式支持。"
                              "时序来自计划 dataset，而非手工指定时间"))
    parser.add_argument("--gripper-mode", choices=("binary", "continuous"), default="binary",
                        help="binary=稳定开/合语义；continuous=逐帧宽度标签连续重放")
    parser.add_argument("--gripper-open-threshold-m", type=float, default=0.045,
                        help="标签宽度不小于此值并稳定后，判为张开")
    parser.add_argument("--gripper-closed-threshold-m", type=float, default=0.015,
                        help="标签宽度不大于此值并稳定后，判为闭合")
    parser.add_argument("--gripper-min-stable-frames", type=int, default=5,
                        help="同一开合状态连续多少源帧后才触发 J7")
    parser.add_argument("--gripper-open-command-m", type=float, default=0.067,
                        help="标签判为张开时 J7 的物理开口命令")
    parser.add_argument("--gripper-closed-command-m", type=float, default=0.009,
                        help="标签判为闭合时 J7 的物理开口命令")
    parser.add_argument("--gripper-final-width-m", type=float, default=None,
                        help="可选：回放结束/中止后，在机械臂回程前发出的 J7 安全开口")
    parser.add_argument("--gripper-continuous-command-hz", type=float, default=10.0,
                        help="continuous 模式 J7 的最大命令频率；默认 10 Hz")
    parser.add_argument("--gripper-continuous-deadband-mm", type=float, default=1.0,
                        help="continuous 模式低于此开口变化量不重发；默认 1 mm")
    parser.add_argument("--temporary-p-coefficients", default=None,
                        help=("仅本次 replay 临时覆盖电机 P 的 JSON，例如 "
                              "'{\"shoulder_lift\": 48}'；控制器退出时恢复 EEPROM 原值"))
    parser.add_argument("--preview-only", action="store_true",
                        help="只构建并报告连续轨迹，不连接、不通电、不控制机械臂")
    parser.add_argument("--trace-out", default=None,
                        help=("仅 adaptive：保存每个执行周期的目标/实际 J1–J6 读回 JSON；"
                              "拒绝覆盖，供后续 3D 轨迹对比"))
    parser.add_argument("--plot-after-replay", action="store_true",
                        help=("保存 trace 并且机械臂返回基准后，自动生成手持/计划/实际 FK 轨迹图；"
                              "启用 J7 时同一张总览图会包含开度标签、命令和读回对比。"
                              "图片和数值报告写在 --trace-out 旁边，拒绝覆盖已有文件"))
    parser.add_argument("--reference-duration", type=float, default=25.0)
    parser.add_argument("--reference-speed-deg-s", type=float, default=3.0)
    parser.add_argument("--reference-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--pause-after-reference", action="store_true",
                        help=("基准到位后暂停，按 Enter 才开始路径；用于在静止状态下放置尺子/纸上标记。"
                              "Ctrl+C 会安全走现有返回流程"))
    parser.add_argument("--waypoint-speed-deg-s", type=float, default=3.0)
    parser.add_argument("--settle-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--waypoint-timeout-s", type=float, default=30.0)
    parser.add_argument("--waypoint-hold-s", type=float, default=0.0,
                        help="settled 模式每个到位点额外保持的秒数；默认 0")
    parser.add_argument("--return", dest="do_return", action="store_true", default=True)
    parser.add_argument("--no-return", dest="do_return", action="store_false")
    args = parser.parse_args()
    if (args.waypoint_stride <= 0 or args.max_source_frames <= 0 or
            args.reference_duration <= 0 or args.reference_speed_deg_s <= 0 or
            args.reference_tolerance_deg <= 0 or args.waypoint_speed_deg_s <= 0 or
            args.settle_tolerance_deg <= 0 or args.waypoint_timeout_s <= 0 or args.waypoint_hold_s < 0 or
            args.smooth_sample_hz <= 0 or args.max_joint_speed_deg_s <= 0 or
            args.max_joint_accel_deg_s2 <= 0 or args.max_tracking_error_deg <= 0 or
            args.queue_advance_tolerance_deg <= 0 or args.queue_progress_timeout_s <= 0 or
            args.queue_chunk_samples <= 0 or args.queue_min_command_duration_s <= 0 or
            args.queue_reversal_settle_after_s <= 0 or
            args.queue_reversal_settle_duration_s <= 0 or args.adaptive_lead_s <= 0 or
            args.adaptive_slow_error_deg <= 0 or args.adaptive_hold_error_deg <= 0 or
            args.adaptive_hold_error_deg <= args.adaptive_slow_error_deg or
            not 0 < args.adaptive_min_speed_scale <= 1 or
            args.adaptive_progress_timeout_s <= 0 or
            args.gripper_open_threshold_m < 0 or args.gripper_closed_threshold_m < 0 or
            args.gripper_open_threshold_m <= args.gripper_closed_threshold_m or
            args.gripper_min_stable_frames <= 0 or args.gripper_open_command_m < 0 or
            args.gripper_closed_command_m < 0 or
            args.gripper_continuous_command_hz <= 0 or
            args.gripper_continuous_deadband_mm < 0 or
            (args.gripper_final_width_m is not None and args.gripper_final_width_m < 0)):
        parser.error("所有数值参数必须为正")
    if args.enable_gripper and args.execution_mode != "adaptive":
        parser.error("--enable-gripper 目前仅支持 --execution-mode adaptive，以便 J7 与实际路径进度同步")
    if args.trace_out is not None and args.execution_mode != "adaptive":
        parser.error("--trace-out 当前仅支持 --execution-mode adaptive")
    if args.plot_after_replay and args.trace_out is None:
        parser.error("--plot-after-replay 必须与 --trace-out 一起使用")
    temporary_p_overrides = None
    if args.temporary_p_coefficients is not None:
        try:
            temporary_p_overrides = json.loads(args.temporary_p_coefficients)
        except json.JSONDecodeError as exc:
            parser.error(f"--temporary-p-coefficients 不是有效 JSON: {exc}")
        if not isinstance(temporary_p_overrides, dict):
            parser.error("--temporary-p-coefficients 必须是 JSON 对象")
        for motor, value in temporary_p_overrides.items():
            if motor not in MOTOR_ARM_NAMES:
                parser.error(f"--temporary-p-coefficients 包含未知电机: {motor}")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 254:
                parser.error(f"临时 P {motor} 必须是 [0, 254] 的整数")
        print("TEMPORARY_P_REPLAY_TEST:", temporary_p_overrides,
              "; controller will restore prior EEPROM P values on exit.")

    plan_path = Path(args.joint_plan).expanduser().resolve()
    reference_path = Path(args.start_reference).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "am_umi_vjaw_taskspace_retarget_plan_v1":
        raise ValueError("不是受支持的任务空间重定向计划")
    if plan.get("status") != "candidate":
        raise ValueError("计划未被离线验收；拒绝执行")
    if Path(plan.get("reference", "")).resolve() != reference_path:
        raise ValueError("计划与 --start-reference 不属于同一基准；拒绝执行")
    path = np.asarray(plan.get("joint_path_model_deg", []), dtype=float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("计划缺少有效的六轴关节路径")
    reference_q = load_reference(reference_path)
    if float(np.max(np.abs(path[0] - reference_q))) > 0.05:
        raise ValueError("计划首点与保存基准不一致；拒绝执行")

    config = yaml.safe_load(Path(args.robot_config).expanduser().read_text(encoding="utf-8"))
    robot_cfg = config["robots"][0]
    if args.enable_gripper:
        gripper_min = float(robot_cfg.get("gripper_width_min", 0.0))
        gripper_max = float(robot_cfg.get("gripper_width_max", 0.100))
        commands = [args.gripper_open_command_m, args.gripper_closed_command_m]
        if args.gripper_final_width_m is not None:
            commands.append(args.gripper_final_width_m)
        if any(command < gripper_min - 1e-9 or command > gripper_max + 1e-9
               for command in commands):
            raise ValueError(f"J7 开口命令必须位于机器人配置范围 [{gripper_min}, {gripper_max}] m")
    urdf_path = Path(robot_cfg["urdf_path"]).expanduser()
    if not urdf_path.is_absolute():
        urdf_path = (ROOT / urdf_path).resolve()
    mapping = mapping_from_config(robot_cfg, ROOT)
    limits = joint_limits(urdf_path, model_safe_limits_from_config(robot_cfg, mapping))
    if np.any(path < limits[:, 0] - 1e-6) or np.any(path > limits[:, 1] + 1e-6):
        raise ValueError("候选路径超出当前真实六轴安全范围；拒绝执行")
    count = len(path) if args.full_plan else min(len(path), args.max_source_frames)
    path = path[:count]
    plan_fps = float(plan.get("parameters", {}).get("fps", 30.0))
    source_fps = args.source_fps or plan_fps
    if source_fps <= 0:
        raise ValueError("源帧率必须为正")
    indices = list(range(0, len(path), args.waypoint_stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)
    print("VJAW_RETARGET_REPLAY_PREPARE")
    gripper_status = (f"ENABLED ({args.gripper_mode} J7 labels)"
                      if args.enable_gripper else "disabled")
    print("safety: gripper commands are", gripper_status, "; replay mode:", args.execution_mode)
    print("plan:", plan_path)
    print(f"source frames: {len(path)}/{len(plan['joint_path_model_deg'])}; waypoints: {len(indices)}")
    smooth_times = smooth_path = smooth_knots = smooth_source_frame_progress = None
    gripper_events = ()
    gripper_width_trajectory = None
    if args.execution_mode in ("smooth", "queue", "adaptive"):
        smooth_times, smooth_path, smooth_speed, smooth_acceleration, smooth_knots = build_smooth_trajectory(
            path, source_fps, args.smooth_sample_hz, args.max_joint_speed_deg_s,
            args.max_joint_accel_deg_s2)
        # Preserve the exact source-frame location of every retimed 50 Hz
        # sample.  It makes a later replay trace able to plot demo J7 labels
        # against the real replay wall-clock, even if adaptive slowing occurs.
        smooth_source_frame_progress = np.interp(
            smooth_times, smooth_knots, np.arange(len(path), dtype=float))
        print("SMOOTH_TRAJECTORY_READY")
        print(f"  samples: {len(smooth_path)}; duration: {smooth_times[-1]:.1f}s; "
              f"sample rate: {args.smooth_sample_hz:.1f} Hz")
        print(f"  max speed: {smooth_speed:.2f} deg/s; max accel: {smooth_acceleration:.2f} deg/s^2")
        if (smooth_speed > args.max_joint_speed_deg_s * 1.01 or
                smooth_acceleration > args.max_joint_accel_deg_s2 * 1.01):
            raise RuntimeError("连续轨迹重定时未满足速度/加速度上限；未连接机械臂")
        if args.enable_gripper:
            if args.gripper_mode == "binary":
                source_events = load_binary_gripper_events(
                    plan, count,
                    open_threshold=args.gripper_open_threshold_m,
                    closed_threshold=args.gripper_closed_threshold_m,
                    min_stable_frames=args.gripper_min_stable_frames,
                    open_command=args.gripper_open_command_m,
                    closed_command=args.gripper_closed_command_m)
                gripper_events = map_gripper_events_to_smooth(source_events, smooth_knots, smooth_times)
                print("J7_BINARY_EVENTS_READY")
                if not gripper_events:
                    print("  no stable open/closed event in selected source frames; J7 will not move")
                for _, command, state, source_frame, event_time, label_width in gripper_events:
                    print(f"  source_frame={source_frame}; arm_path_time={event_time:.2f}s; "
                          f"state={state}; label_width={label_width * 1000:.1f}mm; "
                          f"J7_command={command * 1000:.1f}mm")
            else:
                source_widths = load_source_gripper_widths(plan, count)
                if (np.min(source_widths) < gripper_min - 1e-6 or
                        np.max(source_widths) > gripper_max + 1e-6):
                    raise ValueError(
                        f"连续 J7 标签超出机器人配置范围 [{gripper_min}, {gripper_max}] m；"
                        "请先校正手持夹爪标定")
                gripper_width_trajectory = map_continuous_gripper_widths_to_smooth(
                    source_widths, smooth_knots, smooth_times)
                print("J7_CONTINUOUS_TRAJECTORY_READY")
                print(f"  source width range: {np.min(source_widths) * 1000:.1f}.."
                      f"{np.max(source_widths) * 1000:.1f}mm; "
                      f"command rate cap: {args.gripper_continuous_command_hz:.1f}Hz; "
                      f"deadband: {args.gripper_continuous_deadband_mm:.1f}mm")
                probe_indices = np.linspace(0, len(source_widths) - 1,
                                           num=min(5, len(source_widths)), dtype=int)
                print("  source samples:", ", ".join(
                    f"{index}:{source_widths[index] * 1000:.1f}mm" for index in probe_indices))
    print("No robot command has been sent during this validation.")
    if args.preview_only:
        print("PREVIEW_ONLY_OK: no robot connection was opened.")
        return

    trace_path = None
    if args.trace_out is not None:
        trace_path = Path(args.trace_out).expanduser().resolve()
        if trace_path.exists():
            raise ValueError(f"拒绝覆盖已有 replay trace: {trace_path}")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        if args.plot_after_replay:
            plot_stem = trace_path.with_suffix("")
            plot_paths = (
                plot_stem.with_name(plot_stem.name + ".trajectory.png"),
                plot_stem.with_name(plot_stem.name + ".trajectory.json"),
            )
            existing_plots = [str(path) for path in plot_paths if path.exists()]
            if existing_plots:
                raise ValueError("拒绝覆盖已有 replay 图或报告: " + ", ".join(existing_plots))
    trace_samples: list[dict] = []
    trace_status = "not_started"

    with AM2ProInterpolationController(
        robot_usb_port=robot_cfg.get("robot_usb_port", "/dev/ttyACM0"), frequency=50,
        receive_latency=robot_cfg.get("robot_obs_latency", 0.005),
        server_python=robot_cfg.get("robot_python"),
        ik_backend=robot_cfg.get("ik_backend", "placo"), urdf_path=str(urdf_path),
        flip_joints=robot_cfg.get("joint_flip", []),
        joint_model_candidate=robot_cfg.get("joint_model_candidate"),
        gripper_width_min=robot_cfg.get("gripper_width_min", 0.0),
        gripper_width_max=robot_cfg.get("gripper_width_max", 0.100),
        gripper_servo_closed=robot_cfg.get("gripper_servo_closed", 0.0),
        gripper_servo_open=robot_cfg.get("gripper_servo_open", 0.0),
        gripper_width_servo_table=robot_cfg.get("gripper_width_servo_table"), verbose=True,
        position_p_coefficients=robot_cfg.get("position_p_coefficients"),
        position_p_overrides=temporary_p_overrides,
    ) as robot:
        deadline = time.monotonic() + 5.0
        while robot.get_state() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        start_q = preposition(robot, reference_q, args.reference_duration,
                              args.reference_speed_deg_s, args.reference_tolerance_deg)
        try:
            if args.pause_after_reference:
                input("REFERENCE_PAUSED: 机械臂保持在基准。确认工作区清空、纸上十字对准后按 Enter 开始；Ctrl+C 取消。\n")
            print("VJAW_RETARGET_REPLAYING")
            if args.execution_mode == "smooth":
                execute_smooth(robot, smooth_path, args.smooth_sample_hz,
                               args.max_tracking_error_deg)
            elif args.execution_mode == "queue":
                execute_queue(robot, smooth_path, args.smooth_sample_hz,
                              args.queue_advance_tolerance_deg,
                              args.max_tracking_error_deg,
                              args.queue_progress_timeout_s,
                              args.queue_chunk_samples,
                              args.max_joint_speed_deg_s,
                              args.queue_min_command_duration_s,
                              args.queue_reversal_settle_after_s,
                              args.queue_reversal_settle_duration_s)
            elif args.execution_mode == "adaptive":
                trace_status = "running"
                trace_started_at = time.monotonic()
                execute_adaptive(robot, smooth_path, args.smooth_sample_hz,
                                 args.adaptive_lead_s, args.adaptive_slow_error_deg,
                                 args.adaptive_hold_error_deg,
                                 args.adaptive_min_speed_scale,
                                 args.adaptive_progress_timeout_s,
                                 args.max_tracking_error_deg,
                                 args.queue_advance_tolerance_deg,
                                 gripper_events=gripper_events,
                                 gripper_width_trajectory=gripper_width_trajectory,
                                 gripper_command_hz=args.gripper_continuous_command_hz,
                                 gripper_deadband_m=args.gripper_continuous_deadband_mm / 1000.0,
                                 source_frame_progress_trajectory=smooth_source_frame_progress,
                                 trace_samples=trace_samples,
                                 trace_started_at=trace_started_at)
            else:
                for step, index in enumerate(indices):
                    error = wait_waypoint(robot, path[index], args.waypoint_speed_deg_s,
                                          args.settle_tolerance_deg, args.waypoint_timeout_s)
                    print(f"  waypoint {step + 1}/{len(indices)} (source frame {index}): "
                          f"settled max error={error:.2f} deg", flush=True)
                    if args.waypoint_hold_s > 0:
                        print(f"    holding {args.waypoint_hold_s:.1f}s", flush=True)
                        time.sleep(args.waypoint_hold_s)
            print("VJAW_RETARGET_REPLAY_FINISHED")
            trace_status = "finished"
        except KeyboardInterrupt:
            print("VJAW_RETARGET_REPLAY_INTERRUPTED")
            trace_status = "interrupted"
        except RuntimeError as exc:
            print("VJAW_RETARGET_REPLAY_ABORTED:", exc)
            trace_status = "aborted"
        finally:
            if trace_path is not None:
                trace = {
                    "schema": "am_umi_vjaw_adaptive_replay_trace_v1",
                    "safety": "recorded readback only; it does not measure physical TCP calibration error",
                    "status": trace_status,
                    "plan": str(plan_path),
                    "reference": str(reference_path),
                    "sample_hz": args.smooth_sample_hz,
                    "samples": trace_samples,
                }
                trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
                print(f"REPLAY_TRACE_SAVED: {trace_path} ({len(trace_samples)} samples)")
            if args.enable_gripper and args.gripper_final_width_m is not None:
                robot.schedule_gripper(args.gripper_final_width_m, time.time() + 0.05)
                print(f"J7_FINAL_COMMAND: {args.gripper_final_width_m * 1000:.1f}mm", flush=True)
                time.sleep(0.20)
            if args.do_return:
                print("Returning to verified start joints ...")
                robot.servoJ(start_q, duration=4.0)
                time.sleep(4.2)
                print("Returned.")

    if args.plot_after_replay and trace_path is not None:
        generate_replay_plots(plan=plan, plan_path=plan_path,
                              robot_config=Path(args.robot_config).expanduser().resolve(),
                              trace_path=trace_path)


if __name__ == "__main__":
    main()
