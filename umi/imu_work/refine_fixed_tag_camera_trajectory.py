#!/usr/bin/env python3
"""Refine a direct fixed-Tag camera trajectory without touching raw results.

The input must be produced by ``export_fixed_tag_camera_trajectory.py``.  This
tool rejects isolated pose spikes, interpolates only short internal gaps, and
applies a small Savitzky-Golay smoothing window independently to every valid
segment.  Long gaps remain ``is_lost=true`` so downstream code cannot silently
use fabricated motion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


POSE_COLUMNS = ["x", "y", "z", "q_x", "q_y", "q_z", "q_w"]
REQUIRED_COLUMNS = ["frame_idx", "timestamp", "state", "is_lost",
                    "is_keyframe", *POSE_COLUMNS]


def bool_array(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(copy=True)
    return series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(True).to_numpy(dtype=bool)


def valid_segments(valid: np.ndarray):
    padded = np.pad(valid.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    return [(int(start), int(stop)) for start, stop in edges.reshape(-1, 2)]


def reject_isolated_spikes(
        timestamps: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
        valid: np.ndarray,
        max_position_error_m: float,
        max_rotation_error_deg: float) -> np.ndarray:
    rejected = np.zeros(len(valid), dtype=bool)
    # Two passes catch a pair of adjacent bad PnP estimates without allowing a
    # rejected sample to become an interpolation anchor in the second pass.
    for _ in range(2):
        current_valid = valid & ~rejected
        scores = np.zeros(len(valid), dtype=float)
        for index in range(1, len(valid) - 1):
            if not current_valid[index]:
                continue
            previous = index - 1
            while previous >= 0 and not current_valid[previous]:
                previous -= 1
            following = index + 1
            while following < len(valid) and not current_valid[following]:
                following += 1
            if previous < 0 or following >= len(valid):
                continue
            # Spike rejection is deliberately local.  Missing spans are dealt
            # with by the explicit gap interpolation policy below.
            if index - previous > 2 or following - index > 2:
                continue
            span = timestamps[following] - timestamps[previous]
            if span <= 0:
                continue
            alpha = (timestamps[index] - timestamps[previous]) / span
            expected_position = ((1.0 - alpha) * positions[previous]
                                 + alpha * positions[following])
            position_error = np.linalg.norm(positions[index] - expected_position)
            slerp = Slerp(
                [timestamps[previous], timestamps[following]],
                Rotation.from_quat(quaternions[[previous, following]]),
            )
            expected_rotation = slerp([timestamps[index]])[0]
            rotation_error = np.degrees(
                (expected_rotation.inv()
                 * Rotation.from_quat(quaternions[index])).magnitude())
            scores[index] = max(
                position_error / max_position_error_m,
                rotation_error / max_rotation_error_deg)
        candidates = [
            index for index in range(1, len(valid) - 1)
            if (scores[index] > 1.0
                and scores[index] >= scores[index - 1]
                and scores[index] >= scores[index + 1])]
        if not candidates:
            break
        rejected[candidates] = True
    return rejected


def interpolate_short_gaps(
        timestamps: np.ndarray,
        positions: np.ndarray,
        quaternions: np.ndarray,
        valid: np.ndarray,
        max_gap_frames: int) -> np.ndarray:
    interpolated = np.zeros(len(valid), dtype=bool)
    if max_gap_frames <= 0:
        return interpolated
    for start, stop in valid_segments(~valid):
        gap = stop - start
        if (gap > max_gap_frames or start == 0 or stop == len(valid)
                or not valid[start - 1] or not valid[stop]):
            continue
        left, right = start - 1, stop
        query_times = timestamps[start:stop]
        span = timestamps[right] - timestamps[left]
        if span <= 0:
            continue
        alpha = ((query_times - timestamps[left]) / span)[:, None]
        positions[start:stop] = ((1.0 - alpha) * positions[left]
                                 + alpha * positions[right])
        slerp = Slerp(
            [timestamps[left], timestamps[right]],
            Rotation.from_quat(quaternions[[left, right]]),
        )
        quaternions[start:stop] = slerp(query_times).as_quat()
        interpolated[start:stop] = True
        valid[start:stop] = True
    return interpolated


def smooth_segments(
        positions: np.ndarray,
        quaternions: np.ndarray,
        valid: np.ndarray,
        window: int) -> None:
    if window <= 1:
        return
    for start, stop in valid_segments(valid):
        length = stop - start
        local_window = min(window, length if length % 2 else length - 1)
        if local_window < 3:
            continue
        polyorder = min(2, local_window - 1)
        positions[start:stop] = savgol_filter(
            positions[start:stop], local_window, polyorder,
            axis=0, mode="interp")

        # Keep quaternion signs continuous before converting to a local
        # rotation-vector chart.  The hand-held task motion is well below the
        # chart's pi discontinuity inside a short smoothing window.
        local_quat = quaternions[start:stop].copy()
        for index in range(1, len(local_quat)):
            if np.dot(local_quat[index - 1], local_quat[index]) < 0:
                local_quat[index] *= -1
        reference = Rotation.from_quat(local_quat[0])
        local_rotvec = (reference.inv()
                        * Rotation.from_quat(local_quat)).as_rotvec()
        local_rotvec = savgol_filter(
            local_rotvec, local_window, polyorder,
            axis=0, mode="interp")
        quaternions[start:stop] = (
            reference * Rotation.from_rotvec(local_rotvec)).as_quat()


def run(args: argparse.Namespace) -> dict:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    report_path = output_path.with_suffix(".refinement_report.json")
    if output_path == input_path:
        raise SystemExit("拒绝覆盖原始轨迹；--output 必须是新文件")
    if output_path.exists() or report_path.exists():
        raise SystemExit(f"为保护已有结果，拒绝覆盖：{output_path} 或 {report_path}")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise SystemExit("--smooth-window 必须是正奇数")
    if args.max_gap_frames < 0:
        raise SystemExit("--max-gap-frames 不能为负数")

    frame = pd.read_csv(input_path)
    missing_columns = [column for column in REQUIRED_COLUMNS
                       if column not in frame.columns]
    if missing_columns:
        raise SystemExit(f"输入轨迹缺少列：{missing_columns}")
    timestamps = frame["timestamp"].to_numpy(dtype=float)
    if len(frame) < 3 or not np.all(np.diff(timestamps) > 0):
        raise SystemExit("轨迹时间戳必须严格递增且至少包含 3 帧")

    positions = frame[["x", "y", "z"]].to_numpy(dtype=float)
    quaternions = frame[["q_x", "q_y", "q_z", "q_w"]].to_numpy(dtype=float)
    lost = bool_array(frame["is_lost"])
    finite = np.isfinite(positions).all(axis=1) & np.isfinite(quaternions).all(axis=1)
    quat_norm = np.linalg.norm(quaternions, axis=1)
    valid = (~lost) & finite & (quat_norm > 1e-8)
    quaternions[valid] /= quat_norm[valid, None]
    original_valid = valid.copy()
    original_positions = positions.copy()
    original_quaternions = quaternions.copy()

    rejected = reject_isolated_spikes(
        timestamps, positions, quaternions, valid,
        args.max_position_interpolation_error_m,
        args.max_rotation_interpolation_error_deg)
    valid &= ~rejected
    interpolated = interpolate_short_gaps(
        timestamps, positions, quaternions, valid, args.max_gap_frames)
    smooth_segments(positions, quaternions, valid, args.smooth_window)

    refined = frame[REQUIRED_COLUMNS].copy()
    refined.loc[:, ["x", "y", "z"]] = positions
    refined.loc[:, ["q_x", "q_y", "q_z", "q_w"]] = quaternions
    refined.loc[:, "is_lost"] = ~valid
    refined.loc[:, "state"] = np.where(valid, 2, 1)
    refined.loc[:, "is_keyframe"] = False
    refined.loc[~valid, POSE_COLUMNS] = 0.0
    refined.to_csv(output_path, index=False, float_format="%.9f")

    comparison = original_valid & valid & ~rejected
    position_correction = np.linalg.norm(
        positions[comparison] - original_positions[comparison], axis=1)
    rotation_correction = np.degrees((
        Rotation.from_quat(original_quaternions[comparison]).inv()
        * Rotation.from_quat(quaternions[comparison])).magnitude())
    report = {
        "schema": "am_umi_fixed_tag_trajectory_refinement_v1",
        "input": str(input_path),
        "output": str(output_path),
        "frame_count": int(len(frame)),
        "raw_valid_frames": int(original_valid.sum()),
        "rejected_spike_frames": int(rejected.sum()),
        "interpolated_short_gap_frames": int(interpolated.sum()),
        "final_valid_frames": int(valid.sum()),
        "final_valid_ratio": float(valid.mean()),
        "remaining_lost_frames": int((~valid).sum()),
        "parameters": {
            "max_gap_frames": args.max_gap_frames,
            "smooth_window": args.smooth_window,
            "max_position_interpolation_error_m": args.max_position_interpolation_error_m,
            "max_rotation_interpolation_error_deg": args.max_rotation_interpolation_error_deg,
        },
        "smoothing_correction": {
            "position_median_m": float(np.median(position_correction)),
            "position_p99_m": float(np.quantile(position_correction, 0.99)),
            "position_max_m": float(position_correction.max()),
            "rotation_median_deg": float(np.median(rotation_correction)),
            "rotation_p99_deg": float(np.quantile(rotation_correction, 0.99)),
            "rotation_max_deg": float(rotation_correction.max()),
        },
        "safety_note": (
            "Only short internal gaps were interpolated. Remaining gaps stay lost; "
            "the source CSV and all video/IMU files were not modified."),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="剔除固定 Tag 单帧跳变、短缺口插值并轻量平滑")
    parser.add_argument("--input", required=True, help="固定 Tag 原始轨迹 CSV")
    parser.add_argument("--output", required=True, help="新的 refined CSV；拒绝覆盖")
    parser.add_argument("--max-gap-frames", type=int, default=5,
                        help="最多插值的内部连续缺失帧数；默认 5")
    parser.add_argument("--smooth-window", type=int, default=5,
                        help="Savitzky-Golay 平滑窗口，必须是奇数；1=关闭")
    parser.add_argument("--max-position-interpolation-error-m", type=float, default=0.03,
                        help="相对相邻帧插值超过该距离视为单帧跳变")
    parser.add_argument("--max-rotation-interpolation-error-deg", type=float, default=10.0,
                        help="相对相邻姿态插值超过该角度视为单帧跳变")
    args = parser.parse_args()
    report = run(args)
    print("FIXED_TAG_TRAJECTORY_REFINEMENT_OK")
    print(f"raw_valid: {report['raw_valid_frames']} / {report['frame_count']}")
    print(f"rejected_spikes: {report['rejected_spike_frames']}")
    print(f"interpolated_short_gaps: {report['interpolated_short_gap_frames']}")
    print(f"final_valid: {report['final_valid_frames']} "
          f"({report['final_valid_ratio']:.3%})")
    print(f"remaining_lost: {report['remaining_lost_frames']}")
    print("output:", report["output"])
    print("report:", str(Path(report["output"]).with_suffix(
        ".refinement_report.json")))


if __name__ == "__main__":
    main()
