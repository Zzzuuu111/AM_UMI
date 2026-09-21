#!/usr/bin/env python3
"""Calibrate a one-moving-jaw/V-gripper command from two tool ArUco Tags.

Record a short video with the V gripper held still in the camera frame:
first keep it fully closed, then fully open it, then close it again.  The
fixed-jaw Tag is the stable tool reference; the moving-jaw Tag rotates with
the actuated jaw.  This tool is offline only and never opens a serial port.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.common.cv_util import get_relative_tag_rotation_deg  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用固定爪/活动爪 Tag 标定 V 型夹爪的开合命令（离线，不控制机械臂）。")
    parser.add_argument("--input", required=True, help="detect_aruco.py 输出的 tag_detection.pkl")
    parser.add_argument("--output", required=True, help="新的 V 型夹爪范围 JSON（不得已存在）")
    parser.add_argument("--fixed-tag-id", required=True, type=int,
                        help="安装在固定爪或夹爪底座上的 Tag ID")
    parser.add_argument("--moving-tag-id", required=True, type=int,
                        help="安装在活动爪上的 Tag ID")
    parser.add_argument("--max-opening-m", required=True, type=float,
                        help="新夹爪从闭合到最大开口的实测宽度（米）")
    parser.add_argument("--closed-percentile", type=float, default=1.0,
                        help="闭合端使用的原始旋转角百分位，默认 1")
    parser.add_argument("--open-percentile", type=float, default=99.0,
                        help="张开端使用的原始旋转角百分位，默认 99")
    parser.add_argument("--closed-at", choices=("auto", "low", "high"), default="auto",
                        help="哪个角度端点对应闭合；auto 假定视频以闭合—张开—闭合录制")
    parser.add_argument("--edge-fraction", type=float, default=0.15,
                        help="auto 判断闭合端时取开头和结尾各多少比例，默认 0.15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"为保护已有标定，拒绝覆盖：{output}")
    if args.fixed_tag_id == args.moving_tag_id:
        raise SystemExit("fixed-tag-id 和 moving-tag-id 必须不同")
    if args.max_opening_m <= 0:
        raise SystemExit("--max-opening-m 必须为正")
    if not 0 <= args.closed_percentile < args.open_percentile <= 100:
        raise SystemExit("要求 0 <= closed-percentile < open-percentile <= 100")
    if not 0 < args.edge_fraction <= 0.4:
        raise SystemExit("--edge-fraction 必须在 (0, 0.4] 内")

    with source.open("rb") as file:
        detections = pickle.load(file)
    values = []
    indices = []
    for frame_index, frame in enumerate(detections):
        value = get_relative_tag_rotation_deg(
            frame.get("tag_dict", {}), args.fixed_tag_id, args.moving_tag_id)
        if value is not None and np.isfinite(value):
            values.append(float(value))
            indices.append(frame_index)
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 20:
        raise SystemExit("有效的固定爪+活动爪双 Tag 帧少于 20；请确认两个 Tag 都清晰可见")

    low, high = np.percentile(values, [args.closed_percentile, args.open_percentile])
    if high - low < 3.0:
        raise SystemExit(
            f"观察到的活动爪相对转角仅 {high - low:.2f}°；请完整录到闭合和最大张开")
    closed_at = args.closed_at
    if closed_at == "auto":
        total_frames = max(1, len(detections))
        edge = max(1, int(round(total_frames * args.edge_fraction)))
        closed_samples = [value for index, value in zip(indices, values)
                          if index < edge or index >= total_frames - edge]
        if len(closed_samples) < 4:
            raise SystemExit("auto 无法取得足够的首尾闭合样本；改用 --closed-at low 或 high")
        closed_median = float(np.median(closed_samples))
        closed_at = "low" if abs(closed_median - low) <= abs(closed_median - high) else "high"
    raw_closed, raw_open = (low, high) if closed_at == "low" else (high, low)
    result = {
        "schema_version": 1,
        "gripper_model": "v_jaw_one_moving_tag_pair",
        "width_measurement": "relative_tag_rotation_deg",
        "fixed_jaw_tag_id": int(args.fixed_tag_id),
        "moving_jaw_tag_id": int(args.moving_tag_id),
        "valid_pair_frames": int(len(values)),
        "relative_rotation_percentiles_deg": {
            "low_percentile": float(args.closed_percentile), "low_value": float(low),
            "high_percentile": float(args.open_percentile), "high_value": float(high),
        },
        "raw_closed_open": [float(raw_closed), float(raw_open)],
        "physical_gripper_range_m": [0.0, float(args.max_opening_m)],
        "min_width": 0.0,
        "max_width": float(args.max_opening_m),
        "notes": [
            "原始量是固定爪 Tag 与活动爪 Tag 的相对转角，不是两 Tag 的水平距离。",
            "请用一次完整闭合—张开—闭合的视频标定；默认 auto 以首尾静止段判断闭合端。",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("V_JAW_GRIPPER_RANGE_CALIBRATION_OK")
    print("valid_pair_frames:", len(values))
    print(f"relative_rotation_deg: low={low:.3f}, high={high:.3f}; closed_at={closed_at}")
    print(f"physical_opening_range_m: [0.00000, {args.max_opening_m:.5f}]")
    print("saved:", output)


if __name__ == "__main__":
    main()
