#!/usr/bin/env python3
"""Offline time-stretch a single fixed-Tag hand-held UMI zarr episode.

The spatial demonstration is preserved.  Only its clock is lengthened: TCP
translation and gripper width are linearly interpolated, orientation is SLERP
interpolated, and the policy image nearest to each new timestamp is retained.
This lets a physically reachable but too-fast demonstration be screened again
for low-speed AM2Pro replay.  It never opens a serial port or controls a robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="离线拉伸手持 UMI zarr 的时间轴；绝不连接或控制机械臂。")
    parser.add_argument("--input", required=True, help="单 episode 的输入 .zarr")
    parser.add_argument("--output", required=True, help="新的输出 .zarr（必须不存在）")
    parser.add_argument("--time-scale", required=True, type=float,
                        help="时间拉伸倍数，例如 3.2 表示同一空间轨迹以 3.2 倍时长执行")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="输入和输出策略时间轴的帧率，默认 30")
    return parser.parse_args()


def interpolate_pose(pos: np.ndarray, rotvec: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.arange(len(pos), dtype=np.float64)
    out_pos = np.stack([np.interp(query, source, pos[:, axis]) for axis in range(3)], axis=1)
    rotation = Rotation.from_rotvec(rotvec)
    out_rot = Slerp(source, rotation)(query).as_rotvec()
    return out_pos.astype(np.float32), out_rot.astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.time_scale < 1.0:
        raise SystemExit("--time-scale 必须 >= 1；此工具只做减速拉伸")
    if args.fps <= 0:
        raise SystemExit("--fps 必须为正")

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    if output_path.exists():
        raise SystemExit(f"为保护已有输出，拒绝覆盖：{output_path}")

    buffer = ReplayBuffer.create_from_path(str(input_path), mode="r")
    if buffer.n_episodes != 1:
        raise ValueError(f"此工具只接受单 episode zarr，实际为 {buffer.n_episodes} 个")
    episode = buffer.get_episode(0)
    required = ("camera0_rgb", "robot0_eef_pos", "robot0_eef_rot_axis_angle", "robot0_gripper_width")
    missing = [key for key in required if key not in episode]
    if missing:
        raise ValueError(f"输入缺少字段：{missing}")

    image = np.asarray(episode["camera0_rgb"])
    pos = np.asarray(episode["robot0_eef_pos"], dtype=np.float64)
    rotvec = np.asarray(episode["robot0_eef_rot_axis_angle"], dtype=np.float64)
    width = np.asarray(episode["robot0_gripper_width"], dtype=np.float64).reshape(-1)
    count = len(pos)
    if count < 2 or any(len(value) != count for value in (image, rotvec, width)):
        raise ValueError("输入 episode 的帧数不一致或少于 2 帧")

    # New samples lie on the original index axis, but are spaced 1/time_scale
    # frames apart.  This preserves the first/last poses exactly.
    output_count = int(np.ceil((count - 1) * args.time_scale)) + 1
    query = np.linspace(0.0, count - 1, output_count, dtype=np.float64)
    out_pos, out_rotvec = interpolate_pose(pos, rotvec, query)
    out_width = np.interp(query, np.arange(count, dtype=np.float64), width).astype(np.float32)[:, None]
    image_indices = np.clip(np.rint(query).astype(np.int64), 0, count - 1)
    out_image = image[image_indices]
    tcp_pose = np.concatenate((out_pos, out_rotvec), axis=1).astype(np.float32)

    output = ReplayBuffer.create_empty_numpy()
    output.add_episode({
        "action": np.concatenate((out_pos, out_rotvec, out_width), axis=1).astype(np.float32),
        "camera0_rgb": out_image,
        "robot0_eef_pos": out_pos,
        "robot0_eef_rot_axis_angle": out_rotvec,
        "robot0_gripper_width": out_width,
        "robot0_demo_start_pose": np.repeat(tcp_pose[:1], output_count, axis=0),
        "robot0_demo_end_pose": np.repeat(tcp_pose[-1:], output_count, axis=0),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save_to_path(str(output_path), compressors="disk")

    report = {
        "schema": "am_umi_handheld_demo_retime_v1",
        "safety": "offline only; no serial port or robot command was used",
        "input": str(input_path),
        "output": str(output_path),
        "time_scale": float(args.time_scale),
        "fps": float(args.fps),
        "input_frames": int(count),
        "output_frames": int(output_count),
        "input_duration_s": float((count - 1) / args.fps),
        "output_duration_s": float((output_count - 1) / args.fps),
        "image_sampling": "nearest source policy image; repeated frames preserve the slowed visual state",
        "pose_sampling": "linear translation/width plus SO(3) SLERP orientation",
    }
    report_path = output_path / "retime_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("HANDHELD_DEMO_RETIME_OK")
    print("output:", output_path)
    print(f"time_scale: {args.time_scale:.3f}; frames: {count} -> {output_count}")
    print(f"duration_s: {report['input_duration_s']:.3f} -> {report['output_duration_s']:.3f}")
    print("report:", report_path)


if __name__ == "__main__":
    main()
