#!/usr/bin/env python3
"""Read-only quality gate for an AM_UMI training zarr.

This tool intentionally runs before physical replay or training.  It verifies
the dataset container, episode boundaries, action/state agreement, numeric
sanity, obvious trajectory spikes and a representative sample of decoded RGB
frames.  It never connects to a robot and never modifies the dataset.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import zarr
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


REQUIRED = {
    "action": (2, 7),
    "camera0_rgb": (4, 3),
    "robot0_eef_pos": (2, 3),
    "robot0_eef_rot_axis_angle": (2, 3),
    "robot0_gripper_width": (2, 1),
}


def open_root(path: Path):
    if path.is_dir():
        return zarr.open(str(path), mode="r"), None
    store = zarr.ZipStore(str(path), mode="r")
    return zarr.group(store=store), store


def add_issue(issues, level, code, message, **details):
    item = {"level": level, "code": code, "message": message}
    if details:
        item["details"] = details
    issues.append(item)


def rotation_steps(rotvec):
    if len(rotvec) < 2:
        return np.zeros(0, dtype=np.float64)
    rotations = Rotation.from_rotvec(rotvec)
    relative = rotations[1:] * rotations[:-1].inv()
    return relative.magnitude()


def sampled_indices(n, count):
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, count, dtype=np.int64))


def validate(args):
    path = Path(args.dataset).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    register_codecs()
    root, store = open_root(path)
    issues = []
    report = {
        "schema": "am_umi_dataset_quality_gate_v1",
        "dataset": str(path),
        "parameters": {
            "fps": args.fps,
            "min_episode_frames": args.min_episode_frames,
            "max_position_step_m": args.max_position_step_m,
            "max_rotation_step_rad": args.max_rotation_step_rad,
            "max_gripper_step_m": args.max_gripper_step_m,
            "image_samples": args.image_samples,
        },
        "issues": issues,
    }
    try:
        if "data" not in root or "meta" not in root:
            add_issue(issues, "FAIL", "missing_groups",
                      "zarr 必须包含 data 和 meta 两个组")
            return report
        data = root["data"]
        meta = root["meta"]
        if "episode_ends" not in meta:
            add_issue(issues, "FAIL", "missing_episode_ends",
                      "meta/episode_ends 不存在")
            return report

        ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
        if len(ends) == 0 or np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
            add_issue(issues, "FAIL", "invalid_episode_ends",
                      "episode_ends 必须为严格递增的正整数", values=ends.tolist())
            return report
        n = int(ends[-1])
        starts = np.r_[0, ends[:-1]]
        lengths = ends - starts
        report.update({
            "frames": n,
            "episodes": int(len(ends)),
            "episode_lengths": lengths.tolist(),
            "duration_s_at_declared_fps": float(n / args.fps),
            "arrays": {},
        })

        for name, (ndim, final_dim) in REQUIRED.items():
            if name not in data:
                add_issue(issues, "FAIL", "missing_array",
                          f"缺少训练必需数组 data/{name}", array=name)
                continue
            arr = data[name]
            report["arrays"][name] = {
                "shape": list(arr.shape), "dtype": str(arr.dtype),
                "chunks": list(arr.chunks),
            }
            if len(arr.shape) != ndim or arr.shape[-1] != final_dim:
                add_issue(issues, "FAIL", "bad_array_shape",
                          f"data/{name} 形状不符合 AM_UMI 约定",
                          actual=list(arr.shape), expected_ndim=ndim,
                          expected_last_dim=final_dim)
            if len(arr.shape) == 0 or arr.shape[0] != n:
                add_issue(issues, "FAIL", "length_mismatch",
                          f"data/{name} 与 episode_ends 的总帧数不一致",
                          array_frames=int(arr.shape[0]) if arr.shape else None,
                          expected_frames=n)

        if any(x["level"] == "FAIL" for x in issues):
            return report

        short = np.flatnonzero(lengths < args.min_episode_frames)
        if len(short):
            add_issue(issues, "FAIL", "episode_too_short",
                      "episode 短于策略 horizon，不能形成有效训练样本",
                      episode_ids=short.tolist(), lengths=lengths[short].tolist())

        action = np.asarray(data["action"][:], dtype=np.float64)
        pos = np.asarray(data["robot0_eef_pos"][:], dtype=np.float64)
        rot = np.asarray(data["robot0_eef_rot_axis_angle"][:], dtype=np.float64)
        width = np.asarray(data["robot0_gripper_width"][:], dtype=np.float64)
        expected_action = np.concatenate([pos, rot, width], axis=1)

        numeric = {"action": action, "robot0_eef_pos": pos,
                   "robot0_eef_rot_axis_angle": rot,
                   "robot0_gripper_width": width}
        for name, values in numeric.items():
            bad = int(np.size(values) - np.isfinite(values).sum())
            if bad:
                add_issue(issues, "FAIL", "non_finite_values",
                          f"data/{name} 含 NaN 或 Inf", count=bad)

        if np.all(np.isfinite(action)) and np.all(np.isfinite(expected_action)):
            mismatch = np.max(np.abs(action - expected_action))
            report["action_state_max_abs_error"] = float(mismatch)
            if mismatch > args.action_tolerance:
                add_issue(issues, "FAIL", "action_state_mismatch",
                          "action 与末端位姿/夹爪状态不一致",
                          max_abs_error=float(mismatch),
                          tolerance=args.action_tolerance)

        bad_width = np.flatnonzero((width[:, 0] < args.gripper_min_m) |
                                   (width[:, 0] > args.gripper_max_m))
        report["gripper_width_range_m"] = [float(np.nanmin(width)),
                                             float(np.nanmax(width))]
        if len(bad_width):
            add_issue(issues, "FAIL", "gripper_out_of_range",
                      "夹爪开度超出允许范围", count=int(len(bad_width)),
                      allowed=[args.gripper_min_m, args.gripper_max_m])

        episode_reports = []
        for episode_id, (start, end) in enumerate(zip(starts, ends)):
            p = pos[start:end]
            r = rot[start:end]
            w = width[start:end, 0]
            p_step = np.linalg.norm(np.diff(p, axis=0), axis=1)
            try:
                r_step = rotation_steps(r)
            except ValueError as exc:
                r_step = np.array([math.inf])
                add_issue(issues, "FAIL", "invalid_rotation",
                          f"episode {episode_id} 旋转向量无效", error=str(exc))
            w_step = np.abs(np.diff(w))
            metrics = {
                "episode": episode_id,
                "frames": int(end - start),
                "duration_s": float((end - start) / args.fps),
                "position_path_m": float(p_step.sum()),
                "max_position_step_m": float(p_step.max(initial=0)),
                "max_rotation_step_rad": float(r_step.max(initial=0)),
                "max_gripper_step_m": float(w_step.max(initial=0)),
            }
            episode_reports.append(metrics)
            limits = (("position", metrics["max_position_step_m"],
                       args.max_position_step_m, "m"),
                      ("rotation", metrics["max_rotation_step_rad"],
                       args.max_rotation_step_rad, "rad"),
                      ("gripper", metrics["max_gripper_step_m"],
                       args.max_gripper_step_m, "m"))
            for kind, value, limit, unit in limits:
                if value > limit:
                    add_issue(issues, "FAIL", f"{kind}_step_spike",
                              f"episode {episode_id} 存在明显的 {kind} 单帧跳变",
                              value=value, limit=limit, unit=unit)
                elif value > limit * 0.5:
                    add_issue(issues, "WARN", f"{kind}_step_high",
                              f"episode {episode_id} 的 {kind} 单帧变化偏大",
                              value=value, limit=limit, unit=unit)
        report["episode_metrics"] = episode_reports

        images = data["camera0_rgb"]
        indices = sampled_indices(n, args.image_samples)
        flat = 0
        clipped = 0
        image_metrics = []
        for idx in indices:
            image = np.asarray(images[int(idx)])
            if image.shape[-1] != 3 or image.dtype != np.uint8:
                add_issue(issues, "FAIL", "bad_image",
                          "camera0_rgb 解码后的格式不是 uint8 RGB",
                          frame=int(idx), shape=list(image.shape), dtype=str(image.dtype))
                continue
            std = float(image.std())
            dynamic_range = int(image.max()) - int(image.min())
            clip_fraction = float(np.mean((image <= 1) | (image >= 254)))
            is_flat = std < args.min_image_std or dynamic_range < args.min_image_range
            is_clipped = clip_fraction > args.max_image_clip_fraction
            flat += int(is_flat)
            clipped += int(is_clipped)
            image_metrics.append({"frame": int(idx), "std": std,
                                  "range": dynamic_range,
                                  "clip_fraction": clip_fraction})
        sample_count = max(len(image_metrics), 1)
        report["image_sampling"] = {
            "sampled_frames": int(len(image_metrics)),
            "flat_frames": flat,
            "clipped_frames": clipped,
            "flat_fraction": flat / sample_count,
            "clipped_fraction": clipped / sample_count,
            "samples": image_metrics,
        }
        if flat / sample_count > args.max_bad_image_fraction:
            add_issue(issues, "FAIL", "flat_images",
                      "过多图像接近纯色/灰片，可能是错误解码或坏数据",
                      fraction=flat / sample_count)
        if clipped / sample_count > args.max_bad_image_fraction:
            add_issue(issues, "WARN", "clipped_images",
                      "过多图像严重过曝、欠曝或大面积遮罩",
                      fraction=clipped / sample_count)
        return report
    finally:
        if store is not None:
            store.close()


def main():
    parser = argparse.ArgumentParser(
        description="训练前离线检查 UMI zarr；不会连接或移动机械臂")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", help="JSON 报告路径；默认放在数据集旁边")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-episode-frames", type=int, default=16)
    parser.add_argument("--image-samples", type=int, default=64)
    parser.add_argument("--action-tolerance", type=float, default=1e-5)
    parser.add_argument("--gripper-min-m", type=float, default=-0.002)
    parser.add_argument("--gripper-max-m", type=float, default=0.10)
    parser.add_argument("--max-position-step-m", type=float, default=0.08)
    parser.add_argument("--max-rotation-step-rad", type=float, default=1.0)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.03)
    parser.add_argument("--min-image-std", type=float, default=8.0)
    parser.add_argument("--min-image-range", type=int, default=32)
    parser.add_argument("--max-image-clip-fraction", type=float, default=0.85)
    parser.add_argument("--max-bad-image-fraction", type=float, default=0.05)
    parser.add_argument("--strict", action="store_true",
                        help="将 WARN 也视为不通过")
    args = parser.parse_args()
    if args.fps <= 0 or args.min_episode_frames <= 0 or args.image_samples <= 0:
        parser.error("fps、min-episode-frames 和 image-samples 必须为正数")

    try:
        report = validate(args)
    except Exception as exc:
        report = {
            "schema": "am_umi_dataset_quality_gate_v1",
            "dataset": str(Path(args.dataset).expanduser().resolve()),
            "result": "FAIL",
            "issues": [{"level": "FAIL", "code": "read_error",
                        "message": f"无法完整读取数据集：{exc}"}],
        }

    levels = [item["level"] for item in report["issues"]]
    failed = "FAIL" in levels or (args.strict and "WARN" in levels)
    report["result"] = "FAIL" if failed else ("WARN" if "WARN" in levels else "PASS")
    dataset = Path(args.dataset).expanduser().resolve()
    default_report = (dataset / "dataset_quality_report.json" if dataset.is_dir()
                      else dataset.with_name(dataset.name + ".quality_report.json"))
    report_path = Path(args.report).expanduser().resolve() if args.report else default_report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print("AM_UMI_DATASET_QUALITY_" + report["result"])
    print("dataset:", report["dataset"])
    if "frames" in report:
        print(f"episodes: {report['episodes']}; frames: {report['frames']}")
    for issue in report["issues"]:
        print(f"[{issue['level']}] {issue['code']}: {issue['message']}")
    print("report:", report_path)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
