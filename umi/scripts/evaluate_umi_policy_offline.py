#!/usr/bin/env python3
"""Evaluate a UMI policy checkpoint against recorded Zarr trajectories offline.

This tool only reads a checkpoint and a dataset.  It does not create cameras,
controllers, serial ports, or robot commands.  Predictions and targets are in
the policy's relative action frame: position (m), 6D orientation, and gripper
width (m).
"""

import argparse
import json
import os
import pathlib
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Register the resolver used by saved UMI configs before torch.load restores
# their OmegaConf objects.
import diffusion_policy.workspace.train_diffusion_transformer_timm_workspace  # noqa: E402,F401
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from umi.common.pose_util import pose10d_to_mat  # noqa: E402


def parse_indices(text: str | None, dataset_length: int, count: int) -> list[int]:
    if text:
        indices = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        count = min(count, dataset_length)
        indices = np.linspace(0, dataset_length - 1, count, dtype=int).tolist()
    if not indices:
        raise ValueError("至少需要一个评估样本")
    invalid = [index for index in indices if index < 0 or index >= dataset_length]
    if invalid:
        raise ValueError(f"样本索引超出范围 [0, {dataset_length - 1}]: {invalid}")
    return indices


def rotation_error_deg(pred_pose10d: np.ndarray, target_pose10d: np.ndarray) -> np.ndarray:
    """Geodesic rotation error for 6D action rotations, in degrees."""
    pred_rotation = pose10d_to_mat(pred_pose10d)[..., :3, :3]
    target_rotation = pose10d_to_mat(target_pose10d)[..., :3, :3]
    relative = pred_rotation @ np.swapaxes(target_rotation, -1, -2)
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
                     -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def set_equal_3d_axes(axis, points: np.ndarray) -> None:
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 0.005)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plot_comparison(target: np.ndarray, prediction: np.ndarray, sample: dict,
                    output: pathlib.Path) -> None:
    target_pos_mm = target[:, :3] * 1000.0
    prediction_pos_mm = prediction[:, :3] * 1000.0
    figure = plt.figure(figsize=(13, 9))
    axis_3d = figure.add_subplot(2, 2, 1, projection="3d")
    axis_xy = figure.add_subplot(2, 2, 2)
    axis_xz = figure.add_subplot(2, 2, 3)
    axis_width = figure.add_subplot(2, 2, 4)

    for axis in (axis_3d, axis_xy, axis_xz):
        if axis is axis_3d:
            axis.plot(*target_pos_mm.T, "o-", color="#2878b5", label="recorded target")
            axis.plot(*prediction_pos_mm.T, "o-", color="#e07a2d", label="policy prediction")
            axis.set_xlabel("X (mm)")
            axis.set_ylabel("Y (mm)")
            axis.set_zlabel("Z (mm)")
            set_equal_3d_axes(axis, np.vstack((target_pos_mm, prediction_pos_mm)))
        elif axis is axis_xy:
            axis.plot(target_pos_mm[:, 0], target_pos_mm[:, 1], "o-", color="#2878b5")
            axis.plot(prediction_pos_mm[:, 0], prediction_pos_mm[:, 1], "o-", color="#e07a2d")
            axis.set_xlabel("X (mm)")
            axis.set_ylabel("Y (mm)")
            axis.axis("equal")
            axis.set_title("XY")
        else:
            axis.plot(target_pos_mm[:, 0], target_pos_mm[:, 2], "o-", color="#2878b5")
            axis.plot(prediction_pos_mm[:, 0], prediction_pos_mm[:, 2], "o-", color="#e07a2d")
            axis.set_xlabel("X (mm)")
            axis.set_ylabel("Z (mm)")
            axis.axis("equal")
            axis.set_title("XZ")
        axis.grid(True, alpha=0.3)

    steps = np.arange(len(target))
    axis_width.plot(steps, target[:, 9] * 1000.0, "o-", color="#2878b5", label="recorded target")
    axis_width.plot(steps, prediction[:, 9] * 1000.0, "o-", color="#e07a2d", label="policy prediction")
    axis_width.set_xlabel("prediction horizon step")
    axis_width.set_ylabel("gripper width (mm)")
    axis_width.grid(True, alpha=0.3)
    axis_width.legend(loc="best")
    axis_3d.legend(loc="best")
    figure.suptitle(
        "Offline policy evaluation — sample {sample_index}; "
        "position RMSE {position_rmse_mm:.2f} mm, rotation MAE {rotation_mae_deg:.2f} deg, "
        "gripper RMSE {gripper_rmse_mm:.2f} mm".format(**sample),
        fontsize=11, y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMI checkpoint 的只读离线轨迹评估；不会连接机械臂"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--samples", type=int, default=8,
                        help="均匀抽取的样本数，默认 8")
    parser.add_argument("--split", choices=("train", "val"), default="val",
                        help="评估训练或保留验证划分；默认 val")
    parser.add_argument("--sample-indices",
                        help="逗号分隔的 dataset sample 索引；提供时覆盖 --samples")
    parser.add_argument("--seed", type=int, default=42,
                        help="每个扩散采样的可复现实验种子")
    parser.add_argument("--device", default="cuda:0",
                        help="cuda:0 或 cpu；默认使用 GPU")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples 必须为正数")

    checkpoint_path = pathlib.Path(args.checkpoint).expanduser().resolve()
    dataset_path = pathlib.Path(args.dataset).expanduser().resolve()
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    if not checkpoint_path.is_file():
        parser.error(f"checkpoint 不存在: {checkpoint_path}")
    if not dataset_path.exists():
        parser.error(f"dataset 不存在: {dataset_path}")
    if out_dir.exists():
        parser.error(f"拒绝覆盖已有输出目录: {out_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("请求 CUDA，但当前环境未检测到 CUDA；可改用 --device cpu")

    payload = torch.load(checkpoint_path, map_location="cpu", mmap=True,
                         weights_only=False)
    cfg = payload["cfg"]
    OmegaConf.update(cfg, "task.dataset_path", str(dataset_path), merge=False)
    OmegaConf.update(cfg, "task.dataset.dataset_path", str(dataset_path), merge=False)

    policy = hydra.utils.instantiate(cfg.policy)
    state_dicts = payload["state_dicts"]
    state_name = "ema_model" if cfg.training.use_ema and "ema_model" in state_dicts else "model"
    policy.load_state_dict(state_dicts[state_name], strict=True)
    del payload

    train_dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataset = train_dataset if args.split == "train" else train_dataset.get_validation_dataset()
    # The physical inference path supplies an exact episode-start pose; it
    # does not add the training-only augmentation used by UmiDataset.
    if hasattr(dataset, "episode_start_pose_noise_std"):
        dataset.episode_start_pose_noise_std = 0.0
    indices = parse_indices(args.sample_indices, len(dataset), args.samples)
    device = torch.device(args.device)
    policy.to(device).eval()
    out_dir.mkdir(parents=True)

    records = []
    examples = []
    for ordinal, index in enumerate(indices):
        item = dataset[index]
        batch = default_collate([item])
        observation = dict_apply(batch["obs"], lambda value: value.to(device))
        target = batch["action"][0].numpy().astype(np.float64)
        torch.manual_seed(args.seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + index)
        with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda"):
            result = policy.predict_action(observation)
        prediction = result["action_pred"][0].detach().cpu().numpy().astype(np.float64)
        if prediction.shape != target.shape:
            raise RuntimeError(f"sample {index}: 预测形状 {prediction.shape} 不等于目标 {target.shape}")
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"sample {index}: 策略输出包含 NaN/Inf")

        position_error_mm = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1) * 1000.0
        rotation_error = rotation_error_deg(prediction[:, :9], target[:, :9])
        gripper_error_mm = np.abs(prediction[:, 9] - target[:, 9]) * 1000.0
        record = {
            "sample_index": int(index),
            "position_rmse_mm": float(np.sqrt(np.mean(position_error_mm ** 2))),
            "position_mae_mm": float(np.mean(position_error_mm)),
            "rotation_mae_deg": float(np.mean(rotation_error)),
            "gripper_rmse_mm": float(np.sqrt(np.mean(gripper_error_mm ** 2))),
            "gripper_mae_mm": float(np.mean(gripper_error_mm)),
        }
        records.append(record)
        examples.append((target, prediction, record))
        print("sample {}/{}: index={} position_rmse={:.2f}mm rotation_mae={:.2f}deg gripper_rmse={:.2f}mm".format(
            ordinal + 1, len(indices), index, record["position_rmse_mm"],
            record["rotation_mae_deg"], record["gripper_rmse_mm"]), flush=True)

    median_example = sorted(examples, key=lambda example: example[2]["position_rmse_mm"])[len(examples) // 2]
    plot_path = out_dir / "median_position_error.trajectory.png"
    plot_comparison(*median_example, plot_path)
    summary = {
        "schema": "am_umi_policy_offline_evaluation_v1",
        "safety": "offline only; no camera, serial port, controller, or robot command was created",
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(device),
        "split": args.split,
        "episode_start_pose_noise_std": 0.0,
        "samples": records,
        "aggregate": {
            "position_rmse_mm_median": float(np.median([r["position_rmse_mm"] for r in records])),
            "position_rmse_mm_mean": float(np.mean([r["position_rmse_mm"] for r in records])),
            "rotation_mae_deg_median": float(np.median([r["rotation_mae_deg"] for r in records])),
            "gripper_rmse_mm_median": float(np.median([r["gripper_rmse_mm"] for r in records])),
        },
        "comparison_plot": str(plot_path),
        "plotted_sample": median_example[2]["sample_index"],
    }
    report_path = out_dir / "offline_policy_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AM_UMI_OFFLINE_POLICY_EVALUATION_OK")
    print("report:", report_path)
    print("plot:", plot_path)
    print("aggregate:", json.dumps(summary["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
