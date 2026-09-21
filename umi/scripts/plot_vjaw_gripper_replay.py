#!/usr/bin/env python3
"""Plot source J7 width labels against replay command and AM2Pro readback.

The replay trace is recorded on the adaptive replay wall-clock.  Each trace
sample includes its corresponding fractional source-frame position, so this
plot puts the source visual label, J7 command and J7 encoder readback on the
same horizontal time axis even when adaptive replay slows down.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.filter_am2pro_replayability import open_replay_buffer


DEMO_COLOR = "#1f77b4"
COMMAND_COLOR = "#ff7f0e"
ACTUAL_COLOR = "#98df8a"


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="绘制 demo J7 标签、replay J7 命令与实际回读开口宽度。")
    parser.add_argument("--plan", required=True, help="用于 replay 的重定向计划 JSON")
    parser.add_argument("--trace", required=True, help="带 J7 字段的 adaptive replay trace JSON")
    parser.add_argument("--out", required=True, help="输出 PNG；拒绝覆盖")
    parser.add_argument("--report", default=None, help="可选 JSON 数值报告")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    if out.exists():
        parser.error(f"拒绝覆盖已有图片: {out}")
    trace_path = Path(args.trace).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("schema") != "am_umi_vjaw_adaptive_replay_trace_v1":
        parser.error("不是受支持的 adaptive replay trace")
    samples = trace.get("samples", [])
    required_fields = {"elapsed_s", "source_frame_progress", "actual_gripper_width_m",
                       "gripper_command_width_m"}
    if not samples or not required_fields.issubset(samples[0]):
        parser.error("此 trace 未记录 J7 数据；请用更新后的 replay 脚本重新运行并加入 --trace-out")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dataset_path = resolve_project_path(str(plan.get("dataset", "")))
    if not dataset_path.exists():
        parser.error(f"计划中的 dataset 不存在: {dataset_path}")
    replay, store = open_replay_buffer(dataset_path)
    try:
        episode = replay.get_episode(int(plan.get("episode", 0)))
        source_widths_m = np.asarray(episode["robot0_gripper_width"], dtype=float).reshape(-1)
    finally:
        if store is not None:
            store.close()
    if not len(source_widths_m) or not np.isfinite(source_widths_m).all():
        parser.error("dataset 缺少有效 robot0_gripper_width 标签")

    elapsed_s, source_progress, command_m, actual_m = [], [], [], []
    for sample in samples:
        try:
            elapsed = float(sample["elapsed_s"])
            source_frame = float(sample["source_frame_progress"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite([elapsed, source_frame]).all():
            continue
        elapsed_s.append(elapsed)
        source_progress.append(np.clip(source_frame, 0.0, len(source_widths_m) - 1.0))
        command = sample.get("gripper_command_width_m")
        actual = sample.get("actual_gripper_width_m")
        command_m.append(float(command) if command is not None else np.nan)
        actual_m.append(float(actual) if actual is not None else np.nan)
    elapsed_s = np.asarray(elapsed_s, dtype=float)
    source_progress = np.asarray(source_progress, dtype=float)
    command_m = np.asarray(command_m, dtype=float)
    actual_m = np.asarray(actual_m, dtype=float)
    if not len(elapsed_s):
        parser.error("trace 中没有有效 J7 时间样本")
    demo_m = np.interp(source_progress, np.arange(len(source_widths_m), dtype=float), source_widths_m)

    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    axis.plot(elapsed_s, demo_m * 1000.0, color=DEMO_COLOR, lw=2,
              label="hand-held demo width label")
    if np.isfinite(command_m).any():
        axis.plot(elapsed_s, command_m * 1000.0, color=COMMAND_COLOR, lw=1.8,
                  label="replay J7 command")
    if np.isfinite(actual_m).any():
        axis.plot(elapsed_s, actual_m * 1000.0, color=ACTUAL_COLOR, lw=1.8,
                  label="AM2Pro J7 readback")
    axis.set_title("V-jaw opening during adaptive replay")
    axis.set_xlabel("replay elapsed time (s)")
    axis.set_ylabel("opening width (mm)")
    axis.grid(alpha=.3)
    axis.legend(loc="best")
    fig.text(.5, .01,
             "Blue is a visual hand-held label; orange is the commanded width; light green is the gripper encoder readback.",
             ha="center", fontsize=8, color="0.30")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)

    paired = np.isfinite(command_m) & np.isfinite(actual_m)
    report = {
        "schema": "am_umi_vjaw_gripper_replay_comparison_v1",
        "plan": str(plan_path),
        "trace": str(trace_path),
        "dataset": str(dataset_path),
        "trace_status": trace.get("status"),
        "samples": int(len(elapsed_s)),
        "demo_label_range_mm": [float(np.min(demo_m) * 1000.0), float(np.max(demo_m) * 1000.0)],
        "command_readback_abs_error_mm": (
            {"median": float(np.median(np.abs(command_m[paired] - actual_m[paired])) * 1000.0),
             "max": float(np.max(np.abs(command_m[paired] - actual_m[paired])) * 1000.0)}
            if np.any(paired) else None),
        "limitations": [
            "The blue source width is a visual V-jaw label, not force/contact sensing.",
            "The light-green trace is the AM2Pro gripper encoder readback, not an external physical caliper measurement.",
        ],
    }
    report_path = Path(args.report).expanduser().resolve() if args.report else out.with_suffix(".json")
    if report_path.exists():
        parser.error(f"拒绝覆盖已有报告: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_GRIPPER_REPLAY_COMPARISON_OK")
    print("plot:", out)
    print("report:", report_path)
    if report["command_readback_abs_error_mm"]:
        errors = report["command_readback_abs_error_mm"]
        print(f"J7 command-readback median/max: {errors['median']:.2f}/{errors['max']:.2f} mm")


if __name__ == "__main__":
    main()
