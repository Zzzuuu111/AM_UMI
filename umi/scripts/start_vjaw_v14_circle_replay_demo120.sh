#!/usr/bin/env bash
# Supervised empty-gripper replay of one accepted V14 circle demonstration.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--full" && "$1" != "--preview" ) ]]; then
    echo "用法: bash scripts/start_vjaw_v14_circle_replay_demo120.sh [--preview | --full]" >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

plan="data/handheld_demos_vjaw/v14_circle30_20261013/vjaw_demo_120_v14_circle_030/fixed_tip_constrained_lookahead_plan.json"
reference="calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json"
if [[ ! -f "$plan" || ! -f "$reference" ]]; then
    echo "缺少已筛选的关节计划或 v6 起始基准" >&2
    exit 2
fi

mode="short"
mode_args=(--max-source-frames 120)
if [[ "${1:-}" == "--full" ]]; then
    mode="full"
    mode_args=(--full-plan)
fi
if [[ "${1:-}" == "--preview" ]]; then
    mode_args+=(--preview-only)
else
    mode_args+=(--pause-after-reference)
fi

trace="data/outputs/vjaw_v14_circle_demo120_replay_${mode}_$(date +%Y%m%d_%H%M%S).json"
echo "演示: vjaw_demo_120_v14_circle_030；模式: $mode；J7 连续开合已启用"
if [[ "${1:-}" != "--preview" ]]; then
    echo "空载监督回放；低速回到 v6 基准后暂停，确认路径清空再按 Enter。"
    echo "进入回放后，完成或 Ctrl+C 中断会尝试返回基准；保持人在机械臂旁可急停。"
    echo "轨迹读回: $trace"
fi

args=(
    --joint-plan "$plan"
    --start-reference "$reference"
    --execution-mode adaptive
    --enable-gripper
    --gripper-mode continuous
    "${mode_args[@]}"
)
if [[ "${1:-}" != "--preview" ]]; then
    args+=(--trace-out "$trace" --plot-after-replay)
fi
python -u scripts/am2pro_replay_retarget_plan.py "${args[@]}"
