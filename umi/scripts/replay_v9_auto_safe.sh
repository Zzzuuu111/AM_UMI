#!/usr/bin/env bash
# Safe AM2Pro demonstration replay.
#
# Always performs, in one uninterrupted controller session:
#   1. low-speed move to the V9 saved reference;
#   2. verify the joint readback only after that move has completed;
#   3. replay a short, reduced-amplitude prefix while holding the gripper;
#   4. return to V9.
#
# Usage:
#   bash scripts/replay_v9_auto_safe.sh data/handheld_demos/<demo>.zarr [frames] [scale]
#
# `frames` defaults to 90.  At --speed 0.2, that is roughly 15 seconds of
# physical time, which is long enough to see the motion but remains a first
# direction test rather than a full task execution. `scale` defaults to 0.3;
# pass 1.0 to retain the recorded translation amplitude.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "用法: bash scripts/replay_v9_auto_safe.sh <demo.zarr> [回放帧数] [平移倍率]" >&2
    exit 2
fi

dataset="$1"
max_frames="${2:-90}"
scale="${3:-0.3}"

if [[ ! -d "$dataset" ]]; then
    echo "找不到 zarr 数据集目录: $dataset" >&2
    exit 2
fi

python -u scripts/am2pro_replay_episode.py \
    -rc example/eval_robots_config.yaml \
    --dataset "$dataset" \
    --episode 0 \
    --fps 30 \
    --source-fps 30 \
    --start-reference calibration/robot_wrist_camera/view_references/follow_umi_task_start_v9_centered_elbow/reference.json \
    --move-to-reference \
    --reference-joint-duration 14 \
    --reference-max-joint-speed-deg-s 6 \
    --reference-tolerance-deg 4 \
    --speed 0.2 \
    --scale "$scale" \
    --max-tcp-speed 0.01 \
    --max-tcp-rot-speed-deg 5 \
    --max-frames "$max_frames" \
    --hold-gripper \
    --return \
    --ik-iterations 1 \
    --wrist-flex-min-deg 20 \
    --wrist-flex-max-deg 70
