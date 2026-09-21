#!/usr/bin/env bash
# Original-amplitude, original-timing AM2Pro replay.
# The saved V9 reference is always reached first in the same torque-enabled
# process.  It retains only outlier guards; those do not time-scale nominal
# demo motion, but prevent a corrupt one-frame trajectory spike from being
# sent to the arm.
#
# Usage:
#   bash scripts/replay_v9_auto_raw_motion.sh <demo.zarr> [frames]
# Pass 180 for the first 6 seconds of the recorded motion; pass 1031 only
# after that direction test is confirmed safe.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "用法: bash scripts/replay_v9_auto_raw_motion.sh <demo.zarr> [回放帧数]" >&2
    exit 2
fi

dataset="$1"
max_frames="${2:-180}"

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
    --speed 1.0 \
    --scale 1.0 \
    --max-tcp-speed 0.06 \
    --max-tcp-rot-speed-deg 30 \
    --max-frames "$max_frames" \
    --hold-gripper \
    --return \
    --ik-iterations 1 \
    --wrist-flex-min-deg 20 \
    --wrist-flex-max-deg 70
