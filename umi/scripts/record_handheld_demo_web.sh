#!/usr/bin/env bash
# Start one hand-held demo with the known-good web preview configuration.
# Usage: bash scripts/record_handheld_demo_web.sh <demo_name> [reference_dir]
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "用法: bash scripts/record_handheld_demo_web.sh <演示名称> [基准目录]"
    echo "示例: bash scripts/record_handheld_demo_web.sh test_demo_001 calibration/robot_wrist_camera/view_references/follow_umi_task_start_v8_with_basket"
    exit 2
fi

cd "$(dirname "$0")/.."
output_dir="data/handheld_demos/$1"
reference_dir="${2:-calibration/robot_wrist_camera/view_references/follow_umi_task_start_v9_centered_elbow}"
if [ -e "$output_dir" ]; then
    echo "拒绝覆盖已存在目录: $output_dir"
    exit 2
fi

# Ordinary demos have no automatic cutoff; end a take with the browser's S button.
exec python -u imu_work/record_handheld_umi_session.py \
    --output-dir "$output_dir" \
    --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
    --imu-port /dev/ttyUSB0 \
    --imu-baud 460800 \
    --max-record-seconds 0 \
    --web-preview \
    --reference-dir "$reference_dir" \
    --monitor-tag-ids 13,14 \
    --motion-tag-map calibration/shared_tags/table_tag_map_v2_13_14.json \
    --live-tcp-monitor \
    --live-arm-ik-monitor
