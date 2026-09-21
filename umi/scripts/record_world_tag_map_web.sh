#!/usr/bin/env bash
# Usage: bash scripts/record_world_tag_map_web.sh <session_name>
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "用法: bash scripts/record_world_tag_map_web.sh <session_name>"
    echo "示例: bash scripts/record_world_tag_map_web.sh table_tags_v1"
    exit 2
fi

cd "$(dirname "$0")/.."
output_dir="calibration/shared_tags/world_map_sessions/$1"
if [ -e "$output_dir" ]; then
    echo "拒绝覆盖已存在目录: $output_dir"
    exit 2
fi

exec python -u imu_work/record_handheld_umi_session.py \
    --output-dir "$output_dir" \
    --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
    --imu-port /dev/ttyUSB0 \
    --imu-baud 460800 \
    --max-record-seconds 20 \
    --web-preview \
    --monitor-tag-ids 13,14
