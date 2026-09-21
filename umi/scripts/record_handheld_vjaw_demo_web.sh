#!/usr/bin/env bash
# Record one visual-only V-jaw handheld demonstration.
# Usage: bash scripts/record_handheld_vjaw_demo_web.sh <demo_name> <vjaw_reference_dir>
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "用法: bash scripts/record_handheld_vjaw_demo_web.sh <演示名称> <V型夹爪基准目录>"
    exit 2
fi

cd "$(dirname "$0")/.."
output_dir="data/handheld_demos_vjaw/$1"
reference_dir="$2"
if [ -e "$output_dir" ]; then
    echo "拒绝覆盖已存在目录: $output_dir"
    exit 2
fi
if [ ! -f "$reference_dir/reference.json" ]; then
    echo "找不到 V 型夹爪 reference.json: $reference_dir/reference.json"
    exit 2
fi

exec python -u imu_work/record_handheld_umi_session.py \
    --output-dir "$output_dir" \
    --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
    --no-imu \
    --max-record-seconds 0 \
    --web-preview \
    --web-preview-update-hz 10 \
    --web-recording-preview-update-hz 4 \
    --recording-capture-priority \
    --recording-monitor-update-hz 5 \
    --video-encoder-preset ultrafast \
    --reference-dir "$reference_dir" \
    --alignment-robot-config example/eval_robots_config_vjaw_ros2.yaml \
    --monitor-tag-ids 13,14 \
    --motion-tag-map calibration/shared_tags/table_tag_map_v2_13_14.json \
    --motion-camera-tcp-geometry calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_vjaw_ros2_right_tcp_v1_candidate.json \
    --allow-candidate-motion-tcp \
    --live-tcp-monitor \
    --live-arm-ik-monitor
