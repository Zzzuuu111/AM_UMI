#!/usr/bin/env bash
# Second fixed-point TCP attempt. Use a small ball seated in a shallow dimple.
set -euo pipefail

cd "$(dirname "$0")/.."
python -u imu_work/record_handheld_umi_session.py \
  --output-dir calibration/handheld_gripper_camera/gripper_tag_sessions/tcp_pivot_v2 \
  --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
  --imu-port /dev/ttyUSB0 \
  --imu-baud 460800 \
  --max-record-seconds 45 \
  --web-preview \
  --monitor-tag-id 13 \
  --guided-tcp-pivot
