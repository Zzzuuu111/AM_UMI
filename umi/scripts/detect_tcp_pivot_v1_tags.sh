#!/usr/bin/env bash
# Extract ArUco poses for the fixed-point TCP pivot recording; no hardware use.
set -euo pipefail

cd "$(dirname "$0")/.."
python -u scripts/detect_aruco.py \
  --input calibration/handheld_gripper_camera/gripper_tag_sessions/tcp_pivot_v1/raw_video.mp4 \
  --output calibration/handheld_gripper_camera/gripper_tag_sessions/tcp_pivot_v1/tag_detection.pkl \
  --intrinsics_json calibration/handheld_gripper_camera/intrinsics/emeet_handheld_1920x1080_30fps_fisheye_20260831.json \
  --aruco_yaml calibration/shared_tags/aruco_config.yaml \
  --num_workers 2
