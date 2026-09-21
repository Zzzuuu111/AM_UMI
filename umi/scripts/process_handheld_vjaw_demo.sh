#!/usr/bin/env bash
# Convert one freshly recorded V-jaw hand-held demo into a policy-ready zarr.
# This is offline only: no serial port is opened and no robot command is sent.
# Usage:
#   bash scripts/process_handheld_vjaw_demo.sh [--resume] <demo_name> <vjaw_reference_dir> [output_zarr]
set -euo pipefail

resume=0
if [ "${1:-}" = "--resume" ]; then
    resume=1
    shift
fi

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "用法: bash scripts/process_handheld_vjaw_demo.sh [--resume] <演示名称> <V型夹爪基准目录> [输出zarr]"
    exit 2
fi

cd "$(dirname "$0")/.."
demo_name="$1"
reference_dir="$2"
session_dir="data/handheld_demos_vjaw/$demo_name"
output_zarr="${3:-data/handheld_demos_vjaw/${demo_name}.zarr}"

if [ ! -f "$session_dir/raw_video.mp4" ]; then
    echo "找不到演示视频: $session_dir/raw_video.mp4"
    exit 2
fi
if [ ! -f "$reference_dir/reference.json" ]; then
    echo "找不到 V 型夹爪 reference.json: $reference_dir/reference.json"
    exit 2
fi
if [ -e "$output_zarr" ]; then
    echo "拒绝覆盖已有最终 zarr: $output_zarr"
    exit 2
fi
if [ "$resume" -eq 0 ]; then
    for output in \
        "$session_dir/tag_detection.pkl" \
        "$session_dir/camera_trajectory_multi_tag_robust.csv" \
        "$session_dir/camera_trajectory_multi_tag_refined.csv"; do
    if [ -e "$output" ]; then
        echo "拒绝覆盖已有结果: $output"
            echo "如需从已有中间结果继续，请显式加 --resume。"
        exit 2
    fi
    done
fi

intrinsics="calibration/handheld_gripper_camera/intrinsics/emeet_handheld_1920x1080_30fps_fisheye_20260831.json"
tag_map="calibration/shared_tags/table_tag_map_v2_13_14.json"
# Formal hand-held labels use the accepted, physically measured fixed-jaw
# inner-front tip. Robot IK later converts this named task point back to the
# URDF right_tcp through an explicit right_tcp→task-TCP transform.
camera_tcp="calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_vjaw_fixed_tcp_v1_accepted.json"
# The hand-held and robot V-jaws share the same 84 mm physical maximum.  The
# Tag-angle mapping is still candidate for intermediate widths, but never emit
# the old impossible 100 mm endpoint into a formal zarr again.
gripper_range="calibration/handheld_gripper_camera/gripper_tag_sessions/vjaw_gripper_range_v3_branch_fixed_84mm_candidate.json"
branch_fixed_detection="tag_detection_vjaw_branch_fixed.pkl"

if [ -e "$session_dir/tag_detection.pkl" ]; then
    echo "[1/5] ArUco 检测：复用已有 tag_detection.pkl"
else
    echo "[1/5] ArUco 检测"
    python -u scripts/detect_aruco.py \
        --input "$session_dir/raw_video.mp4" \
        --output "$session_dir/tag_detection.pkl" \
        --intrinsics_json "$intrinsics" \
        --aruco_yaml calibration/shared_tags/aruco_config.yaml \
        --num_workers 2
fi

if [ -e "$session_dir/camera_trajectory_multi_tag_robust.csv" ]; then
    echo "[2/5] 米制相机轨迹：复用已有 robust CSV"
else
    echo "[2/5] 固定 Tag 13/14 米制相机轨迹"
    python -u imu_work/export_fixed_tag_camera_trajectory.py \
        --session-dir "$session_dir" \
        --intrinsics "$intrinsics" \
        --tag-map "$tag_map" \
        --trajectory-name camera_trajectory_multi_tag_robust.csv \
        --min-visible-ratio 0.95
fi

if [ -e "$session_dir/camera_trajectory_multi_tag_refined.csv" ]; then
    echo "[3/5] 轨迹平滑：复用已有 refined CSV"
else
    echo "[3/5] 短缺口处理与轻量平滑"
    python -u imu_work/refine_fixed_tag_camera_trajectory.py \
        --input "$session_dir/camera_trajectory_multi_tag_robust.csv" \
        --output "$session_dir/camera_trajectory_multi_tag_refined.csv" \
        --max-gap-frames 5 \
        --smooth-window 5
fi

if [ -e "$session_dir/$branch_fixed_detection" ]; then
    echo "[4/5] V 型夹爪 Tag 平面姿态分支：复用已有修复 pkl"
else
    echo "[4/5] V 型夹爪 Tag 平面姿态分支修复"
    python -u scripts/fix_vjaw_tag_pose_branches.py \
        --input "$session_dir/tag_detection.pkl" \
        --output "$session_dir/$branch_fixed_detection" \
        --intrinsics "$intrinsics" \
        --aruco-yaml calibration/shared_tags/aruco_config.yaml \
        --fixed-tag-id 0 \
        --moving-tag-id 1 \
        --max-relative-angle-deg 60
fi

echo "[5/5] V 型夹爪 zarr 转换"
python -u scripts/convert_handheld_fixed_tag_demo.py \
    --session-dir "$session_dir" \
    --output "$output_zarr" \
    --trajectory-name camera_trajectory_multi_tag_refined.csv \
    --tag-detection-name "$branch_fixed_detection" \
    --camera-tcp-geometry "$camera_tcp" \
    --gripper-range "$gripper_range" \
    --reference "$reference_dir/reference.json" \
    --gripper-max-m 0.084 \
    --gripper-smooth-window-frames 9

echo "VJAW_HANDHELD_DEMO_PROCESSING_OK"
echo "zarr: $output_zarr"
