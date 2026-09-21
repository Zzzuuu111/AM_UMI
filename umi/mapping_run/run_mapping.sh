#!/bin/bash
# ============================================================================
# 历史 Osmo Action 4 建图一键脚本
#
# 功能：给定录制视频 + Gyroflow CSV，自动完成：
#   1. 搭 session 目录结构
#   2. 视频拷贝 + IMU 数据生成（gyroflow_csv_to_imu_json.py）
#   3. SLAM 建图（02_create_map.py，docker）
#   4. ArUco tag 检测（04_detect_aruco.py）
#   5. SLAM-tag 手眼标定（05_run_calibrations.py，可选）
#   6. 结果摘要
#
# 用法：
#   conda activate AM_UMI
#   bash mapping_run/run_mapping.sh [视频.mp4] [CSV.csv]
#   （不带参数时用默认的 0014 视频 + Wide3.csv）
# ============================================================================
set -e

REPO=/home/zzzjh/AM_UMI/umi
PYTHON=/home/zzzjh/anaconda3/envs/AM_UMI/bin/python
# This script is retained only for historical Osmo video. Do not use it for
# the new 1920x1080 EMEET handheld camera until EMEET SLAM settings exist.
INTRINSICS=$REPO/calibration/archive/legacy_handheld_cameras/osmo4_intrinsics_wide.json
ARUCO_YAML=$REPO/calibration/shared_tags/aruco_config.yaml

VIDEO="${1:-/home/zzzjh/Osmo Action 4/DJI_20260824104053_0014_D.MP4}"
CSV="${2:-/home/zzzjh/Osmo Action 4/Osma Action 4 Wide3.csv}"

if [ ! -f "$VIDEO" ]; then echo "❌ 视频不存在: $VIDEO"; exit 1; fi
if [ ! -f "$CSV" ];   then echo "❌ CSV 不存在: $CSV";   exit 1; fi

# 每次运行用新 session（带时间戳），不会覆盖上次结果
SESSION=$REPO/mapping_run/session_$(date +%m%d_%H%M%S)
MAPPING_DIR=$SESSION/demos/mapping
mkdir -p "$MAPPING_DIR" "$SESSION/calibration"

echo "============================================================"
echo " 视频: $VIDEO"
echo " CSV:  $CSV"
echo " 输出: $SESSION"
echo "============================================================"

# ---- 1. 视频 + 内参 ----
echo "[1/5] 拷贝视频 + 内参 ..."
cp "$VIDEO" "$MAPPING_DIR/raw_video.mp4"
cp "$INTRINSICS" "$SESSION/calibration/"
cp "$ARUCO_YAML" "$SESSION/calibration/"

# ---- 2. IMU 数据 ----
echo "[2/5] 生成 imu_data.json ..."
$PYTHON $REPO/scripts/gyroflow_csv_to_imu_json.py \
    -i "$CSV" -o "$MAPPING_DIR/imu_data.json"

# ---- 3. SLAM 建图（docker）----
echo "[3/5] SLAM 建图（docker, 约 2~5 分钟）..."
# 优先用自定义镜像（免 IMU 初始化，见 build_nomu_image.sh）；没有则回退官方
if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^umi_orb_slam3_nomu:latest$"; then
    DOCKER_IMAGE=umi_orb_slam3_nomu:latest
    echo "    使用自定义镜像: $DOCKER_IMAGE"
else
    DOCKER_IMAGE=chicheng/orb_slam3:latest
    echo "    ⚠️ 未找到自定义镜像，回退官方 $DOCKER_IMAGE（可能受 IMU 初始化影响）"
fi
$PYTHON $REPO/scripts_slam_pipeline/02_create_map.py \
    -i "$MAPPING_DIR" -d "$DOCKER_IMAGE" -np

# ---- 4. tag 检测 ----
echo "[4/5] ArUco tag 检测（约 5 分钟）..."
$PYTHON $REPO/scripts_slam_pipeline/04_detect_aruco.py \
    -i "$SESSION/demos" \
    -ci "$INTRINSICS" \
    -ac "$ARUCO_YAML" \
    -n 4

# ---- 5. 手眼标定（可选，失败不中断）----
echo "[5/5] SLAM-tag 手眼标定 ..."
$PYTHON $REPO/scripts_slam_pipeline/05_run_calibrations.py "$SESSION" \
    && echo "    标定成功" || echo "    ⚠️ 标定未完成（不影响建图结果）"

# ---- 摘要 ----
echo ""
echo "============================================================"
echo " 结果摘要"
echo "============================================================"
echo "session: $SESSION"
echo ""
echo "[SLAM] 地图信息（找 'There are X maps'）:"
grep -E "There are|Map 0 has|KeyFrame" "$MAPPING_DIR/slam_stdout.txt" | tail -3 || true
echo ""
echo "[SLAM] 轨迹长度（非 0 的行数 = 跟踪成功的帧数）:"
$PYTHON - <<EOF
import csv
p = "$MAPPING_DIR/mapping_camera_trajectory.csv"
try:
    with open(p) as f:
        rows = list(csv.DictReader(f))
    tracked = sum(1 for r in rows if r.get('is_lost') == 'false')
    print(f"  总帧 {len(rows)}, 跟踪成功 {tracked} 帧 ({tracked*100//max(len(rows),1)}%)")
except Exception as e:
    print("  轨迹文件读取失败:", e)
EOF
echo ""
echo "[tag 检测]:"
$PYTHON - <<EOF
import pickle
p = "$MAPPING_DIR/tag_detection.pkl"
try:
    with open(p, 'rb') as f:
        d = pickle.load(f)
    n = sum(1 for e in d if isinstance(e, dict) and e.get('tag_dict'))
    print(f"  检测到 tag 的帧数: {n} / {len(d)}")
except Exception as e:
    print("  读取失败:", e)
EOF
echo ""
echo "完成！后续步骤：录 demo 数据（tag 可以收起来了）"
