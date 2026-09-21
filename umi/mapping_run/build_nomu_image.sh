#!/bin/bash
# ============================================================================
# 构建自定义 ORB-SLAM3 docker 镜像（去掉"IMU 必须初始化"约束）
#
# 背景：DJI Osmo 4 的合成 IMU（纯重力加速度）无法稳定完成 ORB-SLAM3 的
# IMU 初始化，导致地图反复被重置。本镜像打上 Tracking.cc 补丁：
#   - 跟踪阈值不再因 IMU 未初始化而提高（统一 15 内点）
#   - IMU 未初始化时丢跟踪不再重置地图（尺度由 tag 13 初始化提供）
#
# 构建时长：约 20~60 分钟（Pangolin + ORB-SLAM3 全量编译）
# 用法：
#   bash mapping_run/build_nomu_image.sh
# ============================================================================
set -e

REPO=/home/zzzjh/universal_manipulation_interface
SRC=$REPO/ORB_SLAM3_umi
IMAGE=umi_orb_slam3_nomu:latest

if [ ! -d "$SRC/src" ]; then
    echo "❌ 源码目录不存在: $SRC"
    exit 1
fi

# 检查 Pangolin 子模块（浅克隆不会自动拉子模块，缺失会导致 Step 6 构建失败）
if [ ! -f "$SRC/Thirdparty/Pangolin/scripts/install_prerequisites.sh" ]; then
    echo "❌ 缺少 Pangolin 子模块（Step 6 会失败）。"
    echo "   修复（GitHub 被墙时用 ghfast.top 镜像）："
    echo "   cd $SRC"
    echo "   rm -rf Thirdparty/Pangolin"
    echo "   git clone --depth 50 https://ghfast.top/https://github.com/stevenlovegrove/Pangolin.git Thirdparty/Pangolin"
    echo "   cd Thirdparty/Pangolin && git fetch --depth 1 origin d484494645cb7361374ac0ef6b27e9ee6feffbd7 && git checkout d484494645cb7361374ac0ef6b27e9ee6feffbd7"
    exit 1
fi

echo "=== 构建镜像 $IMAGE（源码: $SRC）==="
echo "    （-j2 编译 + 9GB 内存上限 + 分步缓存：失败不用重头编译）"
cd "$SRC"
docker build --memory 9g -t "$IMAGE" .

echo ""
echo "✅ 构建完成。验证:"
echo "   docker images | grep umi_orb_slam3_nomu"
echo ""
echo "后续：run_mapping.sh 会自动优先使用该镜像。"
