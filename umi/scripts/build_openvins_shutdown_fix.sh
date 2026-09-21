#!/usr/bin/env bash
# Incremental isolated build: no packages installed in the host environment.
set -euo pipefail
openvins_task_root="$(cd "$(dirname "$0")/.." && pwd)"
openvins_builder_name="am_umi_openvins_shutdown_builder_$(date +%Y%m%d_%H%M%S)"
docker run --name "$openvins_builder_name" --network none --cpus 2 \
  --memory 4g --memory-swap 4g \
  -v "$openvins_task_root/third_party/open_vins:/workspace/src/open_vins:ro" \
  am_umi_openvins_built:latest /bin/bash -c \
  'source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && cmake --build /workspace/build/ov_msckf --target run_subscribe_msckf -- -j1 && cmake --install /workspace/build/ov_msckf'
docker commit "$openvins_builder_name" am_umi_openvins_offline:20260905
printf '隔离修复镜像已生成: am_umi_openvins_offline:20260905\n构建容器（已停止，保留诊断）: %s\n' "$openvins_builder_name"
