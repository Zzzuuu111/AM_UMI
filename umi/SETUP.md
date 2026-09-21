# 新机环境搭建清单（从零到可用）

> 目标：一台新 Ubuntu 22.04 电脑，拉齐本项目（UMI × Osmo Action 4 × AM2Pro）全部代码与环境。
> 项目档案见 `PROJECT_PROGRESS.md`。最后更新：2026-08-25。

## 0. 系统基础（一次性）

- Ubuntu 22.04 + [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- docker 安装并把用户加入 `docker` 组；国内网络需配镜像源 `/etc/docker/daemon.json`（从老机器抄）
- NVIDIA GPU + 驱动 + CUDA 12.x：**仅训练需要**；采集/建图/机器人控制不需要
- SSH 公钥添加到 GitHub（拉私有仓库用）：
  ```bash
  ssh-keygen -t ed25519
  cat ~/.ssh/id_ed25519.pub   # 粘贴到 https://github.com/settings/ssh/new
  ```

## 1. 拉代码（3 个仓库）

```bash
# 主仓库（私有）：全部管线适配 + AM2Pro 控制器 + 文档 + 标定
git clone git@github.com:Zzzuuu111/umi-ampro2.git

# ORB_SLAM3 补丁（公开）：IMU 初始化开关等 4 个 commit
git clone https://github.com/Zzzuuu111/ORB_SLAM3.git umi-ampro2/ORB_SLAM3_umi

# lerobot AM2Pro 版（公开第三方）+ 本地配置补丁
git clone https://github.com/liyiteng/lerobot_alohamini.git
cd lerobot_alohamini && git apply ../umi-ampro2/patches/lerobot_config_alohamini.patch && cd ..
```

## 2. conda 环境（3 个 yaml 都在主仓库里）

```bash
conda env create -f umi-ampro2/environment_umi_full.yml              # umi（py3.9，主环境）
conda env create -f umi-ampro2/environment_lerobot_alohamini.yml     # lerobot_alohamini（py3.12，硬件服务器）
conda env create -f umi-ampro2/environment_lerobot.yml               # lerobot（通用，可选）

# lerobot 本体还需要 editable 安装（在 lerobot_alohamini 环境里）
conda activate lerobot_alohamini && cd lerobot_alohamini && pip install -e . && cd ..

# exiftool perl 修复（老机器踩过的坑，见 PROJECT_PROGRESS §4）
ln -sfn ~/anaconda3/envs/umi/lib/perl5/site_perl ~/anaconda3/envs/umi/bin/lib
```

## 3. Docker

```bash
docker pull chicheng/orb_slam3:latest        # 官方镜像
# 自定义镜像 umi_orb_slam3_nomu（含 IMU 初始化开关补丁），二选一：
#   A（推荐）从脚本重建，自动处理 Pangolin 子模块和 make -j4：
bash umi-ampro2/mapping_run/build_nomu_image.sh
#   B 老机 docker save → 新机 docker load（省编译时间，需大 U 盘/局域网）
```

## 4. 第三方工具

```bash
sudo apt install libc++1 libc++abi1          # Gyroflow 依赖
# Gyroflow：下载 Linux 版解压到 ~/Gyroflow（老机上 ~/Gyroflow 整个拷贝也行）
```

## 5. 不在 git 里的大文件（按需拷贝，U 盘 / rsync 局域网）

| 内容 | 大小 | 用途 |
|---|---|---|
| `example_demo_session/`（官方样例视频） | 731MB | 可选 |
| `example_slam_test/demos/mapping/raw_video.mp4` | 530MB | 复现建图 |
| `mapping_run/session_0824_104746/demos/mapping/raw_video.mp4` | 526MB | 复现建图 |
| 标定/测试原始视频 `DJI_2026*.MP4`（本地视频目录） | ~1GB | 复现标定 |
| Gyroflow 导出的 CSV | 小 | IMU 管线输入 |

```bash
# 老机执行：
rsync -av --progress example_demo_session example_slam_test mapping_run 用户名@新机IP:~/umi-ampro2/
```

## 6. 验证清单

```bash
conda activate umi && python -c "import torch, cv2; from exiftool import ExifToolHelper"  # umi 环境
docker run --rm chicheng/orb_slam3:latest echo ok                                       # docker
python scripts_slam_pipeline/04_detect_aruco.py --help                                   # 管线脚本
ls /dev/ttyACM0                                                                          # 机器人串口
```

## 7. 已整合 vs 未整合（状态总览）

| 内容 | 位置 | 状态 |
|---|---|---|
| 全部适配代码/脚本/控制器 | GitHub `umi-ampro2` | ✅ |
| 全部文档/计划/进度档案 | GitHub `umi-ampro2` | ✅ |
| umi / lerobot / lerobot_alohamini 环境定义 | `environment_*.yml`（3 个） | ✅ |
| ORB_SLAM3 补丁 | GitHub `Zzzuuu111/ORB_SLAM3` | ✅ |
| lerobot 本地配置改动 | `patches/lerobot_config_alohamini.patch` | ✅（备份，需手动 apply） |
| 自定义 docker 镜像 | `mapping_run/build_nomu_image.sh` | ✅（脚本化，重建 ~30min） |
| 超大视频/原始数据 | 仅本地 | ⚠️ 设计如此，见第 5 节清单 |
| Gyroflow / 系统包 | 文档 + 第 4 节 | ✅ 第三方，不捆绑 |
