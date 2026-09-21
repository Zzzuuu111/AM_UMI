# AM_UMI 环境配置

本项目在当前机器上使用名为 `AM_UMI` 的 Conda 环境。它用于手持数据采集、离线处理、训练和受监督的 AM2Pro replay 工具。

> 设备控制前请先确认机械臂周围无人、无障碍物，并始终从低速、空载和人工监督开始。

## 1. 前置条件

- Ubuntu/Linux，已安装 Miniconda 或 Anaconda；
- NVIDIA 驱动已正确安装（训练需要 GPU）；
- UVC 相机、AM2Pro/ROS2 硬件按项目标定文档连接；
- 在仓库根目录 `AM_UMI/` 下操作。

检查基础环境：

```bash
conda --version
nvidia-smi                 # 仅训练时需要
cd ~/AM_UMI/umi
```

## 2. 创建环境

环境定义文件为 [`environment_lerobot_alohamini.yml`](environment_lerobot_alohamini.yml)，其中固定了 Python 3.12、PyTorch、TorchVision、CUDA 运行时、OpenCV、Zarr、Timm 与 W&B 等依赖。

```bash
cd ~/AM_UMI/umi
conda env create -n AM_UMI -f environment_lerobot_alohamini.yml
conda activate AM_UMI
python -m pip install -r requirements-am-umi.txt
```

`requirements-am-umi.txt` 只补充 UMI/相机/机器人接口所需依赖；不要随意用 `pip install --upgrade` 升级 Torch、TorchVision、NumPy、Zarr、Timm、OpenCV 或 W&B，以免破坏已验证的组合。

如果 `AM_UMI` 已存在，更新补充依赖即可：

```bash
conda activate AM_UMI
cd ~/AM_UMI/umi
python -m pip install -r requirements-am-umi.txt
```

## 3. 验证

```bash
conda activate AM_UMI
cd ~/AM_UMI/umi

python - <<'PY'
import sys
import torch
import cv2
import zarr
import timm

print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
print('opencv:', cv2.__version__)
print('zarr:', zarr.__version__)
print('timm:', timm.__version__)
PY
```

运行数据采集或实机 replay 前，还应分别确认相机设备、ROS2/机器人服务和相关标定文件可用。环境验证通过不代表硬件已安全就绪。

## 4. 日常使用

每次新终端执行：

```bash
conda activate AM_UMI
cd ~/AM_UMI/umi
```

训练、数据集和模型产物均保存在本地 `umi/data/`、`umi/wandb/` 与 `umi/outputs/` 下，默认不会提交到 Git 仓库。代码、脚本、配置和 Markdown 文档才是版本控制对象。

## 5. 可复现性说明

- `environment_lerobot_alohamini.yml` 是环境的锁定式定义；
- `requirements-am-umi.txt` 是项目额外 Python 包清单；
- 显卡驱动由系统安装，通常不会被 Conda 环境文件完整复现；
- 相机编号、机器人 IP/串口、ROS2 服务和真实标定结果均依赖本机硬件，不能仅依靠 Git 克隆恢复；
- 如需导出当前机器的完整快照，可执行：

```bash
conda env export -n AM_UMI > environment_am_umi_snapshot.yml
```

该快照可能包含平台相关的 build 信息，适合备份，不建议替代仓库内的主环境定义文件。
