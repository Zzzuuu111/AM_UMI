# UMI × Osmo Action 4 × AM2Pro

> 把 [Universal Manipulation Interface(UMI)](https://umi-gripper.github.io/) 的数据采集/训练管线从原版
> **GoPro + UR5/Franka** 迁移到 **DJI Osmo Action 4 + AlohaMini2/pro(AM2Pro)**。
>
> 手持夹爪采集人类遥操作演示 → 训练扩散策略 → 单臂 AM2Pro 通过 IK 自主执行。

[[UMI 项目页]](https://umi-gripper.github.io/) [[论文]](https://umi-gripper.github.io/#paper) [[原版硬件指南]](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU)

---

## 管线总览

```
① 采集          ② 处理               ③ 训练             ④ 部署
AM2UMI 手持夹爪  Gyroflow IMU 导出    扩散策略(相对10D)  单臂 AM2Pro
+ Osmo Action 4  SLAM 建图/跟踪       umi_torch27        USB 直连 + IK 执行
录遥操作 demo    00→07 合成 zarr       绝对→相对10D        相对10D × FK → IK → 关节角
```

- **AM2UMI**:手持示教夹爪(纯机械),夹爪宽度由手指 tag 视觉估计,仅用于采集
- **AM2Pro**:机器人本体(6-DOF 臂 + 夹爪,Feetech 舵机),仅在推理期通过 IK 执行
- **坐标系**:数据存储为绝对位姿,训练转为相对 10D 增量,推理时与实时 FK 位姿合成还原

## 快速开始

```bash
# 1. 环境(详见 SETUP.md / RUN_FROM_ZERO.md)
conda env create -f environment_umi_full.yml              # umi: 管线主环境(py3.9)
conda env create -f environment_lerobot_alohamini.yml     # 硬件服务器(py3.12, 舵机+IK)
conda activate lerobot_alohamini && pip install -e <lerobot_alohamini 仓库>
conda create -n umi_torch27 --clone umi && conda activate umi_torch27 \
  && pip install torch==2.7.1 torchvision==0.22.1 accelerate "huggingface_hub==0.23.2"

# 2. SLAM 管线(00→07, 采集后处理)
python run_slam_pipeline.py <session_dir>
python scripts_slam_pipeline/07_generate_replay_buffer.py -o <dataset.zarr> <session_dir>

# 3. 训练
python train.py --config-name train_diffusion_unet_image_workspace \
  task=umi task.dataset_path=<dataset.zarr> training.num_epochs=300

# 4. 部署(AM2Pro 实机推理)
python eval_real.py -i <checkpoint.ckpt> -o data/eval_demo -rc example/eval_robots_config.yaml
```

> ⚠️ 完整命令(含 `-nz 0.088`、batch_size=4 等必带参数与踩坑清单)见
> [RUN_FROM_ZERO.md](RUN_FROM_ZERO.md)。

## 文档索引

| 文档 | 内容 |
|---|---|
| [RUN_FROM_ZERO.md](RUN_FROM_ZERO.md) | **从零运行手册**:环境、代码地图、00→07 全流程命令、训练/部署、13 条避雷清单 |
| [SETUP.md](SETUP.md) | 新机环境搭建清单(系统/仓库/conda/docker/大文件) |
| [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) | 项目进度档案:验证记录、历史决策、问题排查 |
| [HARDWARE_ADAPTATION_PLAN.md](HARDWARE_ADAPTATION_PLAN.md) | AM2Pro 硬件适配设计(控制器架构、IK 方案) |
| [UMI_AND_AM2PRO.md](UMI_AND_AM2PRO.md) | UMI 与 AM2Pro 结合方案说明 |
| [NEXT_STEPS.md](NEXT_STEPS.md) | **当前 EMEET + JY901B + AM2Pro 剩余工作清单** |
| [VIEW_REFERENCE.md](VIEW_REFERENCE.md) | 起始视角基准：手持画面对齐与机械臂画面/姿态复位 |
| [ENVIRONMENT_AM_UMI.md](ENVIRONMENT_AM_UMI.md) | 当前 `AM_UMI` Conda 环境的创建、验证与日常使用 |

## 目录结构

```
scripts_slam_pipeline/   采集后处理管线 00→07(视频归档/SLAM/tag检测/标定/zarr合成)
scripts/                 工具脚本(IMU 转换、内参标定、机器人层测试、手眼标定)
umi/real_world/          AM2Pro 控制器(客户端/服务器/协议)与推理位姿转换
diffusion_policy/        扩散策略训练(dataset/workspace/config)
eval_real.py             实机部署入口
train.py                 训练入口
alohamini2pro_right_arm_kinematics.urdf   AM2Pro 运动学 URDF(placo IK 用)
example/                 标定与机器人配置文件
```

## 与原版 UMI 的差异

- 相机:GoPro → **Osmo Action 4**(Wide/4:3/2.7K/60fps,IMU 经 Gyroflow CSV 导出)
- 机器人:UR5/Franka → **AM2Pro**(LeRobot Feetech 舵机总线;笛卡尔控制,控制器内 placo IK)
- 采集:原版 UMI 手持夹爪 → **AM2UMI**(AM2Pro 夹爪手持化,宽度手指 tag 视觉估计)
- 训练环境:umi(py3.9)与硬件运行时 lerobot_alohamini(py3.12)经 Unix socket 隔离

## 致谢

本项目基于 [Universal Manipulation Interface](https://umi-gripper.github.io/),由
Cheng Chi, Zhenjia Xu, Chuer Pan, Eric Cousineau, Benjamin Burchfiel, Siyuan Feng,
Russ Tedrake, Shuran Song 开发。SLAM 部分基于 UMI fork 的 [ORB_SLAM3](https://github.com/cheng-chi/ORB_SLAM3),
硬件运行时基于 [LeRobot](https://github.com/huggingface/lerobot) 的 AlohaMini 适配。

## License

[MIT](LICENSE)
