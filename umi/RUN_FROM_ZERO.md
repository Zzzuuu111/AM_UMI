# 从零运行指南(UMI × Osmo Action 4 × AM2Pro)

> 本文档回答一个问题:**把整个项目从零跑起来,需要哪些环境、哪些代码、按什么顺序操作。**
>
> 与现有文档的关系:
> - `SETUP.md` —— 新机环境搭建清单(本文第 4、5 节的展开版)
> - `PROJECT_PROGRESS.md` —— 项目进度档案(问题排查、历史决策、验证记录)
> - `HARDWARE_ADAPTATION_PLAN.md` / `UMI_AND_AM2PRO.md` —— 硬件适配设计文档
> - 本文 = 从零到跑通"采集 → 处理 → 训练 → 部署"的总入口

---

## 1. 项目全景与数据流

### 1.1 四阶段管线

```
① 采集          ② 处理                ③ 训练              ④ 部署
手持夹爪 AM2UMI  Gyroflow IMU 导出     扩散策略训练        单臂 AM2Pro
+ Osmo Action 4  SLAM 建图/跟踪       (umi_torch27)       USB 直连电脑
录遥操作 demo    00→07 合成 zarr       相对10D动作         推理 + IK 执行
```

- **AM2UMI**:手持示教夹爪(纯机械无电子),夹爪宽度由手指 tag 视觉估计。只在采集阶段出现。
- **AM2Pro**:机器人本体,只在推理期通过 IK 执行。**同一台机器人也提供夹爪用于手持示教**(拆下来当 AM2UMI 用)。

### 1.2 位姿的数据流(绝对/相对,务必理解)

```
合成(zarr)    存【绝对】TCP 位姿: SLAM 相机轨迹 × tag标定 × 相机→TCP固定偏移
              → [pos3, rotvec3, width1] 每帧绝对 (07_generate_replay_buffer.py)

训练时        转【相对10D】: 以当前帧绝对位姿为基准
              pos3 + rot6d + width1 = 相对增量 (diffusion_policy/dataset/umi_dataset.py)

推理时        还原【绝对目标】: 相对10D × 当前FK实测TCP位姿
              (umi/real_world/real_inference_util.py: get_real_umi_action)

执行          绝对目标位姿 → 笛卡尔插值 → IK → 6关节角(度) → 舵机
```

### 1.3 推理闭环(50Hz,每 tick)

```
读舵机 Present_Position(原始tick)
  → 标定换算(减 homing 零点) → 6 关节角
  → FK(URDF/DH 模型矩阵连乘) → 实际TCP位姿  ──┐
                                               │ 上行: 作为策略观测基准
策略输出 相对10D(相对当前TCP的增量)            │
  → × 当前FK位姿 = 绝对目标TCP位姿             │
  → IK(placo, 从当前关节角迭代5次) ──────────┘
  → 6 关节角 → Goal_Position → 串口写舵机
  (夹爪独立一条线: 宽度[0,0.09]m → servo[0,100]线性映射)
```

- **FK(正运动学)**:关节角 → TCP 位姿。输入来自读舵机,本身是纯计算。
- **TCP(工具中心点)**:夹爪上被追踪的物理点(现为 `right_Fixed_Jaw` 定爪)。换夹爪只改这一个偏移,6 关节链不变。
- **IK(逆运动学)**:目标 TCP 位姿 → 6 关节角。策略只会说"手去哪",IK 负责翻译成"关节转多少"。

---

## 2. 硬件清单

| 硬件 | 说明 |
|---|---|
| AM2Pro 单臂 | USB 串口 `/dev/ttyACM0`(QinHeng 1a86:55d3);7 舵机:ID1~6 关节(角度模式)+ ID7 夹爪(0-100 模式);肩 pan/wrist*/gripper = sts3250,肩 lift/elbow = sts3095 |
| Osmo Action 4 | 采集相机,统一设置见 6.1;防抖**必须关** |
| 夹爪 tag | 手指上 tag 0/1(宽度视觉估计;左右互换配置已支持) |
| 桌面 tag | tag 13(160mm),建图/定位基准 |
| charuco 板 | 内参标定(30mm 方格/18mm tag,A4 可打印) |
| SpaceMouse | 部署期人手接管(移动 EE、C 键开始策略、S 键停止) |
| NVIDIA GPU | **仅训练需要**(当前 8GB 显存/15GB 内存,有限制,见 6.7) |
| 部署相机 | 150° 非鱼眼(部署用;需另标内参 + 图像预处理适配,见待办) |

---

## 3. 代码与目录

### 3.1 仓库(3+1 个)

| 仓库 | 本机路径 | 作用 | 必需 |
|---|---|---|---|
| **umi-ampro2**(主仓库,私有) | `~/universal_manipulation_interface`(本文根目录) | 全部管线适配、AM2Pro 控制器、标定、文档 | ✅ |
| ORB_SLAM3 补丁(公开 fork) | 主仓库内 `ORB_SLAM3_umi/` | SLAM(IMU 初始化开关等补丁) | ✅ |
| **lerobot_alohamini**(公开) | `~/lerobot_alohamini` | 物理运行时权威:舵机总线、标定、watchdog、placo IK | ✅ |
| alohamini_ros2 | `~/alohamini_ros2` | 权威机器人描述(URDF/DH/MoveIt)+ 纯 numpy IK | 可选(用作 URDF/TCP 校验参照) |

### 3.2 主仓库关键代码地图

```
scripts_slam_pipeline/          # 采集后处理管线 00→07(umi 环境)
  00_process_videos.py          # 视频归档(DJI 时间码)
  02_create_map.py              # SLAM 建图(容器内)
  03_batch_slam.py              # 批量 SLAM 跟踪
  04_detect_aruco.py            # tag 检测
  05_run_calibrations.py        # 手眼/tag/夹爪标定
  06_generate_dataset_plan.py   # 轨迹→数据集计划(绝对位姿)
  07_generate_replay_buffer.py  # 视频+轨迹 合成 zarr
scripts/
  gyroflow_csv_to_imu_json.py   # Gyroflow CSV → imu_data.json
  calibrate_fisheye_intrinsics.py
  test_am2pro_layer1.py         # 机器人只读测试(lerobot_alohamini 环境)
  test_am2pro_layer4.py         # IK 闭环测试(umi 环境)
  record_robot_world_hand_eye.py# 机器人基座↔训练系对齐
umi/real_world/
  am2pro_interpolation_controller.py  # 客户端:7维位姿→Unix socket(umi 环境)
  am2pro_controller_server.py         # 服务器:插值+IK+舵机写读(lerobot_alohamini 环境)
  am2pro_protocol.py                  # 两端通信协议
  real_inference_util.py              # 相对10D ↔ 绝对位姿 转换
train.py                        # 训练入口(hydra)
eval_real.py                    # 部署/推理入口
diffusion_policy/
  dataset/umi_dataset.py        # 训练时 绝对→相对10D
  workspace/train_diffusion_unet_image_workspace.py  # 训练循环(epoch 日志在此)
alohamini2pro_right_arm_kinematics.urdf  # mesh-free 运动学 URDF(placo 用)
```

### 3.3 数据目录

```
imu_work/          # 正式数据集(imu_work/demo_batch3_dataset.zarr = v2 训练集)
mapping_run/       # 建图 session
data/outputs/      # hydra 训练输出(日志/checkpoint)
wandb/             # wandb 离线日志
calibration/  # 内参/tag 标定文件(当前有效)
```

---

## 4. 环境清单(conda + Docker + 工具)

### 4.1 conda 环境(实际存在的 4 个 + 1 个可选)

| 环境 | Python | 用途 | 定义文件 |
|---|---|---|---|
| `umi` | 3.9 | **主环境**:00→07 管线、eval_real 主进程 | `environment_umi_full.yml` |
| `umi_torch27` | 3.9 | **训练**:torch 2.7.1 + accelerate + hub 0.23.2(由 umi clone 而来) | 见 5.3 步骤 |
| `lerobot_alohamini` | 3.12 | **硬件服务器**:Feetech 舵机 + placo IK/FK | `environment_lerobot_alohamini.yml` |
| `lerobot` | — | 通用 LeRobot(可选,主流程不用) | `environment_lerobot.yml` |
| (FastUMI) | — | 其他实验环境,主流程**不用** | — |

**为什么分两个 Python 环境**:UMI 训练/推理栈要 Python 3.9 + numpy 1.x;lerobot 的电机/运动学代码要 Python 3.12 + numpy 2.x,无法共存。二者靠 **Unix domain socket** 通信(`am2pro_protocol.py`),服务器由客户端用 `robot_python` 参数拉起。

### 4.2 Docker

- `chicheng/orb_slam3:latest`(官方基础镜像)
- `umi_orb_slam3_nomu`(自建,含 IMU 初始化开关补丁;由 `mapping_run/build_nomu_image.sh` 构建)
- SLAM 02/03 脚本在容器内跑;`settings` YAML 和掩码通过挂载进入

### 4.3 系统工具

- Ubuntu 22.04 + conda + docker(用户加入 docker 组;国内配镜像源)
- Gyroflow(Linux 版,`~/Gyroflow/`,依赖 `libc++1`/`libc++abi1`;启动:`cd ~/Gyroflow && LD_LIBRARY_PATH=$PWD/lib ./gyroflow`)
- exiftool 修复:`ln -sfn ~/anaconda3/envs/umi/lib/perl5/site_perl ~/anaconda3/envs/umi/bin/lib`(或 export `PERL5LIB`)
- PyExifTool==0.5(`from exiftool import ExifToolHelper`)

---

## 5. 从零搭建(一台新机)

> 详细版见 `SETUP.md`,此处是完整摘要。

```bash
# 0. 系统: Ubuntu 22.04 + miniconda + docker(+docker组/镜像源) + NVIDIA驱动(训练机)
#    SSH 公钥加入 GitHub(私有仓库)

# 1. 拉代码(3 个仓库)
git clone git@github.com:Zzzuuu111/umi-ampro2.git ~/universal_manipulation_interface
git clone https://github.com/Zzzuuu111/ORB_SLAM3.git ~/universal_manipulation_interface/ORB_SLAM3_umi
git clone https://github.com/liyiteng/lerobot_alohamini.git ~/lerobot_alohamini
cd ~/lerobot_alohamini && git apply ~/universal_manipulation_interface/patches/lerobot_config_alohamini.patch && cd ..

# 2. conda 环境(3 个)
conda env create -f ~/universal_manipulation_interface/environment_umi_full.yml
conda env create -f ~/universal_manipulation_interface/environment_lerobot_alohamini.yml
conda env create -f ~/universal_manipulation_interface/environment_lerobot.yml   # 可选

# 3. lerobot editable 安装(在 lerobot_alohamini 环境里)
conda activate lerobot_alohamini
pip install -e ~/lerobot_alohamini

# 4. 训练环境 umi_torch27(umi clone + 升级 torch)
conda create -n umi_torch27 --clone umi
conda activate umi_torch27
pip install torch==2.7.1 torchvision==0.22.1 accelerate "huggingface_hub==0.23.2"   # hub 必须降级,见坑 8.7

# 5. exiftool 修复
ln -sfn ~/anaconda3/envs/umi/lib/perl5/site_perl ~/anaconda3/envs/umi/bin/lib

# 6. Docker 镜像
bash ~/universal_manipulation_interface/mapping_run/build_nomu_image.sh   # 构建 umi_orb_slam3_nomu,~30min

# 7. 大文件(不在 git 里,老机 rsync 拷贝,见 SETUP.md 第 5 节)
#    example_demo_session / example_slam_test / mapping_run / 标定测试视频 / Gyroflow
```

验证:

```bash
conda activate umi && python -c "import torch, cv2; from exiftool import ExifToolHelper"
docker images | grep orb_slam3
python scripts_slam_pipeline/04_detect_aruco.py --help
ls /dev/ttyACM0   # 机器人串口
```

---

## 6. 全流程操作步骤(按顺序)

### 6.1 相机设置与内参标定

所有录制统一设置:**广角 Wide、4:3、2.7K(2688×2016)、60fps、防抖关、畸变校正关**。

内参标定(相机固定,只移动 charuco 板;扫满四角+倾斜,1~2 分钟):

```bash
conda activate umi
python scripts/calibrate_fisheye_intrinsics.py -i "<标定视频.mp4>" -o calibration/osmo4_intrinsics_wide.json
```

- 当前有效内参:f=1054.20,cx=1344.19,cy=990.63,重投影 0.74px(够用)
- 设置不变则内参长期有效;改 FOV/画幅/防抖/换镜头/碰撞后才需重录重标

### 6.2 录像规范(三类视频)

| 类型 | 规范 |
|---|---|
| mapping(建图) | 桌面贴 tag 13 + 有纹理物体;**平移为主、慢而稳**,不要原地旋转/甩动;tag 尽量入画;1~2 分钟 |
| gripper calibration | 相机在夹爪上、画面见 tag;缓慢移动 + **开合夹爪** |
| demo(遥操作) | **中距离、任务节奏**,避免贴脸凑近物体;夹爪 tag 0/1 尽量入画 |

### 6.3 session 目录结构

```
<session>/demos/
  mapping/{raw_video.mp4, imu_data.json}     # 建图视频(00 归档后)
  demo_XXXX.mp4                              # 遥操作 demo
  <每个视频>/tag_detection.pkl, camera_trajectory.csv   # 04/03 产出
  calibration/                               # 内参 json
  gripper_calibration/                       # 夹爪标定
  mapping/tx_slam_tag.json                   # 手眼/tag 标定(05 产出)
dataset_plan.pkl                             # 06 产出
```

### 6.4 IMU 导出(Gyroflow 工作流,每段视频)

```bash
# 1. Gyroflow GUI 打开视频 → 确认波形 → Export data 导出 CSV
# 2. 转成 gopro_slam 格式
python scripts/gyroflow_csv_to_imu_json.py -i "<导出.csv>" -o "<视频同目录>/imu_data.json"
```

原理:CSV 四元数微分→陀螺仪;加速度=合成重力。**加速度是合成的,非真实线性加速度**(已知限制)。

### 6.5 SLAM 管线 00→07(umi 环境)

```bash
conda activate umi
export PERL5LIB=~/anaconda3/envs/umi/lib/perl5/site_perl   # 若未做 symlink

# 00 视频归档(按 DJI 时间码)
python scripts_slam_pipeline/00_process_videos.py <session>

# 02 建图(容器内跑;session 目录需含 mapping/{raw_video.mp4, imu_data.json})
python scripts_slam_pipeline/02_create_map.py -i <session>/demos/mapping -np

# 03 批量 SLAM 跟踪(用自建镜像 + 地图 + IMU 初始化)
python scripts_slam_pipeline/03_batch_slam.py -i <session>/demos -m <map.osa> \
    -d umi_orb_slam3_nomu -ml 600 -n 3

# 04 tag 检测
python scripts_slam_pipeline/04_detect_aruco.py -i <session>/demos \
    -ci calibration/osmo4_intrinsics_wide.json -ac calibration/aruco_config.yaml

# 05 标定(手眼 tag / 夹爪宽度)
python scripts_slam_pipeline/05_run_calibrations.py <session>

# 06 数据集计划(绝对 TCP 位姿)
python scripts_slam_pipeline/06_generate_dataset_plan.py -i <session> -nz 0.088
#   ⚠️ -nz 0.088 必须带(tag 深度 z≈0.087,默认 0.072 会把夹爪宽度全部滤掉)

# 07 合成 zarr(视频 + 轨迹 → 训练数据集)
python scripts_slam_pipeline/07_generate_replay_buffer.py -o imu_work/dataset.zarr.zip <session>
```

### 6.6 zarr 验证(必做!)

**v1 教训**:v1 zarr 的 `camera0_rgb` 解码后是平坦灰图(与视频帧 MSE>2600),loss 降到 0.03 也是在垃圾图像上跑的,模型无效。**合成后必须逐 chunk 验证**:

- 图像:随机 chunk 解码,与对应视频帧对照,**MSE 必须很小(v2 实测 ≈2.5-2.9,仅 JPEG-XL 有损噪声)**
- 宽度:应在 [0.007, 0.090] m(= 标定范围 闭0/开0.09)
- 动作:7D 无 NaN,姿态范围合理

### 6.7 训练(umi_torch27 环境)

```bash
conda activate umi_torch27
cd /home/zzzjh/universal_manipulation_interface
nohup python train.py --config-name train_diffusion_unet_image_workspace \
  task=umi task.dataset_path=imu_work/demo_batch3_dataset.zarr \
  training.device=cuda:0 training.num_epochs=300 training.lr_warmup_steps=100 \
  training.checkpoint_every=10 logging.mode=offline \
  dataloader.batch_size=4 dataloader.num_workers=2 exp_name=am2umi_v2 \
  > imu_work/train_am2umi_v2.log 2>&1 &
```

注意事项:

- **8GB 显存机器的硬限制**:batch_size=4、num_workers=2(更高会触发 NVIDIA 驱动 OOM → **整机冻结**,不是普通 CUDA OOM);训练期间别开浏览器等重应用
- checkpoint 每 10 epoch 保存,`resume: True` 可中断续训;保存已改串行 + 原子写
- 日志解读:`[epoch X end]` 行里的 `train/val_action_mse_error*` 每 `sample_every=5` 个 epoch(0、5、10…)才出现一次,中间 epoch 只有 train_loss——**正常**
- 训练输出在 `data/outputs/<时间戳>_train_diffusion_unet_image_workspace/checkpoints/`

### 6.8 部署(推理)

**第 0 步:硬件连通验证**

```bash
# 只读测试(lerobot_alohamini 环境)
conda activate lerobot_alohamini
python scripts/test_am2pro_layer1.py

# IK 闭环测试(umi 环境;启动控制器→读位姿→+2cm Z→读回误差)
conda activate umi
python scripts/test_am2pro_layer4.py
```

**第 1 步:机器人配置**(`example/eval_robots_config.yaml` 的 am2pro 块)

```jsonc
{
  "robot_type": "am2pro",
  "robot_usb_port": "/dev/ttyACM0",
  "robot_python": "/home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python",
  "robot_obs_latency": 0.005, "robot_action_latency": 0.02,
  "gripper_servo_closed": 97.1, "gripper_servo_open": 1.4,
  "gripper_width_min": 0.0, "gripper_width_max": 0.09
}
```

**第 2 步:启动推理**

```bash
conda activate umi
python eval_real.py -i <checkpoint.ckpt> -o data/eval_<名字> -rc example/eval_robots_config.yaml
```

- 人手接管:SpaceMouse 移动 EE(默认锁 xy 平面,右键解锁 z,左键解锁旋转);**C 键**交给策略,**S 键**抢回控制
- 架构:eval 主进程(umi py3.9)自动拉起硬件服务器(lerobot_alohamini py3.12),Unix socket 通信,50Hz 闭环
- **第 3 步(部署前必做)**:机器人基座 ↔ 训练系的位姿对齐(`scripts/record_robot_world_hand_eye.py`);否则策略的"相对动作"从错误的基准点出发,整体偏移
- 部署相机(150° 非鱼眼):标内参 + 图像预处理适配(与训练时 224×224 中心裁剪、无 rect 对齐)

---

## 7. 核心原理速览(团队内传阅用)

| 概念 | 一句话 |
|---|---|
| FK | 关节角 → TCP 位姿(读舵机角度,乘几何模型)。观测上行的入口 |
| TCP | 夹爪上被追踪的物理点(现为定爪 `right_Fixed_Jaw`)。换夹爪只改这个偏移 |
| IK | 目标 TCP 位姿 → 关节角。策略与执行之间的翻译层(每 tick 迭代 5 次) |
| 绝对位姿 | zarr 里存的形式;SLAM 轨迹 × 标定得到 |
| 相对10D | 模型输入/输出的形式;相对当前 TCP 位姿的增量(pos3+rot6d+width1) |
| 推理一步 | 相对10D × 当前FK = 绝对目标 → 插值 → IK → 关节角 → 舵机 |

---

## 8. 常见坑与避雷清单(全部实测)

1. **v1 zarr 图像是垃圾**(灰图,与视频 MSE>2600)→ 合成后必须对照视频帧验证(见 6.6);只有 lowdim 正常不算数
2. **NVIDIA 驱动 OOM 整机冻结**(8GB 卡,batch 8 + 4 workers)→ batch_size=4、num_workers=2,训练时不开重应用
3. **checkpoint 损坏**(双线程拷贝峰值内存 + 中断截断)→ 已改串行保存 + 原子写(tmp+os.replace);resume 时遇到 "failed finding central directory" 删 latest.ckpt
4. **06 必须 `-nz 0.088`**,否则夹爪宽度全被滤掉
5. **SLAM CSV 时间戳重复块**(demo_0085/0089 踩过)→ 用 tag_detection.pkl 的严格递增时间覆写 timestamp 列
6. **SLAM 丢帧 28/50 条**(中途手挡镜头/太近/超出地图)→ 录 demo 保持中距离、tag 入画(见 6.2)
7. **hub 版本**:diffusers 0.18.2 与 huggingface_hub 0.34 不兼容(`cached_download` 被删)→ hub 固定 0.23.2
8. **exiftool**:`from exiftool import ExifToolHelper` 报 perl 错 → symlink 修复(见 4.3)
9. **lerobot 读舵机前**必须先 `bus.calibration = bus.read_calibration()`,否则位置是原始 tick
10. **无反光镜**:AM2Pro 采集掩码用底部 35% 横条,mirror 遮罩已去掉;`04_detect_aruco.py` 同步去掉 mirror
11. **tag 左右互换**:夹爪标定支持 json 里 left=1/right=0 互换,不用重贴重录
12. **推理期换夹爪**:6 关节链不变,只需重测 TCP 偏移(腕法兰→指尖中点)并重标宽度;权威 URDF/DH 参照在 `~/alohamini_ros2/src/alohamini_description/`
13. **SpaceMouse 权限**:部署机 `sudo chmod -R 777 /dev/bus/usb`(USB 设备访问)

---

*最后更新:与 `PROJECT_PROGRESS.md` 2026-08-27 状态一致。数据/命令以各节实测记录为准,新改动请同步更新本文件与 PROJECT_PROGRESS。*
