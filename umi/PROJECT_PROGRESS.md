# UMI × Osmo Action 4 × AM2Pro 适配项目日志

> **本文档是项目的唯一进度档案**：记录所有已完成工作、关键决策、文件变更和待办。
> **规则：之后的每次修改、测试、决策都必须同步更新本文档**（尤其「更新日志」一节）。
>
> 配套文档：
> - `OSMO_ACTION4_ADAPTATION_PLAN.md` —— 相机/建图管线的适配计划（含当前状态表）
> - `HARDWARE_ADAPTATION_PLAN.md` —— AM2Pro 机器人控制适配计划（待实施）
> - `UMI_AND_AM2PRO.md` —— UMI 与 AM2Pro 结合的背景文档
>
> 仓库地址（2026-08-25 起）：
> - 主仓库（私有）：https://github.com/Zzzuuu111/umi-ampro2
> - ORB_SLAM3 fork（IMU 初始化开关等 4 个补丁）：https://github.com/Zzzuuu111/ORB_SLAM3
> - 新机搭建清单：`SETUP.md`（代码+环境+大文件+验证，一站式）

---

## 1. 项目目标

**一句话目标**：手持夹爪上安装 DJI Osmo Action 4 摄像头采集人类遥操作演示数据 → 用这些数据训练操作策略模型 → 用训练好的策略驱动单臂 AM2Pro 或双臂树莓派 AM2Pro 机器人自主执行桌面操作任务。

**完整数据闭环**：

```
① 采集端        ② 数据处理        ③ 训练           ④ 部署
手持夹爪   →   UMI 管线      →   训练策略模型   →   驱动 AM2Pro
+ Action 4     （SLAM 建图 + 手眼  （diffusion     （单臂 USB 直连 /
（第一视角拍摄）  标定 + 数据打包）   policy）        双臂树莓派）自主执行
```

**三段硬件与目的**：

| 阶段 | 硬件 | 目的 |
|---|---|---|
| ① 数据采集 | **手持夹爪 + DJI Osmo Action 4**（腕部相机，朝下拍操作区域）| 录制人类演示：视频 + IMU + 夹爪轨迹/开合 |
| ②③ 数据处理与训练 | 电脑（conda `umi` 环境 + UMI 管线）| 演示数据 → replay buffer → 训练扩散策略 |
| ④ 策略部署 | **单臂 AM2Pro**（USB 直连电脑）或 **双臂树莓派 AM2Pro**（LeRobot AlohaMini2/pro，Feetech 舵机）| 训练好的策略驱动机器人自主复现任务 |

**迁移背景**：把 UMI（Universal Manipulation Interface）的数据采集/训练管线从原版 GoPro + UR5/Franka 迁移到 Osmo Action 4 + AM2Pro。**当前推进顺序**：先单臂 USB 直连验证 → 迁树莓派双臂 → 批量录遥操作 demo → 跑完整管线训练 → `eval_real.py` 部署。

---

## 2. 关键决策记录（按时间）

| # | 决策 | 结论 | 理由 |
|---|---|---|---|
| 1 | 相机 FOV 模式 | **用 Wide（广角 130°）**，弃用超广角 155° | Gyroflow 只支持 Wide 的陀螺仪；超广角 dbgi 是 DJI 私有编码，逆向多轮无果。Wide 130° 视场对桌面操作够用 |
| 2 | 内参标定方案 | 自写 scipy 鱼眼 BA（`scripts/calibrate_fisheye_intrinsics.py`） | OpenCV `cv2.fisheye.calibrate` 在该相机/板子上反复失败（OpenCV 4.7 已知 bug + 大板子收敛问题）|
| 3 | IMU 提取方案 | **Gyroflow 导出 CSV → 自写转换脚本**，弃用 dbgi 逆向 | Wide 模式 Gyroflow 官方支持；四元数微分出陀螺仪（已验证积分≈90°）；加速度用合成重力近似 |
| 4 | SLAM 模式 | 沿用 fork 的 `IMU_MONOCULAR`（gopro_slam 硬编码，无纯单目）| `--input_imu_json` 必填，故必须提供 IMU 数据 |
| 5 | 尺度来源 | IMU 重力（合成）+ ArUco tag 13 初始化（gopro_slam 默认 `init_tag_id=13`，size=0.16）| fork 自带 tag 初始化 |
| 6 | 机器人接入顺序 | **先单臂 USB 直连电脑 → 验证后再迁树莓派双臂** | 核心代码（URDF/IK/Controller）与连接方式无关；直连调试快、变量少；迁移只是改 lerobot 配置端口 |
| 7 | 训练硬件 | ⚠️ **待确认 GPU 型号**（`nvidia-smi` 未查）| 决定训练方案与速度 |

---

## 3. 当前状态总览

### ✅ 已完成（实测验证）

| 环节 | 状态 | 关键结果 |
|---|---|---|
| Docker + SLAM 镜像 | ✅ | `chicheng/orb_slam3:latest` 已拉取（配国内镜像源）|
| Wide 内参标定 | ✅ | f=1054.20，cx=1344.19，cy=990.63，k1..k4=0.174/0.079/0.002/-0.044，**重投影 0.74px** |
| IMU 管线 | ✅ | Gyroflow CSV → `imu_data.json`（gopro_slam 格式），SLAM 接受无格式错误 |
| SLAM 建图（02）| ✅ 链路通 | 4434 帧处理，**建出 1 个地图（21 KF）**；⚠️ 跟踪仅末尾 ~2.7s（录像动作问题）|
| ArUco 检测（04）| ✅ | Wide 内参 + 0012 视频：**3948/4434 帧（89%）检测到 tag 13** |
| 管线脚本 00~07 | ✅ 全部适配 | 见第 6 节明细 |

### ⏳ 待办（按顺序）

1. ~~AM2Pro 单臂 USB 直连~~ → **✅ 已完成（2026-08-19）：第 1~4 层验证全过，IK 闭环误差 3.6mm**（见第 12 节）
2. 相机装上 AM2Pro → 按实际视场重画 SLAM 掩码
3. 重录规范 mapping 视频（平移为主、慢而稳）
4. 录 gripper calibration（相机在夹爪、开合夹爪）
5. 双臂迁树莓派 → 录遥操作 demo（每任务 50~100 条）
6. 跑完整管线 02→07 → 训练 → `eval_real.py` 部署

### ⚠️ 已知遗留问题（不阻塞，影响精度）

1. **加速度是合成重力**（无真实线性加速度）→ SLAM 偶发 `scale too small`。~~后续可选：解 dbgi field5 真加速度~~ / 外置 IMU。**2026-08-25 定论：dbgi 不含真加速度（见第 13 节），文件内只能拿到 ~1000Hz 融合姿态；真加速度只能外置 IMU**
2. **`IMU.T_b_c1` 是单位矩阵占位**（settings YAML）——手持慢速扫图可容忍，精度要求高时需 Kalibr
3. **SLAM 掩码按 GoPro 腕带结构画的**（mirror/finger 区域），AM2Pro 实装后需重画
4. 当前 mapping 视频跟踪短 → 录像动作需改进（见第 8 节）

---

## 4. 环境配置

- **系统**：Ubuntu 22.04.5 LTS，x86_64，ASUS V16 笔记本
- **Python**：conda env `umi`（`/home/zzzjh/anaconda3/envs/umi`）
- **Docker**：已安装 + 用户已加入 docker 组 + 国内镜像源（`/etc/docker/daemon.json`）
- **PyExifTool**：`pip install PyExifTool==0.5`（导入名 `from exiftool import ExifToolHelper`）
- **exiftool 修复**：`ln -sfn ~/anaconda3/envs/umi/lib/perl5/site_perl ~/anaconda3/envs/umi/bin/lib`
- **Gyroflow**：Linux 版装在 `~/Gyroflow/`，启动：`cd ~/Gyroflow && LD_LIBRARY_PATH=$PWD/lib ./gyroflow`（依赖 `libc++1`/`libc++abi1`，apt 安装）
- **lerobot**：`/home/zzzjh/lerobot_alohamini`（AM2Pro 专用分支）

---

## 5. 相机与标定

### 相机设置（所有录制统一）

| 参数 | 值 |
|---|---|
| FOV | **广角（Wide）** |
| 画幅 | **4:3**，2.7K（2688×2016）|
| 帧率 | 60fps（实际 59.94）|
| 防抖 | **关闭**（RockSteady/EIS 全关）|
| 畸变校正 | 关 |

### Wide 内参（当前有效）

- 文件：`calibration/osmo4_intrinsics_wide.json`（UMI `FISHEYE/focal_length` 格式）
- f=1054.1976，cx=1344.1871，cy=990.6321，k1..k4 = 0.17367/0.07852/0.00181/-0.04352
- 标定视频：`DJI_20260819095555_0008_D.MP4`，重投影误差 0.74px（最佳 0.45px）
- SLAM 用 960×720 缩尺版本：`calibration/osmo4_fisheye_setting_v1_720.yaml`（fx=fy=376.5，cx=480.07，cy=353.80）

### 标定重录条件

相机设置不变则内参长期有效。以下情况需重录标定视频并重跑标定：
- 改了 FOV/画幅/防抖设置；换镜头；机械碰撞后
- 对精度不满意（当前 0.74px 已接近原版 0.29px 水平，够用）

---

## 6. 管线脚本适配明细（00→07）

| 脚本 | 改动 | 验证 |
|---|---|---|
| `scripts_slam_pipeline/00_process_videos.py` | ① `timecode_util.py` 支持 DJI 时间码 `;` 分隔 ② 序列号缺失回退 `'OSMO4'` | ✅ 实测：视频正确归档为 mapping.mp4 |
| `01_extract_gopro_imu.py` | **弃用**，由 Gyroflow 工作流替代（见第 7 节）| — |
| `scripts_slam_pipeline/02_create_map.py` | ① 掩码尺寸 2028×2704→**2016×2688** ② 挂载 `osmo4_fisheye_setting_v1_720.yaml` 进容器 | ✅ 实测：建出 21 KF 地图 + 轨迹 CSV |
| `scripts_slam_pipeline/03_batch_slam.py` | 同 02（掩码 + settings 挂载）| ✅ 语法检查通过 |
| `scripts_slam_pipeline/04_detect_aruco.py` | 无需改（内参 json 已兼容）| ✅ 实测：3948/4434 帧检出 tag 13 |
| `scripts_slam_pipeline/05_run_calibrations.py` | 无需改 | ✅ 核查无相机硬编码 |
| `scripts_slam_pipeline/06_generate_dataset_plan.py` | 序列号 `.get(..., 'OSMO4')` 回退（2 处）| ✅ 语法检查通过 |
| `scripts_slam_pipeline/07_generate_replay_buffer.py` | 内参文件名自动查找（gopro → osmo4_wide → osmo4_2_7k）| ✅ 语法检查通过 |

其他修改：
- `umi/common/timecode_util.py`：时间码 `;` → `:` 修复
- `umi/common/cv_util.py`：charuco 板默认 30mm 方格/18mm tag（A4 可打印）
- `calibration/aruco_config.yaml`：修复 `predefind` 拼写错误
- `scripts/gen_aruco_tag_pdf.py`：修复同款拼写错误

---

## 7. IMU 工作流（每段视频都要走）

```
1. Gyroflow 打开视频 → 确认波形 → Export data 导出 CSV
2. python scripts/gyroflow_csv_to_imu_json.py -i <csv> -o <video同级目录>/imu_data.json
3. 02_create_map.py / 03_batch_slam.py 自动使用同目录的 imu_data.json
```

- 原理：CSV 的 `org_quat` 四元数**微分**→ 陀螺仪（rad/s）；加速度 = **合成重力**（R⁻¹×[0,0,-9.81]）
- 输出格式满足 gopro_slam `LoadTelemetry`：`{"1":{"streams":{"ACCL":{"samples":[...]}, "GYRO":{...}, "CORI":{...}}}}`
- 已用 009 视频验证：四元数积分 -83.7° ≈ 90° ✓；SLAM 接受无格式错误 ✓

---

## 8. 录像规范

### mapping（建图）视频

- Wide、4:3、2.7K、60fps、防抖关
- 桌面贴 **tag 13（160mm）**，放有纹理的物体（SLAM 要特征点，纯白桌面会失败）
- 手持/机上相机朝下 ~45° 看桌面
- **平移为主**（前后左右移动），慢而稳，**不要原地旋转、不要甩动**
- 全程 tag 尽量入画，1~2 分钟

### gripper calibration 视频

- 相机装在夹爪上，画面能看见 tag
- 缓慢移动 + **开合夹爪**（供夹爪开度标定）

### 标定（内参）视频

- 固定相机不动，**只移动 charuco 板**（卷帘快门：相机动会毁标定）
- 板子扫满画面四角 + 倾斜，1~2 分钟

---

## 9. 工作速查命令

```bash
# 环境
conda activate umi
export PERL5LIB=~/anaconda3/envs/umi/lib/perl5/site_perl   # 若未做 symlink

# 内参标定
python scripts/calibrate_fisheye_intrinsics.py -i "<标定视频.mp4>" -o calibration/osmo4_intrinsics_wide.json

# IMU 转换
python scripts/gyroflow_csv_to_imu_json.py -i "<Gyroflow导出.csv>" -o "<视频同目录>/imu_data.json"

# 建图（session 目录结构: <session>/demos/mapping/{raw_video.mp4, imu_data.json}）
python scripts_slam_pipeline/02_create_map.py -i <session>/demos/mapping -np

# tag 检测（-i 指向 demos 目录）
python scripts_slam_pipeline/04_detect_aruco.py -i <session>/demos -ci calibration/osmo4_intrinsics_wide.json -ac calibration/aruco_config.yaml

# 手眼标定
python scripts_slam_pipeline/05_run_calibrations.py <session>
```

---

## 10. 文件清单

### 新建（本项目中）

| 文件 | 用途 |
|---|---|
| `PROJECT_PROGRESS.md` | 本文档（项目进度唯一档案）|
| `OSMO_ACTION4_ADAPTATION_PLAN.md` | 相机适配计划 + 状态表 |
| `HARDWARE_ADAPTATION_PLAN.md` | AM2Pro 机器人适配计划 |
| `scripts/calibrate_fisheye_intrinsics.py` | scipy 鱼眼 BA 内参标定（输出 UMI 格式 json）|
| `scripts/gyroflow_csv_to_imu_json.py` | Gyroflow CSV → gopro_slam imu json |
| `scripts/extract_dji_imu.py` | DJI dbgi 提取器（**已弃用**，保留供参考）|
| `calibration/osmo4_intrinsics_wide.json` | Wide 内参（当前有效）|
| `calibration/osmo4_intrinsics_2_7k.json` | 超广角内参（已弃用）|
| `calibration/osmo4_fisheye_setting_v1_720.yaml` | SLAM settings（Wide 960×720）|

### 修改（UMI 原版）

`umi/common/cv_util.py`、`umi/common/timecode_util.py`、`scripts_slam_pipeline/00_process_videos.py`、`02_create_map.py`、`03_batch_slam.py`、`06_generate_dataset_plan.py`、`07_generate_replay_buffer.py`、`calibration/aruco_config.yaml`、`scripts/gen_aruco_tag_pdf.py`

### 测试数据

- `DJI_20260819095555_0008_D.MP4` —— Wide 标定视频
- `DJI_20260819100546_0009_D.MP4` —— Wide 90° 旋转（IMU 验证）
- `DJI_20260819110551_0012_D.MP4` —— Wide mapping 尝试（跟踪短）
- `Osma Action 4 Wide.csv` / `Osma Action 4 Wide2.csv` —— Gyroflow 导出
- `example_slam_test/` —— SLAM 测试 session

---

## 11. 更新日志

| 日期 | 内容 |
|---|---|
| 2026-08（前期）| 内参标定脚本（scipy BA）、超广角尝试、charuco 板、tag 打印 |
| 2026-08-18 | 00 脚本适配 + 测试通过；docker 安装 + 镜像拉取；发现超广角陀螺仪不可用 |
| 2026-08-18 | 决策切 Wide；重录标定（0.74px）；Gyroflow 验证波形 + 导出 CSV |
| 2026-08-18 | 写 `gyroflow_csv_to_imu_json.py`（四元数微分 + 合成重力）；02 适配 + SLAM 冒烟测试 |
| 2026-08-19 | 0012 mapping 视频：SLAM 建出 21 KF 地图（跟踪短，录像动作问题）|
| 2026-08-19 | 04 实测通过（3948/4434 帧检出 tag）；03/06/07 适配完成；session calibration 目录建好 |
| 2026-08-19 | 确立后续路线：机器人控制先行（单臂直连 → 树莓派双臂）；建本文档 |
| 2026-08-19 | **AM2Pro 单臂直连验证全过**：串口 ttyACM0 ✅、7 舵机标定读取 ✅、位置读回 ✅、**IK 闭环 3.6mm 误差** ✅、**夹爪开合误差 <0.3mm** ✅（详见第 12 节）|
| 2026-08-24 | 相机装机完成（`DJI_20260824094153_0013_D.MP4` 测试视频）；**SLAM 掩码定制**：用帧时序方差分析测得"相机刚体区（夹爪+臂）"= 画面底部 ~30%，据此把 02/03 掩码改为**底部 35% 横条**、**去掉 mirror 遮罩**（AM2Pro 无反光镜）；`detect_aruco.py` 同步去掉 mirror 遮罩 |
| 2026-08-24 | 新建 **`mapping_run/` 一键建图文件夹**：`run_mapping.sh`（视频+CSV → session → IMU → 02 建图 → 04 tag 检测 → 05 手眼标定 → 结果摘要）+ README；待用 0014 视频 + Wide3.csv 跑正式建图 |
| 2026-08-24 | **0014 建图结果：地图仅 5 KF、跟踪 1%**。定位根因：合成加速度=纯重力 → IMU 初始化不收敛 → Tracking.cc:1832 的"IMU 未初始化即重置地图"逻辑反复清图（tag 检测 20%、05 标定因轨迹差而崩）。**纠正录制建议：合成 IMU 需要旋转激励**（重力方向变化才有加速度信号），"平移为主"对合成 IMU 是错的。定两条路：A=重录旋转丰富的建图视频（免费快试）；B=改 fork C++ 去掉 IMU 初始化硬要求（tag 13 定尺度），重编自定义 docker 镜像（根治）。顺手修 `05_run_calibrations.py` 不检查子进程返回码的 bug |
| 2026-08-24 | **路线 B 补丁完成（共 5 处，3 个 commit）**：克隆 fork 至 `ORB_SLAM3_umi/`。commit da8bb68（Tracking.cc×3）：TrackLocalMap 统一 15 内点；丢跟踪<1s 且 IMU 未初始化不重置；LOST 时小图重置阈值 10→2 KF。commit 4a67541→151be71（LocalMapping/LoopClosing）：**IMU 初始化改为环境变量开关** `ORB_SLAM3_DISABLE_IMU_INIT=1`（合成重力会污染 tag 定好的尺度；同一镜像真 IMU 时不设变量即恢复原版行为）+ 地图合并惯性优化仅 IMU 已初始化时执行。02/03 新增 `--disable_imu_init/--enable_imu_init` 选项（默认开，自动给 docker 传环境变量，对官方镜像无害）。新增 `mapping_run/build_nomu_image.sh`；`run_mapping.sh` 自动优先用新镜像、无则回退官方。**效果：建图不再需要 IMU 初始化，尺度由 tag 13 锚定；代价是单目尺度漂移 ~1~3%（桌面任务可接受）；真 IMU 拿到后无需重编镜像** |
| 2026-08-24 | **镜像构建踩坑**：首次构建 Step 6 失败——`Thirdparty/Pangolin` 是 git 子模块，浅克隆未拉取（目录为空）。GitHub 直连超时，经 **ghfast.top 镜像**克隆并 checkout 到父仓库要求的精确版本 d4844946（install_prerequisites.sh 确认存在）。`build_nomu_image.sh` 已加子模块缺失检查 + 修复提示。**二次踩坑：`make -j` 无限并行 → 12 核 15GB 内存机器卡死**，改 `make -j4`（commit a128050）后重跑 |
| 2026-08-25 | **镜像构建成功 ✅**：`umi_orb_slam3_nomu:latest`（built d422ca329ff3）。三次踩坑全解决：① Pangolin 子模块经 ghfast.top 镜像补齐；② `make -j2` 防主机卡死；③ `--memory 6g` 太小导致 `Killed cc1plus` → 放宽 9g；④ Dockerfile 拆 6 个独立 RUN 步（DBoW2/g2o/Sophus/Pangolin/词袋/ORB-SLAM3 各自缓存，失败不重头编译）。**下一步：用 0014+Wide3 重跑建图验证补丁效果（免重录）** |
| 2026-08-25 | **补丁镜像建图验证成功 🎉**（session_0824_152417，0014 平移视频）：跟踪 **68%（2925/4283 帧，此前 1%）**、最大地图 **19 KF**、日志确认 `[patch] IMU initialization disabled; map scale is tag-anchored` + `keeping map (patched)`、**05 手眼标定成功产出 `tx_slam_tag.json`**（tag≈11cm，物理合理）。整条建图链路（视频→IMU→SLAM→tag→手眼标定）全部打通。遗留：atlas 含 4 个子图（3 次丢跟踪后重定位建新图），更平滑的录制可收敛为单图 |
| 2026-08-25 | **dbgi 深度逆向定论：文件里没有原始 IMU（见第 13 节）**。`imu_work/` 新增分析脚本与数据：`analyze_dbgi.py`、`analyze2.py`、`grid_search.py`、`parse_djmd.py`、`gyro_stream_0009.npz`、`dbgi_0009.bin`、`djmd_0009.bin` |
| 2026-08-25 | **代码上传 GitHub**：主仓库 `Zzzuuu111/umi-ampro2`（私有，适配 commit 5f4aa3f，73 文件）+ ORB_SLAM3 fork（4 补丁，master→a128050）。`.gitignore` 新增：超 100MB 原始视频、`ORB_SLAM3_umi/`、`spnav/`；推送方式 `git push github main` |
| 2026-08-25 | **环境/代码整合补全**：新增 `environment_umi_full.yml`、`environment_lerobot_alohamini.yml`、`environment_lerobot.yml`（3 个 conda 环境完整导出）、`SETUP.md`（新机一站式搭建清单）、`patches/lerobot_config_alohamini.patch`（lerobot 本地未提交改动备份，apply 方式见 SETUP.md）；git 增加 >100MB 文件 pre-commit 拦截钩子（本地生效） |
| 2026-08-26 | **重写第 1 节「项目目标」**：明确完整数据闭环——手持夹爪 + Action 4 采集演示 → UMI 管线处理 → 训练扩散策略 → 驱动单臂 AM2Pro / 双臂树莓派 AM2Pro 自主执行 |

## 13. DJI dbgi 逆向结论（2026-08-25，路径 A 定论）

**问题**：能否从 OA4 文件解出真 IMU（原始陀螺仪/加速度）？

**方法**（0009 视频，8.44s）：解析 MP4 `dbgi`/`djmd`/`tmcd` 流 → 逐字节结构分析 + 以 CSV 四元数微分角速度 ω(t) 与重力分量 R(t)·g 为"预言机"做布局网格搜索（~2000 种步长×相位×编码组合，含低通）→ 与官方 proto（telemetry-parser `dvtm_library.proto`）交叉验证。

**复查（同日二次确认，dbgi 包体 15 字段逐一枚举）**：

| 字段 | 内容 | 变化率 |
|---|---|---|
| 1 | 时间戳 varint (µs) | 每帧 |
| 2 | 传感器配置串 `OV48C40_BIN2_4032_3024_5994P_CPHY` | 恒定 |
| 3 | 拓扑串 `TPLG_OV48C40_VIDEO_4X3_2_7K_5994_QE` | 恒定 |
| 4 | 原"gyro 块"：`[8192Hz时钟24bit][段索引][int16]` 表格 | 30Hz，与运动无关 |
| 5 | 原"accel 块"：33B 头 + 低刷新数据 | 与运动无关（相关≤0.205）|
| 8 | 逐帧 float，按 ±1/256 单调斜坡（时钟漂移类）| 与 ω 无关 |
| 9 | 空 | — |
| 11 | **每帧姿态：vsync 计数 + 融合四元数 + 稳定化四元数**（与 djmd 一致的 60fps 版）| 每帧 |
| 13 | 小整数计数器组 | 与 ω 无关 |
| 14/17 | 静态查找表（65534×16、0/4/8/12…斜坡）| 恒定 |
| 15/16/18/19 | 校准曲线/参数表/64位计数器 | 缓慢更新 |

其余流与文件也已查尽：`djmd` = ~1000Hz 融合姿态 + 曝光/ISO/变焦/WB/朝向 + "DJI AC003" 设备元数据（无 accel/gyro 字段）；`tmcd` = SMPTE 时间码；SD 卡无 .dbgi/.srt 旁车文件；exiftool 无 IMU 标签。

**0014 深度交叉验证（2026-08-25）**：0014（71.5s，4283 帧）的 dbgi/djmd 结构与 0009 逐字段一致——dbgi 包体同样 15 字段（field 2/3 配置串、field 4 表头 `4064653333b340800101a001...` 与 0009 逐字节相同、field 5=3570B、field 11 逐帧姿态+vsync）；djmd FrameMeta 同样 [1,2,3,4] 字段、每帧 16-17 个融合四元数（~1000Hz）。**结论对全部录制数据成立，0009 非特例**。

**结论**：

| 流 | 内容 | 证据 |
|---|---|---|
| `djmd` | **~1000Hz 融合姿态四元数**（每帧 DeviceAttitude 数组 ≈19 个，全程 8426 个；ClipMeta `imu_sampling_rate=1000`）+ 每帧相机元数据（曝光/ISO/变焦/WB/朝向）+ "DJI AC003" 设备元数据 | protobuf 按 `dvtm_ac203.proto` 结构成功解析；四元数物理合理（初始倾角 ~60°）；**CSV 的 org_quat 与其相对旋转恒定**（同信号不同坐标系）|
| `dbgi` field 4（原称"gyro 块"）| **传感器校准/调试表，非 IMU**：6B 记录 `[A24 时钟][段索引][int16]`，A24 以 2^19 步进（8192Hz 时钟），内容 30Hz 刷新但**与运动无关** | 静止/旋转包统计不变；全部布局与 ω 相关 ≤0.48（噪声）；记录内部呈表格分段结构 |
| `dbgi` field 5（原称"accel 块"）| **同属校准表**：33B 固定头 + 低刷新数据 | 与旋转幅度相关 ≤0.205；无重力签名；逐字节位置变化率 ≤17% |

**含义**：
1. **相机不记录原始 IMU，只记录融合姿态**——独立印证了 Gyroflow 官方文档（"DJI doesn't record the IMU data directly. It only contains Quaternions"）。
2. **路径 A（解 dbgi 真加速度）判定不可行**：数据不存在于文件中。真加速度只能靠外置 IMU。
3. 现有管线（Gyroflow CSV → 四元数微分陀螺仪 + 合成重力）**已经拿到了相机能提供的最好数据**（~1000Hz 融合姿态）；可优化项：写 `djmd 直解析脚本`替代 Gyroflow GUI 导出（免 GUI 步骤、拿精确 µs 时间戳），但不增加新信息。
4. **SLAM 侧根治**（路线 B，已实现）：`ORB_SLAM3_DISABLE_IMU_INIT=1` 环境变量开关 + 自定义镜像，尺度由 tag 13 锚定。与本节结论一致：没有真 IMU 可用时这就是终点方案。

**遗留**：若未来精度仍不足 → 外置 IMU（ESP32+BNO085 等，刚性装于相机，LED 闪光/时间码同步，Kalibr 标 T_b_c1）。

## 14. 外置 IMU 路线（JY901B，2026-08-25 起）

**背景**：第 13 节定论 OA4 文件无原始 IMU → 真加速度只能外置。用户现有 **维特智能 JY901B**（亚博渠道，CP2102 USB 串口）。

**已攻克**：
| 项 | 结果 |
|---|---|
| 设备识别 | ✅ CP2102 → `/dev/ttyUSB0`（ModemManager 曾抢占端口，已 stop+disable；sudo 配置文件属主曾损坏，已修复）|
| 数据协议 | WitMotion JY901 系：`0x55 0x51/0x52/0x53/0x54` 11 字节帧+校验和 |
| **速率** | ⚠️ 串口写寄存器 0x03 无效（官方时序/双字节序/9 个地址全试）→ **Windows 官方 MiniIMU 上位机改成功**：200Hz+460800 已保存 |
| 输出内容 | 加速度+角速度+欧拉角（3 类，6600B/s @460800 容量内）|
| 摇晃测试 | ✅ **202.2Hz；加速度峰值 9.38g（>1.3g 门禁通过，含真实线性加速度）；角速度峰值 1031°/s** |

**脚本**（`imu_work/`）：`read_imu_serial.py`（扫描/解析）、`record_jy901b.py`（录制+主机时间戳+读秒提示+摇晃测试）、`jy901b_config*.py`/`jy901b_diag.py`/`jy901b_ack.py`/`jy901b_bruteforce.py`（配置尝试，存档）。

**待办（按顺序）**：
1. 刚性安装到 OA4 相机（双面胶/扎带初版，正式版 3D 打印支架）；
2. **视频-IMU 时间同步**：拍击尖峰法（视频见手拍、IMU 加速度见尖峰对齐）或 LED 闪光法；
3. 转换脚本：JY901B 记录 → gopro_slam `imu_data.json`（ACCL/GYRO，µs 时间戳对齐视频）；
4. **T_b_c 标定**：Kalibr（亚博 ROS 驱动出 topic + 相机 rosbag）；
5. 接入 `02_create_map.py`：真 IMU + 关闭 `ORB_SLAM3_DISABLE_IMU_INIT`，对比路线 B（tag 定尺度）效果。

**踩坑（2026-08-25 冒烟测试）**：`--enable_imu_init` 首跑段错误（returncode 139）。根因：`gopro_slam.cc` 主循环读 IMU **无越界检查**（原版假设 GoPro IMU 与视频同长覆盖全程）；我们的 IMU 因对齐偏移只覆盖视频 [0, 56.26s]/60.66s，播到尾部越界崩溃。修复：① `align_imu_video.py` 尾部用末样本填充至视频全长；② fork `gopro_slam.cc` 加 `last_imu_idx < imuTimestamps.size()` 越界保护（**需重建自定义镜像生效**）。正式录制规范：**IMU 先于视频开始、晚于视频停止，各留 ≥5s 余量**。

**踩坑 2（时间戳压缩 bug）**：填充后二跑仍崩 + "Empty IMU measurements vector!!!" 刷屏。仿真定位：录制脚本按 `波特率/10` 给字节赋时间戳，而模块实际数据流仅 ~6.6kB/s（差 7 倍）→ 时间轴被压缩、每帧样本不足。修复：`record_sync_session.py`/`record_jy901b.py` 改为按**每次读操作的 [t_before, t_after] 区间**给包均分时间戳（自适应速率）。需重录验证。

**真 IMU 建图验证（2026-08-25）**：
- 正式建图视频（63s，tag 13 桌面扫视）：对齐峰 0.641（峰形尖锐，±30ms→0.59）✓；IMU 全覆盖视频 ✓
- 官方镜像 + `--enable_imu_init`：72 次地图重置、14 KF → 元凶 = 原版"IMU 未初始化即重置"逻辑
- 自定义镜像 `umi_orb_slam3_nomu`：**IMU 初始化成功**，但撞上新坑——补丁分支"IMU 已初始化但 InertialBA2 未完成 → 丢跟踪重置地图"（Tracking.cc:1836）反复清图 → 2 KF
- **修复**：Tracking.cc 两处补丁改为"IMU_MONOCULAR 丢跟踪一律保留地图靠重定位"（1836 处去重置 + LOST 分支跳过 ResetActiveMap/CreateMapInAtlas）；镜像重建中（job 后台）
- 新增 `calib_accel.py`：六面静态标定加速度计 scale/bias（待跑，修正 ~14% 尺度误差）

**真 IMU 建图全链路收官（2026-08-25）**：
- 时间戳修复 + 标定方向修正（`a_true=(a-b)/s`，曾写反导致轨迹飞 15m/92% 丢帧）后重跑：
  - session2（黑胶带格子桌面，63s）：**305 KFs、丢帧 0.8%、0 重置、VIBA1/2 完成**；但收尾 13s 轨迹漂移 → tag 一致性 σ≈14/10/8.5cm、首尾漂 20~74cm
  - session4（重录 62s）：**249 KFs**；tag σ≈13.6/16.7/12.5cm、无单调漂移 → **05 标定成功，tx_slam_tag.json 生成**（tag std 0.9 分位 [11.2, 15.3, 11.6]cm）
- 剩余已知限制：① T_b_c 仍为单位阵占位（Kalibr 后置）；② `calibrate_slam_tag.py` 已适配 Osmo（画面 2688×2016、中心距离阈值 0.6→1.1，否则 95% 帧被滤掉）；③ 开局需**横向平移**制造视差（原地晃动视差不足 → Wrong initialization 清图）；④ 录制定式：IMU 先开后停、全程慢、tag 常驻、收尾更要稳
- 工具链就绪：录制（`record_sync_session.py` 带读秒/Ctrl+C）→ 对齐（`align_imu_video.py` 互相关+标定+尾部填充）→ 建图（新镜像 `umi_orb_slam3_nomu` + `--enable_imu_init`）→ 04/05 标定

**建图质量改进清单（已归档，按收益排序，暂缓执行）**：
1. **回环录制**（最高收益、零成本）：录制定式加"结尾回到开头视角"——开放式扫视漂移无法修正（session5 尾 20s 漂 4m 即此因）；黑胶带格子重复纹理还会干扰回环检测，建议格子略不规则
2. **Kalibr 标 T_b_c**（精度天花板）：当前外参单位阵占位；用亚博 ROS 驱动出 topic + 相机 rosbag 跑标准流程
3. **只录 45 秒**：时长越短漂移越少（60s+ 收尾漂移是本次主要失败模式）
4. 网格桌面若仍反复漂移 → 改"哑光不规则图案 + 立体固定物"，减少重复纹理
5. 开局横向平移 10~15cm（已验证有效：对齐峰 0.859）；原地旋转无效
6. `calibrate_slam_tag.py` 已适配 Osmo（0.6→1.1 中心阈值）；`05_run_calibrations.py` 需 PATH 含 umi env（子进程用裸 python，缺 skfda/av 即此因）
7. fork 仍有未补丁的 reset 路径（Tracking.cc:2293 Wrong initialization、1634 LOST 小图重置）——当前用录制规范规避；根治需补丁+重编译
8. session 结论：**采用 session4（249 KF、σ12~17cm、已出 tx_slam_tag.json）**；session2 尾漂、session5 尾漂 4m 弃用

**demo 重定位死锁根因与修复（2026-08-25）**：
- 现象：demo 视频（真 IMU）重定位成功但随即丢跟踪 → 反复"丢-重定位-插新 KF"循环 → 地图膨胀 + BA 每轮重跑 → 11s 视频 30 分钟跑不完；官方镜像同现象（排除补丁）；A/B 测试（建图视频重定位自己地图）同样循环
- 根因链：fork `INIT_RELOCALIZE` 成功分支的**时间戳平移 hack**（地图 KF 时间轴整体平移至负数以衔接当前帧）+ 分体 IMU 的毫秒级时间误差 → 边界处预积分 dt 失真 → 新 KF 继承垃圾速度（日志实测 1324 m/s）→ `PredictStateIMU` 逐帧外推垃圾 Vwb → 每帧运动模型预测崩 → 立即再丢
- **修复（Tracking.cc INIT_RELOCALIZE 分支）**：① 建 KF 前 `mCurrentFrame.SetVelocity(Zero)`；② 建 KF 后 `pKFcur->SetVelocity(Zero)`；③ `mLastFrame.SetPose(当前位姿)` + `mbVelocity=false` 中和 SE3 运动模型——重定位后从干净视觉状态重新起步
- 备选方案（已实现工具，未采用）：`djmd_to_imu_json.py` 从视频自带 djmd 直出合成 IMU（demo 免 JY901B/免对齐），钝化该 bug
- 镜像重建中；重建后 demo 用真 IMU + `--enable_imu_init` + `-ml 600` 重测

**demo 重定位最终根治（2026-08-25，三层补丁 + 定位模式）**：
- 诊断打印（TLMDBG）定位：重定位后前 3 帧视觉优化内点健康（58/37/15），**第 4 帧切到惯性位姿优化后内点崩到 2**（mnFramesToResetIMU=3 窗口结束）→ 位姿被坏预积分拉崩
- **最终修复：`gopro_slam.cc` 加载地图时启用 `ActivateLocalizationMode()`**（上游备而未用的正统"复用地图"模式）：纯视觉优化 + 关闭局部建图 + 不再插新 KF/不平移时间戳 → hack 整条绕过
- 三层补丁全览：① 重定位新 KF 速度清零；② TrackLocalMap 重定位后内点闸门 30→10；③ 定位模式
- 验证：demo 11s 视频 **31 秒跑完**（此前 30+ 分钟/死循环）；重定位 195→23 次；轨迹有真实运动（米级）但该首条 demo 录制偏快（IMU 中位角速度 14°/s、峰值 71°/s），丢帧 60%——**录制规范改为任务节奏的平稳动作后应显著改善**

**session6 回环建图 + demo 重验（2026-08-26）**：
- 回环配方重录（77s IMU / 70s 视频，对齐峰 **0.959** 历史最佳）：**277 KF、丢帧 0.2%**；但 LoopClosing 未触发回环（0 次 loop detected）
- 05 标定成功：tag σ 0.9 分位 [15.9, 10.7, 5.2]cm；稳健 σ x16.8/y7.5/z6.6cm（与 session4 同量级）
- **demo 2 用 session6 地图重处理：丢帧 45%（原 59%），但轨迹仍不可用**（范围 5.4m、最大跳变 5m）→ 位姿跳变来自**网格对称性误匹配**（规则黑胶带格子 BoW 认错格 → PnP 位姿飞走）+ 地图局部扭曲 σ10-17cm
- **下一杠杆（按序）**：① 胶带格子改不规则（斜线/双线/不等距，消除误匹配）；② Kalibr T_b_c；③ 回环真正触发（结尾慢扫开头区域更久）
- 遗留：Tracking.cc 的 TLMDBG 诊断打印待清理（无害）；demo 采集简化——定位模式对 IMU 依赖弱

**session7 不规则标记建图（2026-08-26，失败）+ 多子图 atlas 真 bug 修复**：
- 用户给格子胶带加了不规则标记后重录（90s）：**建图 57.5% 丢帧**（前 52s 初始化一直失败 → keep-map 补丁陆续建了 4 张子图，最后 38.5s 才跟踪成功，158 KF）；tag σ [22.3, 23.2, 18.2]cm 劣于 session6
- demo2 用 session7 图：**0 内点、600/600 全丢**。根因：`System.cc` 加载 atlas 写死 `ChangeMap(map_vector.at(0))`，而多子图图集里 map 0 是初始化失败留下的空图；重定位只在当前子图搜 → 必挂。session6 单子图所以从未暴露
- **修复**：加载 atlas 改为选**关键帧最多的子图**（`System.cc`）；顺手清理 Tracking.cc 的 TLMDBG/TLMDBG2 逐帧调试打印 → 重建镜像 `umi_orb_slam3_nomu:latest`
- **session7 已删除**（清理磁盘）；session4/6 保留为备用地图

**session8 重录（2026-08-26 10:28，已建出优质地图）**：
- 按修正配方重录（视频 71.8s / IMU 81s，开头横向平移）；IMU↔视频对齐峰 **1.018 历史最佳**，Δ=-6.523s；imu_data.json 14415 样本@200Hz 时间轴 3.5ms-71.7s 完美覆盖
- tag13 检测：**86.2% 帧可见**（0-71.7s 全程），利于 tag 定尺度初始化 + 05 标定
- 教训：**ffmpeg remux 会丢 djmd/dbgi 私有流**（时间戳越界被 mp4 muxer 丢弃），视频处理必须用 `cp` 原文件（用 ffprobe 验证 6 条流齐全再对齐）

**建图初始化根因链 + 补丁（2026-08-26 下午，4 个补丁 + 2 个崩溃修复）**：
- session8 首次建图（真 IMU 初始化）：**100% 丢帧**。对照组（禁用 IMU 初始化）38.9% 丢帧 → 证明 VI 初始化在只有 2 KF/0.55s 数据时草率跑出垃圾尺度，污染后续每帧（`IMU initialized, InertialBA2 pending` 即罪证）
- 补丁①：**tag 优先初始化**（tag 在前两帧都可见时先跑 `ReconstructWithTwoViewsAndTags`，不再让普通双目初始化抢跑——session8 里普通初始化第 0.55s 抢先成功但尺度任意、后续帧只 0-8 内点即死）
- 补丁②：初始化参考帧窗口 **0.3s→1.5s**（60fps 平稳运动攒不够视差）
- 补丁③：TrackLocalMap 内点闸门 **15→10**（新地图后续帧只有 8-14 内点）
- 补丁④：**LOST 不再杀小地图**（上游 `KeyFramesInMap()<10 → ResetActiveMap` 把每个刚初始化的 2-KF 地图立刻清空 → 死循环）；改为保留地图并**重定位回图**；RECENTLY_LOST 在 IMU 未初始化时不再无条件失败
- 崩溃①：LOST 失败分支清空 `mpLastKeyFrame` → 重定位成功后 `NeedNewKeyFrame` 解引用 NULL → segfault（catchsegv 回溯确认 `NeedNewKeyFrame+0x2ef`）；修复：不清空
- 崩溃②：定位模式加载"无 IMU 初始化"地图时 `NeedNewKeyFrame` 的惯性分支在 `mbOnlyTracking` 检查**之前**解引用 `mpLastKeyFrame`（新会话为 NULL）→ segfault；修复：`mbOnlyTracking` 提前返回 + 空指针守卫
- **最终建图结果（disable_imu_init 模式）**：丢帧 46.5%（前 33s），122 KF，帧 2000-4300 连续跟踪；**05 标定 tag σ = [1.46, 1.37, 0.87]cm —— 历史最佳**（session6 为 16/11/5cm，好一个数量级）
- **demo3**（22.1s 视频 + demo_03.npz，对齐峰 1.030）已就绪，待崩溃②修复后重定位验证
- 待办：新镜像建图 → 05 标定 → demo2 重定位验证；若达标则开始批量录 demo

**注意事项**：① 加速度尺度有 ~2-14% 误差（静止读数 1.14g），Kalibr/ORB-SLAM3 初始化会细化；② 拖线问题：USB 线随臂动，采集时沿线夹固定；③ JY901B 轴序/符号需在 T_b_c 标定时一并确定。

**GPU 与 torch 升级计划（2026-08-26 记，待执行）**：
- 硬件：**NVIDIA RTX 5050 8GB**（Blackwell sm_120），驱动 580.173.02 / CUDA 13.0，nvidia-smi 正常（仅本 agent 沙箱不可见；训练须在用户终端跑）
- umi 环境 torch 2.1.0(cu121) 太旧，不支持 Blackwell → 需升 torch≥2.7（cu128 自带）
- 步骤：`conda create -n umi_torch27 --clone umi` → `pip install torch==2.7.1 torchvision==0.22.1 accelerate "huggingface_hub==0.23.2"`（清华/上交镜像；diffusers 0.18.2 与 hub 0.34 不兼容，`cached_download` 被删 → 必须降 hub）→ 冒烟 train.py/eval_real.py → 修 diffusion_policy 旧 API（预计十几行：accelerate/transforms 等）→ 固化 `environment_umi_torch27.yml`
- 训练命令模板：`python train.py --config-name train_diffusion_unet_image_workspace task=umi_image task.dataset_path=imu_work/demo_batch_dataset.zarr training.device=cuda:0 training.num_epochs=... dataloader.batch_size=... exp_name=...`（config 无 pretrained 键，resnet weights 默认 null；zarr 为 ZipStore，UmiImageDataset 直接支持）
- 时机：先用老版本跑通第一个模型（可 CPU 冒烟），再升级，便于区分数据问题 vs 升级问题
- 部署侧（训练后）：AM2Pro FK 位姿 + IK 执行；补机器人基座↔训练系对齐（scripts/record_robot_world_hand_eye.py）；部署相机（150° 非鱼眼）标内参 + 图像预处理适配；必要时用部署相机补录微调

**session9 建图尝试 + 决策回滚（2026-08-26 下午，重要）**：
- session9（78s，近景补拍配方）：三轮建图全部失败。① tag 回退初始化在第 4 帧（基线≈0）成功 → 地图缩成 8cm 小球；② 加"初始化最小年龄 30 帧"后，双目初始化抢在第 13 帧赢 → 无尺度锚，局部 BA 中段塌缩（42 KF 全在原点）；③ 改"30 帧后 tag 优先"仍 KF 全在原点，demo4 重定位输出全零。
- 验证关键事实：**用最新镜像重跑 session8 视频（曾出 σ1.5cm 好图）直接段错误 → 三个初始化守卫补丁引入回归**。
- **决策：回滚守卫补丁，恢复被验证的配置**（tag 优先 → 双目 → tag 回退，无年龄/基线守卫；该配置 = session8 好图 + demo3 0% 丢帧）。镜像重建中（bash-40）。
- **生产路线：用 session8 地图（σ1.5cm）+ demo3 式录制**。demo3（中距离任务节奏）= 0% 丢帧；demo4（抓取阶段凑近物体）= 30% 丢帧（地图缺近景）。**批量录 demo 的录制规范：中距离、任务节奏、避免贴脸凑近物体；夹爪 tag 0/1 尽量入画。**
- 遗留课题（下次会话）：session9 近景补图路线——需先解决"tag 回退初始化在零基线时成功出退化图"+"双目初始化无尺度锚致 BA 塌缩"，再重新录近景建图视频；守卫补丁为何段错误待查（崩溃点在初始化路径）。

**demo4 全链路验证（2026-08-26）**：对齐峰 0.963；**夹爪宽度完美**（开 8.8cm→5-9s 合拢 1.1cm 抓取→9-13s 持物→13-16s 张开 7.8cm 释放）；夹爪标定采用 tag 左右互换配置（json 里 left=1/right=0，06 脚本已支持），不用重贴重录。轨迹用 session8 图 30% 丢帧（近景抓取段）。

**首批批量 demo ×10（2026-08-26 16:52，视频 0038-0047 + demo_05~14.npz）**：
- 批量录制器 `imu_work/record_demo_batch.py`（Enter 结束每条、自动编号保存）交付使用
- 处理结果：**9/10 条完美**（0.0% 丢帧、0 跳变、x 范围 0.40-0.52m）；demo_0038 中段有 ~3s 空洞（29.8% 丢，首批第一条，可 06 时视质量决定保留）
- 全部 10 条夹爪 tag 0/1 检出 100%，宽度曲线完整（合 0.005-0.012m → 开 0.074-0.090m，抓-持-放全程覆盖）
- 数据位于 `imu_work/demo_batch/demos/`（含 gripper_calibration）；IMU 对齐峰 0.48-0.87（demo_13/14 低于 0.5 但定位模式视觉为主，不影响）
- 待办：06+07 生成 zarr → train.py 扩散策略训练（GPU 仍待解决）

## 12. AM2Pro 机器人线验证记录（2026-08-19）

**命名约定（2026-08-26 起）**：手持示教夹爪命名为 **AM2UMI**（AM2Pro 夹爪 + UMI 手持示教范式，纯机械无电子，夹爪宽度用手指 tag 视觉估计，与原版 UGripper 同构）；AM2Pro 指机器人本体，仅在训练后的推理期通过 IK 执行。

**架构**（之前会话已写好代码，本轮实测）：UMI 主进程（umi 环境 py3.9）↔ Unix socket ↔ 硬件服务器（lerobot_alohamini 环境 py3.12，Feetech 舵机 + RobotKinematics IK/FK）。

| 层 | 验证内容 | 结果 |
|---|---|---|
| 1 | USB 串口识别 | ✅ `/dev/ttyACM0`（QinHeng 1a86:55d3）|
| 2 | 7 舵机 ID 映射 + EEPROM 标定读取 | ✅ ID 1~6 关节 + ID 7 夹爪，homing 偏移全部读出 |
| 3 | 位置读回（度）| ✅ 7 个角度连续合理 |
| 4 | **IK 闭环**：命令 6D 位姿 → IK → 舵机 → FK 读回 | ✅ **误差 3.6mm**（目标毫米级）|
| 5 | **夹爪开合**：宽度命令 → 舵机 → 读回 | ✅ 开 0.04m→0.0399，夹 0.005m→0.0052（误差 <0.3mm）|

**舵机标定表（实测，写死在 `am2pro_controller_server.py` 的 `_MOTOR_SPEC` 之前已确认一致）**：
shoulder_pan(ID1)/shoulder_lift(ID2)/elbow_flex(ID3)/wrist_flex(ID4)/wrist_yaw(ID5)/wrist_roll(ID6) 为角度模式，gripper(ID7) 为 0-100 模式；模型 sts3250（pan/wrist*/gripper）+ sts3095（lift/elbow）。

**测试脚本**：
- `scripts/test_am2pro_layer1.py` —— 只读测试（连总线/读标定/读位置），lerobot_alohamini 环境跑
- `scripts/test_am2pro_layer4.py` —— IK 闭环测试（启动控制器→读位姿→+2cm Z→读回误差→移回），umi 环境跑

**踩坑记录**：① lerobot `FeetechMotorsBus` 读取位置前必须先 `bus.calibration = bus.read_calibration()`；② lerobot `Motor` 对象无 `motor_id` 属性（打印时勿用）。

**URDF**：`alohamini2pro_right_arm_kinematics.urdf`（mesh-free 版，placo 可解析）已存在且本次验证通过，无需修改。

## 13. 第二批 demo ×50 处理 + v2 数据集（2026-08-27）

**50 条新 demo（视频 0048-0097，demo_01..50.npz）处理结果**：
- IMU 对齐 49/50 峰>0.5（demo_18=0.475 边缘）；tag 检测 50 条全部完成（tag0/1 检出 99-100%）
- 03 对 session8 图重定位（`umi_orb_slam3_nomu` + `-ml 600` + 3 workers，全部 exit 0）
- **质量报告**（`imu_work/quality_report.py`，阈值 = 06 的"丢帧>10 整条丢弃"）：
  - ✅ **合格 22 条**（全部 0 丢帧，3 条 ≤3 帧）：0048 0049 0051 0052 0053 0057 0060 0063 0064 0066 0067 0069 0070 0071 0073 0078 0082 0083 0085 0086 0089 0097
  - ❌ 28 条中途丢帧（18~470 帧），丢帧段多在中途（0074 丢 29-93% 整段）→ 录制姿态问题（手挡镜头/太近超出地图/镜头朝空白处），非地图问题
  - 宽度曲线全部完整：x 差 0.046-0.129 → 校准后缝 0.7-1.5cm 合 / ~9cm 开，与首批一致；`get_gripper_calibration_interpolator` 证实 actual=measured−0.039（闭=0/开=0.09）
- **决策（用户）**：先用 22+9=31 条训练，训练期间用户补录 28 条，之后合并重训

**⚠️ 重大发现：v1 zarr 图像是垃圾**：
- `imu_work/demo_batch_dataset.zarr` 的 camera0_rgb 解码出来全是近灰色平坦图（chunk0 min121/max147/std9，chunk300 min22/max63），与任何视频帧都不匹配（全视频搜索 MSE>2600），chunk 体积仅 600-3500 字节
- 结论：v1 训练（loss 1.23→0.03）是在垃圾图像上跑的，**模型无效**；lowdim（姿态/宽度）没问题
- 已用当前 07 重新生成并逐 chunk 验证（对照视频帧 MSE 必须很小）后才允许训练

**v2 数据集（imu_work/demo_batch3）**：
- 结构：demos/mapping/tx_slam_tag.json（复用 demo_batch 的 session8 标定）+ gripper_calibration + calibration + 31 条 demo（9 旧 0039-0047 + 22 新）
- 06 修复：demo_0085/0089 的 SLAM CSV 时间戳有重复块（SLAM 输出 bug），用 tag_detection.pkl 的严格递增时间覆写 timestamp 列 → 06 通过（31 集、99% 数据使用、0 丢弃）
- 06 必须用 `-nz 0.088`（tag 深度 z≈0.087，默认 0.072 会把宽度全部滤掉；v1 的宽度正常说明当时也用了 0.088）
- 07 输出 `imu_work/demo_batch3_dataset.zarr`（224×224 中心裁剪，无 rect，与 eval_real 默认一致）
- 待办：验证 zarr（图像对照视频帧、宽度、动作）→ 训练 v2（checkpoint_every=10，电源插着）

**v2 zarr 验证（2026-08-27，全部通过）**：
- 图像对照视频帧 **MSE≈2.5-2.9**（帧级完全一致，仅 JPEG-XL 有损噪声；v1 是 >2600 的垃圾）→ v1 图像损坏已解决，当前 07 管线正确
- 23331 帧 / 32 集（demo_0085 因 2 丢帧被 06 切成 2 集）；ep1-9 结构 = v1 完全一致
- 宽度 0.007-0.090m（=校准范围[闭0/开0.09]），姿态 x[-0.16,0.29] y[-0.62,-0.17] z[-0.12,0.13]，rotvec 有限，action 7D，全部无 NaN
- 训练命令（用户终端，umi_torch27，电源插着）：
  ```
  conda activate umi_torch27
  cd /home/zzzjh/universal_manipulation_interface
  nohup python train.py --config-name train_diffusion_unet_image_workspace \
    task=umi task.dataset_path=imu_work/demo_batch3_dataset.zarr \
    training.device=cuda:0 training.num_epochs=300 training.lr_warmup_steps=100 \
    training.checkpoint_every=10 logging.mode=offline \
    dataloader.batch_size=8 dataloader.num_workers=4 exp_name=am2umi_v2 \
    > imu_work/train_am2umi_v2.log 2>&1 &
  ```
  - 23331 帧 → 每 epoch ~2916 步 ≈ 11 分钟；300 epoch ≈ 57 小时（可随时停，checkpoint 每 10 epoch 保存，之后可 resume）

**v2 训练第一次运行死亡（2026-08-27 13:48-14:02，已定位修复）**：
- 症状：epoch 0 末尾整机冻结→重启，进程无 traceback；上一轮 boot 日志抓到元凶：`NVRM: Check failed: Out of memory [NV_ERR_NO_MEMORY] @ mem_desc.c`（13:49，训练启动 1 分钟）
- 原因：8GB 显卡上 模型~1.1GB + EMA + Adam 状态 ≈5.5GB，batch8 时加上系统侧 4×dataloader worker(~1.3GB/个)+主进程 2.6GB，NVIDIA 驱动系统内存分配失败 → 驱动卡死 → 桌面冻结；且 2.3GB checkpoint 双拷贝（latest+topk 两个后台线程并存）再叠 ~4.6GB 峰值
- 修复：① batch_size 8→4、num_workers 4→2（用户决策）② workspace 两处 `save_checkpoint` 改 `use_thread=False`（串行保存，避免双 payload 峰值）③ base_workspace 保存改**原子写**（tmp+os.replace，防止中断产生截断 ckpt 破坏 resume）④ 已删损坏的 latest.ckpt（PytorchStreamReader "failed finding central directory"）
- 教训：NVIDIA 驱动 OOM 不抛 CUDA OOM 异常，直接整机冻结；这台 15GB/8GB 机器训练期间不要开浏览器等重应用

---

## 14. V 型夹爪固定 Tag → AM2Pro replay 进展（2026-09-15）

### 14.1 当天目标与结论

- 目标：建立手持 V 型夹爪录制、固定桌面 Tag 轨迹、离线重定向、真实六轴安全 replay 的闭环；随后让 J7 的开合也来自同一条 demo。
- 当天结论：**J1–J6 的 fixed-tip 任务空间 replay 已成功；J7 的二值开/合时序回放已通过空载短测。**
- 尚未完成：将手持 Tag 的每帧宽度作为可靠毫米值，逐帧连续控制 J7；当前不得把该候选宽度直接视作正式训练标签。

### 14.2 相机、录制与数据质量

- 两个 EMEET WXSJ GC02 1080P 相机为同型号，但控制器状态不同导致手持画面曾较亮。手持端已改为 `auto_exposure=3`（Aperture Priority）和 `exposure_dynamic_framerate=1`；检查方式：`v4l2-ctl -d /dev/video0 --list-ctrls-menus`。
- 录制网页工具增加只读 IK/关节余量预览、相对 TCP 运动预览，以及 `X` 丢弃本次录制（不保存）的交互；录制不会控制机器人。
- demo 007/008 出现 Tag 缺失，未作为正式样本；demo 009 后固定 Tag 可见性达到 100%。
- 后续尝试 demo 010/011/015/016 时，完整 6D 方向约束会使 IK 触及关节边界；位置-only 可行但缺少工具朝向约束，不能作为最终训练数据。
- demo 030 `vjaw_demo_030_v11_tags_stable`：固定 Tag 轨迹有效；采用 fixed-tip TCP 后，离线 constrained-lookahead 候选通过。该 demo 是当天 J1–J6/J7 测试对象。

### 14.3 TCP 与离线 IK/重定向

- 明确区分：URDF `right_tcp` 是机器人固定爪上的运动学参考帧，不是机器人底座，也不必然是抓取接触点。
- 用户选择的任务点为 `fixed_jaw_inner_front_tip`（固定爪内侧前端）。手持端几何已接受：
  `calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_vjaw_fixed_tcp_v1_accepted.json`。
- 新增候选桥接变换：
  `calibration/robot_wrist_camera/tcp/vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json`。
  它仅用于把 fixed-tip demo 目标转换回 URDF `right_tcp` 给 IK；仍需要尺子/低速外点实验验证后才可晋升 accepted。
- `scripts/retarget_vjaw_demo_taskspace.py` 已支持
  `--right-tcp-to-source-tcp` 和 `--allow-provisional-tcp-frame-transform`；保留计划中的 demonstrated TCP 信息。
- demo 030 fixed-tip 候选结果：位置误差 median/max `0.000/0.000 mm`；方向偏差 median/max `5.287/8.922 deg`；最小关节余量 `11.099 deg`。这是当前可 replay 的六轴候选计划。
- lookahead IK 的“无连续安全分支”通常表示给定位置 + 方向 + 余量约束下，连续关节支路不可行；不等同于桌面点本身绝对不可达。完整 6D 方向约束曾在大横向搬运 demo 上触及关节界限，主要受工作空间、起点和姿态共同限制。

### 14.4 六轴真实 replay 与执行策略

- queue/FIFO 模式曾因 J2/J3 在反向段约 `1.5–2 deg` 跟踪滞后而超时；缩小 FIFO 会增加顿挫，增大块又可能无法及时到点。
- 现采用 `adaptive`：以实物读回相对路径锚点的误差连续减速/暂停，并以小前视量连续流式发送；它不会跳过未来路径点，因此比 FIFO 平滑。
- demo 030 fixed-tip 的 120 帧 adaptive 空载段成功，J1–J6 最大命令误差保持在 5° 保护阈值内；完整旧 fixed-tip 六轴 replay 也正常结束并返回基准。当前基准：
  `calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json`。
- 所有真实 replay 均先 `REFERENCE PREPOSITION`，到位检查通过后再开始；`--return` 在结束、中止或 Ctrl+C 时回到经验证的起始关节位。

### 14.5 J7 物理开口标定与控制

- J7 空载尺量锚点（2026-09-15，候选）：

  | 真实固定爪内前端开口 | J7 0–100 命令 |
  |---:|---:|
  | 0 mm | 4 |
  | 9 mm | 9 |
  | 28 mm | 20 |
  | 47 mm | 30 |
  | 67 mm | 40 |
  | 84 mm | 50 |

- 已写入 `example/eval_robots_config_vjaw_ros2.yaml` 的 `gripper_width_servo_table`。控制器现能做分段线性 `开口 m ↔ J7 servo` 映射，而不是旧的错误 0–100 mm 线性范围。
- `scripts/test_am2pro_gripper.py` 已做配置感知测试：命令 `67 mm` 后读回 `66.8 mm`；命令 `9 mm` 后读回 `9.2 mm`，空载通过。
- `scripts/am2pro_replay_retarget_plan.py` 默认仍禁用夹爪。显式 `--enable-gripper` 时，可在 `adaptive` 模式按 demo dataset 标签和实际路径相位发 J7 命令；不会手工写死第几秒开/关。
- 二值 J7 短测（demo 030 前 140 帧）成功：source frame 87 发闭合 `9 mm`，frame 104 发张开 `67 mm`，轨迹最终误差 `0.37 deg`，正常返回；最后按 `--gripper-final-width-m 0.009` 回到 9 mm。

### 14.6 手持宽度标签的已知限制与修复第一步

- 原手持范围文件 `vjaw_gripper_range_v1_provisional_100mm.json` 将固定爪/活动爪 Tag 相对转角线性映射为 `0–100 mm`，并在上限裁剪。`100 mm` 是旧软件上限，不是可靠的实测 100 mm；实际同构 V 型夹爪最大开口约为 `84 mm`。
- 抓住物体后，视觉标签常为约 `16–23 mm`：这表示物体将夹爪撑在该实际缝隙，不应被误解为“没有闭合”。无触觉传感器时仍可发位置闭合命令；物体会机械阻挡夹爪，但系统无法可靠确认接触、夹力或抓取成功。
- 对旧 demo30，二值阈值 `closed<=15 mm` 漏掉了第一次抓取段；改用 `closed<=30 mm`、`open>=45 mm`、连续 10 帧去抖后，离线识别为 frame 9 初始闭合、frame 115 打开、frame 187 闭合。该二值方案仅是安全动作语义测试。
- 新增候选范围文件：
  `calibration/handheld_gripper_camera/gripper_tag_sessions/vjaw_gripper_range_v2_shared_84mm_candidate.json`；它复用两套同构 V 型夹爪的 84 mm 上限。
- `scripts/process_handheld_vjaw_demo.sh` 已改用上述候选文件和 `--gripper-max-m 0.084`，未来不再生成不可能的 100 mm 上限标签。
- 已用 `--resume` 重建：
  `data/handheld_demos_vjaw/vjaw_demo_030_v11_fixed_tip_width84.zarr`；577 帧、fixed Tag 轨迹完整。其 `endpoint_clipped_frames=202`，说明旧相对转角端点仍有大量饱和，**仅改最大值不能证明中间开口准确**。
- 用 width84 zarr 生成的新 offline plan：
  `data/handheld_demos_vjaw/vjaw_demo_030_v11_fixed_tip_width84_constrained_lookahead_v1.json`，结果仍为位置 `0/0 mm`、方向 `5.287/8.922 deg`、最小余量 `11.099 deg`。
- 2026-09-16：`scripts/am2pro_replay_retarget_plan.py` 增加显式
  `--enable-gripper --gripper-mode continuous`。该模式读取计划 dataset 的每帧
  `robot0_gripper_width`，在机械臂重定时后的 path clock 上做**线性**重采样，并只在
  adaptive 路径实际推进时以默认 `10 Hz`、`1 mm` 死区发 J7；arm 因跟踪误差减速时，J7
  同步减速，不会按墙钟提前开合。默认仍为 J7 禁用，旧 binary 模式未改变。
- continuous 模式的无硬件验证通过：demo30 width84 前 120 帧离线预览范围
  `0–84 mm`，代表帧为 `0:24.3 mm, 29:24.7 mm, 59:30.4 mm, 89:1.9 mm, 119:84.0 mm`；
  fake-controller 测试确认仅按路径进度调度首点与末点 J7 目标。未打开串口。

### 14.7 下一次继续的严格顺序

1. 不连接物体，先对 width84 计划执行 J7 的**连续宽度短段空载 replay**：建议前 120 源帧；观察 J7 是否与手持视频的开口变化方向、幅度相符。不可使用旧 100 mm zarr。
2. 若出现 J7 抖动或开口幅度不符，优先检查 Tag 标签及 `10 Hz/1 mm` 命令限速，不要直接提高电机速度或放宽关节保护。
3. 用尺子验证至少 0、28、47、67、84 mm 五个手持 Tag 相对转角对应点；若线性映射不符，替换为多点插值或 V 型夹爪几何模型，再重建正式 zarr。
4. 只有连续宽度与尺量相符、六轴计划通过离线关节余量/位置误差门槛、空载完整 replay 稳定后，才录制/保留含抓取物的正式训练 demo。
5. 训练集中的 J7 标签应采用同一 `fixed_jaw_inner_front_tip` 开口定义；对于夹住物体段，需明确使用“实测实际缝隙”还是“闭合意图”作为 action 标签，不能混用。

### 14.8 手持 V 型夹爪平面 Tag 姿态分支修复（2026-09-16）

- 现象：demo30 的每帧相对 Tag 角本应在约 `7–24°` 连续变化，却有 162/576 个双 Tag 帧突跳到 `100–108°`。旧的线性角度→宽度映射把大于开口端点 `42.586°` 的值裁到物理上限 `84 mm`，因此 continuous J7 replay 会出现无实际对应的反复满开/闭合。
- 根因：小型平面 ArUco 方形 Tag 的 PnP 有两个 IPPE 位姿解；旧的单 Tag pose API 逐帧按重投影误差选解，个别帧切到另一个平面姿态分支。不是 J7 电机、开口尺量或训练标签本身的非线性问题。
- 新增离线脚本 `scripts/fix_vjaw_tag_pose_branches.py`：从现有 `tag_detection.pkl` 的角点重新求每个 Tag 的两个 `IPPE_SQUARE` 候选，按正深度、物理相对角上限（默认 `60°`）和相邻帧连续性选固定爪/活动爪的候选对。没有物理合理候选时仅移除 0/1 工具 Tag，让下游宽度在相邻有效帧间插值；桌面 13/14 Tag 不会改动。
- `scripts/process_handheld_vjaw_demo.sh` 已增加该离线步骤，使用会话内 `tag_detection_vjaw_branch_fixed.pkl` 计算宽度；同时 `convert_handheld_fixed_tag_demo.py` 支持 `--tag-detection-name`，相机轨迹仍可继续使用原始 Tag 检测导出的 CSV。
- 开合标定视频 `vjaw_range_v1_retry` 经同一修复后没有发生分支切换，仍给出闭合/最大张开代理角 `11.284°/42.586°`。已生成 `calibration/handheld_gripper_camera/gripper_tag_sessions/vjaw_gripper_range_v3_branch_fixed_84mm_candidate.json`，并把正常处理流程切换到它；物理上限仍为尺量得到的 `84 mm`。
- 重建 demo30：`data/handheld_demos_vjaw/vjaw_demo_030_v11_fixed_tip_width84_branch_fixed.zarr`。宽度范围为 `0–33.7 mm`、中位数 `16.4 mm`，不再有 `83.5 mm` 以上的伪满开帧；仅 43 帧低于闭合端点而安全裁为 `0 mm`。这个裁剪是物理下限保护，不是把普通值强制变成满开。
- 对应 fixed-tip constrained-lookahead 计划 `vjaw_demo_030_v11_fixed_tip_width84_branch_fixed_constrained_lookahead_v1.json` 通过：位置 `0/0 mm`、方向 `5.287/8.922°`、最小关节余量 `11.099°`、无分支不连续区间。
- 连续 J7 的无硬件预览通过：完整 577 帧经 arm 重定时后 J7 标签仍为 `0–33.7 mm`，按 `10 Hz`、`1 mm` 死区在实际 adaptive 路径进度上调度；`--preview-only` 未打开串口。
- 结论：**可以进行一次低速、无物体的 continuous-J7 + adaptive-J1–J6 实物验证，但该 J7 中间宽度仍是 candidate。** 通过后再用尺子抽查若干实际开口（尤其 0、约 20、约 34 mm）并决定是否把该标签用于正式训练集。

### 14.9 连续 J7 标签的时域平滑（2026-09-16）

- 首次 branch-fixed continuous-J7 空载试验仍观察到小幅反复开合。量化发现原始视觉标签的相邻帧中位变化虽仅 `0.79 mm`，但 95% 为 `5.03 mm`、最大 `15.40 mm`；整段总来回量 `768 mm`，而首尾净变化仅 `1.6 mm`。这是视觉测量噪声被逐帧命令忠实执行的结果。
- `convert_handheld_fixed_tag_demo.py` 新增 `--gripper-smooth-window-frames`：对已经做过姿态分支修复和物理上下限裁剪的宽度，先做居中中值滤波、再做同窗口均值。离线处理允许居中窗口，不会给部署推理添加延迟；最终值仍强制在物理开口范围内。
- `process_handheld_vjaw_demo.sh` 对新正式处理默认传入 9 帧（约 0.3 s）窗口。demo30 已重建为 `vjaw_demo_030_v11_fixed_tip_width84_branch_fixed_smooth9.zarr`：总来回量从 `768 mm` 降至 `125 mm`（约 84%），范围仍为 `0–31.8 mm`；43 个闭合端下界裁剪帧保持不变。
- 对应计划 `vjaw_demo_030_v11_fixed_tip_width84_branch_fixed_smooth9_constrained_lookahead_v1.json` 已离线通过。短段 replay 建议使用更保守的 J7 `5 Hz`、`2 mm` 死区；140 帧预览预计只发 18 次 J7 目标，保留约 `24→32 mm` 的打开和随后至约 `0 mm` 的闭合。
- 若该平滑版本仍出现“已经闭合后又小幅张开”，不要继续降低 J7 电机速度；应在视觉连续值上增加可审计的闭合保持（hysteresis/latch）状态。该属于“闭合意图”标签设计，须与“实际缝隙”训练标签明确区分。

## 15. V12 正式训练候选集：30 条合格 demo 的采集规则（2026-09-16）

### 15.1 当前可冻结并复用的配置

- 机械臂基准：`follow_umi_vjaw_start_v6_safe_midpoint`。
- 手持演示任务点：`fixed_jaw_inner_front_tip`。
- 手持固定 Tag 相机轨迹：桌面 Tag 13/14；录制时优先保证二者均可见，至少不能出现长缺口。
- 手持双 Tag 夹爪宽度：先修正 planar ArUco 的 IPPE 分支，再以 84 mm 物理上限、9 帧视觉宽度平滑生成标签。
- 六轴重定向：fixed-tip constrained-lookahead；使用 `right_tcp → fixed_jaw_inner_front_tip` 候选桥接变换。
- 真实空载执行：`adaptive` J1–J6；J7 连续标签使用 `5 Hz`、`2 mm` 死区。该组合目前方向正确，J7 短段测试可接受。

### 15.2 每条样本的动作和录制要求

1. 从同一 v6 基准画面和同一桌面布置开始；开始录制后再做动作。
2. 靠近左侧物体，张开、夹取、抬起，搬运至右侧，放下；动作可有自然小差异。
3. 避免剧烈翻腕、突然加速、夹爪/手遮住桌面 Tag、移出当前已验证工作区。
4. 录制结束后保留原始目录；不通过的样本不进入训练集合，但不删除原始视频和报告。

### 15.3 三层筛选和保留标准

| 阶段 | 通过条件 | 不通过时处理 |
|---|---|---|
| 视觉/转换 | Tag 轨迹无长丢失、无未修复跳变、zarr 正常生成 | 保留原始会话，标记拒绝；必要时重录 |
| 离线 IK | 位置/方向报告通过、关节余量正、无支路不连续 | 不进入训练集；检查录制姿态或工作区 |
| 空载 replay | 无保护中止、运动方向正确、无明显危险、J7 开合合理 | 不进入训练集；检查标签、TCP 或执行参数 |

只有三层都通过的样本计入“30 条”。因此目标是**30 条通过样本**，实际录制次数预计高于 30。

### 15.4 当前仍是 candidate、允许后续优化但必须留档的项

- `right_tcp → fixed_jaw_inner_front_tip` 固定偏差仍需多点尺量验证。当前小的系统落点偏差可先接受；若出现抓取失败、碰撞风险或明显数厘米偏差，必须先校正再继续采集。
- J7 中间宽度是视觉估计的实际缝隙，尚无触觉/力反馈；夹住物体时不能从标签单独确认夹力或抓取成功。
- 实时策略部署入口尚未接入同一 fixed-tip ↔ right_tcp 变换。训练 smoke test 前必须实现；它不改变已保存的 fixed-tip 训练标签。
- 当前 9 帧宽度平滑和 J7 5 Hz/2 mm 参数用于抑制视觉噪声；若闭合后仍出现不合理小开，需要增加“闭合保持”语义，而不是调快 J7。

### 15.5 批量登记表

正式记录写入 `data/handheld_demos_vjaw/v12_training_manifest.md`。每条样本应填入录制目录、三层结果、问题和最终保留状态。

### 15.6 单程序批量录制

- `scripts/record_handheld_vjaw_demo_batch.sh` 将连续启动多个交互式录制会话，默认生成登记表对应的 30 个名称：`vjaw_demo_031_v12_train_001` 至 `vjaw_demo_060_v12_train_030`。
- 每条在网页中按 `R` 开始、`S` 保存；`X` 删除当前未保存会话并自动重录同一编号；在运行该批处理的终端按 `Ctrl+C` 可提前停止整批。
- 该程序不会自动执行机器人 replay；在当前默认模式下，保存后会自动执行视觉/转换与离线 IK 筛选，只有这两层通过才推进下一编号。实物空载 replay 仍由人在场执行后填写登记表。

### 15.7 自动录制—离线筛选闭环（2026-09-16）

- `scripts/record_handheld_vjaw_demo_batch.sh` 默认已升级为逐条闭环：每次网页 `S` 保存后，自动运行固定 Tag 轨迹/夹爪 Tag 分支修复/zarr 转换，以及 fixed-tip constrained-lookahead 离线 IK；全程不打开串口、不通电、不执行机器人。
- `scripts/check_vjaw_training_episode.py` 是明确的离线判定器：要求固定 Tag 可见率至少 `95%`、refined 轨迹完整且剩余丢失帧为 `0`、计划状态为 `candidate`、最小关节余量至少 `5°`、无关节分支不连续。结果写入每条目录的 `v12_auto_screen.json`。
- 自动通过的样本保留为原登记名称，其 `processed.zarr` 路径追加到 `data/handheld_demos_vjaw/v12_auto_accepted_sessions.txt`，并在 `v12_auto_screen_ledger.jsonl` 追加一条可审计记录；随后才推进到下一条登记编号。
- 自动拒绝的录制不会删除：目录自动重命名为 `__rejected_attempt_XX` 并保留原视频、转换/IK 日志和报告，随后程序重录**同一个**登记编号。这保证目标是 30 条离线通过样本而不是 30 次录制。
- 物理空载 replay 仍不能无人自动化：它涉及真实机械臂、场地和物体状态，必须由人在场监督；因此该闭环仅自动化 15.3 的前两层，不能把“离线通过”误称为最终实物通过。
- 批处理可加 `--collection-dir <单层目录名>`，使一批样本、其 `accepted_sessions.txt` 和 `auto_screen_ledger.jsonl` 完全放在一个新的集合目录下；会话编号可以从 `000` 开始，后续扩容时沿用同一集合目录并继续编号即可。

### 15.8 `right_tcp → fixed-tip` 候选变换的第一项实体验证（2026-09-16）

- 新增 `scripts/plan_vjaw_fixed_tip_pivot_probe.py`。它离线生成一个“尖端定点旋转”计划：在模型中保持 `fixed_jaw_inner_front_tip` 的位置不变，仅将 `right_tcp` 绕其局部轴小幅旋转。若候选变换正确，实物固定爪内前端在纸上十字/尺子参考下应保持不动；若变换偏差存在，尖端会出现可测的弧形漂移。
- 初始只使用局部 `y` 轴的 `±10°`：生成计划 `calibration/robot_wrist_camera/tcp/vjaw_fixed_tip_pivot_probe_y10_v1.json`。离线 FK/IK 误差为 `0.000 mm`，最小关节余量 `18.6°`，最大单关节变化 `12.8°`；尚未执行实体动作。
- `am2pro_replay_retarget_plan.py` 增加 `--pause-after-reference`：机械臂基准到位后停住，操作者可放置纸上十字或视觉尺标，按 Enter 才开始测试；Ctrl+C 进入原有返回流程。该暂停也可用于其他受监督 replay。
- 初步判据：两个端点相对基准纸上标记的实物尖端漂移都不超过约 `2 mm` 才可视为支持该候选；`2–5 mm` 继续保留 candidate 并复测，超过 `5 mm` 则先校正变换，不能晋升 accepted。完成局部 y 后，再做局部 z 的同类测试；单一轴通过不足以验收完整 6D 变换。

### 15.9 夹持圆柱的抓取中心标定：从“点”改为“竖直轴”（2026-09-16）

#### 已做的工作

- 已完成 fixed-tip 旋转 probe 的首次实物观察：局部 y 方向基本符合预期，但固定爪尖端仍可能整体向左约 `7 mm`。用户决定当前先继续采用原有约 `26 mm` 的 `right_tcp → fixed_jaw_inner_front_tip` 固定偏差；**没有**因此修改 URDF、桥接变换或已经通过的 replay 计划。该偏差仍保留为 candidate，后续需独立复测。
- 明确 `right_tcp` 的职责：它是固定爪上的固定运动学参考帧，当前由 `right_Fixed_Jaw` 沿局部 x 偏移 `236 mm` 定义（闭合两指尖中点的候选定义）。它不能因为 J7 开合而移动。
- 对 V 型夹爪，实际“物体抓取中心”会随 J7 宽度改变。因此若未来要改善抓取落点，应建立独立的候选函数 `grasp_center(width)`；不能把它直接写回 URDF `right_tcp`，也不能覆盖已验证的 fixed-tip bridge。
- `scripts/record_am2pro_hand_eye.py` 已增加 `--manual-torque-controls`，用于人工夹住桌面圆柱后采样：
  - `G`：J1–J7 全部失能，可人工摆臂/调 J7；
  - `H`：以各电机当前编码器位置锁住 J1–J7，避免沿用旧目标；
  - `C`：仅在 `H` 后保存一帧图像、J1–J6、固定爪位姿和 J7 原始位置；
  - 保存样本新增 `gripper_position_range_0_100`，旧读取程序可忽略该字段。
  这套工具只采样，不会自动移动机械臂。每次 `G` 前仍必须用手托住机械臂；圆柱只夹持在桌面上，不抬起。
- 新增纯离线脚本
  `scripts/calibrate_am2pro_grasp_center_cylinder_axis.py`。它利用桌面 Tag 13、已验证的腕相机手眼关系和采样关节位姿，拟合“固定爪坐标系中的一点 + 桌面中一根竖直圆柱轴线”。优化只检查该点到圆柱轴的横向（XY）距离，允许不同姿态下沿圆柱高度 Z 的变化。

#### 为什么原 point-pivot 模型不适用

- 传统 point-pivot 的前提是：每个姿态都让**同一个世界固定点**重合（例如球头、尖锥凹槽或圆柱上同一高度的标记点）。
- 当前是从不同姿态、常带倾角地夹住一根竖直圆柱。即使横向抓取中心正确，接触/中心也会落在同一根圆柱的不同高度；把它强行当作一个固定三维点会把正常的 Z 高度差误判为 TCP 大误差。
- 因此“抓取位置看起来没有差很多，但 point-pivot 报几十毫米误差”不必然表示手眼、FK 或夹爪完全错误；对当前夹持方式，轴线模型才匹配几何约束。

#### 当前数据与结果（都**未**被采用为生产标定）

| 采样组 | 文件 | axis 模型最终内点 | 横向残差 mean / max | J7 内点对应实际开口 | 结论 |
|---|---|---:|---:|---:|---|
| 小圆柱 | `calibration/robot_wrist_camera/tcp_pivot/grasp_center_small_cylinder_v2.pkl` | 4 / 9（原序号 3, 4, 5, 8） | 1.845 / 3.609 mm | 约 `30.1 mm`，跨度约 `1.0 mm` | 拒绝：内点不足 |
| 大圆柱 | `calibration/robot_wrist_camera/tcp_pivot/grasp_center_large_cylinder_v1.pkl` | 4 / 10（原序号 4, 6, 7, 9） | 1.344 / 2.301 mm | 约 `54.1 mm`，跨度约 `5.6 mm` | 拒绝：内点不足且宽度不一致 |

- 当前脚本的最低接收条件是：rank 完整、至少 `6` 个内点、内点横向平均误差不超过 `3 mm`、最大误差不超过 `4 mm`。两组均只有 4 个内点，所以输出状态为 `rejected_insufficient_axis_consensus`。
- 旧的固定三维点拟合也不通过：小圆柱 v2 全部样本 mean/max 为 `10.36/20.39 mm`；大圆柱为 `8.47/16.96 mm`。这与上述“圆柱轴而非固定点”的问题一致。
- axis 脚本从各自 4 个内点计算出的局部点仅用于诊断，**禁止**把它们写入 `right_tcp`、fixed-tip bridge 或训练集；它们还没有足够共识来构成 `grasp_center(width)` 的标定点。

#### 目前遇到的问题与后续严格顺序

1. 当前主要瓶颈不是 IK 求解失败，而是手工夹持采样的一致性：横向没有每次夹在同一条圆柱轴线上；大圆柱组的 J7 开口变化尤其大。Tag/手眼噪声也会贡献误差，但现有紧密子集说明轴模型方向正确，不能用少量内点直接下结论。
2. 保留小/大圆柱原始 `.pkl` 和诊断 JSON，不删除、不从中挑选少量样本当正式标定。圆柱在同一 `.pkl` 采样期间必须固定不动；下一组可以换圆柱或换位置。
3. 对小圆柱和大圆柱各补拍/重拍，目标为每组至少 `8–10` 个姿态并得到至少 `6` 个 axis 内点。允许斜着夹、也允许沿圆柱高度不同；关键是从正面把夹爪横向中心对准圆柱轴，并把同一组 J7 实际开口控制在约 `±1 mm`。每次人工摆好后应先 `H` 锁住、确认 Tag 13 可见，再 `C` 采样。
4. 用第三个（中等半径）圆柱做**独立验证**，而不是立即参与拟合。只有小/大两组分别通过后，才拟合候选 `grasp_center(width)`；再由中圆柱确认横向残差和 replay 落点是否改善。
5. 在动态抓取中心通过独立验证前，正式数据采集仍使用当前已通过的 fixed-tip 任务点和 candidate bridge。该新标定是改善抓取接触点的后续优化，不应阻塞已经可用的视觉、IK、J1–J6 adaptive replay 与 J7 连续标签链路。

### 15.10 圆柱轴补采与 TCP 实物叠加诊断（2026-09-20）

- 小圆柱续拍合并后的候选
  `grasp_center_small_cylinder_v3plusv4_axis.json`：15 个可见样本中 7 个内点，横向残差 mean/max 为 `1.84/3.84 mm`，rank=5，满足 axis 候选条件；其 J7 开口约 `27.7 mm`。它仅说明**同一小圆柱、同一夹持规则**下存在稳定的中心点，不是通用 `grasp_center(width)`。
- 大圆柱使用 `v2+v3` 后也得到候选
  `grasp_center_large_cylinder_v2plusv3_axis.json`：7/13 内点、`2.08/3.16 mm`、J7 约 `54.8 mm`。旧 v1 与 v2 的混合组因夹持规则差异而拒绝，不能与新候选混用。
- 中圆柱独立验证组本身可拟合（7/12 内点、`1.37/2.15 mm`、约 `41.2 mm` 开口），但以小/大候选对其做线性宽度插值时误差为 `43.97 mm`，远大于 `5 mm` 限制。因此当前不能把“物体抓取中心”表示为仅由 J7 开口决定的函数；主要原因是三种直径的人工接触深度/固定爪接触规则不一致，而非单组 axis 拟合失败。
- 结论不变：不覆盖 AM_UMI 的 `right_tcp`，不把上述抓取中心写入 fixed-tip bridge。若要建立可泛化的动态抓取中心，必须先增加可重复的物理夹持基准（例如固定爪贴合深度挡块）再采样。
- 已添加 `scripts/preview_am2pro_tcp_overlay.py`：用**活动 AM_UMI FK**、现有腕相机手眼和相机内参，把橙色 `right_tcp` 与浅绿色 `fixed_jaw_inner_front_tip` 候选点实时投影到真实腕部相机画面。它只读 `Present_Position` 和相机画面，不创建控制器、不写目标位置、不开扭矩、不控制 J7。用于直观看候选固定爪点是否落在实际夹爪接触位置。
- `alohamini_ros2` 的可视化 URDF `right_tcp` 与活动 AM_UMI 运动学模型的 TCP 定义不同，因此其 RViz 模型只能用于理解坐标关系，不能替代上述实物相机叠加或用于覆盖活动配置。ROS 静态 TF 对 bridge 数值已可读出；本机 RViz 仍受 GLX/OpenGL 上下文故障阻塞。

### 15.11 Replay 后自动轨迹报告（2026-09-20）

- `scripts/am2pro_replay_retarget_plan.py` 新增可选 `--plot-after-replay`。它要求同时提供 `--trace-out`，并在 replay trace 保存、J7 最终安全开口（如启用）和机械臂返回基准后才调用离线绘图，绝不会在机械臂带扭矩执行期间等待 matplotlib。
- 该选项会在 trace 同目录自动生成一张 `*.trajectory.png` 和一份 `*.trajectory.json`：图中合并显示手持 fixed-tip、离线 IK 计划、实际关节读回 FK 的 3D/分轴对比；若启用 J7，底部面板同时显示视觉开度标签、J7 命令和编码器读回。所有产物拒绝覆盖，绘图失败只报告诊断，不改变已完成的 replay 或返回流程。
- 2026-09-20 完整复跑 `demo030` 的 smooth9 fixed-tip 计划并启用连续 J7：`replay_trace_repeat_full.json` 状态为 `finished`，全 577 源帧、11,097 个 50 Hz 读回样本完成；最大 anchor/command 关节误差为 `1.57°/1.90°`（低于 `5°` 保护阈值）。自动总览报告的 replay-FK 相对离线计划位置误差 median/max 为 `4.22/12.18 mm`，基准到首个读回的模型偏移为 `3.46 mm`；可作为开始 V12 训练**候选**采集的实体回放证据，但每条新样本仍须完成视觉/IK/空载 replay 三层门禁。
- 首条 V12 候选 `vjaw_demo_031_v12_train_001` 已完成三层验证：固定 Tag 可见率与 refined 有效率均为 `100%`、离线 IK 最小余量 `17.87°` 且无支路不连续；全 659 源帧空载 replay 的 `replay_trace_empty_full.json` 状态为 `finished`，11,178 个 50 Hz 读回样本的最大 anchor/command 关节误差为 `1.70°/1.93°`（低于 `5°` 保护阈值），replay-FK 相对计划位置误差 median/max 为 `5.30/12.94 mm`。这说明该样本的标签转换、离线运动学计划和本机执行跟踪闭环均通过；仍不将 FK 读回曲线解释为外部测量的绝对 TCP 精度或抓取成功证据。
- 离线数据回放器 `render_umi_dataset_replay.py` 的轨迹面板升级为默认的组合视图：等比例 X-Y-Z 等距投影与 XY/XZ 正投影同时显示，直接呈现完整 3D TCP 路径及其两个易读的正交分量；仍可用 `--trajectory-view xyz3d` 或 `orthographic` 单独查看其中一种。所有模式都只读取 zarr，不连接机械臂。

### 15.12 V12 离线候选训练集与 6 GB 训练配置（2026-09-20）

- 当前 `accepted_sessions.txt` 中的 11 条 V12 会话（031–041）都已通过视觉/转换和离线 IK 筛选；`scripts/merge_umi_training_sessions.py` 已把它们合并为训练器需要的 ZipStore：`data/handheld_demos_vjaw/v12_train_011.zarr.zip`，共 11 episodes / 5,083 frames。`scripts/validate_umi_dataset.py` 对合并结果为 PASS；训练数据加载 smoke test 的样本形状为 action `(16, 10)`、相机 `(2, 3, 224, 224)`。
- 录制器保存的 action 是绝对 7D（位置、rotvec、宽度），应使用 `task=umi`，而不是 `task=umi_image`；`UmiDataset` 会将它转换为策略的相对 10D（位置、rotation-6D、宽度）表示。
- 新增 `+experiment=am_umi_gpu_6gb`：保留 AdamW，但使用 FP16、batch size 1、8 次梯度累积、关闭 EMA，并把 UNet 降为 `[256, 512, 1024]`、将 CUDA allocator 限制为显存的 90%，为约 6 GB 的设备保留驱动/桌面余量。它是可训练配置，不是 4 GB 配置中仅用于路径验证的 SGD fallback。
- 可在尚未做每条实体 replay 的情况下把这 11 条作为**离线候选训练/流程验证**数据；这不改变 15.3 的实物安全结论：未经受监督空载 replay 的会话不能作为最终实机部署验收数据。训练过程不连接或移动机械臂。

### 15.13 小型视觉 Transformer 的离线策略评估（2026-09-21）

- `vjaw_v12_011_transformer_12h_20260920_183136` 已正常完成 300 epochs（epoch 0–299）；模型为 ViT-Tiny 视觉编码器 + 4 层、256 宽动作 Transformer，共约 `9.81M` 参数。推荐 checkpoint 为 `checkpoints/latest.ckpt`（也等同于 epoch 295 的最后一次保存）。
- 新增 `scripts/evaluate_umi_policy_offline.py`：只读 checkpoint 与 zarr，在固定随机种子下输出 16 步预测与记录目标的 X-Y-Z / XY / XZ / 夹爪宽度对比图，以及位置 RMSE、旋转 MAE、夹爪 RMSE 的 JSON 报告；它不创建相机、串口、控制器或机器人命令。
- 为保证推理与实际部署一致，`TransformerObsEncoder` 中的 RandomCrop/Rotation/ColorJitter 现仅在训练模式使用；`UmiDataset` 的 episode-start pose 噪声保留训练默认值 `0.05`，但离线评估明确设为 `0.0`（真实推理也不添加该噪声）。这两项仅改变评估/推理的随机增强行为，不要求重新训练。
- 在保留验证划分（11 条中随机保留 1 条 episode）均匀抽 8 个上下文的结果：位置 RMSE median/mean 为 `16.90/17.88 mm`，旋转 MAE median 为 `2.55°`，夹爪 RMSE median 为 `3.72 mm`；单段位置 RMSE 范围 `6.52–30.57 mm`。报告为 `data/outputs/vjaw_v12_011_transformer_12h_20260920_183136/offline_eval_val_final/offline_policy_report.json`，中位样本轨迹图同目录。
- 结论：策略已能产生有限、连续且大体合理的相对轨迹，姿态预测相对稳定；但较长预测段的平移转弯与夹爪闭合时序仍会出现明显偏差。此 checkpoint 适合作为离线流程/模型基线，**不应直接用于无保护的实机动作**；应先增加多样示教并在后续受监督低速测试前继续筛选。

### 15.14 V12-30 训练启动、推理安全边界与版本控制（2026-09-21）

- 新集合目录为 `data/handheld_demos_vjaw/v12_train30_20260921/`；30 条离线合格会话（031–060）已合并并验证为 `v12_train_030.zarr.zip`，共 **30 episodes / 13,335 frames**。该 ZipStore 是新的训练输入；原始视频、单条处理结果、W&B 和 checkpoint 继续只保存在本机，不进入 Git。
- 6 GB 级 GPU 的视觉 Transformer 配置是 `+experiment=am_umi_transformer_gpu_6gb`：ViT-Tiny、动作 Transformer `n_emb=256`/4 层/4 头、FP16、batch size 1、梯度累积 8、关闭 EMA，并将 CUDA 可用显存比例限制为 90%。这不是原版 ViT-Base 预训练模型配置。
- 新增 `scripts/start_vjaw_transformer_12h.sh`，在 `AM_UMI` 环境下启动**全新**训练，使用 12 小时硬上限和最多 160 epochs；每 5 epochs 保存 checkpoint，保留 5 个 top-k。启动与看日志：
  ```bash
  conda activate AM_UMI
  cd ~/AM_UMI/umi
  bash scripts/start_vjaw_transformer_12h.sh

  C=data/handheld_demos_vjaw/v12_train30_20260921
  tail -f "$C/latest_transformer_12h.log"
  ```
  训练不连接或移动机械臂。12 小时是硬停止条件，160 是最大 epoch 数而非保证完成数。
- 已明确区分两条执行链路：replay 使用**已知完整** fixed-tip 示教轨迹，经 `right_tcp ↔ fixed-tip` 桥接、离线 constrained-lookahead IK、连续关节支路/限位/裕量检查后才执行；策略推理则按当前相机和机器人状态滚动预测未来短段相对末端动作，模型本身不执行该离线 IK 规划。推理输出仍需转成目标末端位姿并经过 IK/控制器；当前不得把它等同于已验证 replay 的整段安全门禁。实机策略部署前应把预测短轨迹接入同一 TCP 桥接、连续 IK 和限位检查，并先低速、空载、人工监督验证。
- 项目根目录 `~/AM_UMI` 已初始化为 Git 仓库，远程为 `https://github.com/Zzzuuu111/AM_UMI.git`。标准结构为：`main` 保存初始稳定基线，`am-umi-vjaw-training` 为当前开发分支；当前训练启动脚本提交 `e27365d` 位于开发分支，尚需在具有 GitHub 凭据的终端执行 `git push` 后同步该最新提交。`.gitignore` 已排除数据集、视频、W&B、训练输出、权重和本地标定原始文件；环境说明见 `ENVIRONMENT_AM_UMI.md`。
