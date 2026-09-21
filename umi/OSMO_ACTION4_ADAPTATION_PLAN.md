# Plan: UMI 建图 pipeline 适配 DJI Osmo Action 4

> 目标：把 UMI 离线建图/数据生成 pipeline 从 GoPro 迁到 DJI Osmo Action 4（手腕相机），
> 用于 AM2Pro 双臂的数据采集与训练。
> 本文件只覆盖「建图 / 数据生成」链路，机器人控制适配见 `HARDWARE_ADAPTATION_PLAN.md`。

---

## 0. 当前状态（2026-08 实测更新）

**已打通（实测）**：

| 环节 | 状态 | 说明 |
|---|---|---|
| 相机模式 | ⚠️ 用 **Wide（广角）**，不用超广角 | 超广角 dbgi 陀螺仪格式 Gyroflow 不支持且逆向不通；Wide 是官方支持模式 |
| 内参标定 | ✅ `scripts/calibrate_fisheye_intrinsics.py` | Wide 2.7K 4:3：f=1054.20, cx=1344.19, cy=990.63, k1..4=0.174/0.079/0.002/-0.044，重投影 0.74px |
| 00_process_videos | ✅ 适配 | DJI 时间码（`;` 分隔）+ 序列号缺失回退 'OSMO4' |
| IMU 提取 | ✅ **Gyroflow CSV 路线**（弃用 dbgi 逆向） | `scripts/gyroflow_csv_to_imu_json.py`：四元数微分→陀螺仪 + 合成重力→加速度 |
| 02_create_map | ✅ 适配 | 挂载 `osmo4_fisheye_setting_v1_720.yaml`，掩码尺寸 2016×2688；SLAM 跑通（21 KF 地图 + 轨迹）|
| 04_detect_aruco | ✅ 格式兼容 | 内参 json 已按 UMI `FISHEYE/focal_length` 格式输出 |
| 05_run_calibrations | ✅ 无需改动 | 无相机硬编码 |
| 06_generate_dataset_plan | ✅ 适配 | 序列号 `.get(..., 'OSMO4')` 回退 |
| 07_generate_replay_buffer | ✅ 适配 | 内参文件名自动回退（gopro → osmo4）|
| Docker / orb_slam3 镜像 | ✅ 已装已拉 | 需国内镜像源 |

**已知遗留（影响精度，不阻塞）**：

1. **建图视频动作质量**：当前 mapping 视频跟踪只有末尾 ~2.7s（平移不足/有甩动）。
   重录要求：平移为主、慢而稳、全程平滑、tag 13 入画。
2. **加速度是合成重力**（无真实线性加速度）→ 偶发 `scale too small`。
   后续可选：解 dbgi field5 真实加速度，或外置 IMU。
3. **IMU T_b_c1 外参是单位矩阵占位**（settings YAML 中 `IMU.T_b_c1`）。
   手持慢速扫图可容忍；若要精度需 Kalibr 标定。
4. 掩码 `mirror=True` 的镜像区域是按 GoPro 腕带结构画的，AM2Pro 实际安装后需按真实视场重画。

**下一步（按顺序）**：

1. 相机装上 AM2Pro 手腕/夹爪 → 重画掩码
2. 重录规范 mapping 视频 → 建出高质量地图（`There are 1 maps` + 长轨迹）
3. 录 gripper calibration（相机在夹爪上、tag 可见、开合夹爪）
4. 跑 05 手眼标定 → 06 数据计划 → 录 demo → 07 replay buffer
5. 训练（diffusion policy）

---

## 0. 结论（TL;DR）

- **建图算法本身不用改**：ORB-SLAM3 单目惯导 + ArUco 定位 + SLAM-tag 手眼标定，这套流程与相机品牌无关。
- **但 pipeline 里塞满了 GoPro 专属的标定文件、硬编码常量、元数据/tag 解析和时间码逻辑**，
  换 Osmo Action 4 后这些必须逐一替换，否则不是精度崩就是直接 `KeyError` 跑不通。
- 风险最高、最可能卡住的是两个点：**① DJI 陀螺仪遥测的提取与 IMU→相机外参（Tbc）标定**；
  **② GoPro Labs 时间码缺失导致的多相机亚秒级同步**。这两个点有「绕开」的替代方案（见第 4 节）。

---

## 1. 现状：pipeline 里所有 GoPro 专属点

| # | 位置 | 内容 | 换 Osmo 后要做什么 |
|---|---|---|---|
| 1 | `calibration/gopro_intrinsics_2_7k.json` | GoPro 10 Max Lens Mod 2.7K（2704×2028, f=796.85）鱼眼内参 | 重新标定 Osmo 内参 |
| 2 | `scripts_slam_pipeline/02_create_map.py` L80 / `03_batch_slam.py` L117 | ORB-SLAM3 settings `gopro10_maxlens_fisheye_setting_v1_720.yaml`（在 docker 镜像内） | 生成 Osmo 版 settings（内参 + Tbc） |
| 3 | `scripts_slam_pipeline/01_extract_gopro_imu.py` | openicc `extract_metadata_single.js` 按 GoPro GPMF 结构抽陀螺/加速度 | 换支持 DJI 的提取器 |
| 4 | `scripts_slam_pipeline/02_create_map.py` L62 / `03_batch_slam.py` L99 | `np.zeros((2028, 2704))` 掩码尺寸 | 改实际分辨率 |
| 5 | `umi/common/cv_util.py`（`get_mirror/gripper/finger_canonical_polygon`） | 按 GoPro 视场 + UMI 夹爪 + 镜子画的多边形 | 按 Osmo 视场 + AM2Pro 夹爪重画 |
| 6 | `scripts/calibrate_slam_tag.py` L81 | `img_center = np.array([2704, 2028])/2` | 从实际帧尺寸取 |
| 7 | `scripts_slam_pipeline/06_generate_dataset_plan.py` L102/L104 | `cam_to_center_height=0.086`、`cam_to_mount_offset=0.01465`（GoPro 光心到安装螺丝） | 实测 Osmo 安装几何 |
| 8 | `scripts_slam_pipeline/06_generate_dataset_plan.py` L85 | `tcp_offset=0.205`（UMI 夹爪尖到安装螺丝） | 换 AM2Pro 夹爪实测值 |
| 9 | `00_process_videos.py` L65 / `06_generate_dataset_plan.py` L163 / `umi/common/exiftool_util.py` L3 | `QuickTime:CameraSerialNumber`（区分左右手相机） | DJI 是专有 UUID/com.dji atoms，换 tag 名 |
| 10 | `scripts/check_gopro_orientation.py` L32 | `QuickTime:AutoRotation` | 适配或跳过 |
| 11 | `umi/common/timecode_util.py` L35 | `stream.metadata['timecode']` + `creation_time`（GoPro Labs 时间码） | DJI 无 timecode track，换同步方案 |
| 12 | `scripts_slam_pipeline/07_generate_replay_buffer.py` L59-62 | 硬编码 `gopro_intrinsics_2_7k.json` | 改路径 |
| 13 | `run_slam_pipeline.py` L84 | 硬编码 `gopro_intrinsics_2_7k.json` | 改路径 |
| 14 | `umi/common/cv_util.py` `convert_fisheye_intrinsics_resolution` | 假设「垂直不裁剪、水平对称裁剪」（GoPro SuperView） | Osmo FOV 档位裁剪方式不同，尽量按录制分辨率直接标定 |

> 说明：内参其实有**两套**——aruco 检测用全分辨率 `gopro_intrinsics_2_7k.json`；
> ORB-SLAM3 用 720p settings（文件名 `v1_720`）。Osmo 要分别准备两套，不要混用。

---

## 2. 重新适配步骤

### 阶段 A：固定参数 + 标定（无 GoPro 依赖，最稳）

**A0 固定相机设置（一次性）**
- Osmo Action 4 装到 AM2Pro 夹爪上，安装件固定后不再动。
- 固定：FOV 档（建议 Ultra Wide 155°）、分辨率（建议 2.7K）、帧率 60fps。
- **关闭 RockSteady / HorizonSteady / EIS 全部电子防抖**（SLAM 要原始帧，EIS 会裁剪并改变内参）。
- 验收：录一段确认无防抖、无镜头切换、元数据可读。

**A1 标定鱼眼内参（产出 `osmo4_intrinsics_2_7k.json`）**
- 用 `scripts/gen_charuco_board.py` 打印 Charuco 板，录多角度标定视频（**四角要入画**，鱼眼边缘最关键）。
- 用普通 OpenCV `cv2.fisheye.calibrate` 即可（`gen_charuco_board.py` + `draw_charuco_detection.py` 已配套），
  **不必依赖 openicc**（openicc 的价值在相机-IMU 联合标定，纯内参不需要它，见第 4 节）。
- 输出 UMI 格式 JSON：`intrinsic_type: FISHEYE` + 4 个径向系数（照 `gopro_intrinsics_2_7k.json` 结构）。
- 验收：`final_reproj_error` 与 GoPro 的 ~0.29 同级；`draw_charuco_detection.py` 可视化角落 tag 对齐。
- ⚠️ 直接按录制分辨率标定，不要跨分辨率换算（`convert_fisheye_intrinsics_resolution` 的 GoPro 假设不成立）。

### 阶段 B：IMU + ORB-SLAM3 settings（建图核心，风险最高）

**B1 提取 DJI 陀螺仪（产出 `imu_data.json`）**
- Osmo Action 4 把陀螺/加速度写进 MP4（GPMF 格式，[Gyroflow 支持 DJI Action 3/4](https://docs.gyroflow.xyz/app/getting-started/supported-cameras/dji.md)），
  但 `01_extract_gopro_imu.py` 用的 openicc `extract_metadata_single.js` 是 GoPro 专用，大概率解析不出 DJI。
- 二选一：
  - 改 extractor JS 支持 DJI 的 GPMF 设备 ID / 流布局；
  - 用 Gyroflow 的 parser 或 [DJI 遥测提取工具](https://goprotelemetryextractor.com/tools-for-dji-action) 抽数据，转成 openicc 相同的 `imu_data.json` schema。
- 验收：`imu_data.json` 有合理时间戳序列，**单位/坐标系与 B2 的 settings IMU 模型一致**。

**B2 生成 ORB-SLAM3 settings（产出 `osmo4_..._fisheye_setting_v1_720.yaml`）**
- 仿 `gopro10_maxlens_fisheye_setting_v1_720.yaml` 写 Osmo 版：
  - `Camera.type: KannalaBrandt8` + 换算到 SLAM 处理分辨率的内参；
  - `Camera.fps: 60`；
  - **`IMU.Tbc`（IMU→相机外参）**：用相机-IMU 联合标定（openicc 支持 DJI 后，或 Kalibr）标定；标不出来就用机械测量近似，但误差会传导到轨迹尺度。
- 改 `02_create_map.py` / `03_batch_slam.py` 的 `--setting`，把 YAML volume 挂载进 `chicheng/orb_slam3:latest`。
  （⚠️ 本机当前无 docker，这是前置条件。）
- 验收：`02_create_map.py` 跑通，`slam_stdout.txt` 无大量丢帧，`mapping_camera_trajectory.csv` 轨迹平滑。

> **若 B1/B2 的 IMU 链搞不定，直接走第 4 节的「纯单目 + ArUco 定尺度」替代方案**，跳过 B1/B2。

### 阶段 C：掩码 + 元数据 + 几何（标签精度）

**C1 重画掩码**
- 装好相机后录一段含夹爪（和镜子，如有）的视频，在 canonical 坐标下重测
  `get_mirror/gripper/finger_canonical_polygon` 的多边形。
- `02_create_map.py` L62 / `03_batch_slam.py` L99 的 `np.zeros((2028, 2704))` 改实际分辨率。
- 验收：`scripts/gen_image_mask.py` 输出的掩码叠加视频，夹爪/手指被正确盖住且不误伤场景。

**C2 适配元数据 + 时间码**
- `exiftool -a -G1 -s <osmo4.mp4>` 找序列号、录制方向、创建时间的实际 tag 名（DJI 是专有 UUID/com.dji atoms）。
- 改 `00_process_videos.py`、`06_generate_dataset_plan.py`、`umi/common/exiftool_util.py` 的 tag 读取。
- **`timecode_util.py` 是 GoPro Labs 专属**：DJI 无 `timecode` track。多相机亚秒同步改方案（见第 4 节）。
- 验收：`00_process_videos.py` 能正确分出 `mapping / demo_<serial>_<时间> / gripper_calibration_*`。

**C3 实测安装几何**
- 实测 Osmo 光心到安装螺丝（`cam_to_mount_offset`）、光心高度（`cam_to_center_height`）、
  AM2Pro 夹爪尖到安装螺丝（`tcp_offset`）。
- 验收：`dataset_plan.pkl` 里 `tcp_pose` 数值与夹爪实际运动量级一致。

**C4 改 `calibrate_slam_tag.py` L81**：`img_center` 从实际帧尺寸读取。

**C5 改硬编码内参路径**：`run_slam_pipeline.py` L84、`07_generate_replay_buffer.py` L59-62。

### 阶段 D：端到端 + 部署

**D1 小规模闭环**
1. 录一个最小 session：mapping + 1 个 demo + gripper_calibration（夹爪上贴 tag 0/1）。
2. `python run_slam_pipeline.py <session>`，逐段检查：SLAM 丢帧率低、`calibrate_slam_tag.py` 标准差 < 1cm 量级、gripper 检测率 > 90%。
3. 训练小 checkpoint，`eval_real.py` 部署回放验证。

**D2 部署观测一致性（易踩坑）**
- `eval_real.py` 的 `FisheyeRectConverter`（`-sf sim_fov -ci <intrinsics>`）必须用 Osmo 新内参。
- GoPro 原方案是 **clean HDMI → 采集卡 → UVC**（`MultiUvcCamera` 读 `/dev/video*`）。
  Osmo Action 4 走 **USB-C UVC 网络摄像头模式**，其 FOV/分辨率/帧率与原生录制不一致，内参也不同。
  要么标定 UVC 模式内参，要么部署先用录制回放（replay 脚本）绕开，保证训练/部署观测分布一致。

---

## 3. 上一轮方案里的错误与遗漏（修正）

1. **遗漏了两套内参**：之前只说「换 `gopro_intrinsics_2_7k.json`」，实际还有 ORB-SLAM3 的 720p settings
   （#2）和 `07_generate_replay_buffer.py` 里另一个硬编码路径（#12）。现已补齐。
2. **遗漏了时间码同步问题（重要）**：`umi/common/timecode_util.py` 读 `stream.metadata['timecode']`，
   这是 **GoPro Labs 固件**才写的时间码。DJI 没有 timecode track，`00_process_videos.py` 会在
   `mp4_get_start_datetime` 处 `KeyError`，多相机亚秒级对齐失效。之前完全没提到，这是第 2 大风险点。
3. **「用 HDMI 采集卡」不适用于 Osmo Action 4**：之前类比 GoPro 的部署方式不严谨。Osmo Action 4 没有
   HDMI 输出，部署靠 USB-C UVC 模式，需单独处理其内参。
4. **内参标定不必依赖 openicc**：纯鱼眼内参用普通 OpenCV `fisheye` + Charuco 板即可，openicc 只在做
   相机-IMU 联合标定（Tbc）时才需要。之前把两者混在一起了。
5. 其余（掩码尺寸、多边形、`calibrate_slam_tag.py` 硬编码、几何常量、`QuickTime:*` tag）上一轮判断正确，保留。

---

## 4. 更好的替代方案

### 方案一（推荐）：纯单目 SLAM + ArUco 定尺度，跳过 IMU 链
- 桌面 tag（`aruco_config.yaml` 里 marker 12，边长 0.16m）本来就是用来定义世界系的，**尺度可以由 tag 提供**，
  IMU 不是必须。
- 好处：直接跳过第 B1/B2 步（DJI IMU 提取 + Tbc 标定），这两个是最大风险源；也省掉 openicc/GoPro 依赖。
- 代价：快速旋转 / 低纹理时单目 SLAM 比单目惯导鲁棒性差。桌面操作场景通常够用，值得先试。

### 方案二：同步方案替代（针对 GoPro 时间码缺失）
- 采集端「打板 / 闪光」做视觉同步，把 `06_generate_dataset_plan.py` 的 wall-clock 对齐改为视觉事件对齐；
- 或运行时用 `MultiUvcCamera` 统一采集（所有相机同一时钟），彻底绕开离线时间码同步。
  （但 UVC 模式视场/帧率受限，见 D2。）

### 方案三：内参标定用 OpenCV，相机-IMU 用 Kalibr
- 纯内参：OpenCV `cv2.fisheye.calibrate`（零 docker、零 GoPro 依赖）。
- 若坚持 IMU 方案：IMU 数据用 Gyroflow parser 抽成 rosbag，相机-IMU 外参用 Kalibr 标，比硬改 openicc 更通用。

### 建议顺序
`A0 → A1 →（先试方案一绕开 B1/B2）→ C → D`，把 IMU 方案留作「单目不够稳时」的升级项。

---

## 5. 风险清单

| 风险 | 影响 | 缓解 |
|---|---|---|
| DJI 陀螺遥测提取失败 | 建图卡在 B1 | 方案一（纯单目 + ArUco 尺度） |
| Tbc 标不准 | 轨迹尺度/漂移 | 方案一，或 Kalibr + 充分激励轨迹 |
| 时间码缺失 → 多相机不同步 | 双臂/多相机 demo 对不齐 | 方案二（打板视觉同步 / UVC 统一采集） |
| Osmo FOV 档位裁剪方式与 `convert_fisheye_intrinsics_resolution` 假设不符 | 跨分辨率内参错 | 按录制分辨率直接标定 |
| UVC 部署内参 ≠ 录制内参 | 训练/部署分布漂移 | 标定 UVC 模式，或先回放部署 |
| 本机无 docker | SLAM/IMU 容器跑不了 | 装 docker，或本地编译 ORB-SLAM3 fork |
| 掩码/几何常量标错 | 标签带系统偏差 | 逐项可视化验收（C1/C3） |

---

## 6. 参考

- [UMI SLAM fork（cheng-chi/ORB_SLAM3）](https://github.com/cheng-chi/ORB_SLAM3) / [SLAM docker](https://hub.docker.com/r/chicheng/orb_slam3)
- [OpenImuCameraCalibrator（GoPro 专用，作者 Steffen Urban）](https://github.com/urbste/OpenImuCameraCalibrator)
- [Gyroflow 支持的 DJI 相机](https://docs.gyroflow.xyz/app/getting-started/supported-cameras/dji.md)
- [DJI Action 遥测提取工具](https://goprotelemetryextractor.com/tools-for-dji-action)
- [DJI Osmo Action 4 规格](https://www.dji.com/hk/osmo-action-4/specs) / [网络摄像头模式说明](https://support.dji.com/help/content?customId=zh-cn03400006962&spaceId=34&re=CN&lang=zh-CN&documentType=artical&paperDocType=paper)
