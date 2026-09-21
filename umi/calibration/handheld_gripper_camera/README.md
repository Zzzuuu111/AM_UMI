# 手持夹爪相机：建图与训练数据准备

此目录只放手持夹爪上的相机文件。相机、手持夹爪和夹爪 Tag 的相对安装关系从建图到全部示范录制结束都必须保持不变。

## 已有 / 待完成

| 项目 | 状态 | 说明 |
|---|---|---|
| 相机内参 | 已有候选 | `intrinsics/emeet_handheld_1920x1080_30fps_fisheye_20260831.json`，误差 0.740 px。先确认它对应实际那台 EMEET。 |
| IMU 加速度标定 | 已完成并独立验证 | `imu/jy901b_accel_v1.json`；独立静止姿态校正后重力为 9.8083±0.0134 m/s²。 |
| IMU 陀螺零偏 | 已完成 | `imu/jy901b_gyro_bias_v1.json`；最终安装状态下测得零偏约 ±0.003°/s。 |
| IMU ↔ 相机轴方向 | 已完成（粗核验） | `imu/jy901b_camera_axis_check_v2.json`：相机 +X→IMU +Y、相机 +Y→IMU -Z、相机 +Z→IMU +X；不替代后续精确相机-IMU 外参标定。 |
| 相机 ↔ IMU 精确旋转与时间偏移 | 已完成并独立验证 | 当前采用 `camera_imu/emeet_jy901b_rotation_time_v4_uvc_source.json`：偏移 -22 ms。独立 `session_v13_uvc_source_rotation_validation/` 验证相关性 0.9847、RMSE 0.0944 rad/s。 |
| 相机 ↔ IMU 平移外参 / 加速度噪声候选 | 已完成并独立验证 | `camera_imu/emeet_jy901b_translation_noise_v4_uvc_source.json`；独立验证的重力拟合 RMSE 0.1025 m/s²。 |
| 相机 + IMU 同步原始采集 | 已通过 | EMEET 1920×1080 MJPG 以 PyAV V4L2 读取，实测 29.90 fps；JY901B 加速度/陀螺同步保存。见 `sync_tests/emeet_jy901b_smoketest_v7/`。 |
| 相机与夹爪固定 | 已完成 | 固定支架后不要再转动或拆卸相机。 |
| EMEET + JY901B SLAM 适配 | 已完成视觉链路与时间轴修复 | UVC PTS/SCR、严格递增帧时间轴、IMU 导出与 ORB-SLAM3 初始化空指针已修复；固定 Tag 视觉路线可用。 |
| 固定 Tag 米制相机轨迹 | 已通过 v5 离线验证 | `data/handheld_mapping/emeet_jy901b_mapping_v5_balanced_motion/camera_trajectory_fixed_tag13_metric_v5.csv`。世界坐标系为桌面 Tag 13，3456/3542 帧有效（97.57%），只适用于 Tag 13 保持可见的录制。 |
| 通用 ORB-SLAM 地图 / 重定位 | 暂不用于正式数据 | 发现 Tag 初始化中存在未初始化畸变参数，已在新的 `am_umi_orb_slam3_emeet_tagfix:latest` 镜像中修复；真米制尺度下，旧的两帧稀疏初始化仍不稳定，需单独改进后再启用。旧 ORB CSV 不可视为米制真值。 |
| 完整 VIO（视觉 + IMU） | 暂不用于正式数据 | 已消除初始化段错误；当前验证集可通过 IMU 初始化和 VIBA1，但 VIBA2 出现 `scale too small`，随后视觉内点不足而掉跟踪。暂时优先使用固定 Tag 的视觉直测轨迹。 |
| 夹爪 Tag / 开度标定 | 待执行 | 正式示范前把指端 Tag 装好；它用于恢复夹爪位姿和开度，不能省略。 |
| SLAM-Tag 对齐 | 待执行 | 用固定的桌面 Tag 将 SLAM 轨迹变为有尺度的任务坐标。 |
| 相机延迟 | 建议测量 | 录制动作很快或要精确时间同步时再做。 |

## 录制会话

使用 `imu_work/record_handheld_umi_session.py`，它先打开实时预览；在预览窗口按 `r` 才会开始保存 `raw_video.mp4`、`frame_timestamps.csv` 与 `imu_raw.npz`，按 `s` 停止视频并保留 2 秒 IMU 尾部。默认还会同时读取 EMEET 的配对 UVCH 元数据节点：保存 `raw_uvc_payload_headers.bin`，并从 PTS/SCR 生成 `frame_timestamps_uvc_source.csv`。原始 `frame_timestamps.csv` 永远保留，便于审计或回退。完成相机-IMU 外参和相机延迟标定前，输出仍是**原始会话**，不能直接当作正式视觉惯性 SLAM 输入。

后续的相机—IMU 旋转/平移标定、IMU 导出与固定 Tag 轨迹导出，会自动优先采用 `frame_timestamps_uvc_source.csv`；旧会话没有该文件时则自动回退到原始主机接收时间轴。可对旧的已录制会话离线补建该文件：`python -u imu_work/align_uvc_payload_timestamps.py --session-dir <会话目录>`。

该派生时间轴必须严格递增：一张解码视频帧只能匹配一条递增的 UVC PTS 事件。旧版本的“逐帧最近邻”配对会偶尔把相邻两帧映射到同一 PTS，导致零时长帧间隔和空 IMU 预积分；当前脚本已改为在录制窗口内一对一、单调匹配。对已经用旧脚本处理过的会话，应重新运行上面的对齐命令，再重新导出 `imu_data`，保留旧 JSON 作为审计记录。

每个 mapping/demo 会话都应先引用已保存的 AM2Pro 任务基准完成画面对齐，再按 `r` 开始保存。手持画面本身可另存为参考，但它不替代包含 AM2Pro TCP/关节状态的机械臂基准。

手持夹爪如何引用 AM2Pro 的任务起始基准、何时按 `R` 开始录制，以及每个命令参数的说明，见
[VIEW_REFERENCE.md](../../VIEW_REFERENCE.md) 的“手持夹爪录制前”部分。

## 当前推荐：固定桌面 Tag 13 的米制轨迹

当任务桌面上的 **Tag 13** 在录制全过程中可见时，不必等待通用 SLAM 或 VIO：相机可从已知尺寸的固定 Tag 直接得到米制位姿。该方式的世界原点就是 Tag 13，不依赖 IMU，也不会创建可在无 Tag 画面中重定位的通用地图。

在 `(AM_UMI)` 环境、`~/AM_UMI/umi` 目录中，对一次已录制会话运行：

```bash
python -u imu_work/export_fixed_tag_camera_trajectory.py \
  --session-dir data/handheld_mapping/<session_name> \
  --intrinsics calibration/handheld_gripper_camera/intrinsics/emeet_handheld_1920x1080_30fps_fisheye_20260831.json \
  --tag-id 13 \
  --trajectory-name camera_trajectory_fixed_tag13_metric.csv
```

命令只读取录制文件。它会拒绝覆盖已有 CSV；若 Tag 13 可见率低于 95%，会保留诊断结果但以失败退出，此时不要把轨迹用于正式示范处理。

导出后应生成一个新文件做孤立跳变剔除、短缺口插值和轻量平滑。原始 CSV
不会被覆盖；超过 5 帧的缺口继续标记为 lost：

```bash
python -u imu_work/refine_fixed_tag_camera_trajectory.py \
  --input data/handheld_mapping/<session_name>/camera_trajectory_fixed_tag13_metric.csv \
  --output data/handheld_mapping/<session_name>/camera_trajectory_fixed_tag13_metric_refined.csv \
  --max-gap-frames 5 \
  --smooth-window 5
```

当前 `mapping_v5_balanced_motion` 的 refined v2 结果为：3,479/3,542 帧有效
（98.22%）；剔除 5 个孤立跳变、插补 28 个短缺口，剩余 63 帧分布在四段
12–20 帧的长缺口中并继续保持 lost。99% 的平滑修正小于 3.8 mm 和 0.26°。

### 接入 UMI 数据集计划

固定 Tag CSV 已经位于 Tag 世界坐标系，不需要也不能再次应用
`tx_slam_tag.json`。`06_generate_dataset_plan.py` 已增加显式的轨迹来源选择，并在
fixed_tag 模式检查 `.fixed_tag_report.json` 或 `.refinement_report.json`，防止把
ORB/VIO 轨迹误当成固定 Tag 轨迹。

正式生成前还必须完成 EMEET 手持相机到夹爪 TCP 的固定几何标定，文件说明见
`gripper_geometry/README.md`。标定完成后使用：

```bash
python -u scripts_slam_pipeline/06_generate_dataset_plan.py \
  --input <任务项目目录> \
  --camera-trajectory-name camera_trajectory_fixed_tag13_metric_refined.csv \
  --trajectory-frame fixed_tag \
  --camera-tcp-geometry calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_camera_to_tcp_v1.json
```

fixed_tag 模式默认允许每个选定视频区间最多 5% lost 帧，但不会使用这些帧：长
缺口会把演示切成多个有效片段，短于 `--min_episode_length` 的片段会被丢弃。
脚本同时生成 `dataset_plan.provenance.json`，记录实际使用的轨迹文件、坐标系与
相机-TCP 几何文件。

### 保留 VIO 的要求

固定 Tag 是当前主路线，但每段新会话仍需保留 `raw_video.mp4`、
`frame_timestamps.csv`、`frame_timestamps_uvc_source.csv`、原始 JY901B 数据和
`metadata.json`。不要把 refined CSV 重命名覆盖原始轨迹；这样以后更新 IMU
带宽、时间模型或 VIO 算法时，可以直接重处理同一段原始数据。

## 原版 UMI 中 IMU 的作用

原版 UMI 从运动相机的视频中提取内置 IMU；它的用途是**离线视觉惯性里程计（VIO）**，用于把手持相机视频恢复为有米制尺度的相机轨迹。视觉负责观察桌面/物体纹理并建立地图；陀螺仪帮助快速转动时的姿态估计，加速度计结合重力帮助单目系统恢复尺度，并在短暂模糊或弱纹理时提供帧间运动预测。

恢复出的相机轨迹再结合相机到夹爪的固定几何关系及指端 Tag，生成训练所需的夹爪动作标签。IMU **不是** Diffusion Policy 当前的观测输入，也**不参与** AM2Pro 的逆运动学或部署控制；部署时使用机器人自身状态和腕部相机。

本项目的 JY901B 对应原版运动相机的内置 IMU。当前已完成原始记录、时间轴、零偏和相机—IMU 外参标定；完整 VIO 仍在验证中。因此固定桌面 Tag 13 直测轨迹是现阶段正式处理路线，IMU 原始数据仍应随每段视频保存，以便未来重新处理为完整 VIO。

### 当前完整 VIO 的状态与限制

我们已使用 JY901B 的设备时间轴（约 200 Hz）而不是仅依赖串口到达时间；设备时钟与主机时钟比例稳定在约 `0.9949`，相机角速度与陀螺仪角速度的独立验证相关性约 `0.99`，估计的 IMU 相对相机时间偏移约 `-22 ms`。这证明整体时间轴已经基本正确，但**不等于完整 VIO 已经可用于正式处理**。

当前仍可能限制 VIO 的因素包括：

- EMEET 是普通 UVC 相机，没有与 IMU 共用的硬件触发。它已确认提供 UVC PTS/SCR 源时钟，可显著优于仅使用 Python 解码到达时间；但 SCR 对应源数据进入 USB 链路的时刻，不是可证明的传感器曝光开始/中点，因此真实曝光时刻仍可能有毫秒级偏移与抖动。
- JY901B 的加速度噪声、内部滤波和真实动态响应未必严格符合 ORB-SLAM3 使用的惯性噪声模型；仅有静态噪声估计不足以保证高动态初始化稳定。
- 相机—IMU 平移外参目前仍是经独立验证的候选值，不是硬件测量真值。
- 小范围、近距离和近似平面的桌面场景会使视觉初始化/局部地图脆弱；稳定的非重复纹理与不同深度的固定物体可改善视觉部分，但不能单独修复惯性问题。

因此，“同时按下视频和 IMU 录制”只能减少开始时间的不确定性，不能得到每一帧的真实曝光时刻。现阶段不将 JY901B 数据用于逆运动学、策略输入或正式动作标签；它被保存用于未来的 VIO 重处理与时间质量检查。

即使 EMEET 与 JY901B 都通过 USB 接到同一台电脑，也仍有两条独立的时间链路：

```text
相机：曝光 → 传感器读出 → MJPEG 编码 → UVC 缓冲 → USB → 主机收到帧
IMU ：采样 → 设备打包 → USB 串口缓冲 → USB → 主机收到数据包
```

主机单调时钟只能给两条链路一个共同的“到达电脑”参考。相机的曝光、读出/编码与 UVC 缓冲，以及 IMU 的分包与 USB 串口缓冲，都会让该时间与真实采样时刻不同。JY901B 的设备时间戳已改善 IMU 一侧；EMEET 的 SCR/PTS 则允许在相机侧重建稳定的“源时钟映射到主机单调时钟”时间轴，但仍需要以视觉—陀螺运动相关性估计最终相对偏移和剩余抖动。

当前 EMEET (`/dev/video2`) 已确认支持 `exposure_time_absolute` 控制；按 V4L2 约定其单位是 100 微秒，但该数值只表示**配置的曝光时长**，不是每帧曝光时刻。相机默认处于自动曝光模式时，该控制会标记为 inactive，不能把默认显示值当作实际快门。为使帧间视觉时序更稳定，可在光照稳定后测试固定曝光、关闭动态帧率；任何这类修改都必须先用短视频验证 1920×1080 MJPEG 30 FPS 和图像亮度，不能直接用于正式采集。

设备还声明了 `Metadata Capture` 能力，探测 `emeet_uvch_probe_v1` 已确认 `/dev/video3` 的 UVCH 流在全部 79,127 个有效 payload 中均提供 PTS 与 SCR。正式录制脚本会自动使用该节点（EMEET 图像为 `/dev/video2` 时自动配对 `/dev/video3`）；可用 `--uvc-metadata-device none` 显式关闭。SCR 通常比主机到达时间更接近采集时间，但不必然等于传感器的曝光开始时间。

在 `(AM_UMI)` 环境、项目根目录运行 `scripts/capture_emeet_uvc_metadata_probe.py` 可同时启动 `/dev/video2` 的 MJPEG 30 FPS 流与 `/dev/video3` 的 UVCH 元数据流，并报告 EMEET 是否真的填写了 PTS/SCR。该探测不修改相机控制项。

### VIO 改进路线

先做的软件/标定工作是：用多段旋转视频估计相机—IMU 偏移在不同时间窗口内的分布（得到偏移均值和抖动，而不只得到单个 `-22 ms`）；再使用更丰富、具有深度层次的固定工作区验证视觉与完整 VIO。

若这些步骤后仍不稳定，硬件层面的根治路线是使用带**硬件同步相机与 IMU**的视觉惯性设备，或使用支持外部硬件触发/时间戳的相机与 IMU 组合。仅更换为“更高精度但仍通过 USB/UART 独立传输”的 IMU，可能降低噪声，却不能消除相机曝光时间的不确定性。

## 工作区纹理要求

SLAM/VIO 需要稳定的视觉特征。**纯白板不会增加特征，反而可能使视觉跟踪更差。** 若要加板，使用硬质、哑光、固定不动的白色底板，并在其上粘贴一张不重复的高对比纹理图（照片拼贴、随机图形、文字与图标混合均可）；避免规则棋盘格、重复网格和反光覆膜。

纹理板会提升视觉前端的匹配质量，因此可能间接提高完整 VIO 的成功率；但它不能修复 IMU 时间同步、相机—IMU 外参、IMU 噪声模型或惯性初始化问题。纹理板、桌面 Tag 13 和任务桌面的相对位置应从 mapping 到全部正式 demo 保持不变。

## 不属于本流程

- 不做 AM2Pro 的机械臂手眼标定。
- 不使用 `robot_wrist_camera/` 中的任何内参或手眼结果。

## 重要限制

旧 Osmo/GoPro 设置仍归档在 `../archive/legacy_handheld_cameras/`，不能用于 EMEET。当前完整 VIO 候选设置为 `slam/emeet_jy901b_orbslam3_v6_uvc_source_candidate.yaml`；它使用 UVC 源时钟、独立验证的旋转/平移候选外参和 -22 ms 时间偏移。内参文件本身不能替代 SLAM 设置。
