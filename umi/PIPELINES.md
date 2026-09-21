# 手持夹爪两条轨迹路线：文件与数据隔离

本项目保留两条路线，但它们的**产物绝不能互相替换**。两条路线都可保存同一段
原始视频与 IMU；差别在于用哪一种方式生成相机/夹爪轨迹，以及这些轨迹能否进入
正式训练数据集。

## 1. 正式路线：固定 Tag 13 视觉直测

这是当前唯一可以用于正式 demo、zarr 和训练的路线。

- 世界坐标：固定不动的 ArUco Tag 13。
- 位姿来源：每帧以 Tag 13 PnP 直接测量相机位姿；不是 SLAM 地图。
- 夹爪：Tag 0/1 开度标定
  `calibration/handheld_gripper_camera/gripper_tag_sessions/tag_visibility_v2/gripper_range_v1.json`。
- 相机到 TCP：
  `calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_camera_to_tcp_v2.json`。
- 正式会话建议保存到：
  `data/handheld_sessions/fixed_tag/<task_name>/<demo_name>/`。
- 轨迹文件必须命名为
  `camera_trajectory_fixed_tag13_metric.csv`，细化后为
  `camera_trajectory_fixed_tag13_metric_refined.csv`；必须保留相邻的
  `.fixed_tag_report.json` 或 `.refinement_report.json`。
- 数据集计划必须显式带 `--trajectory-frame fixed_tag` 和
  `--camera-tcp-geometry ...camera_to_tcp_v2.json`。脚本会写 provenance，避免误把
  其他 CSV 当作固定 Tag 轨迹。

硬约束：Tag 13 在正式演示中应持续可见；长缺口必须丢弃相应片段，不能由 VIO/SLAM
结果补填。

## 2. 实验路线：EMEET + JY901B IMU/VIO

这是保留的研发路线，**当前不能产生正式训练标签**。

- 相机—IMU 标定只在
  `calibration/handheld_gripper_camera/camera_imu/` 中管理。
- ORB-SLAM3/OpenVINS 运行、配置、日志和评估只放在
  `calibration/handheld_gripper_camera/openvins/` 或会话自身的
  `vio_experimental/` 子目录。
- 新的 VIO 采集建议保存到：
  `data/handheld_sessions/vio_experimental/<session_name>/`。
- VIO 的候选输出必须使用明确名称，例如
  `camera_trajectory_openvins_candidate.csv`；不得使用
  `camera_trajectory_fixed_tag13_metric*.csv` 的名字，也不得放进 fixed_tag demo
  目录。
- VIO 结果必须带独立误差报告；在达到验收精度前，禁止传给
  `06_generate_dataset_plan.py --trajectory-frame fixed_tag` 或生成正式 zarr。

当前状态：时间轴、外参与 OpenVINS 运行链已完成，但与固定 Tag 对照仍为厘米级误差，
因此仅供诊断和未来硬件/算法改进使用。

## 3. 共享但不可混淆的文件

以下属于硬件或原始传感器信息，可被两条路线共同保存：

- `raw_video.mp4`、`frame_timestamps.csv`、`frame_timestamps_uvc_source.csv`；
- `imu_raw.npz`、UVC payload header、录制 `metadata.json`；
- 相机内参、Tag 字典、IMU 零偏和 Allan 记录。

共享原始数据不等于共享轨迹标签。正式处理时只以固定 Tag CSV 及其 provenance 为准。

## 4. 旧文件处理

已有的 `data/handheld_mapping/`、`camera_imu_extrinsic_sessions/` 和
`camera_imu_translation_sessions/` 是历史标定/试验档案，保留但不移动，也不作为新的
正式 demo 目录。新数据从本文件规定的两个根目录开始，避免路径和 CSV 名称冲突。
