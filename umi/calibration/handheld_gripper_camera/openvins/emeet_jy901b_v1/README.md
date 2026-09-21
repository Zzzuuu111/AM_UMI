# OpenVINS EMEET + JY901B candidate v1

This configuration is only for offline OpenVINS comparison.  It is isolated
from AM2Pro control and Diffusion Policy deployment.

Inputs fixed here:

- 1920x1080 EMEET fisheye intrinsics;
- camera-to-IMU rotation/time from `emeet_jy901b_rotation_time_v4_uvc_source.json`;
- camera-to-IMU translation and conservative dynamic noise from
  `emeet_jy901b_translation_noise_v4_uvc_source.json`;
- camera clock uses strict UVC source timestamps.  The offline exporter shifts
  JY901B device time onto this camera timeline, therefore this replay config
  uses OpenVINS **`cam0.timeshift_cam_imu: 0.0`**. The previous comment
  referring to `imu0.time_offset` was incorrect: that field did not set the
  OpenVINS camera offset. Changing it did not fix the earlier divergence.

The first experiment keeps calibration fixed.  If it initializes and tracks,
we can separately test OpenVINS online time-offset calibration.  A successful
OpenVINS run is still an experimental VIO result; fixed-Tag metric pose remains
the production trajectory source until independently validated.

## 2026-09-05 接口审计与离线测试

完整使用方法和测试结论见 [OPENVINS_OFFLINE.md](../../../../OPENVINS_OFFLINE.md)。

不要直接用本目录 YAML 启动旧的后台 `ros2 run` 命令。使用
`scripts/run_emeet_openvins.py`：它会把配置复制到独立输出目录，统一
960×540 图像/内参/遮罩，使用 `uvc_source_relative_s` 并核对 CORI 时间戳，
通过 ROS 参数启用状态保存。原始输入只读，不覆盖标定或旧轨迹。

`runs/full_v1`、`runs/full_v2_camera_timeline` 结果已发散，不能使用。
旧回放误读 `relative_s` 为 UVC 源时间轴，且没有遮罩；两轮结果不能用于
判定 JY901B 硬件不适合 VIO。

修复后的默认镜像为 `am_umi_openvins_offline:20260905`。
最终完整测试 [final_full_v8](../audited_runs/final_full_v8/summary.json)
保存 3303 个状态并正常退出；固定 Tag 对照
位置 RMSE 约 1.67 cm，但尺度与姿态仍有偏差，尚不用于正式训练轨迹。
