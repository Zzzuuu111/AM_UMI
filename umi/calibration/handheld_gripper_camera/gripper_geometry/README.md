# 手持相机到夹爪 TCP 几何

固定世界 Tag 能直接给出相机在 Tag 坐标系中的米制位姿，但训练动作需要的是夹爪
TCP 位姿。因此还需要一个固定变换 `T_camera_tcp`：TCP 在相机坐标系中的位置与
姿态。

本目录只存放**手持夹爪**的相机到 TCP 几何，不能放机械臂腕部相机的手眼结果。
相机、夹爪或支架一旦拆装，该结果立即失效。

文件格式：

```json
{
  "schema": "am_umi_camera_tcp_geometry_v1",
  "pose_cam_tcp": [x_m, y_m, z_m, rx_rad, ry_rad, rz_rad]
}
```

前三项是米，后三项是旋转向量（axis-angle，弧度）。该变换的方向是
`camera -> tcp`，即 `T_world_tcp = T_world_camera @ T_camera_tcp`。

`emeet_handheld_camera_to_tcp_TEMPLATE.json` 只是结构模板，`pose_cam_tcp` 为
`null`，不能直接用于生成数据集。等整套夹爪回到现场后，应通过实测/标定生成带
版本号的正式文件，例如 `emeet_handheld_camera_to_tcp_v1.json`，并独立验证。

当前通过独立验证的文件为
`emeet_handheld_camera_to_tcp_v2.json`。其平移由 `tcp_pivot_v2` 拟合，并由
独立的 `tcp_pivot_v3_validation` 验证：1,132/1,329 帧（85.2%）在 8 mm 阈值内，
内点 RMSE 为 4.65 mm。`v1` 因只有 23.4% 支点内点而明确标记为拒绝文件，不能使用。

## 当前 AM_UMI 的实测方法：固定点枢轴标定

先完成一次双指 Tag 开合范围标定（生成的 `gripper_range.json` 必须显示
`schema_version: 2`）。随后录制一段 35–45 秒的枢轴视频：固定世界 Tag 13、
夹爪 Tag 0、夹爪 Tag 1 都必须可见；推荐把小圆珠放进瓶盖/橡皮泥的浅凹槽，
再由夹爪全程以同一开口夹住圆珠。圆珠球心即固定支点，手持装置从至少 8 个
不同方向缓慢转动。不要在这一环节使用任务起始画面基准。

对视频完成 ArUco 检测后，运行：

```bash
python -u scripts/calibrate_handheld_camera_tcp_pivot.py \
  --input calibration/handheld_gripper_camera/gripper_tag_sessions/tcp_pivot_v1/tag_detection.pkl \
  --output calibration/handheld_gripper_camera/gripper_geometry/emeet_handheld_camera_to_tcp_v1.json \
  --world-tag-id 13 \
  --left-finger-tag-id 0 \
  --right-finger-tag-id 1
```

它只读取录像和 Tag 位姿，**不会移动机械臂**。输出的平移部分来自实测；默认的
旋转 `[0,0,0]` 是“相机光轴与夹爪工具轴按当前安装方向对齐”的显式假设。若支架
拆装或该假设后来验证错误，重新采集并以新版本文件替换，不能沿用旧文件。
