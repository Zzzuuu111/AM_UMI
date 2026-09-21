# 机械臂腕部相机：AM2Pro 部署与手眼标定

此目录只放固定在 AM2Pro 夹爪/腕部的相机文件。相机支架一旦移动、拆装或松动，应重新检查内参适用性，并重新做手眼标定。

## 已有 / 待完成

| 项目 | 状态 | 说明 |
|---|---|---|
| 部署相机内参 | 已有 | `intrinsics/emeet_wrist_1920x1080_30fps_fisheye_v1.json`，用于当前 1920×1080、30 fps 部署裁剪中心。 |
| 相机采集与实时观测 | 已通过 | EMEET 与 AM2Pro 的实时观测、策略干运行已验证。 |
| 起始视角基准 | 可按任务保存 | 存至 `../view_references/`，用于部署前找回相近视角。 |
| 机械臂手眼标定 | 已完成（当前安装） | `hand_eye/new_gripper_camera_v4_hand_eye.json`：基于固定 Tag 13 与 7 个单关节静态姿态，平均一致性 4.72 mm / 0.63°。相机支架再移动后必须重做。 |
| 机械臂/相机延迟 | 可选 | 若之后追求更精确动态控制再测量。 |

部署起始视角的保存与“画面 + TCP/关节/夹爪”复位检查，见项目根目录的
[VIEW_REFERENCE.md](../../VIEW_REFERENCE.md)。

## 不属于本流程

- 不建手持 SLAM 地图。
- 不需要把 Tag 固定在同一相机/夹爪刚体上；手眼标定的 Tag 应固定在桌面。
- 不使用 `handheld_gripper_camera/` 的 SLAM 或夹爪 Tag 标定结果。
## Follow UMI gripper TCP

`tcp/new_gripper_tcp_v1_from_measurement.json` records the active v1 TCP.
The controller uses the `right_tcp` URDF frame: the midpoint between closed
fingertips, provisionally 236 mm along `right_Fixed_Jaw` +Z.  This does not
change arm joints 1–6, the gripper servo mapping, or the camera hand-eye
calibration.  It is a measured starting value and must be refined by a later
visual TCP-pivot check before precision pick/place work.
