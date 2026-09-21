# 标定文件中心目录

本目录按**设备角色**隔离文件。不要把手持夹爪相机与机械臂腕部相机的内参、外参或结果混用；即使两者是同型号 EMEET，相机的物理个体和安装位置也不同。

```text
calibration/
├── handheld_gripper_camera/          # 手持夹爪录制、SLAM、夹爪 Tag
├── robot_wrist_camera/               # AM2Pro 腕部部署、机械臂手眼标定
├── shared_tags/                      # 两套流程共用的 Tag 定义与可打印 PDF
└── archive/legacy_handheld_cameras/  # 不再用于新采集的 Osmo / GoPro 历史文件
```

## 手持夹爪相机

请先阅读 [handheld_gripper_camera/README.md](handheld_gripper_camera/README.md)。

当前候选内参为 `handheld_gripper_camera/intrinsics/emeet_handheld_1920x1080_30fps_fisheye_20260831.json`：1920×1080、30 fps、关闭防抖、197 张图、重投影误差 0.740 px。只在实际录制相机与该标定相同的物理设备、录像设置完全一致时使用。

## 机械臂腕部相机

请先阅读 [robot_wrist_camera/README.md](robot_wrist_camera/README.md)。

部署使用的 1920×1080、30 fps EMEET 内参在 `robot_wrist_camera/intrinsics/emeet_wrist_1920x1080_30fps_fisheye_v1.json`。后续 AM2Pro 手眼标定结果放入 `robot_wrist_camera/hand_eye/`，而不是手持目录。

## 共用 Tag

`shared_tags/` 中包含 ArUco 字典配置和可打印 Tag：

- `aruco_gripper_0_letter.pdf`：手持夹爪指端 Tag；正式示范数据需要它。
- `aruco_cubes_letter.pdf`：固定桌面 / 世界坐标 Tag；手持流程的 SLAM-Tag 对齐，以及机械臂手眼标定都可使用它。
- `aruco_config.yaml`：检测这些 Tag 时传入的配置文件。

单个手持夹爪使用 `aruco_gripper_0_letter.pdf` 中的 **ID 0 与 ID 1** 两张小型指端 Tag（16 mm）：在最终保存的相机画面中，画面左侧的夹爪指端贴 **ID 0**，画面右侧贴 **ID 1**。不要按操作者自身左右手判断。第二个独立夹爪才使用 ID 6（画面左）与 ID 7（画面右）；ID 2–5、8–11 是立方体 Tag，不是指端 Tag。

新文件采用前缀 `emeet_handheld_...` 或 `emeet_wrist_...`，并保留分辨率、帧率、模型和版本/日期。不要覆盖旧文件。

根目录以前遗留但无法确认归属的标定可视化图片已移至 `archive/unknown_source_visual_reports/`；它们不作为任一相机的有效依据。

`example/calibration` 是指向本目录的兼容符号链接；新命令请使用 `calibration/...` 的新分组路径。
