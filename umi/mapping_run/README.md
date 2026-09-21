# mapping_run —— 建图一键运行文件夹

## 文件说明

| 文件 | 作用 |
|---|---|
| `run_mapping.sh` | **唯一入口**。一条命令跑完：建 session → IMU 生成 → SLAM 建图 → tag 检测 → 手眼标定 → 结果摘要 |
| `session_*/` | 每次运行自动生成的输出目录（时间戳命名，互不覆盖）|

依赖脚本（在仓库 `scripts/` 和 `scripts_slam_pipeline/` 中，入口脚本自动调用）：
`gyroflow_csv_to_imu_json.py`、`02_create_map.py`、`04_detect_aruco.py`、`05_run_calibrations.py`

## 使用方法

```bash
conda activate umi
bash mapping_run/run_mapping.sh
```

默认处理 0014 视频 + Wide3.csv；自定义参数：

```bash
bash mapping_run/run_mapping.sh "/path/视频.mp4" "/path/Gyroflow导出.csv"
```

## 运行前提

1. umi conda 环境已激活
2. docker 可用（`docker run hello-world` 能过）
3. 视频：Wide、4:3、2.7K、60fps、防抖关
4. CSV：Gyroflow 打开该视频后 Export data 导出

## 结果怎么看

运行结束会自动打印摘要：

- **SLAM**：`There are 1 maps in the atlas, Map 0 has N KFs` = 地图建成；跟踪成功帧占比越高越好（>50% 理想）
- **tag 检测**：检测到 tag 的帧数 / 总帧数（>50% 为佳）
- **手眼标定**：`mapping/tx_slam_tag.json` 生成即成功

输出目录结构：

```
session_MMDD_HHMMSS/
├── calibration/            # 内参 + tag 配置（跑 06/07 时要用）
└── demos/
    └── mapping/
        ├── raw_video.mp4               # 拷贝的视频
        ├── imu_data.json               # 生成的 IMU
        ├── slam_mask.png               # SLAM 掩码（底部 35%）
        ├── slam_stdout.txt             # SLAM 日志
        ├── map_atlas.osa               # 建出的地图
        ├── mapping_camera_trajectory.csv  # 相机轨迹
        ├── tag_detection.pkl           # tag 检测结果
        └── tx_slam_tag.json            # 手眼标定结果
```
