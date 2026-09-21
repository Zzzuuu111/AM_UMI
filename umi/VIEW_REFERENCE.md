# 起始视角基准与两种对齐流程

起始基准用于让手持示范的起点与 AM2Pro 部署的起点尽量一致。它不是自动导航或自动回位功能：当前阶段由人根据实时画面和数值提示调整；后续完成固定桌面 Tag 与机械臂手眼标定后，才能做几何意义上的自动找位。

一个基准目录保存两类信息：

- `policy_input_rgb.png`：经过当前部署配置的裁剪、缩放、RGB 转换及夹爪遮罩后的 **224×224 策略输入**；
- `reference.json`：保存时的相机设置、AM2Pro 实际 TCP 位姿、关节角度和夹爪开度。

因此有两种用途不同的对齐流程，不能把它们混用。

| 场景 | 对齐依据 | 是否读取机械臂状态 |
|---|---|---|
| 手持夹爪建图/示范录制前 | 画面构图、固定背景/Tag、物体尺度 | 否 |
| AM2Pro 重新部署前找回起始位 | 策略画面 + TCP 位置/朝向 + 关节 + 夹爪 | 是 |

## 0. 共同前提

保存和使用基准时，任务物体、容器、桌面固定物和光照应尽量处于同一任务起始状态。抓取任务的待抓物应出现在基准画面中。

手持流程中，相机、夹爪和 IMU 必须固定成一个刚体；机械臂流程中，腕部相机支架不得松动或改变。画面分数会受光照、物体材质和遮挡影响，不能单独当作精确三维位姿误差；优先观察桌沿、固定 Tag 和容器是否重合。

## 1. 保存 AM2Pro 起始基准

先用已有的安全方式把机械臂放到希望部署时采用的起始位置和姿态。停止该控制程序后，运行：

```bash
cd ~/AM_UMI/umi

python -u scripts/save_am2pro_view_reference.py \
  --reference-dir calibration/robot_wrist_camera/view_references/task_start_v2 \
  --preview-first
```

窗口左侧是**原始相机广角画面**，用于人工寻找任务构图；右侧是**策略实际输入画面**。窗口获得焦点后：

- `B`：保存右侧策略画面和此刻完整 AM2Pro 状态；
- `Q` / `Esc`：取消，不写入文件。

该脚本只启动保持控制器，**不发送任何轨迹动作**。运行期间不要启动另一个 AM2Pro 控制程序，也不要徒手硬掰仍在保持扭矩的机械臂。同名基准目录不会被覆盖；新基准请使用 `task_start_v3` 等新目录名。

### 保存命令参数

| 参数 | 含义 | 本次建议 |
|---|---|---|
| `--reference-dir` | 新建基准目录，内含 `policy_input_rgb.png` 与 `reference.json`。已有目录会被拒绝写入，保护旧基准。 | `calibration/robot_wrist_camera/view_references/task_start_v2` |
| `--preview-first` | 保存前弹出双画面预览；按 `B` 才真正保存。不带它会在相机缓冲就绪后直接保存。 | 必须带 |
| `--display-width` | 原始广角画面在窗口中的最大显示宽度，默认 `960` 像素；只影响显示，不改变保存图或策略输入。 | 一般不需要设置 |
| `--robot-config` | AM2Pro、相机路径、分辨率和 `camera_crop` 的配置文件。 | 默认 `example/eval_robots_config.yaml` |

## 2. AM2Pro 重新部署前：画面 + 姿态复位检查

机械臂位置或姿态改变后，使用默认的完整检查模式：

```bash
cd ~/AM_UMI/umi

python -u scripts/align_umi_view_reference.py \
  --reference-dir calibration/robot_wrist_camera/view_references/task_start_v2 \
  --show \
  --seconds 3600 \
  --interval 0.2
```

它会显示两个窗口：原始广角画面，以及 `reference | live | overlay | difference` 策略画面对齐面板。面板和终端还会实时报告：

- `TCP position`：当前末端相对基准的空间位置误差，单位 mm；
- `orientation`：当前末端相对基准的真正旋转角误差，单位 degree；
- `max joint`：任一关节相对基准的最大角度误差，单位 degree；
- `gripper`：夹爪开度误差，单位 mm。

`--seconds 3600` 只代表这个**监视程序**最多持续一小时，不是机械臂动作时限；窗口聚焦后按 `Q` / `Esc`，或终端按 `Ctrl+C`，都可以提前退出。

这个模式只读取和保持当前位置，**不会自动移动机械臂回基准**，也不能与另一个 AM2Pro 控制程序同时运行。实际人工调整可采用“安全控制程序粗调 → 退出 → 本工具检查 → 必要时重复”的方式。若要在同一界面中低速微调并连续显示上述位姿误差，需要单独启用未来的安全 jog 工具，不能由本监视器隐式移动机械臂。

每次结果还会写到：

```text
calibration/robot_wrist_camera/view_references/task_start_v2/live_alignment/
```

其中 `latest_alignment.png` 是最新策略画面对齐面板，`alignment_history.csv` 记录每次画面分数。

### 机械臂复位检查参数

| 参数 | 含义 | 本次建议 |
|---|---|---|
| `--reference-dir` | 读取要找回的机械臂基准；其中必须有 `robot_state`，才会显示 TCP、关节与夹爪误差。 | 指向 `task_start_v2` |
| `--show` | 显示原始相机窗口与对齐面板窗口；不带它只输出终端数值和图片文件。 | 建议带 |
| `--seconds` | 监视程序的最长运行时间，单位秒；不是机械臂运动时间限制。 | `3600`，可随时 `Q`/`Esc`/`Ctrl+C` 退出 |
| `--interval` | 两次对齐计算之间的间隔，单位秒。`0.2` 约为每秒 5 次更新；更小会更频繁并增加 CPU 负担。 | `0.2` |
| `--output-dir` | 保存 `latest_alignment.png` 和 `alignment_history.csv` 的目录。 | 不设置，自动写到该基准目录下的 `live_alignment/` |
| `--robot-config` | AM2Pro 与相机配置。 | 默认 `example/eval_robots_config.yaml` |
| `--camera-only` | **不要用于机械臂完整复位检查**；它不读取 TCP、关节或夹爪，只做画面对齐。 | 不带 |
| `--device` | 仅 `--camera-only` 时指定 UVC 相机设备。 | 本模式不需要 |

## 3. 手持夹爪录制前：纯画面对齐，然后手动开始录制

手持夹爪没有 AM2Pro 基座坐标和关节状态，故只能对齐画面。使用集成录制器：

```bash
cd ~/AM_UMI/umi

python -u imu_work/record_handheld_umi_session.py \
  --output-dir data/handheld_sessions/demo_001 \
  --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
  --imu-port /dev/ttyUSB0 \
  --imu-baud 460800 \
  --reference-dir calibration/robot_wrist_camera/view_references/task_start_v2
```

启动后先进入不录制的预览/对齐阶段。拿着固定好的手持夹爪、相机和 IMU，按固定桌面特征、Tag、物体尺度和朝向对齐；稳定后在窗口按：

- `R`：开始保存视频与 IMU；
- `S`：停止视频录制，程序继续记录约 2 秒 IMU 尾部以便时间对齐；
- `T`：试走/重置，只用于检查构图和动作，不保存正式 demo；
- `X`：取消已经开始的本次录制；录制器会关闭视频/metadata，并删除本次新建的 `--output-dir`，可直接复用同一条命令重录；
- 在未开始录制时退出：不产生正式 demo。

输出为 `raw_video.mp4`、`frame_timestamps.csv` 和 `imu_raw.npz`。这些时间戳使用同一主机单调时钟，但在完成相机-IMU 固定外参和时间偏移标定前，它们仍是原始会话，不能直接作为正式视觉惯性 SLAM 输入。

### V 型夹爪当前标准录制入口（视觉-only）

当前 V 型夹爪正式 demo 不使用 IMU 标签，而使用固定桌面 Tag 13/14 的米制轨迹。请使用包装脚本而不是上面的通用 IMU 示例：

```bash
cd ~/AM_UMI/umi

bash scripts/record_handheld_vjaw_demo_web.sh \
  vjaw_demo_XXX_description \
  calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint
```

终端会给出本机网页地址。网页/键盘操作为 `T=试走或重置`、`R=正式录制`、`S=停止并保存`、`X=丢弃本次录制`、`Q=退出`。`R` 的语义很重要：它会把 Tag 相对运动原点和实时 IK 预测的 warm-start 重置为该次正式录制的第一时刻；因此预览时的相对位移或红色提示不会继承为正式 demo 的起点。

该入口在后台约 `5 Hz` 运行 Tag/TCP/IK 的只读预览；后台忙时丢弃诊断帧而非阻塞相机和编码。面板红色表示当前预测关节距安全限位小于 `5°`，它只用于帮助调整录制手法，不连接机械臂、不发机器人命令，也不能替代录制后的完整离线连续 IK 检查。启动过的旧网页不能自动获得新按键；脚本更新后需退出并重启录制器。

对正式 V 型夹爪数据，至少一个已映射的 Tag（13 或 14）必须持续可定位。处理后要求 refined 轨迹没有剩余丢帧；否则 zarr 转换会拒绝该会话。推荐在抓取/放置接触前提早、平缓地完成腕部朝向变化，并在接触和最终下探段保持稳定，避免中途突然转腕造成机器人 IK 分支跳变。

### 手持录制命令参数

| 参数 | 含义 | 本次建议 |
|---|---|---|
| `--output-dir` | 本次会话的新输出目录，保存视频、帧时间戳、IMU 和元数据；每次 demo 使用不同目录。 | `data/handheld_sessions/demo_001`，后续依次递增 |
| `--camera-device` | 手持 EMEET 的稳定 `/dev/v4l/by-id/` 路径；不要写易变化的 `/dev/videoN`。 | 以 `ls -l /dev/v4l/by-id/` 实际结果为准 |
| `--imu-port` | JY901B 的串口设备。 | `/dev/ttyUSB0` |
| `--imu-baud` | JY901B 串口波特率。已实测为 460800。 | `460800` |
| `--reference-dir` | 读入 AM2Pro 基准策略画面，在正式录制前实时显示画面对齐参考。它不会读取机械臂状态。 | 指向 `task_start_v2` |
| `--imu-tail-seconds` | 停止视频后继续采集 IMU 的时长；用于覆盖末帧边界、帮助后续时间对齐。 | 默认 `2` 秒 |
| `--max-record-seconds` | 单个会话允许的最长录制时长；达到后自动停止，避免遗忘录制占满磁盘。 | 默认 `180` 秒，任务较长时再增大 |
| `--width`、`--height`、`--fps`、`--pixel-format` | 手持相机采集模式，必须与相机内参、录制和 SLAM 配置一致。 | 当前为 `1920`、`1080`、`30`、`MJPG` |

## 4. 仅检查手持画面（不录制）

如只想检查手持视角而不录制，可运行：

```bash
python -u scripts/align_umi_view_reference.py \
  --reference-dir calibration/robot_wrist_camera/view_references/task_start_v2 \
  --camera-only --show --seconds 3600 --interval 0.2
```

`--camera-only` 不连接 AM2Pro；它只能与未占用同一 EMEET UVC 设备的程序并行。它输出的 `appearance_score` 越接近 1 越好、`mae` 越接近 0 越好、`correlation` 越接近 1 越好，但没有可通用于所有任务的固定合格阈值。

### 纯手持检查参数

`--reference-dir`、`--show`、`--seconds`、`--interval` 的含义与上表相同。额外参数如下：

| 参数 | 含义 | 本次建议 |
|---|---|---|
| `--camera-only` | 只读 UVC 相机，禁止启动 AM2Pro 控制器；因此只会显示画面对齐。 | 必须带 |
| `--device` | 指定要打开的手持 UVC 相机；不设置时按 `example/eval_robots_config.yaml` 自动寻找 EMEET。 | 有多台 EMEET 时明确指定稳定 by-id 路径 |

## 5. V 型夹爪：低速空载 replay 与自动总览图

以下是当前最新的 demo 030 平滑候选计划的**短段、低速、空载** replay。它会先低速回到
`follow_umi_vjaw_start_v6_safe_midpoint` 基准，暂停等待操作者确认，再回放前 140 帧；结束、
中止或保护 abort 后均会回到基准。`--plot-after-replay` 会在机械臂完成返回、控制器退出后，
自动生成一张轨迹总览图和一份数值报告。

运行前必须清空工作区、移走物体并确认物理急停/断电触手可及。看到
`REFERENCE_PAUSED` 后，确认机械臂处于基准且周围无障碍，再按 Enter；若方向错误、抖动或有
碰撞风险，立即按 `Ctrl+C`。

```bash
conda activate AM_UMI
cd ~/AM_UMI/umi

python -u scripts/am2pro_replay_retarget_plan.py \
  --joint-plan data/handheld_demos_vjaw/vjaw_demo_030_v11_fixed_tip_width84_branch_fixed_smooth9_constrained_lookahead_v1.json \
  --start-reference calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json \
  --robot-config example/eval_robots_config_vjaw_ros2.yaml \
  --execution-mode adaptive \
  --max-source-frames 140 \
  --enable-gripper \
  --gripper-mode continuous \
  --gripper-continuous-command-hz 5 \
  --gripper-continuous-deadband-mm 2 \
  --gripper-final-width-m 0.009 \
  --pause-after-reference \
  --trace-out data/handheld_demos_vjaw/vjaw_demo_030_v11_tags_stable/replay_trace_repeat_140.json \
  --plot-after-replay
```

正常结束时，终端依次输出 `VJAW_RETARGET_REPLAY_FINISHED`、`Returned.` 和
`VJAW_TRAJECTORY_COMPARISON_OK`。结果保存在 trace 同目录：

```text
replay_trace_repeat_140.json             # 实际关节读回 trace
replay_trace_repeat_140.trajectory.png   # 一张总览：3D/XYZ/J7 开度对比
replay_trace_repeat_140.trajectory.json  # 图的数值报告
```

该图的手持 demo/离线计划曲线是相对运动对比，绿色 replay 曲线是由电机读回经 FK 得到的
轨迹；它用于诊断关节跟踪与模型差异，不等同于外部动作捕捉的绝对 TCP 精度。每次重跑必须更换
`--trace-out` 的文件名，脚本会拒绝覆盖已有 trace、图或报告。
