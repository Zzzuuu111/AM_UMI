# EMEET + JY901B：OpenVINS 离线验证

更新：2026-09-05。对象是**手持夹爪**，不是机械臂。只读录好的视频/IMU，
不打开相机、串口，不发送机械臂指令，不运行 DP。

本轮结论：**OpenVINS 离线适配与完整回放已完成，正常退出；正式轨迹精度尚未验收。**
最终结果目录：`calibration/handheld_gripper_camera/openvins/audited_runs/final_full_v8`。
后续独立复测及同规则对照见文末 `independent_rotation_v9` / `baseline_global_control_v10`；
两轮均正常退出，但米制尺度一致性仍未确认。

## 环境与隔离

主机使用 `AM_UMI` 虚拟环境启动脚本即可。ROS2 Humble、OpenVINS、Ceres 等
全部在 Docker 内，不安装到 `AM_UMI`、`lerobot_alohamini` 或原 `umi` 环境。
测试使用 CPU、不使用 GPU；每个容器最多 2 核、4 GiB 内存，禁止额外 swap。
相机/IMU/标定目录以只读方式挂载，只有指定的新输出目录可写。
Docker 镜像和结果会占用磁盘，这不等于修改 Conda 包。

默认运行镜像为 **`am_umi_openvins_offline:20260905`**，含节点退出顺序修复。
`am_umi_openvins_built:latest` 是修复前已完成编译的镜像；
`am_umi_openvins_debug:latest` 是单独增加 gdb 的调试镜像，未升级原环境。
基础构建文件为 `third_party/open_vins/Dockerfile.am_umi_ros2`，它只构建
ROS/依赖基础层；当前 OpenVINS 安装是容器内编译后保存的，不应把基础层当成
已编译的运行镜像。构建时 make 应明确 `-j1`，仅限制 colcon worker 不足以限制内存。

本地 OpenVINS 源版本：`69488123ed9362dd44b6f28e7f4680abbff1442b`。
基于已有运行镜像重新增量生成退出修复镜像（不需要重新下载依赖）：

```bash
bash scripts/build_openvins_shutdown_fix.sh
```

该脚本保留已停止的构建容器，失败时不会保存为成功镜像。

## 本轮修正

1. 回放应读取 `frame_timestamps_uvc_source.csv` 的 `uvc_source_relative_s`，
   不是同一文件的 `relative_s`（接收时间）。与已导出 JSON 的 CORI 逐帧核对，
   不一致或不严格递增时拒绝运行。
2. IMU JSON 已应用时间偏移、加速度校正、陀螺零偏；不再重复校正，也不减去重力。
   生效的 OpenVINS 时间偏移字段是相机配置中的 `cam0.timeshift_cam_imu`，
   本回放为 0。之前把 `imu0.time_offset` 的变化说成修正重复偏移是不准确的。
3. 输入从 1920×1080 缩到 960×540，fx/fy/cx/cy 同比除以 2；畸变系数不变。
4. OpenVINS 遮罩白色表示禁止特征。旧遮罩仅盖住中央底部，没有完整盖住左右
   夹爪。`--gripper-envelope` 补充本验证视频的手指/Tag 区域，输出
   `mask_preview.jpg` 供核对。**这个补充区域不是所有安装位置的通用遮罩**。
5. 状态保存通过节点 ROS 参数开启，不能只写 YAML。直接启动实际节点并管理
   生命周期，不再仅向 `ros2 run` 包装进程发退出信号。
6. 等待两个订阅者就绪后才回放；时间戳采用整数秒/纳秒构造，避免大 epoch
   浮点相加损失精度。半速仅降低墙钟回放速度，不改变传感器时间。
7. gdb 复现定位到 `image_transport::Publisher` 在全局析构阶段的段错误。
   `run_subscribe_msckf.cpp` 现在在 main 结束前主动释放 viz/sys，并从 executor
   移除节点；离线不启动后台图像发布线程。没有修改滤波器、优化器或 IMU 模型。

## 启动命令

虚拟环境：`AM_UMI`；工作目录：`/home/zzzjh/AM_UMI/umi`。
以下是离线诊断命令，**不是正式采集/训练/机械臂部署命令**。

```bash
conda activate AM_UMI
cd /home/zzzjh/AM_UMI/umi
python -u scripts/run_emeet_openvins.py \
  --session-dir calibration/handheld_gripper_camera/camera_imu_translation_sessions/vio_full_validation_v4_uvc_source \
  --output-dir calibration/handheld_gripper_camera/openvins/audited_runs/manual_trial_v1 \
  --gripper-envelope
```

| 参数 | 含义 |
| --- | --- |
| `--session-dir` | 已录制目录，要求 raw_video.mp4、UVC 源时间 CSV、imu_data_strict_uvc.json |
| `--output-dir` | 新结果目录，已有目录会拒绝覆盖；每次换新名字 |
| `--config-dir` | 候选标定配置目录，默认 emeet_jy901b_v1；运行时复制快照 |
| `--image` | Docker 运行镜像，不能填未编译的基础镜像 |
| `--speed 0.5` | 默认半速回放，给离线算法足够计算时间 |
| `--max-seconds 30` | 只测视频前 30 秒；默认 0 表示全部 |
| `--gripper-envelope` | 额外屏蔽此验证视频里的左右夹爪；先看 mask_preview.jpg |
| `--without-mask` | 仅做遮罩消融对照，正式候选不要使用 |
| `--gdb` | 调试镜像内捕获调用栈，正常测试不需要 |

运行中可 Ctrl+C，脚本停止本次容器并保留日志；不要 Ctrl+Z。

## 输出与判断

- `replay.log`：实际发送帧数/IMU 数量，完整回放应有 `OPENVINS_REPLAY_FINISHED`。
- `openvins_node.log`：初始化、估计和退出日志。
- `state_estimate.txt` / `state_std.txt`：估计与标准差，状态位置是 **IMU 原点**。
- `config/`、`mask_preview.jpg`、`run_manifest.json`：本次实际配置、遮罩和输入摘要。
- `summary.json`：状态覆盖、速度/位移粗检及进程退出状态；正常退出不代表精度合格。
- `fixed_tag_comparison.json`：相机中心轨迹与固定 Tag 参考的误差。

比较脚本 `imu_work/compare_openvins_fixed_tag.py` 使用已标定杆臂把 IMU 原点
转换为相机中心，处理 OpenVINS 的 JPL 四元数约定，再做一个刚体 SE(3) 对齐。
报告的位置误差**没有缩放轨迹**。另外输出的 Sim(3) 尺度仅用于发现尺度偏差，
不能拿它修饰米制精度。Tag PnP 与 VIO 共用内参，不是独立高精度真值。

## 已确认结果与限制

- 原先 `full_v1/full_v2` 发生公里级发散，回放接口存在问题，旧结果不用于硬件结论。
- `audited_runs/envelope_full_v4`：完整发送 3542 帧，保存 3303 个状态，
  约从 8.135 s 初始化后覆盖至末尾；最大相对位移 0.207 m。
- 与同段固定 Tag 参考对齐后，3295 个匹配状态：位置 RMSE **0.01672 m**，
  P95 **0.03529 m**，最大 **0.05489 m**；姿态误差中位 **5.63°**。
  诊断尺度系数约 **0.828**，提示仍有尺度偏差，不能称为精确抓取轨迹。
- 这一版本回放结束后收到 SIGINT 会段错误（节点退出 -11），独立于估计是否发散。
  关闭后台图像发布线程的短测试仍复现，不能把它归因于相机或 JY901B。
- 修复镜像 `shutdown_fixed_smoke_v7`：同一段视频前 15 s，203 个估计状态，
  回放与节点退出均为 0（`CLEAN_EXIT`），状态与修复前短测数值基本一致。
- **最终完整复测 `final_full_v8`**：3542 帧全部回放，3303 个状态，
  节点和回放程序退出均为 0；没有遗留运行容器。位置 RMSE **1.672 cm**，
  P95 **3.529 cm**，最大 **5.489 cm**；姿态误差中位 **5.627°**，
  诊断尺度系数 **0.82760**。与修复前完整结果基本一致。
  这里的“通过”只指输入/初始化/状态保存/正常退出链路，不等于精确 VIO 验收。
- 无标记泛化、不同动作幅度/遮挡的独立验证尚未完成；当前固定 Tag 可见的验证
  不能代表任意工作区。不要直接替换正式数据轨迹来源。

## 剩余工作（不需要立刻重录）

1. 已补做独立转动视频诊断复测（见下节）；尺度一致性未通过，不能升级为正式轨迹来源。
2. 排查约 0.828 的诊断尺度系数和姿态差：分别检查初始化、噪声候选、
   动态加速度一致性及镜头模型。当前结果不支持把问题简单归结为“JY901B 不行”。
3. 选择独立精度要求后再做验收，不能通过缩放轨迹、调高阈值把结果变成“合格”。
4. 精度通过后才接入相机到夹爪 TCP 的轨迹转换和正式训练数据流水线。
   目前没有替换已有的固定 Tag 路线，也没有修改 DP/机械臂部署控制。

接口单元测试（AM_UMI 环境，无硬件）：

```bash
python scripts/test_openvins_offline_adapters.py
```

固定 Tag 对照命令示例（AM_UMI 环境，同一工作目录）：

```bash
python -u imu_work/compare_openvins_fixed_tag.py \
  --run-dir calibration/handheld_gripper_camera/openvins/audited_runs/final_full_v8 \
  --reference calibration/handheld_gripper_camera/camera_imu_translation_sessions/vio_full_validation_v4_uvc_source/camera_trajectory_fixed_tag_openvins_reference.csv \
  --rotation-calibration calibration/handheld_gripper_camera/camera_imu/emeet_jy901b_rotation_time_v4_uvc_source.json \
  --translation-calibration calibration/handheld_gripper_camera/camera_imu/emeet_jy901b_translation_noise_v4_uvc_source.json
```

`--run-dir` 是这次 OpenVINS 的新结果目录；`--reference` 是同段视频的固定 Tag
轨迹；后两个文件用于旋转与杆臂换算，应与测试时一致。已有对照报告拒绝覆盖。

## 独立视频复测与时间匹配审计

独立素材：`camera_imu_extrinsic_sessions/session_v13_uvc_source_rotation_validation`，
45 秒三轴转动录制。它曾用于旋转外参验证，但没有参与当前旋转/平移参数拟合，
也不是前面的 120 秒 OpenVINS 调试视频。没有重录，没有改动原始录制文件。

复测前发现该素材的旧 UVC CSV 有 **223 个非递增间隔**。之前的贪心一对一
匹配虽然能消除重复，但可能在局部接收抖动时跳过事件，之后无法回退，导致长期
错开帧编号。在此视频上，贪心重建后有 779 帧的源时钟映射值与接收时间差超过 50 ms。

`imu_work/uvc_payload_timing.py` 已改为全局有序一对一最小平方代价匹配，
不移动实际 UVC PTS 值，不人为插造均匀帧间隔。单元测试用小规模穷举检查全局最优。
独立视频重建后严格递增，接收差值中位 -10.47 ms、绝对值 P95 28.77 ms，
仍有 **6 帧**超过 50 ms（最多约 79 ms），未删除。

这些数值是**源事件与主机接收记录的关联残差，不是已测准的曝光同步误差**。
全局匹配降低了关联代价，但不能证明每一帧都关联到了真实曝光事件。
由于仍有异常，准备脚本默认拒绝继续；本轮明确使用
`--allow-timing-outliers` 做离线诊断。它不修改阈值，也不表示时间对齐验收通过。

准备副本的方法（环境 AM_UMI，目录 `/home/zzzjh/AM_UMI/umi`）：

```bash
python -u imu_work/prepare_openvins_validation_session.py \
  --session-dir calibration/handheld_gripper_camera/camera_imu_extrinsic_sessions/session_v13_uvc_source_rotation_validation \
  --output-dir calibration/handheld_gripper_camera/openvins/prepared_sessions/independent_manual_v1 \
  --allow-timing-outliers
```

`--session-dir` 为原素材，`--output-dir` 必须是新目录。脚本复制视频、原始 IMU、
UVCH 和原始时间记录，再在副本内重建时间轴/导出 JSON。这里只使用现有的 v4
旋转时间、v1 加速度和 v1 陀螺零偏校正，不重新拟合。该脚本为当前候选配置准备数据，
将来更换标定时须同步检查它与运行配置的来源。

结果 `audited_runs/independent_rotation_v9`：

- 1328 帧全部回放，1174 个状态；初始化后帧覆盖率 100%，正常退出。
- 与固定 Tag 匹配 1163 个状态，位置 RMSE **1.972 cm**、P95 **2.974 cm**、
  最大 **9.237 cm**；姿态误差中位 **6.715°**。
- 诊断尺度系数 **1.5409**。该视频以转动为主、平移范围较小，尺度指标应谨慎解释，
  但也不能据约 2 cm 的总体误差就宣称位移尺度可靠。
- 与 `final_full_v8` 的三个实际生效 YAML（去除注释）和遮罩二进制一致。
  滤波器、标定、噪声、初始化设置均未调参。

新的匹配规则也用于原 120 秒素材的对照副本，以区分“换视频”和“时间预处理修正”
的影响。`baseline_global_control_v10` 完整回放 3542 帧、保存 3302 个状态，
初始化后覆盖率 100%，正常退出。该副本还有 2 帧源事件/接收时间差超过 50 ms，
同样明确作为诊断保留。三次对照如下，误差均未进行尺度校正：

| 测试 | 位置 RMSE | 位置 P95 | 姿态误差中位 | 诊断尺度系数 |
| --- | --- | --- | --- | --- |
| 原 120 s，旧贪心规则，v8 | 1.672 cm | 3.529 cm | 5.627° | 0.8276 |
| 原 120 s，全局匹配对照，v10 | 1.389 cm | 2.648 cm | 2.714° | 0.8357 |
| 独立 45 s 转动，全局匹配，v9 | 1.972 cm | 2.974 cm | 6.715° | 1.5409 |

新规则改善了原素材相对 Tag 的姿态误差，但没有消除跨视频尺度差异。
两次复测的容器均已退出，5 项接口单元测试通过；未安装或升级任何 Conda 环境包。
这些指标尚不能证明泛化可靠，也不能等同于完成了地图保存/重定位功能。

旧结果和原始时间轴保留不覆盖。接下来应优先在**同一时间匹配规则**下
离线复核已有旋转/延迟标定和尺度约束，再决定是否需要新增采集；暂不接地图后端。

## 2026-09-06 初始化、加速度时间与真实刷新率复核

以下均为已有录制的离线对照，不连接机械臂、不重录，也不替换默认标定。

- 在全局有序 UVC 时间轴上用 90 秒旋转素材重新拟合，得到候选陀螺时间偏移
  `-6 ms`；但在独立 45 秒素材上，其角速度 RMSE 为 `0.0752 rad/s`，略差于
  原 v4 的 `0.0720 rad/s`。所以 v5 只保留为候选，默认仍用 v4 的 `-22 ms`。
- 原 120 秒素材由动态初始化成功；独立转动素材的默认配置则在特征位移约
  7 px 时用了静态初始化。只把静止判定阈值由 15 px 改为 3 px 后，独立素材
  改用动态初始化，诊断尺度从 `1.5409` 改为 `1.3189`，但姿态误差中位从
  `6.715°` 恶化到 `13.435°`。初始化会影响结果，但该阈值不是可接受的修复。
- 在全局时间轴上单独搜索加速度延迟，候选为 `-32 ms`，而陀螺仍为 `-22 ms`。
  独立 120 秒素材的刚体加速度拟合 RMSE 为 `0.1018 m/s²`；默认共用 `-22 ms`
  时约为 `0.1025 m/s²`，改善很小。
- 将这个加速度延迟与重新估计的杆臂用于 OpenVINS 后，120 秒同素材位置 RMSE
  从 `1.389 cm` 降到 `1.188 cm`，诊断尺度从 `0.8357` 改为 `0.9410`；姿态误差
  中位却从 `2.714°` 恶化到 `5.841°`。在独立 45 秒视频上，位置 RMSE 又变为
  `2.150 cm`、姿态中位 `9.234°`、诊断尺度 `0.7105`，没有跨视频复现原素材
  的尺度改善。因此该候选已被排除，不能提升为正式配置。
- 原始 NPZ 进一步显示：多数录制中，JY901B 每秒发送约 201 包，但 accel/gyro
  向量约每 4 包才改变一次，实际数值刷新率约 `50.2 Hz`，连续重复率约 75%。
  这不是丢包；它表示串口输出包率与传感器有效更新率不同。把重复值当作 201 个
  相互独立的新测量，在“每次刷新彼此独立”的简化假设下，会使由样本标准差换算的
  白噪声密度约低估 2 倍；设备内部滤波会让真实关系更复杂。
- 实际将重复段折叠为约 `50.25 Hz`，并把白噪声密度按采样周期换算为 2 倍后，
  同一 120 秒视频的位置 RMSE 为 `1.556 cm`、姿态中位误差 `5.781°`、诊断尺度
  `0.8655`，比未折叠的加速度延迟候选更差。因此“删除重复包并将噪声翻倍”被
  排除为当前修复方案；重复值更适合按原时间网格上的零阶保持测量处理。

为避免把候选混入正式链路，准备脚本现在允许显式指定
`--accel-time-calibration`，并提供仅诊断用的 `--collapse-repeated-imu`。
后者保留每段相同数值的第一个包；对应 OpenVINS 配置必须同时使用约 50.28 Hz
更新率和重新换算的噪声密度。默认不启用该选项。

### 当前结论

OpenVINS 适配器本身已能完整回放、初始化、输出状态并正常退出；问题已经从
“代码能否运行”缩小为“当前异步 UVC + JY901B 能否产生跨视频一致的米制 VIO”。
现有证据表明答案仍是否定的：全局时间匹配改善了部分误差，但初始化、加速度延迟、
杆臂和有效刷新率候选都没有同时通过两段独立视频。不要从单段视频挑一个最好数字
写回默认配置，也不要靠 Sim3 缩放把轨迹伪装成米制正确。

默认 v4 标定与 v1 OpenVINS 配置保持不变；`*_candidate` 目录和 `v11`–`v14`
结果只用于审计。当前可用于工程推进的路线仍是固定 Tag 直接测量（Tag 持续可见）
或更换为带硬件同步/可靠设备时间戳、真实高频原始 IMU 的相机—IMU组合。若坚持
现有硬件，下一轮应先验证/调整 JY901B 的真实内部刷新率与滤波设置，再做一组从
充分静止开始、含大幅三轴平移且有独立尺度参考的全新标定/验收；继续复用现有视频
调参数已经不能提供独立证据。
