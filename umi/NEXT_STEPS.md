# AM_UMI 当前剩余工作清单

本文是当前 **EMEET UVC 相机 + JY901B 外置 IMU + 手持夹爪 + 单臂 AM2Pro** 的工作清单。旧 `PROJECT_PROGRESS.md` 中关于 Osmo/GoPro/Gyroflow 的内容是历史记录；后续现场工作以本清单、`calibration/` 和 [VIEW_REFERENCE.md](VIEW_REFERENCE.md) 为准。

固定 Tag 正式路线与 IMU/VIO 实验路线的目录、命名与禁止混用规则见
[PIPELINES.md](PIPELINES.md)。

目标闭环为：手持夹爪录制带相机/IMU/夹爪信息的演示 → EMEET 视觉惯性 SLAM 与 Tag 处理 → 生成 zarr 数据集 → 训练策略 → AM2Pro 安全、可重复地部署。

## 当前已验证，不需要重复做

- [x] `AM_UMI` 环境可加载并恢复现有 UMI Diffusion Policy checkpoint。
- [x] GPU 推理在 4 GB 显存约束下可运行；8 次去噪的干运行延迟稳定约 0.46 s。
- [x] EMEET 腕部相机、AM2Pro 状态读取、保持控制器、同步观测和有限时策略实体小幅执行已通过。
- [x] 两台相机的内参文件已按手持/机械臂角色放到 `calibration/`；手持候选内参重投影误差为 0.740 px。
- [x] JY901B 已确认使用 `/dev/ttyUSB0`、`460800` baud，各类数据包回传约 201 Hz；离线检查发现当前加速度/角速度新数值约每 4 包更新一次（约 50 Hz），因此不能把包率直接称为真实采样率。已增加只读配置检查与不保存的 256 Hz 带宽动态测试工具，待手持装置回到现场补测。
- [x] JY901B 六面加速度标定完成并独立验证：`calibration/handheld_gripper_camera/imu/jy901b_accel_v1.json`。
- [x] 最终安装状态下的 JY901B 陀螺零偏和相机-IMU 三轴粗核验完成；EMEET + JY901B 同步原始采集实测 29.90 FPS 且稳定退出。
- [x] EMEET ↔ JY901B 旋转外参和时间偏移已完成并升级到 UVC 源时钟版本：`camera_imu/emeet_jy901b_rotation_time_v4_uvc_source.json`；独立会话角速度相关性 0.9847、RMSE 0.0944 rad/s，采用 IMU 相对相机的时间偏移 -22 ms。
- [x] UVC 的 PTS/SCR 源时钟已接入采集；JY901B 的 0x50 设备时间也已接入 IMU 时间轴。30 分钟静止 Allan 记录与噪声候选已保存于 `imu/allan/`。
- [x] Docker daemon、SLAM 镜像和本机 CPU 环境可用。
- [x] EMEET 手持夹爪遮罩已生成：`calibration/handheld_gripper_camera/slam/emeet_handheld_gripper_mask_960x540_v1.png`。`run_emeet_orbslam3.py` 默认使用它，避免把相机刚性连接的夹爪和夹爪 Tag 当作世界特征。
- [x] 已保存 AM2Pro 起始基准 `calibration/robot_wrist_camera/view_references/task_start_v2/`，并实现手持纯画面对齐、机械臂画面 + 位姿检查两种流程。

## A. 固定硬件与补齐视觉惯性标定（正式采集前必须完成）

- [x] **锁定手持刚体。** 相机、手持夹爪、IMU 和线缆已固定；之后不要拆装或相对转动。
- [x] **确认手持相机内参归属。** 当前使用实际 EMEET 的 1920×1080、30 fps、MJPG 内参 `emeet_handheld_...json`；更换相机或采集模式须重做内参。
- [x] **测陀螺仪静止零偏与坐标轴方向。** 最终刚体安装状态下的零偏和三轴粗核验已完成。
- [x] **标定相机 ↔ IMU 固定旋转外参。** `R_camera_imu` 已由 `session_v5` 求得，并由独立 `session_v6_rotation_validation` 验证。
- [x] **测相机 ↔ IMU 时间偏移。** 已得到 `imu_time_offset_for_camera_s = -0.026 s`；定义和矩阵均保存在同一标定 JSON 中。
- [x] **为 EMEET 创建视觉惯性 SLAM 初始设置与导出器。** `calibration/handheld_gripper_camera/slam/emeet_jy901b_orbslam3_v1.yaml` 已包含 960×540 处理内参、`T_b_c1` 和 JY901B 初始噪声；`imu_work/export_handheld_imu_to_orbslam3.py` 导出真实帧时间与校正 IMU，`am_umi_orb_slam3_emeet:latest` 镜像已通过离线追踪冒烟测试。首段 mapping 后再按实际丢失情况决定是否调整噪声或增加夹爪遮挡 mask。

完成标准：用一小段同步会话可将 JY901B 数据导出为 SLAM 接受的格式，并能启动 EMEET 视觉惯性 SLAM；不要求此时地图已很好。

## 当前 VIO 状态（2026-09-07）

- [x] 相机—IMU 旋转/时间、平移/噪声候选均已完成到 UVC 源时钟版本：`camera_imu/emeet_jy901b_rotation_time_v4_uvc_source.json` 与 `camera_imu/emeet_jy901b_translation_noise_v4_uvc_source.json`；平移标定的独立数学验证已通过。
- [x] ORB-SLAM3 完整惯性模式仍不稳定：曾出现尺度过小、频繁丢失和段错误，因此不作为当前正式链路。
- [x] OpenVINS 离线完整 VIO 可干净结束且覆盖完整后初始化帧。`openvins/audited_runs/final_full_v8` 输出 3,303 个有限状态、111.9 秒轨迹、正常退出；说明输入转换、时间关联、滤波器执行链已经通。
- [ ] **完整 VIO 尚未达到正式标签精度。** `final_full_v8` 与同一会话固定 Tag PnP 轨迹作单个 SE(3) 对齐后，位置 RMSE 为 16.7 mm、P95 为 35.3 mm、姿态 P95 为 6.46°；诊断 Sim(3) 尺度为 0.828。这一比较共享相机内参，仍不是独立计量真值，但已足以说明当前 VIO 不能替代固定 Tag 直测的正式标签。
- [x] JY901B 的包传输约 201 Hz，但加速度/角速度的新数值约每四包更新一次（约 50 Hz）。使用等效 50 Hz 候选的 OpenVINS 虽可完整运行，位置 RMSE 仍约 15.6 mm、P95 33.7 mm，尚未解决精度问题。

当前主路线为“**固定世界 Tag 逐帧直接测量米制相机轨迹**”，不把任何 ORB-SLAM3/OpenVINS 轨迹作为正式 demo 标签。固定 Tag 路线不依赖 IMU；每段正式会话仍保留 IMU 原始数据、JY901B 设备时钟和 UVC PTS/SCR，供以后重处理。

若要继续恢复完整 VIO，应优先取得真正硬件同步、可靠曝光时刻的相机时间戳和真实高频原始 IMU，或用独立运动真值系统对 OpenVINS 做针对性误差诊断。仅继续调当前 YAML/噪声参数预计难以把厘米级误差降到正式抓取标签所需水平。

## B. 手持夹爪 Tag、建图与会话处理（正式 demo 前必须完成）

- [ ] **贴夹爪 Tag。** 单夹爪用 `calibration/shared_tags/aruco_gripper_0_letter.pdf`：最终相机画面左指端是 ID 0，右指端是 ID 1；保持打印比例，Tag 边长 16 mm。
- [ ] **布置固定桌面 Tag。** 选择一个不会被移动、容易看到的位置。它用于把不同示范/地图对齐到同一任务世界坐标，也为后续机械臂手眼标定提供目标。
- [ ] **录制 mapping 会话。** 手持夹爪打开、相机/夹爪/IMU 固定；桌面放有纹理。缓慢平移并从不同角度观察场景，少急转、少纯原地旋转；确保固定桌面 Tag 多次可见。
- [ ] **处理 mapping 并检查质量。** 运行新的 EMEET+JY901B SLAM 处理链，检查连续跟踪、尺度、重投影、Tag 可见率和地图覆盖。地图漂移/丢失时先修正相机-IMU 外参、时间偏移或采集动作，再重录。
- [x] **完成夹爪开度标定。** `gripper_tag_sessions/tag_visibility_v2/gripper_range_v1.json` 使用 977 帧双 Tag 同时检测，采用无符号双 Tag 横向间距；开口标签范围为 0–0.07737 m。后续处理禁止把单 Tag 备用估计混入该范围。
- [ ] **完成 SLAM-Tag 对齐。** 将 SLAM 轨迹转换到固定桌面 Tag 坐标，检查同一静态起点重复录制时的误差。
- [x] **固定 Tag 直接轨迹软件链。** 已实现米制轨迹导出、来源报告、孤立跳变剔除、最多 5 帧短缺口插补和轻量平滑；`mapping_v5` refined v2 有效率 98.22%，长缺口保持 lost。数据集计划脚本可显式选择 `--trajectory-frame fixed_tag` 并记录 provenance。
- [x] **标定手持 EMEET 相机到夹爪 TCP 平移。** 当前有效文件为 `gripper_geometry/emeet_handheld_camera_to_tcp_v2.json`；独立枢轴验证 `tcp_pivot_v3_validation` 得到 85.2% 支点内点、4.65 mm RMSE。fixed_tag 模式可引用该文件，不能沿用原版 GoPro 安装尺寸。相机/支架/夹爪任一拆装后必须重新标定；其中旋转 `[0,0,0]` 仍是当前安装对齐假设，须在首条低风险 demo 中继续观察验证。

完成标准：一段 demo 可得到时间同步的视频、IMU、相机轨迹、夹爪位姿/开度和固定任务坐标，而不只是 `raw_video.mp4`。

## C. 录制正式抓取数据集（在 B 完成后）

- [ ] **定义一个任务版本。** 固定任务物体、容器/目标位置、桌面 Tag、起始夹爪开度和成功判据；一次只做一个清晰任务。
- [ ] **保存并使用起始基准。** 机械臂部署起点保存于 `task_start_vN`；手持录制前先用原始画面和策略画面对齐，稳定后按 `R` 才开始录制。详见 [VIEW_REFERENCE.md](VIEW_REFERENCE.md)。
- [ ] **录 mapping 与 demo。** 每次改变桌面、相机支架、物体尺寸、镜头模式或 Tag 位置后，重新录 mapping；每条 demo 单独目录、单独会话元数据。
- [ ] **做采集质量检查。** 删除/重录严重模糊、丢帧、SLAM 断轨、Tag 长时间不可见、物体被遮住或任务失败原因不明确的示范。
- [ ] **形成训练/验证划分。** 保留未参与训练的完整 episodes 作验证；不要把同一段录制裁成相邻片段后同时放到训练与验证集。
- [ ] **生成 UMI zarr。** 将已校准的相机、机器人/夹爪轨迹和动作表示打包为当前 UMI 配置所需的 zarr；检查 `shape_meta`、单位（m/rad）、时间顺序和相对动作表示。
- [ ] **执行训练前数据质量门禁。** 在 `AM_UMI` 环境运行 `scripts/validate_umi_dataset.py`；结构、episode 边界、RGB 解码、NaN/Inf、动作与状态一致性、夹爪范围和单帧轨迹跳变全部通过后，数据才能进入训练。它是只读离线检查，不连接机械臂。
- [ ] **分级做 Arm Replay 验证。** 先用 `scripts/render_umi_dataset_replay.py` 生成不连接机械臂的 RGB、TCP 轨迹和夹爪开度同步视频，再做离线可达性/IK 检查；最后才在清空工作区、低速、缩放轨迹和有人急停的条件下，仅回放一条已通过质量门禁的 episode。实体回放是验证运动转换，不代替视觉与时间同步检查。
- [ ] **评估 4DGS/场景重放增强。** 第一版真实数据基线通过后，再评估用 3D/4D Gaussian Splatting 重建并生成新视角、光照、背景或物体位置变化。合成图像必须与变换后的动作标签严格几何一致，并与真实验证集分开；它不能修复原始时间戳、轨迹或夹爪标签错误。

建议先做少量端到端样本验证格式，再逐步扩到足够多且多样的成功示范；不要先批量录很多原始视频、最后才发现 SLAM 或 Tag 链路不正确。

训练数据准入顺序固定为：**原始会话质检 → 固定 Tag 轨迹质检 → zarr 质量门禁 → 离线 Arm Replay/IK → 少量实体慢速回放 → 训练**。4DGS 属于通过准入后的数据增强支线，不作为坏数据修复工具。

## D. 机械臂几何标定与部署准备

- [ ] **完成 AM2Pro 手眼标定。** 相机固定在机械臂，桌面 Tag 固定不动；机械臂携带相机到约 10–20 个不同位置/朝向拍 Tag。求相机↔夹爪以及基座↔任务世界关系。
- [ ] **验证手眼结果。** 用未参与求解的 3–5 个姿态检查 Tag 重投影/空间误差；若支架移动，手眼结果立即失效并需重做。
- [ ] **确定起始位容差。** 使用 `align_umi_view_reference.py` 回到 `task_start_v2`，测真实可重复误差。细抓取可先以约 10–20 mm、3–5°为目标，再由实测成功率调整。
- [ ] **决定人工复位方式。** 当前可“安全控制粗调 → 退出 → 复位检查 → 必要时重复”。若需要一个界面内的连续低速微调，应另行实现带速度/工作空间/急停限制的 AM2Pro jog 工具。
- [ ] **复测端到端时延。** 用真实相机帧和 AM2Pro 动作确认相机、观测、策略、控制器延迟；让频率、`steps_per_inference`、执行点数与推理时间相匹配。
- [ ] **准备现场安全流程。** 明确物理急停/断电位置、软件 `Ctrl+C`、最大位置/旋转/夹爪速度、工作空间、人与机械臂最小距离和每次测试最大时长。

## E. 训练、离线评估与实体分级测试

- [ ] **先训练新的 Diffusion Policy 基线。** 使用新 zarr、当前 AM2Pro 相对动作表示和 4 GB 显存安全配置；训练设置仍可使用 AdamW。保留数据版本、配置、随机种子和 checkpoint。
- [ ] **离线验证。** 检查训练/验证损失、动作量级、夹爪开闭范围、从 checkpoint 恢复和 GPU 推理显存；先做不发动作的 dry run。
- [ ] **重复推理基准测试。** 插电并固定性能模式后，多次测试 6/8/16 去噪步；选择既能在控制周期预算内稳定完成、又保持足够任务质量的步数。
- [ ] **实体分级测试。** 依次进行：保持/观测 → dry run → 1 个受限动作点 → 1–2 秒自动 episode → 完整 episode。每阶段成功且确认安全后才进入下一阶段。
- [ ] **记录任务指标。** 至少记录成功率、失败类型、起始对齐误差、推理时间、每次动作长度、是否发生急停/限速，以及视频证据。

## F. 策略比较（基线成功后再做）

- [ ] **Diffusion Transformer。** 仓库有 UMI 配置，但它仍是扩散策略；需用同一数据重新训练，重新测显存、去噪延迟和实体成功率。
- [ ] **ACT。** 可作为 LeRobot 侧备选；需要把 UMI zarr 转为 ACT 所需数据格式、实现 AM2Pro 观测/动作适配、重新训练，并重新做 dry run 和实体分级测试。现有 Diffusion Policy checkpoint 不能直接转换为 ACT。
- [ ] **确定最终策略。** 用相同任务、相同初始视角误差范围、相同安全限速和未见验证 episodes 比较成功率、延迟、恢复能力与稳定性，而不是只比较训练损失。

## 当前推荐的下一步

当前离线软件已经优先接通**固定 Tag 直接米制轨迹**。夹爪回到现场后的顺序是：补做 JY901B 256 Hz 带宽动态测试 → 标定 EMEET 相机到夹爪 TCP → 完成左右指端 Tag 开度标定 → 先录少量 fixed Tag 持续可见的 demo 做端到端 zarr 验证。完整 JY901B VIO 继续作为独立改进项，不阻塞固定 Tag 主路线；所有会话仍保存原始 IMU 与 UVC 时间数据。
