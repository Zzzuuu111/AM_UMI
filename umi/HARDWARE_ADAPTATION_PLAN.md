# Plan: UMI 适配 LeRobot AlohaMini (AM2Pro)

## Context

将 UMI 的部署 pipeline 适配到 LeRobot AlohaMini2/pro (AM2Pro, 6-DOF arm)。现有 UMI 仅支持 UR5/Franka（笛卡尔控制），需要新增 AM2Pro 硬件驱动。用户通过 USB 连接机械臂，使用 lerobot 库控制舵机，选择笛卡尔控制方案（控制器内部做 IK）。

**已确认的 lerobot 代码库情况：**
- lerobot 已安装 (`pip install -e /home/zzzjh/lerobot_alohamini`)
- AM2Pro 通过 Feetech 舵机 SDK（USB 串口，半双工 UART，1Mbps 波特率）通信
- 每只臂 7 个舵机：6 个 arm joint + 1 gripper，共享一条 USB 总线
- **无 URDF 文件**，需要使用 lerobot 自带的 `RobotKinematics`（基于 placo 求解器），需要自建 URDF
- 双机械臂各走各的 USB 端口

## 架构总览

```
eval_real.py                    ← 不改
real_inference_util.py          ← 不改（SE(3) 变换层）
bimanual_umi_env.py             ← 加一个 elif 分支 + 夹爪适配
    ├── AM2ProInterpolationController      ← ★ 新建（核心）
    │   ├── 6 arm joints via FeetechMotorsBus
    │   ├── 6D pose → IK → 6 arm joint angles
    │   └── gripper command → gripper servo (ID 7)
    └── eval_robots_config.yaml           ← 改配置
```

## 需要创建的文件

### 1. `umi/real_world/am2pro_robot_6dof.urdf` — AM2Pro 运动学模型

为 6-DOF AM2Pro 臂编写一个最小 URDF。关节名与 lerobot `am-follower-6dof` profile 保持一致：
- `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_yaw`, `wrist_roll`
- EE frame: `gripper_frame_link`

URDF 供 lerobot 的 `RobotKinematics`（placo 求解器）使用，只需准确的运动学链结构，不需要真实的 CAD mesh。

### 2. `umi/real_world/am2pro_interpolation_controller.py`（核心）

参照 `franka_interpolation_controller.py` 的完整模式。

**类：`AM2ProInterpolationController(mp.Process)`**

统一 API（与现有控制器一致）：
- `schedule_waypoint(pose, target_time)` — 接收 6D 笛卡尔位姿
- `get_state(k, out)` / `get_all_state()` — 返回观测状态
- `start(wait)` / `stop(wait)` — 进程生命周期

**内部组件：**

1. **电机通信层**：直接使用 lerobot 的 `FeetechMotorsBus`
   ```python
   from lerobot.motors.feetech import FeetechMotorsBus
   bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors=arms_and_gripper_motors)
   bus.connect()
   bus.sync_read("Present_Position", motors)   # 读关节角（度）
   bus.sync_write("Goal_Position", goals)       # 写目标角（度）
   ```

2. **FK 求解器**：lerobot `RobotKinematics.forward_kinematics(joint_pos_deg)` → 4×4 矩阵
   - 用于将读取的关节角转为 `ActualTCPPose`（6D pose）

3. **IK 求解器**：lerobot `RobotKinematics.inverse_kinematics(current_joint_pos, desired_ee_pose)`
   - 6-DOF arm 可以完整控制 6 自由度位姿（位置+姿态），不需要 soft-orientation
   - `position_weight=1.0, orientation_weight=1.0`
   - 用上一帧关节角作初始猜测，保证连续性

4. **主控制循环**（`run()` 方法）：
   ```
   while keep_running:
       1. PoseTrajectoryInterpolator 插值当前 6D pose
       2. 构建 4×4 目标矩阵：pos → t, axis_angle → R
       3. IK: 4×4 矩阵 → 6 arm joint angles (degrees)
       4. sync_write Goal_Position (arm joints + gripper)
       5. sync_read Present_Position → 实际关节角
       6. FK: arm joints → ActualTCPPose (6D)
       7. 写入 ring_buffer
       8. 检查 input_queue → 更新插值器
       9. frequency 调节 (precise_wait)
   ```

5. **共享内存**：
   - `SharedMemoryQueue` 输入：cmd, target_pose(6), target_time
   - `SharedMemoryRingBuffer` 输出：ActualTCPPose(6), ActualQ(6), ActualQd(6), timestamps

6. **夹爪控制**：集成在同一进程中
   - 命令 `schedule_gripper(pos, target_time)` → 映射到 gripper servo（motor ID 7）
   - 映射：UMI 的 0-0.09m gripper_width ↔ servo 角度范围（需标定）

**关键参数：**
- 控制频率：50Hz（Feetech 舵机 ~20ms 响应，留 50% 余量）
- 6-DOF arm → 完整 IK，orientation 不受限
- 关节限位：根据 lerobot `am-follower-6dof` profile 定义

## 需要修改的文件

### 3. `umi/real_world/bimanual_umi_env.py`

**import**：新增
```python
from umi.real_world.am2pro_interpolation_controller import AM2ProInterpolationController
```

**robot_type 分发处**（~L244）：新增 elif：
```python
elif rc['robot_type'].startswith('am2pro'):
    this_robot = AM2ProInterpolationController(
        shm_manager=shm_manager,
        robot_usb_port=rc.get('robot_usb_port', '/dev/ttyUSB0'),
        frequency=50,
        receive_latency=rc['robot_obs_latency']
    )
```

**夹爪初始化**（~L247-256）：am2pro 臂无需独立夹爪控制器，gripper 设为 None

**exec_actions()**（~L501-514）：am2pro 的夹爪命令发给 robot：
```python
if rc['robot_type'].startswith('am2pro'):
    robot.schedule_gripper(pos=g_actions, target_time=...)
else:
    gripper.schedule_waypoint(pos=g_actions, target_time=...)
```

### 4. `example/eval_robots_config.yaml`

新增配置示例：
```yaml
"robots": [
  {
    "robot_type": "am2pro",
    "robot_usb_port": "/dev/ttyUSB0",
    "robot_obs_latency": 0.005,
    "robot_action_latency": 0.02
  },
  {
    "robot_type": "am2pro",
    "robot_usb_port": "/dev/ttyUSB1",
    ...
  }
]
```

## 不需要修改的文件

- `eval_real.py` — 完全通过 robot_type 分发，不感知具体硬件
- `umi/real_world/real_inference_util.py` — SE(3) 变换与硬件无关
- `umi/common/pose_trajectory_interpolator.py` — 笛卡尔插值器直接复用
- `umi/shared_memory/*` — 共享内存框架直接复用
- 所有训练/SLAM pipeline

## 依赖

```
lerobot 已安装 (pip install -e /home/zzzjh/lerobot_alohamini)
无需额外安装 pybullet（用 lerobot 自带的 RobotKinematics/placo）
```

## 验证步骤

1. **舵机通信测试**：单独运行 FeetechMotorsBus 连接 → ping 所有电机 → 读 Present_Position
2. **FK 验证**：给定关节角 → FK → 检查输出的 4×4 矩阵是否合理
3. **IK 闭环测试**：给定关节角 → FK → IK → 验证能回到原关节角
4. **遥操作测试**：SpaceMouse 控制 AM2Pro，验证笛卡尔轨迹流畅性
5. **策略部署测试**：加载 UMI checkpoint → `python eval_real.py -rc example/eval_robots_config.yaml ...`

## 风险和注意事项

1. **URDF 精度**：需要手动创建 AM2Pro 6-DOF 运动学 URDF，用 lerobot 的 `RobotKinematics`（placo）加载，URDF 的准确性直接影响 IK 精度
2. **工作空间**：AM2Pro 臂展远小于 UR5/Franka，策略输出的目标位姿可能超出可达范围
3. **USB 延迟**：半双工 UART 串口通信延迟（~5-10ms）比 RTDE/zerorpc 大，需适当调高 latency 补偿参数
4. **夹爪标定**：需要测量 UMI 夹爪宽度 (0-0.09m) 与舵机角度 (例如 0°-90°) 的映射关系
