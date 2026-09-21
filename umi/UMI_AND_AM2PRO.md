# UMI 代码与 AM2Pro 适配讲解

> 第一部分讲 UMI 原始代码的核心（位姿表示 + 轨迹插值器），第二部分详细讲 AM2Pro 适配。
> 核心结论一句话：**UMI 真正重要的就是「相对位姿动作 + 轨迹插值器」，我们的适配只是把它们接到
> Feetech 舵机上的一层接线。**

---

# 第一部分：UMI 原始代码核心

## 1. 三种位姿表示（`umi/common/pose_util.py`）

同一个位姿有三种等价形式：

| 表示 | 维度 | 用在哪 |
|---|---|---|
| `[pos(3), rotvec(3)]` | 6 | 机器人下发、插值器、环境观测 |
| 4×4 齐次矩阵 | 4×4 | 数学变换（求相对位姿） |
| `[pos(3), rot6d(6)]` | 10 | 喂给神经网络的输入/输出 |

网络用 **10D（rot6d）** 而不是 6D（rotvec），因为 rotvec 在旋转角→0 时有奇异性，rot6d（旋转矩阵
前两列铺平）连续无歧义，最适合回归。转换函数：

```python
pose_to_mat / mat_to_pose          # 6D ↔ 4×4
mat_to_pose10d / pose10d_to_mat    # 4×4 ↔ 10D（rot6d 用 Gram-Schmidt 正交化还原旋转矩阵）
```

## 2. relative / abs / delta 转换（`diffusion_policy/common/pose_repr_util.py`）

这是 UMI 动作表示的核心。`convert_pose_mat_rep(pose_mat, base_pose_mat, pose_rep, backward)`：

- `backward=False`：**训练方向**，绝对 → 相对（生成标签）。
- `backward=True`：**推理方向**，相对 → 绝对（下发机器人）。

`base_pose_mat` 通常取当前观测的**最后一帧** `pose_mat[-1]`。

最常用的 `'relative'` 只有两行：

```python
out = np.linalg.inv(base_pose_mat) @ pose_mat   # forward（训练）
out = base_pose_mat @ pose_mat                  # backward（推理）
```

即「`pose` 在 `base` 坐标系下的位姿」。UMI 的 cup 策略用的是 `relative`
（`umi.yaml` 里 `obs_pose_repr: relative, action_pose_repr: relative`）。

另外两种：`'rel'` 是注释里明确写的 legacy buggy 实现（位置在世界系相减、旋转却相对，混用），
只为兼容旧 checkpoint；`'delta'` 是相邻帧差分（增量式控制，反向 cumsum 还原）。

## 3. 轨迹插值器（`umi/common/pose_trajectory_interpolator.py`）— 最核心的类

把**稀疏 waypoint** 插值成任意时刻的稠密位姿，并**自动限速**。

```python
class PoseTrajectoryInterpolator:
    def __init__(self, times, poses):   # poses: (N, 6) = [pos(3), rotvec(3)]
        self.pos_interp = si.interp1d(times, poses[:,:3], axis=0)  # 位置：线性插值
        self.rot_interp = st.Slerp(times, st.Rotation.from_rotvec(poses[:,3:]))  # 旋转：Slerp

    def __call__(self, t):              # 控制循环每帧调用一次
        t = np.clip(t, times[0], times[-1])
        return concat(self.pos_interp(t), self.rot_interp(t).as_rotvec())
```

核心方法是 `schedule_waypoint(pose, time, max_pos_speed, max_rot_speed, curr_time, last_waypoint_time)`：

1. 裁剪掉过期 waypoint；
2. 从当前末端位姿到新路点算最小耗时 `dist / max_speed`；
3. **若目标时刻太近会导致超速，就自动推迟到达时刻**；
4. append 新路点。

这样不管上层怎么排路点，插值器都保证输出轨迹速度 ≤ 上限。`drive_to_waypoint` 是它的简化版
（遥操作/手动测试用，不保留中间路点）。`trim` 用于丢弃已执行过的旧路点。

> 环境层会乘 `√3`（`max_pos_speed * cube_diag`），因为位姿是归一化到单位立方体的，速度要换回真实米/秒。

## 4. 整条链路怎么串起来

**训练侧**（`diffusion_policy/dataset/umi_dataset.py` 的 `__getitem__`）：

```python
# 把 7D 绝对动作 → 10D 相对动作
action_mat = pose_to_mat(action[..., :6])                       # 绝对位姿
action_pose_mat = convert_pose_mat_rep(action_mat,
    base_pose_mat=pose_mat[-1], pose_rep='relative', backward=False)  # 相对最后一帧
action_pose10d = mat_to_pose10d(action_pose_mat)                # 10D
# 动作 = [pos(3), rot6d(6), gripper(1)] = 10D
```

**推理侧**（`umi/real_world/real_inference_util.py`）：

```python
# get_real_umi_obs_dict：观测绝对 → 相对（backward=False）
# get_real_umi_action：动作相对 → 绝对（backward=True）
action_pose_mat = pose10d_to_mat(action_pose10d)
action_mat = convert_pose_mat_rep(action_pose_mat,
    base_pose_mat=当前观测位姿, pose_rep='relative', backward=True)
action_pose = mat_to_pose(action_mat)   # 6D 绝对位姿
```

**执行侧**（`eval_real.py` → 环境 `exec_actions` → 控制器）：

```python
action_timestamps = arange(len(action)) * dt + obs_timestamps[-1]  # 每个动作带绝对时间戳
robot.schedule_waypoint(pose=绝对6D位姿, target_time=时间戳)
```

控制器（原 `rtde_interpolation_controller.py`，我们的 `am2pro_controller_server.py` 照它写的）主循环
每帧 `pose_interp(t_now)` 取插值位姿下发，再做一次 `monotonic - time + target_time` 的时钟转换
（客户端传 wall-clock，插值器用 monotonic）。

---

# 第二部分：AM2Pro 适配（详细）

## 0. 背景与约束

- UMI 部署栈原生只支持 UR5 / UR5e / Franka（RTDE / zerorpc 笛卡尔控制）。
- AM2Pro 用 Feetech 舵机（USB 串口、半双工 UART），6 关节 + 1 夹爪共 7 电机。

**最关键约束：两个 Python 环境无法共存**（语法 + numpy 二进制不兼容）：

| 环境 | Python | 关键依赖 | 用途 |
|---|---|---|---|
| UMI | 3.9 | torch 2.1 / numpy 1.x | 训练、推理、`eval_real.py` |
| lerobot | 3.12 | numpy 2.x / placo / FeetechMotorsBus | 控制舵机、做 IK |

所以适配的地基是：**把「硬件驱动 + IK」拆成独立 server 进程（lerobot 环境），客户端留在 UMI
环境，两者用 Unix domain socket 通信。** 这也是为什么不能像原 `RTDEInterpolationController`
那样用 `mp.Process`——fork 会继承 UMI 的 py3.9 解释器，跑不了 lerobot 代码。

## 1. 架构总览

```
┌──────────────── UMI 进程 (py3.9) ─────────────────┐
│  eval_real.py / bimanual_umi_env.py                 │
│     └─► am2pro_interpolation_controller.py（客户端）│
│            │  pack_command  ←── am2pro_protocol.py  │
└────────────┼───────────────────────────────────────┘
             │  Unix domain socket
┌────────────▼──────── lerobot 进程 (py3.12) ─────────┐
│  am2pro_controller_server.py                        │
│     插值 6D pose → IK(迭代) → 写舵机 → 读回 → FK → 回传 │
│     └─► /dev/ttyACM0 ──► 7 个 Feetech 舵机            │
└─────────────────────────────────────────────────────┘
```

## 2. 通信协议层 `am2pro_protocol.py`

**作用**：定义两个进程「说什么、怎么打包」。零第三方依赖（`struct` + `numpy`），两个环境都能原样
import——这是它被单独抽出来的唯一原因。

### 分帧格式

```
[1 字节: 消息类型][4 字节: 负载长度, 大端][负载 bytes]
```

### 消息类型 + 命令枚举

| 消息 | 方向 | 负载 |
|---|---|---|
| `MSG_STATE` | server→client | 固定 23 个 float64 |
| `MSG_COMMAND` | client→server | int32 + 9 个 float64 |
| `MSG_READY` | server→client | 空 |
| `MSG_ERROR` | server→client | UTF-8 文本 |

```python
CMD_STOP=0, CMD_SERVOL=1, CMD_SCHEDULE_WAYPOINT=2, CMD_SCHEDULE_GRIPPER=3
```

### 状态负载（23 float64）

```python
# pose(6) + q(7) + qd(7) + [gripper_position, recv_ts, robot_ts](3)
_STATE_STRUCT = struct.Struct(">" + "d" * 23)
```

`ActualQ` / `ActualQd` 是 **7 维**——前 6 维是手臂关节角，第 7 维塞了夹爪舵机值，为了沿用 UMI 的
`robot_joint_pos` 结构。

### 命令负载（`>i9d`）

```python
_COMMAND_STRUCT = struct.Struct(">i" + "d" * 9)
# cmd + target_pose(6) + target_time + duration + target_gripper
```

**`target_gripper = -1.0` 表示「夹爪不动」**——这是协议约定。客户端在 `schedule_waypoint` / `servoL`
里传 `-1.0`，只有 `schedule_gripper` 传真实宽度。

### 收发函数（两种读法对应两种场景）

- `send_msg` / `recv_exact` / `recv_msg`：`recv_msg` 是**阻塞读**一条完整消息，给客户端 reader 线程用。
- `try_recv_messages(sock, buf)`：**非阻塞 drain**，给服务端 50Hz 循环用（不能卡住控制循环）。用持久
  `bytearray` 累积，按帧头切出完整消息，半截的留下次；返回 `(messages, alive)`，`alive=False`
  表示对端关 socket。

## 3. 服务端 `am2pro_controller_server.py`（大脑）

跑在 lerobot 环境，做四件事：连舵机、笛卡尔→关节角 IK、50Hz 控制循环、回传状态。

### 3.1 import 纪律

文件头明确**只允许 import**：stdlib/numpy、`umi.common` 里的纯 numpy 工具（插值器/位姿/precise_sleep）、
lerobot 的 motors+model。**绝不能 import torch / diffusion_policy / umi.shared_memory**，否则在
py3.12 + numpy2.x 环境直接崩。

### 3.2 配置常量

```python
URDF_ARM_JOINT_NAMES = ["right_shoulder_pan", ...]  # URDF 关节名（6个）
MOTOR_ARM_NAMES      = ["shoulder_pan", ...]        # lerobot 电机名（6个，顺序一一对应）
TCP_FRAME_NAME = "right_Fixed_Jaw"                  # IK/FK 末端坐标系
```

`_MOTOR_SPEC`：7 个电机 `(名字, ID, 型号, 归一化模式)`，手臂关节用 `DEGREES`，夹爪用 `RANGE_0_100`。

### 3.3 夹爪宽度 ↔ 舵机值映射

```python
def map_width_to_servo(width, args):   # 米 → 0-100
    ratio = (clip(width) - width_min) / (width_max - width_min)
    return servo_closed + ratio * (servo_open - servo_closed)
```

`map_servo_to_width` 是反向（回传状态时把舵机值还原成米）。

### 3.4 `connect_hardware` — 硬件初始化

按顺序五步，失败会抛异常（`main` 转成 `MSG_ERROR` 回给客户端）：

1. `FeetechMotorsBus` + `bus.connect()`
2. `RobotKinematics(urdf, target_frame, joint_names)` 加载 URDF（在 connect 之前构造，提前暴露 URDF 错误）
3. `bus.calibration = bus.read_calibration()`——读 EEPROM 标定（homing 偏移 + 量程），没有它编码器值转不成度数
4. `torque_disabled()` 上下文里 `configure_motors` + 写 PID（P=16/I=0/D=32）+ 夹爪单独降扭矩上限（防烧电机）
5. **关键防抖**：读当前 `Present_Position` 后，立即 `sync_write("Goal_Position", 当前值)`——
   否则上电瞬间舵机会冲向 EEPROM 里残留的上次目标角。

### 3.5 `run_loop` — 50Hz 控制循环

每 20ms 一圈，10 步：

```python
pose_command = pose_interp(t_now)                    # 1. 插值当前 6D 位姿
t_des = pose_to_mat(pose_command)                    # 2. 6D → 4×4

q_target = arm_joint_pos.copy()                      # 3. IK 迭代
for _ in range(ik_iterations):
    q_target = kinematics.inverse_kinematics(q_target, t_des, ...)

bus.sync_write("Goal_Position", goals)               # 4. 写臂+夹爪目标
motor_positions = bus.sync_read("Present_Position")  # 5. 读回实际
actual_pose = FK(arm_joint_pos)                      # 6. 实际位姿
actual_qd = (pos - prev_pos) / dt                    # 7. 差分速度

t_recv = time.time()                                 # 8. 回传状态（wall-clock）
send_msg(conn, MSG_STATE, pack_state(actual_pose, actual_q, actual_qd, ...))

messages, alive = try_recv_messages(conn, cmd_buf)   # 9. 非阻塞处理命令
precise_wait(t_start + (iter_idx+1)*dt)              # 10. 锁频
```

### 3.6 为什么 IK 要迭代

placo 的 `inverse_kinematics()` 内部 `solve(True)` **只做一步速度级 QP，不收敛**（单次对远处目标
误差 1.4~49mm）。修复：用上一次结果做种子反复迭代 ~5 次，每次从「离目标更近处」重新线性化雅可比，
收敛到 0.0mm 级误差。种子用**当前实际关节角**而非上次 IK 输出，同时追踪了舵机未到位的真实偏差。

### 3.7 时钟转换（关键）

```python
# CMD_SCHEDULE_WAYPOINT 分支
target_time = time.monotonic() - time.time() + cmd["target_time"]
```

客户端传的 `target_time` 是 wall-clock（`time.time()`），插值器用 `time.monotonic()`。`monotonic - time`
是两钟偏移量（同机、走时一致、只差零点），加上它完成 wall-clock → monotonic 转换。
**回传状态却用 `time.time()`**，因为 `bimanual_umi_env.get_obs()` 要拿它和摄像头 wall-clock 时间戳对齐。

### 3.8 `main` 生命周期

先 bind socket → accept 客户端 → 再连硬件（慢且可能失败）→ 失败发 `MSG_ERROR` / 成功发 `MSG_READY`
→ `run_loop` → `finally` 断舵机、关 socket、删 socket 文件。**先 bind 再连硬件**是为了让客户端能先
连上等待，不用等慢速硬件初始化。

## 4. 客户端 `am2pro_interpolation_controller.py`（API 镜像）

跑在 UMI 环境，是上层代码直接操作的对象。**API 与原 Franka 控制器一致**，上层无感知。用
`subprocess.Popen` 拉起 server + 维护 socket + reader 线程。

### 4.1 关键字段

```python
DEFAULT_SERVER_PYTHON = "/home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python"  # 硬编码
self.sock_path = f"/tmp/am2pro_{uuid.uuid4().hex}.sock"   # 每实例独立（双臂不撞）
self._states = collections.deque(maxlen=get_max_k)        # 状态环缓冲
```

### 4.2 `start(wait=True)`

1. 校验 server python 存在（否则清晰报错）
2. 把频率/IK/夹爪标定参数拼成 CLI，`Popen` 启动 server
3. `_connect_with_retry()` 重试连 socket（`poll()` 检测 server 秒退；三种连接异常吞掉重试 0.05s）
4. 起 reader 线程
5. `wait=True` 时 `start_wait()` 等 READY

### 4.3 `_reader_loop`（独立线程）

`recv_msg` 阻塞收消息：`MSG_STATE` 加锁追加进 deque；`MSG_READY` 置事件；`MSG_ERROR` 存错误文本。
socket 关闭时**兜底置 READY 事件**，避免 `start_wait` 卡死。

### 4.4 命令 API（加锁 `_send_lock` 保证线程安全）

| 方法 | 命令 | target_time | duration | target_gripper |
|---|---|---|---|---|
| `schedule_waypoint(pose, target_time)` | `CMD_SCHEDULE_WAYPOINT` | 绝对时间戳 | 0.0 | -1.0 |
| `schedule_gripper(pos, target_time)` | `CMD_SCHEDULE_GRIPPER` | 绝对时间戳 | 0.0 | 真实宽度 |
| `servoL(pose, duration)` | `CMD_SERVOL` | 0.0 | 相对秒数 | -1.0 |

### 4.5 状态 API

- `_stack`：`_ARRAY_KEYS`（ActualTCPPose/ActualQ/ActualQd）堆 `(N,d)`，`_SCALAR_KEYS`（gripper_position/
  时间戳）堆 `(N,)`——精确模拟原 `SharedMemoryRingBuffer.get_all()` 返回格式。
- **`TargetTCPPose = ActualTCPPose` hack**：AM2Pro 无独立命令位姿流（插值在 server 内部），用实际位姿
  充当目标，避免 `eval_real.py` 遥操作循环读 `TargetTCPPose` 时 `KeyError`。
- `get_all_state()` 空状态时返回**形状正确的空数组**（而非 `None`），让 `get_obs()` 的对齐逻辑安全度过
  「服务端首帧到达前」的时间窗。

## 5. 夹爪标定 `am2pro_gripper_calibrate.py`

跑在 lerobot 环境，**只驱动夹爪一个电机**（ID 7），手臂不碰。交互式挪夹爪（`c` 记录全闭、`o` 记录全开、
`w <米>` 输入实测爪缝、`q` 退出打印），产出 4 个映射参数。实测 `servo_closed=97.1, servo_open=1.4`。

## 6. 运动学模型 `alohamini2pro_right_arm_kinematics.urdf`

给 placo 的 `RobotKinematics` 提供关节链几何（lerobot 没给现成 URDF）。删掉原 CAD URDF 的 Git-LFS
mesh 引用，只留 kinematic 链。关节链：`shoulder_pan → shoulder_lift → elbow_flex → wrist_flex →
wrist_yaw_joint → wrist_roll → right_Fixed_Jaw`（末端 TCP）。每个 joint 的 `origin`（xyz+rpy，来自 CAD）、
`axis`、`limit`（弧度限位）是运动学关键。夹爪 `right_gripper` 是独立 revolute 关节，不在 `URDF_ARM_JOINT_NAMES`
里，IK 不求解它，由 `map_width_to_servo` 独立控制。

## 7. 改动的 3 个文件

### 7.1 `bimanual_umi_env.py`（改动最多，全是「加分支/判空」）

| 位置 | 改动 | 为什么 |
|---|---|---|
| import | 引入 `AM2ProInterpolationController` | — |
| 机器人构造 | `elif rc['robot_type'].startswith('am2pro')`，从 config 读 USB 端口/标定值 | 分发到我们的控制器 |
| 夹爪构造 | am2pro 设为 `None` | 夹爪集成进臂，无独立 WSG 控制器 |
| `is_ready`/`start`/`stop` 等 | 夹爪循环加 `if gripper is not None` | 防对 `None` 调用崩溃 |
| `get_obs()` | 夹爪状态改从 robot 环缓冲取 `gripper_position` | 夹爪状态存在 robot 数据里 |
| `exec_actions()` | 夹爪命令改走 `robot.schedule_gripper()` | 同上 |
| `get_gripper_state()` | 夹爪 `None` 时返回 `None` | 上层容错 |

核心思想：**AM2Pro 夹爪和手臂共用一条电机总线，没有独立夹爪控制器**，所有夹爪相关 `None` 分支都是
「改用 robot 控制器代劳」。

### 7.2 `eval_real.py`（两处判空）

```python
# 读夹爪状态：gs is None 时改从 robot 状态取
gripper_target_pos[gs_idx] = float(gs['gripper_position']) if gs is not None \
    else float(robot_states[gs_idx].get('gripper_position', 0.0))

# 发夹爪命令：有独立夹爪走原路，否则走 robot.schedule_gripper
if env.grippers[idx] is not None:
    env.grippers[idx].schedule_waypoint(grip, ...)
else:
    env.robots[idx].schedule_gripper(grip, ...)
```

### 7.3 `example/eval_robots_config.yaml`

加注释掉的 `am2pro` 段（`robot_usb_port`、`robot_python`、夹爪标定值），夹爪段加 `{}` 占位——
因为 `assert len(robots) == len(grippers)` 要求两段对齐，AM2Pro 无独立夹爪但得放个占位。

## 8. 关键约定

1. **两个映射**：6D 笛卡尔位姿 ↔ 6 关节角（placo IK 迭代收敛）；UMI 夹爪宽度 [0, 0.09] m ↔
   舵机 0-100（线性映射）。
2. **分层原则**：`eval_real.py` / `real_inference_util.py` / `pose_trajectory_interpolator` 全部复用，
   只改了 3 处「接线」。
3. **已知硬件限制**：shoulder_lift / elbow_flex 用 sts3095（弱舵机），折叠姿态下被重力压住无法
   到位（力矩不足，非配置问题）；测试在伸展姿态下进行。

## 9. 使用方式

```bash
# 1. 标定夹爪（lerobot 环境，一次性）
/home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python \
    umi/real_world/am2pro_gripper_calibrate.py --port /dev/ttyACM0

# 2. 把标定结果填进 example/eval_robots_config.yaml

# 3. 运行推理（UMI 环境）
python eval_real.py -rc example/eval_robots_config.yaml --output <out_dir>
```
