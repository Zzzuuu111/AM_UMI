"""
Standalone AM2Pro controller server.

Runs under the lerobot environment (Python 3.12) as a separate process from the
UMI main process (Python 3.9). The two talk over a Unix domain socket using the
protocol in `am2pro_protocol.py`.

The control loop here is identical in spirit to the original
`AM2ProInterpolationController.run()`: interpolate a 6D Cartesian pose, solve IK,
write goal positions to the Feetech servos, read back actual positions, solve FK,
and stream the resulting state to the client.

IMPORTANT import discipline: this file must ONLY import
  - stdlib / numpy (available everywhere)
  - umi.common.pose_trajectory_interpolator / pose_util / precise_sleep
    (pure numpy/scipy/time)
  - lerobot motors + model (available only in the lerobot env)
It must NOT import torch, diffusion_policy, or umi.shared_memory.
"""

import argparse
import json
import os
import socket
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.pose_util import mat_to_pose, pose_to_mat
from umi.common.precise_sleep import precise_wait
from umi.real_world.am2pro_protocol import (
    CMD_SCHEDULE_GRIPPER,
    CMD_SCHEDULE_WAYPOINT,
    CMD_SERVOJ,
    CMD_SERVOJ_STREAM,
    CMD_SERVOL,
    CMD_STOP,
    MSG_COMMAND,
    MSG_ERROR,
    MSG_READY,
    MSG_STATE,
    pack_state,
    send_msg,
    try_recv_messages,
    unpack_command,
)

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.model import RobotKinematics
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend
from umi.real_world.am2pro_joint_mapping import (
    encoder_to_model,
    load_candidate,
    model_to_encoder,
)

# ---- kinematics / hardware configuration ----

# Mesh-free URDF: identical kinematic chain to the CAD URDF, but without the
# Git-LFS mesh references that placo cannot parse (irrelevant to FK/IK anyway).
URDF_PATH = os.path.join(REPO_ROOT, "alohamini2pro_right_arm_kinematics.urdf")

URDF_ARM_JOINT_NAMES = [
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_flex",
    "right_wrist_yaw_joint",
    "right_wrist_roll",
]

# Motor names corresponding to each URDF joint (same order).
MOTOR_ARM_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
]

# Follow UMI long gripper: midpoint between closed fingertips.  The mounting
# frame and camera extrinsic remain right_Fixed_Jaw; see right_tcp_joint in URDF.
TCP_FRAME_NAME = "right_tcp"
GRIPPER_MOTOR_NAME = "gripper"

# am-follower-6dof-hd profile (detected hardware): shoulder_pan/wrist*/gripper
# are sts3250, shoulder_lift/elbow_flex are sts3095.
_MOTOR_SPEC = [
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
]


def map_width_to_servo(width, args):
    """UMI gripper width (meters) -> servo position (0-100)."""
    w = np.clip(float(width), args.gripper_width_min, args.gripper_width_max)
    if args.gripper_width_servo_table:
        table = np.asarray(args.gripper_width_servo_table, dtype=float)
        return float(np.interp(w, table[:, 0], table[:, 1]))
    denom = args.gripper_width_max - args.gripper_width_min
    if denom <= 0:
        return args.gripper_servo_closed
    ratio = (w - args.gripper_width_min) / denom
    return float(args.gripper_servo_closed + ratio * (args.gripper_servo_open - args.gripper_servo_closed))


def map_servo_to_width(servo, args):
    """Servo position (0-100) -> UMI gripper width (meters)."""
    if args.gripper_width_servo_table:
        table = np.asarray(args.gripper_width_servo_table, dtype=float)
        order = np.argsort(table[:, 1])
        return float(np.clip(np.interp(float(servo), table[order, 1], table[order, 0]),
                             args.gripper_width_min, args.gripper_width_max))
    denom = args.gripper_servo_open - args.gripper_servo_closed
    if abs(denom) < 1e-9:
        return args.gripper_width_min
    ratio = (float(servo) - args.gripper_servo_closed) / denom
    width = args.gripper_width_min + ratio * (args.gripper_width_max - args.gripper_width_min)
    return float(np.clip(width, args.gripper_width_min, args.gripper_width_max))


def connect_hardware(args):
    """Connect the bus, load calibration, configure motors, init kinematics.

    Returns (bus, kinematics, arm_joint_pos, target_gripper_servo,
    original_p_coefficients).
    Raises on any failure (caller reports it to the client via MSG_ERROR).
    """
    motors = {
        name: Motor(motor_id, model, norm_mode)
        for name, motor_id, model, norm_mode in _MOTOR_SPEC
    }
    bus = FeetechMotorsBus(port=args.port, motors=motors)

    kinematics = create_kinematics_backend(
        args.ik_backend,
        urdf_path=args.urdf_path,
        joint_names=URDF_ARM_JOINT_NAMES,
        target_frame_name=TCP_FRAME_NAME,
    )

    bus.connect()

    # Load calibration from motor EEPROM (homing offsets + range limits).
    # Required for raw encoder <-> normalized (degrees / 0-100) conversion.
    bus.calibration = bus.read_calibration()

    # A diagnostic may temporarily override one or more P gains.  P is an
    # EEPROM register on these servos, so remember the prior value and restore
    # it when this server exits.  Normal replay never supplies an override.
    original_p_coefficients = {}
    for motor in args.position_p_overrides:
        original_p_coefficients[motor] = int(bus.read("P_Coefficient", motor))

    # Configure: enable torque, POSITION mode, tune PID (mirrors lerobot SOFollower).
    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # The replacement gripper has a longer/heavier moment arm.  Give
            # only the wrist-flex joint a modestly firmer position loop;
            # retain the existing conservative setting for every other joint.
            default_p = 80 if motor == "wrist_flex" else 24
            configured_p = args.position_p_coefficients.get(motor, default_p)
            p_coefficient = args.position_p_overrides.get(motor, configured_p)
            bus.write("P_Coefficient", motor, p_coefficient)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
            if motor == GRIPPER_MOTOR_NAME:
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)

    all_motor_names = list(motors.keys())
    motor_positions = bus.sync_read("Present_Position", all_motor_names)
    arm_joint_pos = np.array(
        [float(motor_positions[n]) for n in MOTOR_ARM_NAMES], dtype=np.float64
    )
    target_gripper_servo = float(motor_positions[GRIPPER_MOTOR_NAME])

    # Hold current position immediately after torque is enabled so the arm does
    # not jerk toward a stale Goal_Position from a prior session.
    bus.sync_write("Goal_Position", {
        name: float(arm_joint_pos[i]) for i, name in enumerate(MOTOR_ARM_NAMES)
    } | {GRIPPER_MOTOR_NAME: target_gripper_servo})

    if original_p_coefficients:
        print("[am2pro-server] temporary P override:",
              {name: args.position_p_overrides[name]
               for name in original_p_coefficients},
              "; will restore previous EEPROM P on exit", flush=True)

    return (bus, kinematics, arm_joint_pos, target_gripper_servo,
            original_p_coefficients)


def restore_p_coefficients(bus, original_p_coefficients):
    """Restore temporary diagnostic P settings before disconnecting the bus."""
    if not original_p_coefficients:
        return
    with bus.torque_disabled():
        for motor, value in original_p_coefficients.items():
            bus.write("P_Coefficient", motor, int(value))
    print("[am2pro-server] restored previous P_Coefficient values:",
          original_p_coefficients, flush=True)


def run_loop(conn, bus, kinematics, arm_joint_pos, target_gripper_servo, args):
    dt = 1.0 / args.frequency
    all_motor_names = [name for name, _, _, _ in _MOTOR_SPEC]

    def to_model(q):
        return encoder_to_model(q, args.joint_mapping)

    def to_hardware(q):
        return model_to_encoder(q, args.joint_mapping)

    curr_pose = mat_to_pose(kinematics.forward_kinematics(to_model(arm_joint_pos)))
    curr_t = time.monotonic()
    last_waypoint_time = curr_t
    pose_interp = PoseTrajectoryInterpolator(times=[curr_t], poses=[curr_pose])
    # A temporary joint-space move is used only for deterministic setup (for
    # example setting wrist_flex before replay).  A subsequent Cartesian
    # command clears this mode and resumes ordinary IK control.
    joint_move = None
    # A streamed joint target is distinct from joint_move: it is already
    # interpolated client-side and must not restart a smoothstep every tick.
    joint_stream_target = None

    t_start = time.monotonic()
    prev_arm_joint_pos = to_model(arm_joint_pos)
    prev_gripper_servo = target_gripper_servo
    iter_idx = 0
    keep_running = True
    cmd_buf = bytearray()
    next_diagnostic_time = t_start

    conn.setblocking(False)

    def _bus_retry(fn, *fn_args, tries=3, **fn_kwargs):
        """Retry bus operations on transient serial glitches (e.g. servo load
        spikes during motion cause a dropped status packet)."""
        last_exc = None
        for attempt in range(tries):
            try:
                return fn(*fn_args, **fn_kwargs)
            except ConnectionError as e:
                last_exc = e
                print(f"[am2pro-server] bus op failed (attempt {attempt + 1}/{tries}): {e}",
                      file=sys.stderr, flush=True)
                time.sleep(0.5)
        raise last_exc

    while keep_running:
        t_now = time.monotonic()

        # 1. Either execute a short direct joint-space setup move, or
        # interpolate the ordinary Cartesian command and solve IK.
        q_current = to_model(arm_joint_pos)
        if joint_stream_target is not None:
            q_target = joint_stream_target.copy()
        elif joint_move is not None:
            frac = np.clip(
                (t_now - joint_move['start_time']) /
                max(joint_move['end_time'] - joint_move['start_time'], 1e-9),
                0.0, 1.0)
            # Smoothstep avoids a sharp velocity change at either end.
            frac = frac * frac * (3.0 - 2.0 * frac)
            q_target = ((1.0 - frac) * joint_move['start_q'] +
                        frac * joint_move['target_q'])
        else:
            pose_command = pose_interp(t_now)
            t_des = pose_to_mat(pose_command)
            # placo's solve() is a single velocity-level QP step, so a single
            # call does not converge; iterate it, seeding with current joints.
            q_target = q_current
            for _ in range(args.ik_iterations):
                q_target = kinematics.inverse_kinematics(
                    q_target,
                    t_des,
                    position_weight=args.ik_position_weight,
                    orientation_weight=args.ik_orientation_weight,
                )
                # wrist_flex near zero makes the yaw and roll axes nearly
                # collinear.  For the deployment camera's positive-flex
                # starting branch, do not let IK enter that wrist singularity.
                if args.wrist_flex_min_deg is not None:
                    q_target[3] = max(q_target[3], args.wrist_flex_min_deg)
                if args.wrist_flex_max_deg is not None:
                    q_target[3] = min(q_target[3], args.wrist_flex_max_deg)

        # 4. write model-space targets back in calibrated encoder convention.
        q_hw = to_hardware(q_target)
        goals = {name: float(q_hw[i]) for i, name in enumerate(MOTOR_ARM_NAMES)}
        goals[GRIPPER_MOTOR_NAME] = float(target_gripper_servo)
        _bus_retry(bus.sync_write, "Goal_Position", goals)

        # 5. read actual positions
        motor_positions = _bus_retry(
            bus.sync_read, "Present_Position", all_motor_names)
        arm_joint_pos = np.array(
            [float(motor_positions[n]) for n in MOTOR_ARM_NAMES], dtype=np.float64
        )
        gripper_servo = float(motor_positions[GRIPPER_MOTOR_NAME])
        arm_joint_pos_model = to_model(arm_joint_pos)

        # Optional read-only live servo telemetry. Values are raw Feetech
        # register units; ordinary controller state and commands are unchanged.
        if args.diagnostic_motor and t_now >= next_diagnostic_time:
            name = args.diagnostic_motor
            try:
                current = _bus_retry(bus.read, "Present_Current", name)
                load = _bus_retry(bus.read, "Present_Load", name)
                temperature = _bus_retry(bus.read, "Present_Temperature", name)
                print(f"[am2pro-server] telemetry {name}: "
                      f"current={current} load={load} temp_c={temperature}",
                      flush=True)
            except Exception as exc:  # diagnostics must never stop control
                print(f"[am2pro-server] telemetry read failed for {name}: {exc}",
                      file=sys.stderr, flush=True)
            next_diagnostic_time = t_now + args.diagnostic_interval_s

        # 6. FK -> Cartesian pose
        actual_pose = mat_to_pose(kinematics.forward_kinematics(arm_joint_pos_model))

        # 7. approximate joint velocities
        actual_qd = np.zeros(7, dtype=np.float64)
        actual_qd[:6] = (arm_joint_pos_model - prev_arm_joint_pos) / dt
        actual_qd[6] = (gripper_servo - prev_gripper_servo) / dt
        prev_arm_joint_pos = arm_joint_pos_model
        prev_gripper_servo = gripper_servo

        # 8. stream state
        actual_q = np.zeros(7, dtype=np.float64)
        actual_q[:6] = arm_joint_pos_model
        actual_q[6] = gripper_servo
        t_recv = time.time()
        payload = pack_state(
            actual_pose,
            actual_q,
            actual_qd,
            map_servo_to_width(gripper_servo, args),
            t_recv,
            t_recv - args.receive_latency,
        )
        send_msg(conn, MSG_STATE, payload)

        # 9. drain and process commands (non-blocking)
        messages, alive = try_recv_messages(conn, cmd_buf)
        if not alive:
            keep_running = False
            break
        for msg_type, payload in messages:
            if msg_type != MSG_COMMAND:
                continue
            cmd = unpack_command(payload)
            code = cmd["cmd"]

            if code == CMD_STOP:
                keep_running = False
                break
            elif code == CMD_SERVOL:
                # Return from any direct setup move to Cartesian control,
                # seeded from the actual pose reached by that move.
                joint_move = None
                joint_stream_target = None
                actual_now = mat_to_pose(
                    kinematics.forward_kinematics(to_model(arm_joint_pos)))
                pose_interp = PoseTrajectoryInterpolator(
                    times=[t_now], poses=[actual_now])
                # duration is a relative time (seconds), not an absolute timestamp
                curr_time = t_now + dt
                t_insert = curr_time + cmd["duration"]
                pose_interp = pose_interp.drive_to_waypoint(
                    pose=cmd["target_pose"],
                    time=t_insert,
                    curr_time=curr_time,
                )
                last_waypoint_time = t_insert
                if cmd["target_gripper"] >= 0:
                    target_gripper_servo = map_width_to_servo(cmd["target_gripper"], args)
            elif code == CMD_SCHEDULE_WAYPOINT:
                joint_move = None
                joint_stream_target = None
                actual_now = mat_to_pose(
                    kinematics.forward_kinematics(to_model(arm_joint_pos)))
                pose_interp = PoseTrajectoryInterpolator(
                    times=[t_now], poses=[actual_now])
                target_time = time.monotonic() - time.time() + cmd["target_time"]
                pose_interp = pose_interp.schedule_waypoint(
                    pose=cmd["target_pose"],
                    time=target_time,
                    curr_time=t_now + dt,
                    last_waypoint_time=last_waypoint_time,
                )
                last_waypoint_time = target_time
                if cmd["target_gripper"] >= 0:
                    target_gripper_servo = map_width_to_servo(cmd["target_gripper"], args)
            elif code == CMD_SCHEDULE_GRIPPER:
                if cmd["target_gripper"] >= 0:
                    target_gripper_servo = map_width_to_servo(cmd["target_gripper"], args)
            elif code == CMD_SERVOJ:
                joint_stream_target = None
                duration = max(float(cmd["duration"]), dt)
                joint_move = {
                    'start_q': to_model(arm_joint_pos),
                    'target_q': np.asarray(cmd["target_pose"], dtype=np.float64),
                    'start_time': t_now + dt,
                    'end_time': t_now + dt + duration,
                }
            elif code == CMD_SERVOJ_STREAM:
                joint_move = None
                joint_stream_target = np.asarray(cmd["target_pose"], dtype=np.float64)

        # 10. regulate control frequency
        precise_wait(t_start + (iter_idx + 1) * dt, time_func=time.monotonic)
        iter_idx += 1

        if args.verbose and iter_idx % args.frequency == 0:
            print(f"[am2pro-server] {iter_idx / args.frequency:.0f}s, "
                  f"actual ~{1.0 / max(time.monotonic() - t_now, 1e-9):.1f} Hz",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description="AM2Pro Cartesian controller server")
    parser.add_argument("--sock", required=True, help="Unix domain socket path")
    parser.add_argument("--port", default="/dev/ttyACM0", help="USB serial port")
    parser.add_argument("--frequency", type=int, default=50)
    parser.add_argument("--receive-latency", type=float, default=0.0)
    parser.add_argument("--ik-position-weight", type=float, default=1.0)
    parser.add_argument("--ik-orientation-weight", type=float, default=1.0)
    parser.add_argument("--ik-iterations", type=int, default=5)
    parser.add_argument("--wrist-flex-min-deg", default="none",
                        help="Optional positive lower bound for wrist_flex in model "
                             "degrees; keeps replay out of the wrist singularity.")
    parser.add_argument("--wrist-flex-max-deg", default="none",
                        help="Optional upper bound for wrist_flex in model degrees; "
                             "keeps replay away from the positive joint limit.")
    parser.add_argument("--ik-backend", default="ros2_dh",
                        help="IK/FK backend: 'ros2_dh' (default) or 'placo'")
    parser.add_argument("--urdf-path", default=URDF_PATH,
                        help="URDF used for FK/IK. Defaults to the legacy long-gripper model.")
    parser.add_argument("--flip-joints", default="",
                        help="Comma-separated motor names whose physical direction "
                             "is reversed vs the model (e.g. wrist_roll)")
    parser.add_argument("--joint-model-candidate", default=None,
                        help="validated encoder→URDF candidate JSON; overrides legacy --flip-joints")
    parser.add_argument("--gripper-width-min", type=float, default=0.0)
    parser.add_argument("--gripper-width-max", type=float, default=0.09)
    parser.add_argument("--gripper-servo-closed", type=float, default=100.0)
    parser.add_argument("--gripper-servo-open", type=float, default=0.0)
    parser.add_argument("--gripper-width-servo-table", default="[]",
                        help="JSON [[width_m, servo_0_100], ...]；为空时沿用端点线性映射")
    parser.add_argument("--diagnostic-motor", choices=MOTOR_ARM_NAMES + [GRIPPER_MOTOR_NAME],
                        default=None, help="Optional read-only live servo telemetry.")
    parser.add_argument("--diagnostic-interval-s", type=float, default=0.25)
    parser.add_argument("--position-p-overrides", default="{}",
                        help="JSON object of temporary motor P overrides; values are "
                             "restored when the controller exits, for diagnostics only")
    parser.add_argument("--position-p-coefficients", default="{}",
                        help="JSON object of persistent software-default arm P gains")
    parser.add_argument("--accept-timeout", type=float, default=30.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    def parse_p_mapping(raw_json, option_name):
        try:
            raw_mapping = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--{option_name} must be JSON: {exc}")
        if not isinstance(raw_mapping, dict):
            parser.error(f"--{option_name} must be a JSON object")
        parsed = {}
        for motor, value in raw_mapping.items():
            if motor not in MOTOR_ARM_NAMES:
                parser.error(f"--{option_name} only supports arm motors, got: {motor}")
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 254:
                parser.error(f"P for {motor} in --{option_name} must be an integer in [0, 254]")
            parsed[motor] = value
        return parsed

    args.position_p_overrides = parse_p_mapping(
        args.position_p_overrides, "position-p-overrides")
    args.position_p_coefficients = parse_p_mapping(
        args.position_p_coefficients, "position-p-coefficients")

    try:
        table = json.loads(args.gripper_width_servo_table)
    except json.JSONDecodeError as exc:
        parser.error(f"--gripper-width-servo-table must be JSON: {exc}")
    if not isinstance(table, list):
        parser.error("--gripper-width-servo-table must be a JSON list")
    parsed_table = []
    for index, point in enumerate(table):
        if not isinstance(point, list) or len(point) != 2:
            parser.error(f"gripper table point {index} must be [width_m, servo_0_100]")
        width, servo = point
        if (isinstance(width, bool) or isinstance(servo, bool) or
                not isinstance(width, (int, float)) or not isinstance(servo, (int, float)) or
                not np.isfinite(width) or not np.isfinite(servo) or width < 0 or not 0 <= servo <= 100):
            parser.error(f"invalid gripper table point {index}")
        parsed_table.append((float(width), float(servo)))
    if parsed_table:
        if len(parsed_table) < 2:
            parser.error("gripper table needs at least two points")
        widths = np.asarray([point[0] for point in parsed_table])
        servos = np.asarray([point[1] for point in parsed_table])
        if np.any(np.diff(widths) <= 0):
            parser.error("gripper table widths must be strictly increasing")
        servo_delta = np.diff(servos)
        if not (np.all(servo_delta > 0) or np.all(servo_delta < 0)):
            parser.error("gripper table servo values must be strictly monotonic")
        if (abs(widths[0] - args.gripper_width_min) > 1e-6 or
                abs(widths[-1] - args.gripper_width_max) > 1e-6):
            parser.error("gripper table endpoints must match --gripper-width-min/max")
    args.gripper_width_servo_table = parsed_table
    if args.position_p_coefficients:
        print("[am2pro-server] configured P coefficients:",
              args.position_p_coefficients, flush=True)
    args.urdf_path = os.path.abspath(os.path.expanduser(args.urdf_path))
    if not os.path.isfile(args.urdf_path):
        parser.error(f"--urdf-path does not exist: {args.urdf_path}")
    args.flip_joints = [n.strip() for n in args.flip_joints.split(",") if n.strip()]
    if args.joint_model_candidate:
        if args.flip_joints:
            parser.error("--joint-model-candidate 与 --flip-joints 不能同时使用")
        args.joint_mapping = load_candidate(args.joint_model_candidate)
    else:
        legacy_sign = np.ones(6, dtype=np.float64)
        for index, name in enumerate(MOTOR_ARM_NAMES):
            if name in args.flip_joints:
                legacy_sign[index] = -1.0
        args.joint_mapping = {
            "sign": legacy_sign,
            "offset_deg": np.zeros(6, dtype=np.float64),
            "source": "legacy --flip-joints",
        }
    print("[am2pro-server] joint mapping:", args.joint_mapping["source"],
          "sign=", args.joint_mapping["sign"].astype(int).tolist(),
          "offset_deg=", np.round(args.joint_mapping["offset_deg"], 3).tolist(),
          flush=True)
    for option_name in ('wrist_flex_min_deg', 'wrist_flex_max_deg'):
        raw_value = getattr(args, option_name)
        if raw_value.lower() == "none":
            setattr(args, option_name, None)
            continue
        try:
            value = float(raw_value)
        except ValueError:
            parser.error(
                f"--{option_name.replace('_', '-')} must be a number or 'none'")
        if not 0.0 <= value <= 85.0:
            parser.error(f"--{option_name.replace('_', '-')} must be in [0, 85]")
        setattr(args, option_name, value)
    if (args.wrist_flex_min_deg is not None and
            args.wrist_flex_max_deg is not None and
            args.wrist_flex_min_deg > args.wrist_flex_max_deg):
        parser.error("--wrist-flex-min-deg cannot exceed --wrist-flex-max-deg")
    if args.diagnostic_interval_s <= 0:
        parser.error("--diagnostic-interval-s must be positive")

    # Bind the socket first so the client can connect while we do (slow) init.
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(args.sock)
    except OSError as e:
        print(f"[am2pro-server] failed to bind {args.sock}: {e}", file=sys.stderr, flush=True)
        return 1
    server.listen(1)
    server.settimeout(args.accept_timeout)

    conn = None
    try:
        conn, _ = server.accept()
    except socket.timeout:
        print("[am2pro-server] timeout waiting for client", file=sys.stderr, flush=True)
        server.close()
        return 1

    # Hardware init can fail (port busy, calibration missing, IK load error).
    try:
        (bus, kinematics, arm_joint_pos, target_gripper_servo,
         original_p_coefficients) = connect_hardware(args)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            send_msg(conn, MSG_ERROR, str(e).encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        conn.close()
        server.close()
        return 1

    if args.verbose:
        print(f"[am2pro-server] hardware ready on {args.port}", flush=True)

    send_msg(conn, MSG_READY, b"")

    try:
        run_loop(conn, bus, kinematics, arm_joint_pos, target_gripper_servo, args)
    finally:
        try:
            restore_p_coefficients(bus, original_p_coefficients)
        except Exception as exc:  # noqa: BLE001
            print(f"[am2pro-server] failed to restore temporary P values: {exc}",
                  file=sys.stderr, flush=True)
        try:
            bus.disconnect()
        except Exception:  # noqa: BLE001
            pass
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
        server.close()
        try:
            os.unlink(args.sock)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
