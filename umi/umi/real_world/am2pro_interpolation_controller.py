"""
AM2Pro Interpolation Controller (client side).

Cartesian control of the 6-DOF AM2Pro arm (AlohaMini2/pro) with IK solved inside
a *separate* process that runs under the lerobot environment (Python 3.12).
This client lives in the UMI environment (Python 3.9); the two communicate over
a Unix domain socket using the protocol in `am2pro_protocol.py`.

Why the split: UMI's training/inference stack requires Python 3.9 / torch 2.1 /
numpy 1.x, while lerobot's motor/kinematics code uses Python 3.12 syntax
(PEP 695) and numpy 2.x. They cannot coexist in one interpreter, so the hardware
driver is isolated in its own process.

The public API here mirrors the original `AM2ProInterpolationController` (and
`FrankaInterpolationController`) so upper-layer code is unchanged:
    schedule_waypoint(pose, target_time)
    schedule_gripper(pos, target_time)
    get_state(k=None) / get_all_state()
    start() / stop() / is_ready / __enter__ / __exit__
"""

import collections
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from typing import Optional

import numpy as np

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
    pack_command,
    recv_msg,
    send_msg,
    unpack_state,
)

# Python interpreter for the server process (must have lerobot + placo).
DEFAULT_SERVER_PYTHON = "/home/zzzjh/anaconda3/envs/lerobot_alohamini/bin/python"

_SERVER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "am2pro_controller_server.py"
)

# State fields that are per-sample scalars (stacked to (N,) in get_all_state).
_SCALAR_KEYS = ("gripper_position", "robot_receive_timestamp", "robot_timestamp")
# State fields that are per-sample arrays (stacked to (N, d) in get_all_state).
_ARRAY_KEYS = ("ActualTCPPose", "ActualQ", "ActualQd")


class AM2ProInterpolationController:
    """Client wrapper that drives the standalone AM2Pro controller server."""

    def __init__(
            self,
            shm_manager=None,          # unused; kept for API compatibility
            robot_usb_port: str = "/dev/ttyACM0",
            frequency: int = 50,
            launch_timeout: float = 10.0,
            verbose: bool = False,
            get_max_k: Optional[int] = None,
            receive_latency: float = 0.0,
            ik_position_weight: float = 1.0,
            ik_orientation_weight: float = 1.0,
            ik_iterations: int = 5,
            wrist_flex_min_deg: Optional[float] = None,
            wrist_flex_max_deg: Optional[float] = None,
            ik_backend: str = "ros2_dh",
            urdf_path: Optional[str] = None,
            flip_joints=None,
            joint_model_candidate: Optional[str] = None,
            gripper_width_min: float = 0.0,
            gripper_width_max: float = 0.09,
            gripper_servo_closed: float = 97.1,
            gripper_servo_open: float = 1.4,
            gripper_width_servo_table=None,
            diagnostic_motor: Optional[str] = None,
            diagnostic_interval_s: float = 0.25,
            position_p_coefficients=None,
            position_p_overrides=None,
            server_python: str = DEFAULT_SERVER_PYTHON,
            server_script: str = _SERVER_SCRIPT,
        ):
        self.robot_usb_port = robot_usb_port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.receive_latency = receive_latency
        self.ik_position_weight = ik_position_weight
        self.ik_orientation_weight = ik_orientation_weight
        self.ik_iterations = ik_iterations
        self.wrist_flex_min_deg = wrist_flex_min_deg
        self.wrist_flex_max_deg = wrist_flex_max_deg
        self.ik_backend = ik_backend
        self.urdf_path = urdf_path
        self.flip_joints = list(flip_joints) if flip_joints else []
        self.joint_model_candidate = joint_model_candidate
        self.gripper_width_min = gripper_width_min
        self.gripper_width_max = gripper_width_max
        self.gripper_servo_closed = gripper_servo_closed
        self.gripper_servo_open = gripper_servo_open
        self.gripper_width_servo_table = list(gripper_width_servo_table or [])
        self.diagnostic_motor = diagnostic_motor
        self.diagnostic_interval_s = diagnostic_interval_s
        self.position_p_coefficients = dict(position_p_coefficients or {})
        self.position_p_overrides = dict(position_p_overrides or {})
        self.server_python = server_python
        self.server_script = server_script

        if get_max_k is None:
            get_max_k = int(frequency * 5)
        self.get_max_k = get_max_k

        # Unique socket path per controller instance.
        self.sock_path = f"/tmp/am2pro_{uuid.uuid4().hex}.sock"

        # Runtime state.
        self._proc = None
        self._conn = None
        self._reader_thread = None
        self._ready_event = threading.Event()
        self._stop_event = threading.Event()
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._error = None
        self._states = collections.deque(maxlen=get_max_k)

    # ======================================================================
    # Lifecycle
    # ======================================================================

    def start(self, wait=True):
        if self._proc is not None:
            return

        if not os.path.isfile(self.server_python):
            raise RuntimeError(
                f"server_python not found: {self.server_python}. "
                "Pass the lerobot-env Python path via `server_python=` or the "
                "config key `robot_python`."
            )

        cmd = [
            self.server_python, self.server_script,
            "--sock", self.sock_path,
            "--port", self.robot_usb_port,
            "--frequency", str(self.frequency),
            "--receive-latency", str(self.receive_latency),
            "--ik-position-weight", str(self.ik_position_weight),
            "--ik-orientation-weight", str(self.ik_orientation_weight),
            "--ik-iterations", str(self.ik_iterations),
            "--wrist-flex-min-deg", str(self.wrist_flex_min_deg)
            if self.wrist_flex_min_deg is not None else "none",
            "--wrist-flex-max-deg", str(self.wrist_flex_max_deg)
            if self.wrist_flex_max_deg is not None else "none",
            "--ik-backend", self.ik_backend,
            "--gripper-width-min", str(self.gripper_width_min),
            "--gripper-width-max", str(self.gripper_width_max),
            "--gripper-servo-closed", str(self.gripper_servo_closed),
            "--gripper-servo-open", str(self.gripper_servo_open),
            "--gripper-width-servo-table", json.dumps(self.gripper_width_servo_table),
        ]
        if self.joint_model_candidate:
            cmd += ["--joint-model-candidate", os.path.abspath(
                os.path.expanduser(self.joint_model_candidate))]
        else:
            cmd += ["--flip-joints", ",".join(self.flip_joints)]
        if self.urdf_path:
            cmd += ["--urdf-path", self.urdf_path]
        if self.diagnostic_motor:
            cmd += ["--diagnostic-motor", self.diagnostic_motor,
                    "--diagnostic-interval-s", str(self.diagnostic_interval_s)]
        if self.position_p_coefficients:
            cmd += ["--position-p-coefficients",
                    json.dumps(self.position_p_coefficients)]
        if self.position_p_overrides:
            cmd += ["--position-p-overrides", json.dumps(self.position_p_overrides)]
        if self.verbose:
            cmd.append("--verbose")

        self._proc = subprocess.Popen(cmd)

        # Connect to the server socket (retry until it binds).
        self._conn = self._connect_with_retry()

        # Reader thread streams state into a local ring buffer.
        self._reader_thread = threading.Thread(
            target=self._reader_loop, name="AM2Pro-reader", daemon=True)
        self._reader_thread.start()

        if wait:
            self.start_wait()

    def _connect_with_retry(self):
        deadline = time.time() + self.launch_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("AM2Pro server process exited before connecting")
            try:
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.connect(self.sock_path)
                return conn
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                time.sleep(0.05)
        raise RuntimeError(
            f"timed out connecting to AM2Pro server socket {self.sock_path}")

    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                msg_type, payload = recv_msg(self._conn)
            except (ConnectionError, OSError):
                break

            if msg_type == MSG_STATE:
                state = unpack_state(payload)
                with self._lock:
                    self._states.append(state)
            elif msg_type == MSG_READY:
                self._ready_event.set()
            elif msg_type == MSG_ERROR:
                self._error = payload.decode("utf-8", "replace")
                self._ready_event.set()
                break

        # If the socket closed without READY (e.g. server crashed), unblock start_wait.
        self._ready_event.set()

    def start_wait(self):
        self._ready_event.wait(self.launch_timeout)
        if self._error is not None:
            raise RuntimeError(f"AM2Pro controller init failed: {self._error}")
        if not self._ready_event.is_set():
            raise RuntimeError("AM2Pro controller timed out during init")
        if self._proc.poll() is not None:
            raise RuntimeError("AM2Pro controller process exited during init")

    def stop(self, wait=True):
        self._stop_event.set()
        if self._conn is not None:
            try:
                with self._send_lock:
                    send_msg(self._conn, MSG_COMMAND,
                             pack_command(CMD_STOP, np.zeros(6), 0.0, 0.0, -1.0))
            except OSError:
                pass
        if wait:
            self.stop_wait()

    def stop_wait(self):
        if self._proc is not None:
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
            self._proc = None
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass

    @property
    def is_ready(self):
        return self._ready_event.is_set()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ======================================================================
    # Command API
    # ======================================================================

    def _send_command(self, cmd, target_pose, target_time, duration, target_gripper):
        if self._conn is None:
            raise RuntimeError("controller not started")
        payload = pack_command(cmd, target_pose, target_time, duration, target_gripper)
        with self._send_lock:
            send_msg(self._conn, MSG_COMMAND, payload)

    def schedule_waypoint(self, pose, target_time):
        """Schedule a 6D Cartesian waypoint at *target_time* (global time)."""
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)
        self._send_command(CMD_SCHEDULE_WAYPOINT, pose, target_time, 0.0, -1.0)

    def schedule_gripper(self, pos, target_time):
        """Schedule a gripper-width command at *target_time* (global time).

        Args:
            pos: UMI gripper width in meters (0 = closed, ~0.09 = open).
        """
        self._send_command(CMD_SCHEDULE_GRIPPER, np.zeros(6), target_time, 0.0, float(pos))

    def servoL(self, pose, duration=0.1):
        """Smoothly drive to *pose* over *duration* seconds."""
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)
        self._send_command(CMD_SERVOL, pose, 0.0, duration, -1.0)

    def servoJ(self, joints, duration=1.0):
        """Smoothly drive the six arm joints to model-space degrees.

        This is intentionally separate from :meth:`servoL`: it is useful for
        deterministic pre-positioning before a Cartesian replay, where asking
        IK to "bend the wrist" would otherwise allow it to move other joints.
        The next Cartesian command switches the server back to Cartesian IK.
        """
        joints = np.array(joints, dtype=np.float64)
        assert joints.shape == (6,)
        if duration <= 0:
            raise ValueError("servoJ duration must be positive")
        self._send_command(CMD_SERVOJ, joints, 0.0, duration, -1.0)

    def servoJStream(self, joints):
        """Set the latest model-space joint target of a 50 Hz joint stream.

        The server holds this target until a newer sample arrives.  Callers
        must provide their own bounded interpolation and tracking gate; this
        method intentionally performs no Cartesian IK and no gripper action.
        """
        joints = np.array(joints, dtype=np.float64)
        assert joints.shape == (6,)
        self._send_command(CMD_SERVOJ_STREAM, joints, 0.0, 0.0, -1.0)

    # ======================================================================
    # State API
    # ======================================================================

    def _snapshot(self):
        with self._lock:
            return list(self._states)

    @staticmethod
    def _stack(states):
        result = {}
        for key in _ARRAY_KEYS:
            result[key] = np.stack([s[key] for s in states])
        for key in _SCALAR_KEYS:
            result[key] = np.array([s[key] for s in states], dtype=np.float64)
        # AM2Pro has no separate commanded-pose stream (the server interpolates
        # internally). Expose the actual pose as the "target" so eval_real's
        # teleop loop seeds from the current pose instead of KeyErroring.
        result["TargetTCPPose"] = result["ActualTCPPose"].copy()
        return result

    def get_state(self, k=None, out=None):
        """Latest state (k=None) or the last k states stacked."""
        states = self._snapshot()
        if not states:
            return None
        if k is None:
            latest = dict(states[-1])
            latest["TargetTCPPose"] = latest["ActualTCPPose"].copy()
            return latest
        k = min(k, len(states))
        return self._stack(states[-k:])

    def get_all_state(self):
        """All buffered states stacked into a dict of numpy arrays.

        Returns empty arrays (matching SharedMemoryRingBuffer.get_all()) if no
        sample has arrived yet, so downstream obs alignment never sees None.
        """
        states = self._snapshot()
        if not states:
            result = {}
            for key in _ARRAY_KEYS:
                result[key] = np.zeros((0, 7), dtype=np.float64) if key != "ActualTCPPose" \
                    else np.zeros((0, 6), dtype=np.float64)
            for key in _SCALAR_KEYS:
                result[key] = np.zeros((0,), dtype=np.float64)
            return result
        return self._stack(states)
