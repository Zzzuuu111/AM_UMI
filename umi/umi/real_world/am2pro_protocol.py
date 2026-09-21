"""
Wire protocol for the AM2Pro controller.

The AM2Pro controller is split across two Python environments (a UMI main
process on Python 3.9 and a hardware-control server on Python 3.12), so it
cannot use in-process shared memory. Instead the two sides talk over a single
Unix domain socket using a simple length-prefixed framing:

    [1 byte: message type][4 bytes: payload length, big-endian][payload]

Message types
    MSG_STATE   (server -> client): 23 float64 (see `pack_state`)
    MSG_COMMAND (client -> server): int32 cmd + 9 float64 (see `pack_command`)
    MSG_READY   (server -> client): empty payload, sent once init completes
    MSG_ERROR   (server -> client): UTF-8 error text, sent before exit on failure

This module has NO third-party imports (only stdlib + numpy) so it can be
loaded identically in both environments.
"""

import struct

import numpy as np

# ---- message types ----
MSG_STATE = 1
MSG_COMMAND = 2
MSG_READY = 3
MSG_ERROR = 4

# ---- command enum (must match between client and server) ----
CMD_STOP = 0
CMD_SERVOL = 1
CMD_SCHEDULE_WAYPOINT = 2
CMD_SCHEDULE_GRIPPER = 3
CMD_SERVOJ = 4
# Direct joint target for an externally interpolated 50 Hz stream.  Unlike
# CMD_SERVOJ, this does not restart a finite-duration smoothstep on every
# sample; the latest target is held by the controller loop until replaced.
CMD_SERVOJ_STREAM = 5

# ---- state payload layout (all float64, 23 total) ----
#   ActualTCPPose (6), ActualQ (7), ActualQd (7),
#   gripper_position (1), robot_receive_timestamp (1), robot_timestamp (1)
STATE_N_FLOATS = 23
_STATE_STRUCT = struct.Struct(">" + "d" * STATE_N_FLOATS)

# ---- command payload layout ----
#   int32 cmd; float64 target_pose[6], target_time, duration, target_gripper
#   `target_gripper` carries the UMI gripper width in meters (-1.0 = "no change").
COMMAND_N_FLOATS = 9
_COMMAND_STRUCT = struct.Struct(">i" + "d" * COMMAND_N_FLOATS)


def pack_state(pose, q, qd, gripper_position, recv_ts, robot_ts):
    """Serialize one state sample to a fixed 23-float64 payload."""
    pose = np.asarray(pose, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    qd = np.asarray(qd, dtype=np.float64).ravel()
    assert pose.size == 6 and q.size == 7 and qd.size == 7
    vals = np.concatenate([
        pose, q, qd,
        np.array([gripper_position, recv_ts, robot_ts], dtype=np.float64),
    ])
    assert vals.size == STATE_N_FLOATS
    return _STATE_STRUCT.pack(*vals)


def unpack_state(payload):
    """Deserialize a state payload into a dict of scalars / numpy arrays."""
    vals = _STATE_STRUCT.unpack(payload)
    return {
        "ActualTCPPose": np.array(vals[0:6], dtype=np.float64),
        "ActualQ": np.array(vals[6:13], dtype=np.float64),
        "ActualQd": np.array(vals[13:20], dtype=np.float64),
        "gripper_position": float(vals[20]),
        "robot_receive_timestamp": float(vals[21]),
        "robot_timestamp": float(vals[22]),
    }


def pack_command(cmd, target_pose, target_time, duration, target_gripper):
    """Serialize a command to a fixed 9-float64 payload."""
    pose = np.asarray(target_pose, dtype=np.float64).ravel()
    assert pose.size == 6
    return _COMMAND_STRUCT.pack(
        int(cmd), *pose, float(target_time), float(duration), float(target_gripper)
    )


def unpack_command(payload):
    """Deserialize a command payload into a dict."""
    vals = _COMMAND_STRUCT.unpack(payload)
    return {
        "cmd": vals[0],
        "target_pose": np.array(vals[1:7], dtype=np.float64),
        "target_time": float(vals[7]),
        "duration": float(vals[8]),
        "target_gripper": float(vals[9]),
    }


# ---- framing helpers ----

def send_msg(sock, msg_type, payload=b""):
    header = struct.pack(">BI", msg_type, len(payload))
    sock.sendall(header + payload)


def recv_exact(sock, n):
    """Read exactly n bytes, raising ConnectionError if the peer closes early."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    """Blocking read of one framed message."""
    header = recv_exact(sock, 5)
    msg_type, length = struct.unpack(">BI", header)
    return msg_type, recv_exact(sock, length)


def try_recv_messages(sock, buf):
    """Non-blocking drain of all available complete messages.

    Appends whatever bytes are available to `buf`, then extracts and returns
    every complete (msg_type, payload) frame. Partial frames stay in `buf`.
    Returns (messages, alive) where alive=False means the peer closed the socket.
    """
    messages = []
    try:
        while True:
            data = sock.recv(4096)
            if not data:
                return messages, False
            buf.extend(data)
            if len(data) < 4096:
                break
    except BlockingIOError:
        pass

    while len(buf) >= 5:
        msg_type, length = struct.unpack(">BI", buf[:5])
        if len(buf) < 5 + length:
            break
        payload = bytes(buf[5:5 + length])
        del buf[:5 + length]
        messages.append((msg_type, payload))

    return messages, True
