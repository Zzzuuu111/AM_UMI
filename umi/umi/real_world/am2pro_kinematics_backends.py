"""
Pluggable IK/FK backends for the AM2Pro controller server.

Both backends implement the SAME contract, so the server control loop is
backend-agnostic:

    backend.forward(q_deg) -> 4x4 pose of the requested TCP in right_Base (degrees in)
    backend.inverse_kinematics(q_deg, T, position_weight, orientation_weight)
        -> q_new_deg (single velocity-level step; server iterates 5x per tick)

Backends:
    - "placo":   lerobot RobotKinematics + placo (current default)
    - "ros2_dh": alohamini_ros2 AlohaMiniArmKinematics (standard-DH + DLS,
                 pure numpy, joint limits from the URDF)

Cross-checked offline: both backends must agree on FK at the requested
reference point for the same joint vector.
"""
from __future__ import annotations

import math
import os
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

# right_Base -> right_Fixed_Jaw offset in the DH table (wrist_roll row).
# The new Follow UMI gripper's nominal working point is 236 mm further along
# the same +Z direction.  This is intentionally a single named constant so a
# later visual TCP-pivot calibration changes one value, not the arm model.
FIXED_JAW_Z_OFFSET = 0.060992744083
FOLLOW_UMI_TCP_Z_FROM_FIXED_JAW = 0.236


class PlacoBackend:
    def __init__(self, urdf_path, joint_names, target_frame_name):
        from lerobot.model import RobotKinematics  # lazy: umi env may lack lerobot

        self.kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame_name,
            joint_names=list(joint_names),
        )

    def forward_kinematics(self, q_deg):
        return self.kin.forward_kinematics(np.asarray(q_deg, dtype=float))

    forward = forward_kinematics  # alias

    def inverse_kinematics(self, q_deg, T_target, position_weight=1.0, orientation_weight=1.0):
        return self.kin.inverse_kinematics(
            np.asarray(q_deg, dtype=float),
            np.asarray(T_target, dtype=float),
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )


class Ros2DhBackend:
    """alohamini_ros2 standard-DH FK + damped least-squares IK (pure numpy)."""

    def __init__(self, urdf_path, joint_names, target_frame_name,
                 ros2_ws_path=None):
        ros2_ws_path = ros2_ws_path or "/home/zzzjh/alohamini_ros2"
        share = os.path.join(ros2_ws_path, "src", "alohamini_description")

        # stub ament_index_python (only used by from_description)
        if "ament_index_python" not in sys.modules:
            pkg = types.ModuleType("ament_index_python.packages")
            pkg.get_package_share_directory = lambda name, _share=share: _share
            am = types.ModuleType("ament_index_python")
            am.packages = pkg
            sys.modules["ament_index_python"] = am
            sys.modules["ament_index_python.packages"] = pkg

        kin_dir = os.path.join(
            ros2_ws_path, "src", "alohamini_joycon_teleop", "alohamini_joycon_teleop")
        if kin_dir not in sys.path:
            sys.path.insert(0, kin_dir)
        import kinematics as _kin

        # joint limits from the same URDF placo reads
        # NOTE: URDF <limit> values are natively in RADIANS
        root = ET.parse(urdf_path).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        limits = []
        for name in joint_names:
            lim = joints[name].find("limit")
            limits.append((float(lim.get("lower")), float(lim.get("upper"))))

        self.ik = _kin.AlohaMiniArmKinematics(
            dh_path=os.path.join(share, "config", "kinematics", "right_arm_kinematics.yaml"),
            tool_path=os.path.join(share, "config", "kinematics", "kinematics.yaml"),
            joint_limits=limits,
        )
        # The ROS2 class defaults to its own legacy jaw-tip tool point.  Select
        # exactly the frame requested by the AM_UMI URDF instead.  Keeping this
        # explicit prevents a silent mismatch between the numpy and placo FK.
        self.ik.tool_transform = np.eye(4)
        if target_frame_name == "right_Fixed_Jaw":
            self.ik.tool_transform[2, 3] = FIXED_JAW_Z_OFFSET
        elif target_frame_name == "right_tcp":
            self.ik.tool_transform[2, 3] = (
                FIXED_JAW_Z_OFFSET + FOLLOW_UMI_TCP_Z_FROM_FIXED_JAW
            )
        else:
            raise ValueError(
                "ros2_dh supports right_Fixed_Jaw or right_tcp; "
                f"got {target_frame_name!r}. Use the placo backend for another frame."
            )
        self.joint_names = list(joint_names)

    def forward_kinematics(self, q_deg):
        return self.ik.forward(np.deg2rad(np.asarray(q_deg, dtype=float)))

    forward = forward_kinematics  # alias

    def inverse_kinematics(self, q_deg, T_target, position_weight=1.0, orientation_weight=1.0):
        q_new, _info = self.ik.step(
            np.deg2rad(np.asarray(q_deg, dtype=float)),
            np.asarray(T_target, dtype=float),
            dt=0.02,
            position_gain=40.0 * float(position_weight),
            orientation_gain=20.0 * float(orientation_weight),
            # defaults (1.5 rad/s) are too slow for the 5-iteration servo loop;
            # 3 rad/s is still well below the sts3250 no-load speed (~4.5 rad/s)
            max_joint_velocity=3.0,
            max_joint_step=0.2,
        )
        return np.rad2deg(q_new)


def create_kinematics_backend(name, urdf_path, joint_names, target_frame_name,
                              ros2_ws_path=None):
    if name == "placo":
        return PlacoBackend(urdf_path, joint_names, target_frame_name)
    if name in ("ros2_dh", "ros2-dh"):
        return Ros2DhBackend(urdf_path, joint_names, target_frame_name,
                             ros2_ws_path=ros2_ws_path)
    raise ValueError(f"Unknown IK backend '{name}' (choose: placo, ros2_dh)")
