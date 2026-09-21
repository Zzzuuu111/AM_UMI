#!/usr/bin/env python3
"""Derive a hand-held V-jaw camera-to-``right_tcp`` candidate from its ROS2 URDF.

The V-jaw pivot calibration finds a reliable *point* on the fixed jaw, but it
uses the fixed-jaw ArUco axes as the TCP axes.  Those axes are not guaranteed
to be the same as the robot URDF's ``right_tcp`` axes.  This tool keeps the
measured file intact and writes a separate, explicitly provisional geometry
whose camera-to-TCP transform is taken from the matching robot URDF.

It is offline only; it neither opens a camera nor connects to a robot.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.common.pose_util import mat_to_pose, pose_to_mat


def origin_matrix(node: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if node is None:
        return matrix
    matrix[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return matrix


def root_transforms(urdf_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    parents: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parents[child.attrib["link"]] = (parent.attrib["link"], origin_matrix(joint.find("origin")))

    cache: dict[str, np.ndarray] = {}

    def solve(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        if link not in parents:
            cache[link] = np.eye(4, dtype=np.float64)
            return cache[link]
        parent, parent_to_link = parents[link]
        cache[link] = solve(parent) @ parent_to_link
        return cache[link]

    for link in list(parents):
        solve(link)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从匹配的 ROS2 V 型夹爪 URDF 导出手持 camera→right_tcp 候选几何；不控制硬件。")
    parser.add_argument("--measured-geometry", required=True,
                        help="已验收的固定爪 Tag 轴向 pivot 几何 JSON（只读，用于生成差异报告）")
    parser.add_argument("--urdf", required=True, help="同型号机器人 V 型夹爪 URDF")
    parser.add_argument("--camera-link", default="right_camera")
    parser.add_argument("--tcp-link", default="right_tcp")
    parser.add_argument("--out", required=True, help="新的候选 JSON；不得已存在")
    args = parser.parse_args()

    measured_path = Path(args.measured_geometry).expanduser().resolve()
    urdf_path = Path(args.urdf).expanduser().resolve()
    output = Path(args.out).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有输出：{output}")
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    if measured.get("schema") != "am_umi_camera_tcp_geometry_v1":
        raise ValueError("measured-geometry 不是预期的 camera-TCP 几何格式")
    measured_matrix = pose_to_mat(np.asarray(measured["pose_cam_tcp"], dtype=np.float64))

    transforms = root_transforms(urdf_path)
    if args.camera_link not in transforms or args.tcp_link not in transforms:
        raise ValueError(f"URDF 中找不到 {args.camera_link!r} 或 {args.tcp_link!r}")
    camera_to_robot_tcp = np.linalg.inv(transforms[args.camera_link]) @ transforms[args.tcp_link]
    measured_to_robot = np.linalg.inv(measured_matrix) @ camera_to_robot_tcp
    delta_rotation_deg = float(np.degrees(
        Rotation.from_matrix(measured_to_robot[:3, :3]).magnitude()))
    delta_translation_mm = float(np.linalg.norm(measured_to_robot[:3, 3]) * 1000.0)

    result = {
        "schema": "am_umi_camera_tcp_geometry_v1",
        "status": "candidate",
        "transform_direction": "camera_to_tcp",
        "pose_cam_tcp": mat_to_pose(camera_to_robot_tcp).tolist(),
        "tcp_definition": f"ROS2 URDF link {args.tcp_link}",
        "tcp_axes_definition": f"ROS2 URDF axes of {args.tcp_link}",
        "translation_method": "matching_ros2_urdf_camera_to_tcp",
        "rotation_method": "matching_ros2_urdf_camera_to_tcp",
        "source": {
            "measured_fixed_jaw_geometry": str(measured_path),
            "urdf": str(urdf_path),
            "camera_link": args.camera_link,
            "tcp_link": args.tcp_link,
            "measured_tcp_to_ros2_right_tcp": mat_to_pose(measured_to_robot).tolist(),
            "measured_tcp_to_ros2_translation_mm": delta_translation_mm,
            "measured_tcp_to_ros2_rotation_deg": delta_rotation_deg,
        },
        "limitations": [
            "候选仅在手持装置与该 ROS2 URDF 的相机安装座、夹爪型号和镜头朝向完全相同时成立。",
            "保留 measured-geometry 作为固定爪 pivot 的原始测量；本文件不覆盖它。",
            "必须经离线 IK 筛选，随后再进行低速实机验证，才可升级为 accepted。",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VJAW_ROS2_TCP_GEOMETRY_CANDIDATE_OK")
    print("output:", output)
    print("camera_to_right_tcp:", np.array2string(mat_to_pose(camera_to_robot_tcp), precision=6))
    print("measured_tcp_to_right_tcp: translation_mm={:.3f}; rotation_deg={:.3f}".format(
        delta_translation_mm, delta_rotation_deg))


if __name__ == "__main__":
    main()
