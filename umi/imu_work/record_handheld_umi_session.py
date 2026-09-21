"""Record one hand-held EMEET + JY901B session with a shared host clock.

The preview is live before recording starts.  Click the preview window and
press ``r`` only after the rig is aligned to the intended task start view.
Press ``s`` to stop (the IMU continues for a short tail); ``q`` exits without
creating a recording.  The output deliberately keeps raw IMU samples and
per-frame host timestamps.  Camera-IMU time-latency and extrinsics are applied
later, after their calibrations exist.

Example:
    python imu_work/record_handheld_umi_session.py \
      --output-dir data/handheld_sessions/map_v1 \
      --camera-device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0 \
      --imu-port /dev/ttyUSB0 --imu-baud 460800
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from fractions import Fraction
from multiprocessing import Event, Lock

import av
import cv2
import numpy as np
import serial
import yaml
from scipy.spatial.transform import Rotation


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.record_sync_session import parse_reads_precise_with_device_time  # noqa: E402
from imu_work.uvc_payload_timing import align_session_frame_timestamps  # noqa: E402
from umi.common.view_reference import (  # noqa: E402
    compare_policy_images,
    load_reference,
    preprocess_uvc_bgr,
    write_reference,
)
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    mapping_from_config,
    model_safe_limits_from_config,
)
from imu_work.export_fixed_tag_camera_trajectory import (  # noqa: E402
    load_intrinsics,
    marker_area_px2,
    pose_from_marker,
    robust_average_camera_pose,
)
from scripts.filter_am2pro_replayability import (  # noqa: E402
    URDF_JOINTS,
    URDF_PATH,
    load_joint_limits,
    pose_error,
)
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


# 用于采集「相机—IMU 外参 + 时间偏移」的 90 秒动作脚本。时间从按下 R、
# 真正开始写入视频的那一帧起算。平移提供视觉视差，三个旋转轴则是求解
# 相机与 IMU 固定旋转关系时最重要的激励。
EXTRINSIC_GUIDE = (
    (3.0, "0–3 秒：保持整套手持装置完全静止。"),
    (15.0, "3–15 秒：缓慢左右平移整套装置，幅度约 20–40 cm；Tag 始终留在画面里。"),
    (27.0, "15–27 秒：缓慢上下平移整套装置，幅度约 20–40 cm；不要转动夹爪。"),
    (39.0, "27–39 秒：缓慢前后平移（靠近、远离 Tag/桌面），幅度约 20–40 cm。"),
    (51.0, "39–51 秒：夹爪位置大致不动，缓慢上下点头（pitch），来回约 20–45°。"),
    (63.0, "51–63 秒：夹爪位置大致不动，缓慢左右摇头（yaw），来回约 20–45°。"),
    (75.0, "63–75 秒：夹爪位置大致不动，缓慢绕镜头方向滚转（roll），来回约 20–45°。"),
    (84.0, "75–84 秒：小幅混合平移和转动，动作连续、缓慢，仍让 Tag 可见。"),
    (90.0, "84–90 秒：再次完全静止；程序随后会自动停止视频并保留 IMU 尾部。"),
)


ROTATION_VALIDATION_GUIDE = (
    (3.0, "0–3 秒：保持整套手持装置完全静止，固定 Tag 位于画面中央。"),
    (15.0, "3–15 秒：位置大致不动，缓慢上下点头（pitch），来回约 20–45°。"),
    (27.0, "15–27 秒：位置大致不动，缓慢左右摇头（yaw），来回约 20–45°。"),
    (39.0, "27–39 秒：位置大致不动，缓慢绕镜头方向滚转（roll），来回约 20–45°。"),
    (45.0, "39–45 秒：再次完全静止；程序随后会自动停止视频并保留 IMU 尾部。"),
)


VIO_CALIBRATION_GUIDE = (
    (5.0, "0–5 秒：整套装置完全静止；固定 Tag 位于画面中央。"),
    (20.0, "5–20 秒：只做左右平移。每次移动约 20–40 cm，平稳加速、停住、再反向；不要快速甩动。"),
    (35.0, "20–35 秒：只做上下平移。每次移动约 20–40 cm，平稳加速、停住、再反向。"),
    (50.0, "35–50 秒：只做前后平移（靠近、远离 Tag）。每次约 20–40 cm，保持 Tag 清晰可见。"),
    (65.0, "50–65 秒：位置大致不动，反复上下点头（pitch）；每次转 20–45°，中间短暂停住。"),
    (80.0, "65–80 秒：位置大致不动，反复左右摇头（yaw）；每次转 20–45°，中间短暂停住。"),
    (95.0, "80–95 秒：位置大致不动，反复滚转（roll）；每次转 20–45°，中间短暂停住。"),
    (110.0, "95–110 秒：小幅斜向平移并叠加转动；连续但不要模糊，Tag 始终在画面里。"),
    (120.0, "110–120 秒：再次完全静止；程序随后自动停止视频并保留 IMU 尾部。"),
)


# Actual mapping should be gentler than a calibration excitation sequence:
# enough translation/rotation for VIO, but without fast wrist motion that
# introduces blur and makes it impossible to diagnose the visual front-end.
MAPPING_VIO_GUIDE = (
    (5.0, "0–5 秒：整套装置完全静止。Tag 13 必须固定在桌面/世界上、不能装在手持装置上，且不超过画面约 1/4。"),
    (25.0, "5–25 秒：连续匀速左右平移，单程约 30–40 cm、约 1–1.5 秒完成后平滑反向；不要停停走走，也不要甩动。"),
    (45.0, "25–45 秒：连续匀速上下平移，单程约 25–35 cm、约 1–1.5 秒；让桌面上的不同高度物体产生视差。"),
    (65.0, "45–65 秒：连续前后平移（靠近、远离桌面），单程约 30–40 cm、约 1–1.5 秒。"),
    (85.0, "65–85 秒：位置大致不动，连续做点头、摇头、滚转；每轴约 ±20°，每个来回约 2–3 秒。"),
    (105.0, "85–105 秒：沿桌面做连续小弧线移动（半径约 30 cm），保持中等速度；Tag 可在边缘但外部环境必须占大部分画面。"),
    (120.0, "105–120 秒：再次完全静止；程序随后自动停止视频并保留 IMU 尾部。"),
)


# 用于相机到夹爪 TCP 平移标定。两指基本闭合后的尖端中心始终轻触桌面
# 同一小点；相机和夹爪作为刚体围绕该点转动。不要开合夹爪，也不要让该点
# 在桌面上滑动，否则“固定 TCP”的数学前提不成立。
TCP_PIVOT_GUIDE = (
    (5.0, "0–5 秒：夹爪基本闭合；两指尖中心轻触桌面同一个小点，整套装置完全静止。确认 Tag 13、Tag 0、Tag 1 都可见。"),
    (12.0, "5–12 秒：指尖中心不离开该小点，缓慢上下点头（pitch）约 ±15–25°；不要开合夹爪。"),
    (19.0, "12–19 秒：指尖中心仍在同一点，缓慢左右摇头（yaw）约 ±15–25°；不要滑动。"),
    (26.0, "19–26 秒：指尖中心仍在同一点，缓慢绕夹爪前后方向滚转（roll）约 ±15–25°。"),
    (38.0, "26–38 秒：保持该支点不动，混合小幅点头、摇头、滚转；从多个方向观察固定 Tag。"),
    (45.0, "38–45 秒：回到稳定姿态，指尖中心仍压在同一点；程序随后自动停止并保存。"),
)

# V 型单活动爪的 TCP 位于固定爪上，而非会随开合移动的两爪中点。
VJAW_TCP_PIVOT_GUIDE = (
    (5.0, "0–5 秒：固定爪内侧最前端的 TCP 轻触桌面同一个小点，整套装置完全静止。确认桌面 Tag 13 与固定爪 Tag 0 清晰可见。"),
    (12.0, "5–12 秒：固定爪 TCP 不离开该小点，缓慢上下点头（pitch）约 ±15–25°；不要让该点滑动。"),
    (19.0, "12–19 秒：固定爪 TCP 仍在同一点，缓慢左右摇头（yaw）约 ±15–25°；不要开合夹爪。"),
    (26.0, "19–26 秒：固定爪 TCP 仍在同一点，缓慢绕夹爪前后方向滚转（roll）约 ±15–25°。"),
    (38.0, "26–38 秒：保持该支点不动，混合小幅点头、摇头、滚转；让 Tag 13 与 Tag 0 尽量持续可见。"),
    (45.0, "38–45 秒：回到稳定姿态，固定爪 TCP 仍压在同一点；程序随后自动停止并保存。"),
)


# A short, deliberately simple motion record used to establish the sign and
# ordering of hand-held TCP axes before any robot replay.  It is not a task
# demonstration: each motion returns to the same start pose so its direction
# can be checked independently on the robot at very low speed.
TCP_AXIS_VALIDATION_GUIDE = (
    (8.0, "0–8 秒：先不要移动；保持手持夹爪完全静止于基准位置，让你读完屏幕说明并准备。Tag 13 或 14 必须清晰可见。"),
    (16.0, "8–16 秒：保持朝向不变，沿桌面朝篮子/任务前方平移约 10 cm，再回到起点。"),
    (24.0, "16–24 秒：保持朝向不变，向上抬高约 10 cm，再回到起点。"),
    (32.0, "24–32 秒：保持朝向不变，按你面对桌面时的左方平移约 10 cm，再回到起点。"),
    (40.0, "32–40 秒：保持位置大致不变，缓慢将夹爪工具方向向下点头约 15°，再回到原朝向。"),
    (48.0, "40–48 秒：回到原始位置和朝向并保持静止；程序随后自动保存。"),
)


def timed_guide(elapsed_s: float, guide: tuple[tuple[float, str], ...]) -> tuple[int, str]:
    """Return the current numbered guide stage for a recording elapsed time."""
    for index, (end_s, text) in enumerate(guide):
        if elapsed_s < end_s:
            return index, text
    return len(guide), f"{guide[-1][0]:.0f} 秒已完成：等待程序保存数据。"


def draw_legible_text(image: np.ndarray, text: str, origin: tuple[int, int],
                      color: tuple[int, int, int], scale: float = 0.58) -> None:
    """Draw text readable on bright or dark live-camera backgrounds."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def annotate_monitored_tags(image_bgr: np.ndarray, detector,
                            tag_ids: list[int]) -> tuple[np.ndarray, dict[int, bool]]:
    """Draw per-world-Tag status on the full-resolution browser preview."""
    preview = image_bgr.copy()
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    found_ids = set() if ids is None else {int(item) for item in ids.reshape(-1)}
    status = {tag_id: tag_id in found_ids for tag_id in tag_ids}
    if ids is not None:
        chosen = [index for index, item in enumerate(ids.reshape(-1))
                  if int(item) in status]
        if chosen:
            chosen_corners = [corners[index] for index in chosen]
            chosen_ids = np.asarray([[int(ids[index, 0])] for index in chosen], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(preview, chosen_corners, chosen_ids)
    # Keep all statuses inside a fixed high-contrast panel.  The old footer
    # could be below the browser viewport, which made only the first Tag look
    # visible to the operator.
    panel_width = min(520, max(390, preview.shape[1] // 2))
    panel_height = 34 * len(tag_ids) + 16
    x0 = 14
    y0 = 68
    panel = preview[y0:y0 + panel_height, x0:x0 + panel_width]
    if panel.size:
        overlay = panel.copy()
        overlay[:] = (20, 20, 20)
        cv2.addWeighted(overlay, 0.88, panel, 0.12, 0, panel)
    for index, tag_id in enumerate(tag_ids):
        detected = status[tag_id]
        text = f"WORLD TAG {tag_id}: {'DETECTED' if detected else 'MISSING'}"
        color = (0, 230, 255) if detected else (0, 0, 255)  # yellow / red
        draw_legible_text(preview, text, (x0 + 12, y0 + 30 + 34 * index), color,
                          scale=0.82)
    return preview, status


def make_recording_preview(
        frame_bgr: np.ndarray,
        camera_crop: dict,
        detector,
        tag_ids: list[int],
        metrics: dict | None,
        label: str,
        motion: dict | None = None,
        arm_ik: dict | None = None,
) -> tuple[np.ndarray, dict[int, bool]]:
    """Show the full raw camera view, with the policy crop drawn on top.

    The fixed world Tag used for trajectory recovery may intentionally be
    outside the square policy crop.  It therefore has to be checked against
    the complete camera frame rather than a cropped reference-comparison
    panel.
    """
    tag_status: dict[int, bool] = {}
    annotated = frame_bgr
    if detector is not None and tag_ids:
        # Detect at native resolution before resizing, so a Tag near the edge
        # of the wide camera image remains easy to inspect.
        annotated, tag_status = annotate_monitored_tags(frame_bgr, detector, tag_ids)

    scale = 960.0 / annotated.shape[1]
    display = cv2.resize(annotated, (960, round(annotated.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    center_x, center_y = camera_crop["center"]
    size = camera_crop["size"]
    x0 = round((center_x - size / 2) * scale)
    y0 = round((center_y - size / 2) * scale)
    side = round(size * scale)
    cv2.rectangle(display, (x0, y0), (x0 + side, y0 + side), (0, 255, 0), 2)
    draw_legible_text(display, "RAW CAMERA VIEW | green square = policy input crop",
                      (12, 28), (255, 255, 255))
    draw_legible_text(display, label, (12, 54), (255, 255, 255), scale=0.54)
    if metrics is not None:
        text = (f"policy-crop appearance={metrics['appearance_score']:.3f}  "
                f"corr={metrics['correlation']:.3f}")
        draw_legible_text(display, text, (12, 78), (255, 255, 255), scale=0.50)
    if motion is not None:
        if not motion["started"]:
            draw_legible_text(display, "TCP monitor: waiting for one mapped world Tag ...",
                              (12, 106), (0, 230, 255), scale=0.52)
        else:
            delta_cm = np.asarray(motion["delta_m"]) * 100.0
            limits_enabled = (motion["translation_limit_m"] > 0 or
                              motion["rotation_limit_deg"] > 0)
            warning = ((motion["translation_limit_m"] > 0 and
                        motion["distance_m"] > motion["translation_limit_m"]) or
                       (motion["rotation_limit_deg"] > 0 and
                        motion["rotation_deg"] > motion["rotation_limit_deg"]))
            color = (0, 0, 255) if warning else (255, 255, 255)
            state = "GUIDE EXCEEDED" if warning else "relative motion"
            draw_legible_text(
                display,
                f"relative TCP: {motion['distance_m'] * 100:.1f} cm | "
                f"rotation: {motion['rotation_deg']:.1f} deg | {state}",
                (12, 106), color, scale=0.52)
            draw_legible_text(
                display,
                (f"world-Tag delta: x={delta_cm[0]:+.1f}  y={delta_cm[1]:+.1f}  z={delta_cm[2]:+.1f} cm"
                 if not limits_enabled else
                 f"world-Tag delta: x={delta_cm[0]:+.1f}  y={delta_cm[1]:+.1f}  z={delta_cm[2]:+.1f} cm "
                 f"(guide {motion['translation_limit_m'] * 100:.0f} cm / {motion['rotation_limit_deg']:.0f} deg)"),
                (12, 130), color, scale=0.45)
    if arm_ik is not None:
        if not arm_ik["ready"]:
            draw_legible_text(display, "arm IK prediction: waiting for world-Tag start pose ...",
                              (12, 154), (0, 230, 255), scale=0.48)
        else:
            if arm_ik["reachable"]:
                low_margin = arm_ik["joint_margin_deg"] < 5.0
                color = (0, 230, 255) if (arm_ik["baseline_limit_mismatch"] or low_margin) else (255, 255, 255)
                state = "IK POSE FITS - LOW MARGIN" if low_margin else "IK POSE FITS"
            else:
                color = (0, 0, 255)
                state = "IK POSE DOES NOT FIT"
            draw_legible_text(
                display,
                f"arm IK prediction: {state} | pos err {arm_ik['position_error_mm']:.1f} mm | "
                f"rot err {arm_ik['rotation_error_deg']:.1f} deg",
                (12, 154), color, scale=0.47)
            margin_note = ("baseline exceeds nominal URDF limit; limit calibration pending"
                           if arm_ik["baseline_limit_mismatch"] else
                           f"joint-limit margin: {arm_ik['joint_margin_deg']:.1f} deg (offline estimate only)")
            draw_legible_text(display, margin_note, (12, 178), color, scale=0.43)
            # [margin] is the distance to the nearer of the two URDF limits.
            # A red individual field is the direct explanation for a red IK
            # warning; neutral fields remain white so the operator can tell
            # whether it is one joint or a whole-pose fitting failure.
            joint_names = ("J1 pan", "J2 lift", "J3 elbow",
                           "J4 pitch", "J5 yaw", "J6 roll")
            joint_angles = arm_ik["joint_deg"]
            joint_margins = arm_ik["joint_margin_each_deg"]
            for row in range(2):
                y = 202 + row * 24
                for column in range(3):
                    index = row * 3 + column
                    field_color = ((0, 0, 255) if joint_margins[index] < 5.0
                                   else (255, 255, 255))
                    draw_legible_text(
                        display,
                        f"{joint_names[index]} {joint_angles[index]:+.1f} [{joint_margins[index]:.1f}]",
                        (12 + column * 314, y), field_color, scale=0.36)
    return display, tag_status


class LiveTcpMotionMonitor:
    """Low-rate direct multi-Tag TCP displacement monitor for demo recording.

    It is only an operator warning panel.  The later offline trajectory export
    remains the authoritative source and the monitor never sends robot motion.
    """

    def __init__(self, tag_map_path: pathlib.Path, intrinsics_path: pathlib.Path,
                 camera_tcp_geometry_path: pathlib.Path,
                 translation_limit_m: float, rotation_limit_deg: float,
                 allow_candidate_geometry: bool = False):
        tag_map = json.loads(tag_map_path.read_text(encoding="utf-8"))
        if tag_map.get("schema") != "am_umi_world_tag_map_v1":
            raise ValueError("--motion-tag-map 必须是 am_umi_world_tag_map_v1")
        self.known_tags = {}
        for raw_id, item in tag_map.get("tags", {}).items():
            transform = np.asarray(item["T_world_tag"], dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError(f"世界 Tag {raw_id} 的 T_world_tag 无效")
            self.known_tags[int(raw_id)] = {
                "size_m": float(item["size_m"]), "T_world_tag": transform}
        if not self.known_tags:
            raise ValueError("--motion-tag-map 没有可用的世界 Tag")
        geometry = json.loads(camera_tcp_geometry_path.read_text(encoding="utf-8"))
        allowed_status = (None, "accepted", "candidate") if allow_candidate_geometry \
            else (None, "accepted")
        if geometry.get("schema") != "am_umi_camera_tcp_geometry_v1" or \
                geometry.get("status") not in allowed_status:
            raise ValueError("--motion-camera-tcp-geometry 不是已接受的相机-TCP 几何")
        if geometry.get("status") == "candidate" and not allow_candidate_geometry:
            raise ValueError("候选相机-TCP 几何只能配合 --allow-candidate-motion-tcp 用于诊断")
        pose_cam_tcp = np.asarray(geometry.get("pose_cam_tcp"), dtype=float)
        if pose_cam_tcp.shape != (6,) or not np.isfinite(pose_cam_tcp).all():
            raise ValueError("相机-TCP 几何中的 pose_cam_tcp 无效")
        self.tx_cam_tcp = pose_to_mat(pose_cam_tcp)
        self.camera, self.distortion = load_intrinsics(intrinsics_path)
        config = yaml.safe_load((ROOT_DIR / "calibration" / "shared_tags" / "aruco_config.yaml").read_text())
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, config["aruco_dict"]["predefined"]))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.translation_limit_m = float(translation_limit_m)
        self.rotation_limit_deg = float(rotation_limit_deg)
        self.start_tcp = None

    def pose_from_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        candidates = []
        if ids is not None:
            for marker_index, raw_id in enumerate(ids.reshape(-1)):
                tag_id = int(raw_id)
                item = self.known_tags.get(tag_id)
                if item is None:
                    continue
                tag_to_camera = pose_from_marker(
                    corners[marker_index], item["size_m"], self.camera, self.distortion)
                if tag_to_camera is not None:
                    candidates.append((
                        tag_id,
                        item["T_world_tag"] @ np.linalg.inv(tag_to_camera),
                        marker_area_px2(corners[marker_index]),
                    ))
        if not candidates:
            return None
        tx_world_camera, _, _, _ = robust_average_camera_pose(candidates)
        return tx_world_camera @ self.tx_cam_tcp

    def update(self, frame_bgr: np.ndarray) -> dict:
        current = self.pose_from_frame(frame_bgr)
        if current is None:
            return {"started": False, "missing": True}
        if self.start_tcp is None:
            self.start_tcp = current
        relative = np.linalg.inv(self.start_tcp) @ current
        delta = relative[:3, 3]
        rotation_deg = float(np.degrees(Rotation.from_matrix(relative[:3, :3]).magnitude()))
        return {
            "started": True,
            "missing": False,
            "delta_m": delta.tolist(),
            "distance_m": float(np.linalg.norm(delta)),
            "rotation_deg": rotation_deg,
            "translation_limit_m": self.translation_limit_m,
            "rotation_limit_deg": self.rotation_limit_deg,
            "tx_world_tcp": current,
        }


class LiveArmIkMonitor:
    """Read-only estimate of whether the *current* relative demo pose has IK.

    It mirrors the offline replay gate's body-relative transform but runs one
    warm-started local solve at preview rate.  It is explicitly advisory: no
    serial port is opened, and collision checking still needs the later gate
    and physical low-speed replay.
    """

    def __init__(self, reference_metadata: dict, robot_config_path: pathlib.Path):
        state = reference_metadata.get("robot_state", {})
        if reference_metadata.get("tcp_frame") != "right_tcp":
            raise ValueError("reference 必须是新夹爪 right_tcp 基准")
        q = np.asarray(state.get("ActualQ", [])[:6], dtype=float)
        pose = np.asarray(state.get("ActualTCPPose", []), dtype=float)
        if q.shape != (6,) or pose.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(pose).all():
            raise ValueError("reference.json 缺少有效 ActualQ / ActualTCPPose")
        robot_config = yaml.safe_load(robot_config_path.read_text(encoding="utf-8"))
        robots = robot_config.get("robots", [])
        if not robots:
            raise ValueError(f"机器人配置缺少 robots: {robot_config_path}")
        robot = robots[0]
        urdf_path = pathlib.Path(robot.get("urdf_path", URDF_PATH)).expanduser()
        if not urdf_path.is_absolute():
            urdf_path = (ROOT_DIR / urdf_path).resolve()
        if not urdf_path.is_file():
            raise ValueError(f"机器人配置中的 URDF 不存在: {urdf_path}")
        mapping = mapping_from_config(robot, ROOT_DIR)
        model_safe_limits = model_safe_limits_from_config(robot, mapping)
        self.q_start = q
        self.q_previous = q.copy()
        self.tx_robot_start = pose_to_mat(pose)
        self.limits = load_joint_limits(urdf_path, model_safe_limits=model_safe_limits)
        self.baseline_margin_deg = float(np.min(np.minimum(
            self.q_start - self.limits[:, 0], self.limits[:, 1] - self.q_start)))
        self.backend = create_kinematics_backend(
            robot.get("ik_backend", "placo"), str(urdf_path), URDF_JOINTS, "right_tcp")

    def update(self, motion: dict | None, start_tcp: np.ndarray | None) -> dict:
        if motion is None or not motion.get("started") or start_tcp is None:
            return {"ready": False}
        target = self.tx_robot_start @ np.linalg.inv(start_tcp) @ motion["tx_world_tcp"]
        q = self.q_previous.copy()
        for _ in range(40):
            q = self.backend.inverse_kinematics(
                q, target, position_weight=1.0, orientation_weight=0.35)
            # Match the offline gate: retain a calibrated J2 envelope while
            # solving so preview can show a feasible elbow-compensation
            # branch rather than an impossible unconstrained J2 angle.
            q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        actual = self.backend.forward_kinematics(q)
        position_error_m, rotation_error_deg = pose_error(target, actual)
        joint_margins = np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q)
        margin = float(np.min(joint_margins))
        self.q_previous = q
        # The saved AM2Pro start pose is known to be physically usable, but
        # its wrist-flex value lies outside the legacy URDF's nominal limit.
        # Do not mislabel the *start pose itself* as unreachable.  The panel
        # separates geometric IK fit from that pending URDF-limit calibration.
        reachable = (position_error_m <= 0.010 and rotation_error_deg <= 8.0)
        return {
            "ready": True,
            "reachable": bool(reachable),
            "position_error_mm": position_error_m * 1000.0,
            "rotation_error_deg": rotation_error_deg,
            "joint_margin_deg": margin,
            # These are predicted AM2Pro joint angles, not measurements from
            # a robot command.  Showing each value makes a red IK warning
            # actionable: the operator can see *which* joint is near a limit.
            "joint_deg": q.tolist(),
            "joint_margin_each_deg": joint_margins.tolist(),
            "baseline_limit_mismatch": bool(self.baseline_margin_deg < 0.0),
        }


class AsyncRecordingMonitor:
    """Run the advisory Tag/TCP/IK monitor away from camera capture.

    Formal video capture always wins: this worker accepts at most one copied
    frame at a low fixed rate, and drops a diagnostic frame whenever it is
    still solving the preceding one.  It never blocks ``camera.read`` or the
    encoder, and it never opens a serial port or commands the arm.
    """

    def __init__(self, motion_monitor: LiveTcpMotionMonitor | None,
                 arm_ik_monitor: LiveArmIkMonitor | None, update_hz: float):
        self.motion_monitor = motion_monitor
        self.arm_ik_monitor = arm_ik_monitor
        self.period_s = 1.0 / update_hz
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self.lock = threading.Lock()
        self.last_submit_s = 0.0
        self.motion: dict | None = None
        self.arm_ik: dict | None = None
        self.thread = threading.Thread(target=self._run, daemon=True,
                                       name="handheld-recording-ik-monitor")

    def start(self) -> None:
        self.thread.start()

    def reset_origin(self) -> None:
        """Request a new trajectory origin before the next worker frame."""
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
        with self.lock:
            self.motion = None
            self.arm_ik = None
        self.reset_event.set()

    def submit(self, frame_bgr: np.ndarray) -> None:
        now = time.monotonic()
        if now - self.last_submit_s < self.period_s:
            return
        self.last_submit_s = now
        try:
            # The copy occurs at 5 Hz by default; the worker exclusively owns
            # it, so a later UVC dequeue can never overwrite its pixels.
            self.frames.put_nowait(frame_bgr.copy())
        except queue.Full:
            # A stale display result is preferable to delaying a video frame.
            pass

    def snapshot(self) -> tuple[dict | None, dict | None]:
        with self.lock:
            return self.motion, self.arm_ik

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame_bgr = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.reset_event.is_set():
                if self.motion_monitor is not None:
                    self.motion_monitor.start_tcp = None
                if self.arm_ik_monitor is not None:
                    self.arm_ik_monitor.q_previous = self.arm_ik_monitor.q_start.copy()
                self.reset_event.clear()
            motion = (self.motion_monitor.update(frame_bgr)
                      if self.motion_monitor is not None else None)
            arm_ik = (self.arm_ik_monitor.update(
                motion,
                self.motion_monitor.start_tcp if self.motion_monitor is not None else None)
                if self.arm_ik_monitor is not None else None)
            with self.lock:
                self.motion = motion
                self.arm_ik = arm_ik


class ImuChunkReader(threading.Thread):
    """Low-latency serial reader; precise packet times are decoded afterwards."""

    def __init__(self, port: str, baud: int, read_timeout_s: float = 0.002):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.read_timeout_s = read_timeout_s
        self.stop_event = Event()
        self.lock = Lock()
        self.chunks: list[tuple[int, int, bytes]] = []
        self.error: Exception | None = None

    def run(self):
        try:
            with serial.Serial(self.port, self.baud, timeout=self.read_timeout_s) as device:
                # Do not let packets buffered before this capture define the
                # session's device-clock origin.
                device.reset_input_buffer()
                while not self.stop_event.is_set():
                    # Read what the driver has now, rather than waiting for a
                    # large 2048-byte buffer.  JY901B packets are small and
                    # this keeps host delivery latency near the 2 ms timeout.
                    request = max(1, min(int(device.in_waiting), 256))
                    t_before = time.monotonic_ns()
                    chunk = device.read(request)
                    t_after = time.monotonic_ns()
                    if chunk:
                        with self.lock:
                            self.chunks.append((t_before, t_after, bytes(chunk)))
        except Exception as exc:  # returned to main process for a clear failure
            self.error = exc

    def stop(self):
        self.stop_event.set()
        self.join(timeout=2.0)

    def snapshot(self):
        with self.lock:
            return list(self.chunks)


class WebPreview:
    """Local browser preview that avoids OpenCV/Qt HighGUI deadlocks."""

    def __init__(self, port: int, reference_bgr: np.ndarray | None = None):
        self.lock = threading.Lock()
        self.jpeg = None
        self.reference_jpeg = None
        if reference_bgr is not None:
            ok, encoded = cv2.imencode(
                ".jpg", reference_bgr, [cv2.IMWRITE_JPEG_QUALITY, 86])
            if ok:
                self.reference_jpeg = encoded.tobytes()
        self.status = "预览已打开：确认 Tag 在画面中后，按 R 开始录制。"
        self.commands = queue.Queue()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path.startswith("/status.json"):
                    with owner.lock:
                        status = owner.status
                    payload = json.dumps({"status": status}, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.path.startswith("/frame.jpg"):
                    with owner.lock:
                        image = owner.jpeg
                    if image is None:
                        self.send_response(204)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(image)))
                    self.end_headers()
                    self.wfile.write(image)
                    return
                if self.path.startswith("/reference.jpg"):
                    with owner.lock:
                        image = owner.reference_jpeg
                    if image is None:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "public, max-age=3600")
                    self.send_header("Content-Length", str(len(image)))
                    self.end_headers()
                    self.wfile.write(image)
                    return
                if self.path.startswith("/command/"):
                    command = self.path.rsplit("/", 1)[-1].lower()
                    if command in {"t", "r", "s", "x", "q", "b"}:
                        owner.commands.put(command)
                        self.send_response(204)
                    else:
                        self.send_response(400)
                    self.end_headers()
                    return
                reference_panel = "" if owner.reference_jpeg is None else '''<div class="view"><h3>基准原始照片（仅供人工对齐）</h3><img id="reference" src="/reference.jpg"></div>'''
                page = '''<!doctype html><meta charset="utf-8"><title>Handheld UMI preview</title>
<style>body{background:#111;color:#eee;font:18px sans-serif;text-align:center;margin:12px}#views{display:flex;gap:12px;align-items:flex-start;justify-content:center;flex-wrap:wrap}.view{flex:1 1 420px;max-width:960px}.view h3{font-size:18px;margin:8px}.view img{max-width:100%%;max-height:72vh;object-fit:contain}button{font-size:24px;margin:8px;padding:10px 24px}</style>
<h2>手持 UMI 预览</h2><div id="status">正在连接预览…</div><div id="views"><div class="view"><h3>实时原始画面</h3><img id="v" src="/frame.jpg"></div>%s</div><p>
<button onclick="cmd('t')">T: trial / restart</button><button onclick="cmd('r')">R: formal start</button><button onclick="cmd('s')">S: stop &amp; save</button><button onclick="cmd('x')">X: discard</button><button onclick="cmd('q')">Q: exit</button></p>
<script>const v=document.getElementById('v'),s=document.getElementById('status');let shownUrl=null;async function nextFrame(){try{const r=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store'});if(r.ok){const u=URL.createObjectURL(await r.blob());const old=shownUrl;shownUrl=u;v.src=u;if(old)URL.revokeObjectURL(old)}}catch(e){}setTimeout(nextFrame,35)}nextFrame();setInterval(()=>fetch('/status.json?t='+Date.now()).then(r=>r.json()).then(x=>s.textContent=x.status).catch(()=>{}),200);function cmd(x){fetch('/command/'+x)}document.addEventListener('keydown',e=>{const k=e.key.toLowerCase();if(['t','r','s','x','q'].includes(k)){e.preventDefault();cmd(k)}})</script>''' % reference_panel
                page = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def publish(self, image_bgr):
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            with self.lock:
                self.jpeg = encoded.tobytes()

    def set_status(self, status: str):
        with self.lock:
            self.status = status

    def get_command(self):
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


class PyAvUvcCapture:
    """V4L2 MJPEG reader with OpenCV-compatible ``read``/``release`` methods.

    On this EMEET device, OpenCV VideoCapture delivers roughly 15 FPS even
    though V4L2 can supply 30 FPS.  PyAV's V4L2 reader has been measured at
    29 FPS including BGR conversion, so use it for recording.
    """

    def __init__(self, device: str, width: int, height: int, fps: float,
                 pixel_format: str):
        self.resolved = os.path.realpath(device)
        input_format = {"MJPG": "mjpeg", "YUYV": "yuyv422"}.get(
            pixel_format.upper(), pixel_format.lower())
        options = {
            "input_format": input_format,
            "video_size": f"{width}x{height}",
            "framerate": str(int(fps) if fps.is_integer() else fps),
        }
        try:
            self.container = av.open(self.resolved, format="v4l2", options=options)
        except Exception as exc:
            raise RuntimeError(
                f"cannot open UVC camera {device!r} (resolved {self.resolved!r})") from exc
        self.frames = self.container.decode(video=0)

    def read(self):
        try:
            frame = next(self.frames)
        except StopIteration:
            return False, None
        return True, frame.to_ndarray(format="bgr24")

    def release(self):
        # Close the decoder generator before its V4L2 container.  Leaving a
        # live generator to Python's shutdown can segfault in this PyAV build
        # after an otherwise successful capture+encode session.
        if self.frames is not None:
            self.frames.close()
            self.frames = None
        if self.container is not None:
            self.container.close()
            self.container = None


def open_uvc(device: str, width: int, height: int, fps: float, pixel_format: str):
    camera = PyAvUvcCapture(device, width, height, fps, pixel_format)
    return camera, camera.resolved


def infer_uvc_metadata_device(resolved_video_device: str, requested: str) -> str | None:
    """Find the UVC metadata sibling for ``--uvc-metadata-device auto``.

    EMEET's ``/dev/video2`` image stream has ``/dev/video3`` as its UVCH
    metadata stream.  Do not guess for a non-numeric device path: recording
    video+IMU is still valid, only the optional improved timing is skipped.
    """
    if requested != "auto":
        return None if requested.lower() in {"none", "off", "disable"} else requested
    name = pathlib.Path(resolved_video_device).name
    if not name.startswith("video") or not name[5:].isdigit():
        return None
    sibling = f"/dev/video{int(name[5:]) + 1}"
    return sibling if pathlib.Path(sibling).exists() else None


class UvcMetadataCapture:
    """Capture raw paired UVCH headers while the video stream is live.

    This calls only ``v4l2-ctl --stream-to`` on the camera's *metadata* node;
    it never changes image controls or the video capture mode.  Metadata is
    intentionally optional: failure is recorded and must not discard a good
    hand-held demonstration.
    """

    def __init__(self, device: str, output_dir: pathlib.Path):
        self.device = device
        self.raw_path = output_dir / "raw_uvc_payload_headers.bin"
        self.log_path = output_dir / "uvc_metadata_capture.log"
        self.process = None
        self.log = None

    def start(self):
        self.log = self.log_path.open("wb")
        command = [
            "v4l2-ctl", f"--device={self.device}", "--stream-mmap=3",
            f"--stream-to={self.raw_path}",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT)
        # Do not wait here: the next image frame may arrive within 33 ms and
        # the first source-clock event is valuable.  An unavailable node is
        # reported during finalization as ``captured: false`` instead.

    def stop(self) -> dict:
        if self.process is None:
            return {"captured": False, "reason": "not_started"}
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3.0)
        returncode = self.process.returncode
        if self.log is not None:
            self.log.close()
            self.log = None
        size = self.raw_path.stat().st_size if self.raw_path.exists() else 0
        return {"captured": size > 0, "returncode": returncode, "bytes": size,
                "raw_path": self.raw_path.name, "log_path": self.log_path.name}


def make_writer(path: pathlib.Path, width: int, height: int, fps: float,
                encoder_preset: str = "veryfast"):
    container = av.open(str(path), mode="w")
    # PyAV 15 rejects a Python float here (it expects an AVRational-like
    # numerator/denominator object).  Keep 30.0 represented exactly as 30/1.
    stream = container.add_stream("libx264", rate=Fraction(str(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # Keep the encoded MP4 on the camera's stable nominal timeline.  The
    # authoritative per-frame host receive timestamps are saved separately in
    # frame_timestamps.csv; forcing microsecond PTS here is rejected by this
    # particular PyAV/MP4/H.264 combination during muxing.
    stream.options = {"crf": "20", "preset": encoder_preset}
    return container, stream


def write_frame(stream, container, frame_bgr: np.ndarray, relative_s: float):
    video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
    for packet in stream.encode(video_frame):
        container.mux(packet)


def close_writer(stream, container):
    if container is None:
        return
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def save_imu_raw(output_path: pathlib.Path, chunks, baud: int, metadata: dict):
    if not chunks:
        raise RuntimeError("no IMU bytes received; check port and baud")
    reads = [(start / 1e9, end / 1e9, chunk) for start, end, chunk in chunks]
    parsed = parse_reads_precise_with_device_time(reads, baud)
    metadata = dict(metadata)
    metadata["imu_timestamp_method"] = (
        "jy901b_0x50_device_clock_mapped_to_host_v1"
        if parsed["uses_device_clock"] else "continuous_uart_8n1_packet_center_v1")
    np.savez_compressed(
        output_path,
        t_accel_monotonic_s=parsed["t_accel_monotonic_s"],
        accel_raw=parsed["accel_raw"],
        t_gyro_monotonic_s=parsed["t_gyro_monotonic_s"],
        gyro_raw=parsed["gyro_raw"],
        t_accel_uart_host_s=parsed["t_accel_uart_host_s"],
        t_gyro_uart_host_s=parsed["t_gyro_uart_host_s"],
        t_accel_device_s=parsed["t_accel_device_s"],
        t_gyro_device_s=parsed["t_gyro_device_s"],
        uses_jy901b_device_time=parsed["uses_device_clock"],
        device_clock_to_host_scale=parsed["device_clock_to_host_scale"],
        device_time_packet_count=parsed["device_time_packet_count"],
        accel_scale_g_per_lsb=16.0 / 32768.0,
        gyro_scale_deg_s_per_lsb=2000.0 / 32768.0,
        **metadata,
    )
    if parsed["uses_device_clock"]:
        print("JY901B 设备时间已用于 IMU 时间轴："
              f"{parsed['device_time_packet_count']} 个 0x50 包，"
              f"时钟比例={parsed['device_clock_to_host_scale']:.9f}")
    else:
        print("警告：未检测到 JY901B 0x50 时间包，IMU 时间轴退回串口到达时间。")
    return (len(parsed["accel_raw"]), len(parsed["gyro_raw"]),
            bool(parsed["uses_device_clock"]))


def save_metadata(path: pathlib.Path, data: dict):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="new session directory; refuse to overwrite")
    parser.add_argument("--camera-device", required=True,
                        help="prefer the stable /dev/v4l/by-id/... path")
    parser.add_argument("--imu-port", default="/dev/ttyUSB0")
    parser.add_argument("--imu-baud", type=int, default=460800)
    parser.add_argument("--no-imu", action="store_true",
                        help="record camera/UVC data only; intended for visual Tag calibration when no IMU is attached")
    parser.add_argument("--imu-read-timeout-ms", type=float, default=2.0,
                        help="JY901B 串口读取超时；默认 2 ms，降低主机时间戳抖动")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pixel-format", default="MJPG")
    parser.add_argument("--uvc-metadata-device", default="auto",
                        help=("paired UVC UVCH metadata node. Default auto finds the next "
                              "/dev/videoN node; use none to keep legacy host timestamps only"))
    parser.add_argument("--imu-tail-seconds", type=float, default=2.0)
    parser.add_argument("--max-record-seconds", type=float, default=180.0,
                        help="最长录制秒数；设为 0 表示不自动停止，只能按 S 手动停止")
    parser.add_argument("--reference-dir", default=None,
                        help="optional saved hand-held start-view reference")
    parser.add_argument("--alignment-robot-config",
                        default=str(ROOT_DIR / "example" / "eval_robots_config.yaml"),
                        help="camera crop used for reference/alignment; defaults to AM2Pro config")
    parser.add_argument("--save-reference-dir", default=None,
                        help="press b in preview to save a new reference here")
    parser.add_argument("--no-preview", action="store_true",
                        help="for automated camera/IMU checks only; records immediately")
    parser.add_argument("--web-preview", action="store_true",
                        help="show preview/control in a local browser instead of OpenCV Qt")
    parser.add_argument("--web-preview-port", type=int, default=8765)
    parser.add_argument("--web-preview-update-hz", type=float, default=10.0,
                        help="trial/idle browser image, Tag and TCP refresh rate")
    parser.add_argument("--web-recording-preview-update-hz", type=float, default=None,
                        help="formal-recording preview refresh rate; defaults to --web-preview-update-hz. Lower it to prioritize video FPS")
    parser.add_argument("--recording-capture-priority", action="store_true",
                        help="after R, keep video capture highest priority; Tag/TCP/IK diagnostics run only in a dropping low-rate worker")
    parser.add_argument("--recording-monitor-update-hz", type=float, default=5.0,
                        help="正式录制期间后台 Tag/TCP/IK 限位诊断刷新率；默认 5 Hz，诊断帧会丢弃而不阻塞视频")
    parser.add_argument("--video-encoder-preset", default="veryfast",
                        help="FFmpeg libx264 preset used for raw_video.mp4; ultrafast reduces CPU load at the cost of a larger file")
    parser.add_argument("--monitor-tag-id", type=int, default=13,
                        help="网页预览中实时提示的固定桌面 Tag 编号；设为负数关闭")
    parser.add_argument("--monitor-tag-ids", default=None,
                        help="逗号分隔的多个固定世界 Tag，例如 13,14,15；覆盖 --monitor-tag-id")
    parser.add_argument("--live-tcp-monitor", action="store_true",
                        help="录制时在网页预览显示相对起点的固定 Tag TCP 位移/转角提示；只提示，不控制机械臂")
    parser.add_argument("--live-arm-ik-monitor", action="store_true",
                        help="录制时基于 right_tcp 基准做只读 IK 可达性预测；需要 --reference-dir，绝不连接机械臂")
    parser.add_argument("--motion-tag-map",
                        default=str(ROOT_DIR / "calibration" / "shared_tags" / "table_tag_map_v1.json"),
                        help="--live-tcp-monitor 使用的固定世界 Tag 地图")
    parser.add_argument("--motion-intrinsics",
                        default=str(ROOT_DIR / "calibration" / "handheld_gripper_camera" / "intrinsics" /
                                    "emeet_handheld_1920x1080_30fps_fisheye_20260831.json"),
                        help="--live-tcp-monitor 使用的手持相机内参")
    parser.add_argument("--motion-camera-tcp-geometry",
                        default=str(ROOT_DIR / "calibration" / "handheld_gripper_camera" / "gripper_geometry" /
                                    "emeet_handheld_camera_to_tcp_v2.json"),
                        help="--live-tcp-monitor 使用的相机到手持 TCP 几何")
    parser.add_argument("--allow-candidate-motion-tcp", action="store_true",
                        help="仅诊断时允许 --motion-camera-tcp-geometry 使用 candidate；不能作为正式验收")
    parser.add_argument("--live-translation-warning-m", type=float, default=0.0,
                        help="可选的相对起点 TCP 位移提示线；0=关闭（默认）")
    parser.add_argument("--live-rotation-warning-deg", type=float, default=0.0,
                        help="可选的相对起点 TCP 转角提示线；0=关闭（默认）")
    parser.add_argument("--guided-extrinsic", action="store_true",
                        help="after R, show a 90-second Chinese camera-IMU calibration action timeline")
    parser.add_argument("--guided-rotation-validation", action="store_true",
                        help="after R, show a 45-second Chinese independent rotation-validation timeline")
    parser.add_argument("--guided-vio-calibration", action="store_true",
                        help="after R, show a 120-second Chinese camera-IMU translation/noise timeline")
    parser.add_argument("--guided-mapping-vio", action="store_true",
                        help="after R, show a 120-second gentle mapping/VIO test timeline")
    parser.add_argument("--guided-tcp-pivot", action="store_true",
                        help="after R, show a 45-second Chinese fixed-point camera-to-gripper TCP timeline")
    parser.add_argument("--guided-vjaw-tcp-pivot", action="store_true",
                        help="after R, show a 45-second Chinese fixed-point timeline for a V-jaw fixed-jaw TCP")
    parser.add_argument("--guided-tcp-axis-validation", action="store_true",
                        help="after R, show a 35-second Chinese hand-held TCP axis-direction validation timeline")
    args = parser.parse_args()

    if len(args.pixel_format) != 4:
        parser.error("--pixel-format must be a four-character V4L2 code, e.g. MJPG")
    if args.imu_tail_seconds < 0 or args.max_record_seconds < 0 or args.imu_read_timeout_ms <= 0:
        parser.error("invalid duration")
    if (args.web_preview_update_hz <= 0 or
            (args.web_recording_preview_update_hz is not None and
             args.web_recording_preview_update_hz <= 0)):
        parser.error("网页预览刷新频率必须为正")
    if args.recording_monitor_update_hz <= 0:
        parser.error("--recording-monitor-update-hz 必须为正")
    if args.live_translation_warning_m < 0 or args.live_rotation_warning_deg < 0:
        parser.error("实时 TCP 位移/转角提示线不能为负；0 表示关闭")
    if args.live_arm_ik_monitor and not args.reference_dir:
        parser.error("--live-arm-ik-monitor 需要 --reference-dir 的 right_tcp 基准")
    if args.no_preview and args.web_preview:
        parser.error("--no-preview and --web-preview cannot be used together")
    if args.monitor_tag_ids is None:
        monitored_tag_ids = [] if args.monitor_tag_id < 0 else [args.monitor_tag_id]
    else:
        try:
            monitored_tag_ids = [int(item.strip()) for item in args.monitor_tag_ids.split(",")
                                 if item.strip()]
        except ValueError:
            parser.error("--monitor-tag-ids 必须为逗号分隔的整数，例如 13,14,15")
        if not monitored_tag_ids or len(set(monitored_tag_ids)) != len(monitored_tag_ids) \
                or any(tag_id < 0 for tag_id in monitored_tag_ids):
            parser.error("--monitor-tag-ids 需要至少一个互不相同的非负编号")
    guide_flags = sum((args.guided_extrinsic, args.guided_rotation_validation,
                       args.guided_vio_calibration, args.guided_mapping_vio,
                       args.guided_tcp_pivot, args.guided_vjaw_tcp_pivot,
                       args.guided_tcp_axis_validation))
    if guide_flags > 1:
        parser.error("一次只能选择一种时间轴")
    if args.guided_extrinsic and args.max_record_seconds < 90:
        parser.error("--guided-extrinsic requires --max-record-seconds of at least 90")
    if args.guided_rotation_validation and args.max_record_seconds < 45:
        parser.error("--guided-rotation-validation requires --max-record-seconds of at least 45")
    if args.guided_vio_calibration and args.max_record_seconds < 120:
        parser.error("--guided-vio-calibration requires --max-record-seconds of at least 120")
    if args.guided_mapping_vio and args.max_record_seconds < 120:
        parser.error("--guided-mapping-vio requires --max-record-seconds of at least 120")
    if args.guided_tcp_pivot and args.max_record_seconds < 45:
        parser.error("--guided-tcp-pivot requires --max-record-seconds of at least 45")
    if args.guided_vjaw_tcp_pivot and args.max_record_seconds < 45:
        parser.error("--guided-vjaw-tcp-pivot requires --max-record-seconds of at least 45")
    if args.guided_tcp_axis_validation and args.max_record_seconds < 48:
        parser.error("--guided-tcp-axis-validation requires --max-record-seconds of at least 48")
    guide = (EXTRINSIC_GUIDE if args.guided_extrinsic else
             ROTATION_VALIDATION_GUIDE if args.guided_rotation_validation else
             VIO_CALIBRATION_GUIDE if args.guided_vio_calibration else
             MAPPING_VIO_GUIDE if args.guided_mapping_vio else
             TCP_PIVOT_GUIDE if args.guided_tcp_pivot else
             VJAW_TCP_PIVOT_GUIDE if args.guided_vjaw_tcp_pivot else
             TCP_AXIS_VALIDATION_GUIDE if args.guided_tcp_axis_validation else None)
    guide_name = ("相机—IMU 标定" if args.guided_extrinsic else
                  "相机—IMU 旋转验证" if args.guided_rotation_validation else
                  "VIO 平移/噪声标定" if args.guided_vio_calibration else
                  "VIO 建图测试" if args.guided_mapping_vio else
                  "相机—夹爪 TCP 固定点标定" if args.guided_tcp_pivot else
                  "V 型夹爪固定爪 TCP 标定" if args.guided_vjaw_tcp_pivot else
                  "手持 TCP 轴向验证" if args.guided_tcp_axis_validation else
                  "录制")

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    reference = None
    reference_metadata = None
    reference_raw_bgr = None
    if args.reference_dir:
        reference, reference_metadata = load_reference(args.reference_dir)
        raw_reference_path = pathlib.Path(args.reference_dir).expanduser() / "raw_camera.png"
        reference_raw_bgr = cv2.imread(str(raw_reference_path), cv2.IMREAD_COLOR)
        if reference_raw_bgr is None:
            # Older references may only have the 224 px policy image.  Keep
            # the panel available rather than failing a recording session.
            reference_raw_bgr = np.ascontiguousarray(reference[..., ::-1])
            print("WEB_REFERENCE_PANEL_FALLBACK: raw_camera.png missing; showing policy image.",
                  flush=True)
    alignment_config_path = pathlib.Path(args.alignment_robot_config).expanduser()
    alignment_config = yaml.safe_load(alignment_config_path.read_text())
    alignment_crop = alignment_config.get("camera_crop")
    tag_detector = None
    if monitored_tag_ids:
        tag_config_path = ROOT_DIR / "calibration" / "shared_tags" / "aruco_config.yaml"
        tag_config = yaml.safe_load(tag_config_path.read_text())
        tag_dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, tag_config["aruco_dict"]["predefined"]))
        tag_detector = cv2.aruco.ArucoDetector(tag_dictionary, cv2.aruco.DetectorParameters())
    motion_monitor = None
    if args.live_tcp_monitor:
        try:
            motion_monitor = LiveTcpMotionMonitor(
                pathlib.Path(args.motion_tag_map).expanduser().resolve(),
                pathlib.Path(args.motion_intrinsics).expanduser().resolve(),
                pathlib.Path(args.motion_camera_tcp_geometry).expanduser().resolve(),
                args.live_translation_warning_m, args.live_rotation_warning_deg,
                allow_candidate_geometry=args.allow_candidate_motion_tcp)
        except Exception as exc:
            raise RuntimeError(f"cannot start live TCP monitor: {exc}") from exc
    arm_ik_monitor = None
    if args.live_arm_ik_monitor:
        try:
            arm_ik_monitor = LiveArmIkMonitor(reference_metadata, alignment_config_path)
        except Exception as exc:
            raise RuntimeError(f"cannot start live arm IK monitor: {exc}") from exc
    recording_monitor = None
    if args.recording_capture_priority and (motion_monitor is not None or arm_ik_monitor is not None):
        recording_monitor = AsyncRecordingMonitor(
            motion_monitor, arm_ik_monitor, args.recording_monitor_update_hz)
        recording_monitor.start()

    camera, resolved_device = open_uvc(
        args.camera_device, args.width, args.height, args.fps, args.pixel_format)
    metadata_device = infer_uvc_metadata_device(
        resolved_device, args.uvc_metadata_device)
    window_name = "Handheld UMI preview (r=start, s=stop, x=discard, b=save reference, q=quit)"
    # Do not call cv2.namedWindow explicitly.  This AM_UMI Qt build blocks in
    # namedWindow, while the same desktop successfully creates the window via
    # the implicit creation performed by cv2.imshow below.
    imu = None
    if not args.no_imu:
        imu = ImuChunkReader(args.imu_port, args.imu_baud,
                             read_timeout_s=args.imu_read_timeout_ms / 1000.0)
        imu.start()
        time.sleep(0.15)
        if imu.error is not None:
            camera.release()
            raise RuntimeError(f"cannot start IMU reader: {imu.error}")

    writer = stream = None
    uvc_metadata_capture = None
    uvc_metadata_status = {"captured": False, "reason": "not_requested"}
    csv_file = None
    csv_writer = None
    recording = args.no_preview
    recording_cancelled = False
    stop_deadline = None
    start_ns = None
    stop_ns = None
    frame_count = 0
    first_frame_ns = None
    last_frame_ns = None
    saved_reference = False
    preview_first_frame_reported = False
    web_preview = None
    web_preview_last_publish = 0.0
    live_reference = None
    display = None
    tag_visible: dict[int, bool] = {}
    last_guide_stage = None
    live_motion = None
    live_arm_ik = None
    trial_count = 0
    trial_start_ns = None

    print("Camera:", args.camera_device, "->", resolved_device)
    if metadata_device is not None:
        print("UVC payload metadata:", metadata_device,
              "(will capture PTS/SCR during recording)")
    else:
        print("UVC payload metadata: disabled/unavailable; using raw host receive timestamps")
    print("IMU: disabled (--no-imu; visual-only recording)" if args.no_imu else
          f"IMU: {args.imu_port} @ {args.imu_baud}")
    print("alignment crop:", alignment_crop)
    if reference_metadata is not None:
        print("reference:", args.reference_dir,
              "created", reference_metadata.get("created_at", "unknown"))
    if motion_monitor is not None:
        if args.live_translation_warning_m > 0 or args.live_rotation_warning_deg > 0:
            print("LIVE_TCP_MONITOR_READY: optional relative-motion guide at "
                  f"{args.live_translation_warning_m * 100:.0f} cm / "
                  f"{args.live_rotation_warning_deg:.0f} deg")
        else:
            print("LIVE_TCP_MONITOR_READY: displaying relative motion only; no arbitrary distance limit")
    if arm_ik_monitor is not None:
        print("LIVE_ARM_IK_MONITOR_READY: read-only preview prediction; no serial / torque / robot command")
        print("预览阶段即显示 IK：以首次有效 Tag 定位为临时起点；按 R 后重设为录制起点。")
    if not args.no_preview:
        if args.web_preview:
            web_preview = WebPreview(args.web_preview_port, reference_raw_bgr)
            print("WEB_PREVIEW_READY:", web_preview.url, flush=True)
            if reference_raw_bgr is not None:
                print("WEB_REFERENCE_PANEL_READY: live raw view is shown beside the saved raw reference photo.",
                      flush=True)
            print("在浏览器打开该地址；T=试走/重置，R=正式录制，S=停止，Q=退出。")
        else:
            print("Click the preview window. r=start recording; s=stop/save; "
                  "x=discard current recording; b=save reference; q=exit without a recording.")
    else:
        print("--no-preview: recording starts immediately")

    try:
        while True:
            ok, frame_bgr = camera.read()
            receive_ns = time.monotonic_ns()
            if not ok:
                raise RuntimeError("UVC camera stopped returning frames")
            if frame_bgr.shape[:2] != (args.height, args.width):
                raise RuntimeError(
                    f"camera negotiated {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, "
                    f"expected {args.width}x{args.height}")

            if recording and writer is None:
                start_ns = receive_ns
                if recording_monitor is not None:
                    recording_monitor.reset_origin()
                    live_motion = None
                    live_arm_ik = None
                elif motion_monitor is not None:
                    motion_monitor.start_tcp = None
                    live_motion = None
                if recording_monitor is None and arm_ik_monitor is not None:
                    arm_ik_monitor.q_previous = arm_ik_monitor.q_start.copy()
                    live_arm_ik = None
                writer, stream = make_writer(output_dir / "raw_video.mp4",
                                             args.width, args.height, args.fps,
                                             args.video_encoder_preset)
                csv_file = (output_dir / "frame_timestamps.csv").open("w", newline="", encoding="utf-8")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["frame_index", "receive_monotonic_ns", "relative_s"])
                if metadata_device is not None:
                    try:
                        uvc_metadata_capture = UvcMetadataCapture(metadata_device, output_dir)
                        uvc_metadata_capture.start()
                        print("UVC_METADATA_CAPTURE_STARTED", flush=True)
                    except Exception as exc:
                        uvc_metadata_capture = None
                        uvc_metadata_status = {"captured": False, "reason": str(exc),
                                               "device": metadata_device}
                        print("WARNING: UVC payload metadata unavailable; continuing with "
                              f"host receive timestamps only: {exc}", flush=True)
                print("RECORDING_STARTED", flush=True)
                if guide is not None:
                    print(f"{guide_name} 时间轴已开始（共 {guide[-1][0]:.0f} 秒）。", flush=True)
                    if args.guided_tcp_axis_validation:
                        print("[动作总览] 请先阅读；程序会给你前 8 秒准备，8 秒后才开始第一项平移。", flush=True)
                        for _, guide_text in guide:
                            print(f"[动作总览] {guide_text}", flush=True)

            if writer is not None and stop_deadline is None:
                relative_s = (receive_ns - start_ns) / 1e9
                write_frame(stream, writer, frame_bgr, relative_s)
                csv_writer.writerow([frame_count, receive_ns, f"{relative_s:.9f}"])
                frame_count += 1
                first_frame_ns = receive_ns if first_frame_ns is None else first_frame_ns
                last_frame_ns = receive_ns
                if recording_monitor is not None:
                    # Non-blocking: a full one-frame queue drops this preview
                    # input rather than ever slowing camera dequeue/encoding.
                    recording_monitor.submit(frame_bgr)
                if guide is not None:
                    guide_stage, guide_text = timed_guide(relative_s, guide)
                    if guide_stage != last_guide_stage:
                        print(f"[时间轴] {guide_text}", flush=True)
                        last_guide_stage = guide_stage
                    if web_preview is not None:
                        web_preview.set_status(
                            f"录制中 {relative_s:05.1f}s / {guide[-1][0]:.0f}s｜{guide_text}")
                if args.max_record_seconds > 0 and relative_s >= args.max_record_seconds:
                    print("Maximum recording duration reached; stopping.", flush=True)
                    stop_deadline = time.monotonic() + args.imu_tail_seconds
                    stop_ns = receive_ns

            if not args.no_preview:
                if not preview_first_frame_reported:
                    print("PREVIEW_FIRST_FRAME_OK; displaying preview now.", flush=True)
                    preview_first_frame_reported = True
                # The browser needs only the newest preview image.  Building
                # a full-resolution multi-Tag panel on every 30 FPS capture
                # frame made the preview lag behind the live camera.
                preview_hz = (args.web_recording_preview_update_hz
                              if writer is not None and
                              args.web_recording_preview_update_hz is not None
                              else args.web_preview_update_hz)
                refresh_preview = (web_preview is None or
                                   time.monotonic() - web_preview_last_publish >=
                                   1.0 / preview_hz)
                if refresh_preview:
                    label = "PREVIEW - T=trial/restart, R=formal start, Q=quit"
                    if trial_start_ns is not None:
                        trial_s = (receive_ns - trial_start_ns) / 1e9
                        label = (f"TRIAL {trial_count} {trial_s:.1f}s - "
                                 "T=restart trial, R=formal start")
                    if arm_ik_monitor is not None:
                        label = "PREVIEW IK: relative to first Tag pose; R resets origin"
                    if writer is not None and stop_deadline is None:
                        label = f"RECORDING {frame_count} frames - S=save, X=discard"
                    elif stop_deadline is not None:
                        label = "STOPPED - recording IMU tail"
                    capture_priority = writer is not None and args.recording_capture_priority
                    if capture_priority:
                        # Formal video timing matters more than an operator
                        # overlay.  PnP/Tag detection and IK are performed by
                        # a low-rate dropping worker, never in this loop.
                        if recording_monitor is not None:
                            live_motion, live_arm_ik = recording_monitor.snapshot()
                            label += (" | CAPTURE PRIORITY: background IK "
                                      f"{args.recording_monitor_update_hz:g} Hz")
                        else:
                            live_motion, live_arm_ik = None, None
                            label += " | CAPTURE PRIORITY: monitor unavailable"
                        display, tag_visible = make_recording_preview(
                            frame_bgr, alignment_crop, None, [], None, label,
                            live_motion, live_arm_ik)
                    else:
                        # The policy crop is compared to the reference, but the
                        # web preview remains the *full raw view* so a fixed
                        # world Tag may stay outside the square policy crop.
                        live_reference = preprocess_uvc_bgr(frame_bgr, camera_crop=alignment_crop)
                        metrics = (compare_policy_images(reference, live_reference)
                                   if reference is not None else None)
                        if motion_monitor is not None and stop_deadline is None:
                            live_motion = motion_monitor.update(frame_bgr)
                        if arm_ik_monitor is not None and stop_deadline is None:
                            live_arm_ik = arm_ik_monitor.update(
                                live_motion, motion_monitor.start_tcp if motion_monitor is not None else None)
                        display, tag_visible = make_recording_preview(
                            frame_bgr, alignment_crop, tag_detector, monitored_tag_ids,
                            metrics, label, live_motion, live_arm_ik)
                if web_preview is not None:
                    if refresh_preview:
                        if writer is not None and args.recording_capture_priority:
                            if live_arm_ik is not None and live_arm_ik.get("ready"):
                                web_preview.set_status(
                                    "正式视频采集优先｜后台 IK=" +
                                    ("可能可达" if live_arm_ik["reachable"] else "可能不可达") +
                                    f"｜最小余量 {live_arm_ik['joint_margin_deg']:.1f}°｜S 停止录制")
                            else:
                                web_preview.set_status(
                                    "正式视频采集优先：后台 Tag/TCP/IK 正在等待有效世界 Tag；S 停止录制。")
                        elif tag_visible and writer is None:
                            detected = [str(tag_id) for tag_id, present in tag_visible.items() if present]
                            missing = [str(tag_id) for tag_id, present in tag_visible.items() if not present]
                            prefix = (f"试走第 {trial_count} 次（不保存文件；T 重置、R 正式录制）｜"
                                      if trial_start_ns is not None else "")
                            status = (
                                prefix + f"已检测到 Tag: {','.join(detected) or '无'}；"
                                f"未检测到: {','.join(missing) or '无'}")
                            if live_motion is not None and live_motion.get("started"):
                                status += (f"｜TCP {live_motion['distance_m'] * 100:.1f} cm / "
                                           f"{live_motion['rotation_deg']:.1f}°")
                            web_preview.set_status(status)
                        elif live_motion is not None and live_motion.get("started"):
                            status = (f"TCP 相对起点：{live_motion['distance_m'] * 100:.1f} cm，"
                                      f"转角：{live_motion['rotation_deg']:.1f}°")
                            if live_arm_ik is not None and live_arm_ik.get("ready"):
                                status += ("；IK=" + ("可能可达" if live_arm_ik["reachable"] else "可能不可达") +
                                           f"（{live_arm_ik['position_error_mm']:.1f}mm / "
                                           f"{live_arm_ik['rotation_error_deg']:.1f}°）")
                            web_preview.set_status(status)
                        web_preview.publish(display)
                        web_preview_last_publish = time.monotonic()
                    key = web_preview.get_command()
                else:
                    cv2.imshow(window_name, display)
                    key = chr(cv2.waitKey(1) & 0xFF)
                if key == "b":
                    if args.save_reference_dir is None:
                        print("Set --save-reference-dir before pressing b.")
                    elif saved_reference:
                        print("Reference was already saved in this run.")
                    else:
                        metadata = {
                            "created_at": dt.datetime.now().astimezone().isoformat(),
                            "purpose": "handheld UMI recording start-view reference",
                            "camera_device": args.camera_device,
                            "camera_settings": {"width": args.width, "height": args.height,
                                                "fps": args.fps, "pixel_format": args.pixel_format},
                            "preprocessing": "AM2Pro camera crop + RGB conversion + gripper mask",
                            "camera_crop": alignment_crop,
                        }
                        write_reference(args.save_reference_dir, live_reference, metadata)
                        print("HANDHELD_VIEW_REFERENCE_SAVED:", args.save_reference_dir)
                        saved_reference = True
                elif key == "t" and writer is None:
                    trial_count += 1
                    trial_start_ns = receive_ns
                    if motion_monitor is not None:
                        motion_monitor.start_tcp = None
                        live_motion = None
                    if arm_ik_monitor is not None:
                        arm_ik_monitor.q_previous = arm_ik_monitor.q_start.copy()
                        live_arm_ik = None
                    print(f"TRIAL_ROUTE_STARTED: {trial_count}；不写入视频。再次按 T 会丢弃本次试走并重置起点；按 R 才开始正式保存。",
                          flush=True)
                    if web_preview is not None:
                        web_preview.set_status(
                            f"试走第 {trial_count} 次已开始：不保存文件；再次按 T 重置，按 R 正式录制。")
                elif key == "r" and writer is None:
                    trial_start_ns = None
                    recording = True
                elif key == "t":  # formal recording is immutable by design
                    print("TRIAL_ROUTE_IGNORED: 正式视频已经开始；请先按 S 结束，不能在同一 session 中重置。",
                          flush=True)
                    if web_preview is not None:
                        web_preview.set_status("正式视频录制中；T 不会重置。请按 S 结束本条。")
                elif key == "s" and writer is not None and stop_deadline is None:
                    stop_ns = receive_ns
                    stop_deadline = time.monotonic() + args.imu_tail_seconds
                    print("VIDEO_STOPPED; recording IMU tail ...", flush=True)
                elif key == "x" and writer is not None:
                    # Explicitly discard only this fresh session directory.
                    # Cleanup happens after muxer/camera shutdown so no file
                    # handle remains open.  A later identical command can
                    # therefore reuse its original output path.
                    recording_cancelled = True
                    print("RECORDING_CANCEL_REQUESTED: current session will be discarded; no dataset will be saved.",
                          flush=True)
                    break
                elif key == "q":
                    if writer is None:
                        print("Exited before recording; removing empty output directory.")
                        output_dir.rmdir()
                        return
                    if stop_deadline is None:
                        stop_ns = receive_ns
                        stop_deadline = time.monotonic() + args.imu_tail_seconds
                        print("VIDEO_STOPPED; recording IMU tail ...", flush=True)

            if stop_deadline is not None and time.monotonic() >= stop_deadline:
                break
    finally:
        # Finish the H.264 muxer before closing the independent V4L2 decoder.
        # Explicit ordering avoids relying on PyAV object destructors during
        # interpreter shutdown.
        if csv_file is not None:
            csv_file.close()
        close_writer(stream, writer)
        if uvc_metadata_capture is not None:
            try:
                uvc_metadata_status = uvc_metadata_capture.stop()
            except Exception as exc:
                uvc_metadata_status = {"captured": False, "reason": str(exc),
                                       "device": metadata_device}
        if imu is not None:
            imu.stop()
        if recording_monitor is not None:
            recording_monitor.stop()
        camera.release()
        if web_preview is not None:
            web_preview.stop()
        if not args.no_preview and web_preview is None:
            cv2.destroyAllWindows()

    if recording_cancelled:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print("RECORDING_CANCELLED: output directory removed; rerun the same command to record again.")
        return
    if writer is None or start_ns is None or stop_ns is None:
        raise RuntimeError("no video recording was saved")
    if imu is not None and imu.error is not None:
        raise RuntimeError(f"IMU reader failed during recording: {imu.error}")

    if imu is None:
        imu_counts = (0, 0, False)
    else:
        chunks = imu.snapshot()
        imu_counts = save_imu_raw(
            output_dir / "imu_raw.npz", chunks, args.imu_baud,
            {"recording_start_monotonic_ns": start_ns,
             "recording_stop_monotonic_ns": stop_ns,
             "imu_timestamp_method": "auto_detect_jy901b_0x50_v1"})
    duration_s = (last_frame_ns - first_frame_ns) / 1e9 if frame_count > 1 else 0.0
    measured_fps = (frame_count - 1) / duration_s if duration_s > 0 else 0.0
    if uvc_metadata_status.get("captured"):
        try:
            timing_report = align_session_frame_timestamps(output_dir)
            uvc_metadata_status["timing_alignment"] = timing_report
            print("UVC_PAYLOAD_TIME_ALIGNMENT_OK")
            print("source_frame_events:", timing_report["source_frame_events"],
                  "video_frames:", timing_report["video_frames"])
            print("PTS_clock_hz:", f"{timing_report['pts_clock_hz']:.1f}",
                  "p95_source_vs_decode_ms:",
                  f"{timing_report['source_minus_decode_receive_p95_abs_ms']:.3f}")
        except Exception as exc:
            uvc_metadata_status["timing_alignment_error"] = str(exc)
            print("WARNING: UVC raw metadata was saved but frame alignment failed:", exc)
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "kind": "handheld_umi_raw_session",
        "camera": {"requested_device": args.camera_device, "resolved_device": resolved_device,
                   "width": args.width, "height": args.height, "requested_fps": args.fps,
                   "pixel_format": args.pixel_format, "frames": frame_count,
                   "measured_receive_fps": measured_fps},
        "imu": {"enabled": not args.no_imu,
                "port": None if args.no_imu else args.imu_port, "baud": args.imu_baud,
                "accel_samples": imu_counts[0], "gyro_samples": imu_counts[1],
                "read_timeout_ms": args.imu_read_timeout_ms,
                "timestamp_method": ("not_recorded" if args.no_imu else
                                     "jy901b_0x50_device_clock_mapped_to_host_v1"
                                     if imu_counts[2]
                                     else "continuous_uart_8n1_packet_center_v1"),
                "accel_calibration": "calibration/handheld_gripper_camera/imu/jy901b_accel_v1.json"},
        "timing": {"clock": "host monotonic", "recording_start_monotonic_ns": start_ns,
                   "recording_stop_monotonic_ns": stop_ns,
                   "imu_tail_seconds": args.imu_tail_seconds,
                   "camera_imu_latency_calibrated": False,
                   "uvc_payload_timing": uvc_metadata_status},
        "files": {"video": "raw_video.mp4", "frame_timestamps": "frame_timestamps.csv",
                  "imu_raw": None if args.no_imu else "imu_raw.npz",
                  "uvc_payload_headers": ("raw_uvc_payload_headers.bin"
                                          if uvc_metadata_status.get("captured") else None),
                  "uvc_source_frame_timestamps": (
                      "frame_timestamps_uvc_source.csv"
                      if uvc_metadata_status.get("timing_alignment") else None)},
        "reference": str(args.reference_dir) if args.reference_dir else None,
    }
    save_metadata(output_dir / "metadata.json", metadata)
    print("HANDHELD_UMI_SESSION_SAVED")
    print("output_dir:", output_dir)
    print(f"frames: {frame_count}; measured_receive_fps: {measured_fps:.2f}")
    print("IMU samples: not recorded (--no-imu)" if args.no_imu else
          f"IMU samples: accel={imu_counts[0]}, gyro={imu_counts[1]}")
    if measured_fps < args.fps * 0.8:
        print("WARNING: measured camera FPS is far below requested FPS. Do not run formal "
              "SLAM until capture mode / SLAM configuration are made consistent.")
    if uvc_metadata_status.get("timing_alignment"):
        print("This session retains raw host timestamps and also has a derived UVC "
              "PTS/SCR source-clock timeline. It is not yet a final hardware-"
              "synchronized camera-IMU SLAM input.")
    else:
        print("This session contains raw synchronized host-clock data; it is not yet a "
              "camera-IMU extrinsic/latency-calibrated SLAM input.")


if __name__ == "__main__":
    main()
