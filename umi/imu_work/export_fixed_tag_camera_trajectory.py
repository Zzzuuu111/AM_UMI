"""Export a metric camera trajectory from one fixed Tag or a fixed multi-Tag map.

Use this for an EMEET hand-held recording only when the selected tabletop tag
is intentionally visible throughout the segment.  The output uses the same
CSV columns as the local ORB-SLAM3 runner, but its world frame is explicitly
the fixed Tag frame -- it is not an atlas/relocalization trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import av
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import load_preferred_frame_times, preferred_frame_timestamps_path


def load_intrinsics(path: pathlib.Path):
    data = json.loads(path.read_text(encoding="utf-8"))["intrinsics"]
    camera = np.array(
        [[data["focal_length"], 0.0, data["principal_pt_x"]],
         [0.0, data["focal_length"], data["principal_pt_y"]],
         [0.0, 0.0, 1.0]], dtype=float)
    distortion = np.array([data[f"radial_distortion_{i}"] for i in range(1, 5)],
                          dtype=float).reshape(4, 1)
    return camera, distortion


def pose_from_marker(corner: np.ndarray, size_m: float, camera: np.ndarray,
                     distortion: np.ndarray) -> np.ndarray | None:
    """Return the PnP transform that maps this Tag frame into the camera frame."""
    half = size_m / 2.0
    object_points = np.array(
        [[-half, half, 0.0], [half, half, 0.0],
         [half, -half, 0.0], [-half, -half, 0.0]], dtype=float)
    image_points = np.asarray(corner, dtype=float).reshape(4, 1, 2)
    image_points = cv2.fisheye.undistortPoints(image_points, camera, distortion, P=camera)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not np.isfinite(rvec).all() or tvec[2, 0] <= 0:
        return None
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(rvec.ravel()).as_matrix()
    out[:3, 3] = tvec.reshape(3)
    return out


def average_camera_pose(candidates: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    """Average several world<-camera estimates from co-visible fixed Tags."""
    translations = np.stack([item[:3, 3] for item in candidates])
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in candidates]))
    average_rotation = rotations.mean()
    average_translation = translations.mean(axis=0)
    translation_spread_mm = float(np.max(np.linalg.norm(
        translations - average_translation, axis=1)) * 1000.0)
    rotation_spread_deg = float(np.max((average_rotation.inv() * rotations).magnitude()) * 180.0 / np.pi)
    pose = np.eye(4)
    pose[:3, :3] = average_rotation.as_matrix()
    pose[:3, 3] = average_translation
    return pose, translation_spread_mm, rotation_spread_deg


def marker_area_px2(corner: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(corner, dtype=np.float32).reshape(-1, 2))))


def robust_average_camera_pose(candidates: list[tuple[int, np.ndarray, float]]) -> tuple[np.ndarray, int, float, float]:
    """Fuse agreeing Tags, rejecting a blurred/edge-of-frame PnP outlier.

    The largest projected Tag is normally the most precise PnP estimate.  It
    defines a conservative same-frame gate; only agreeing Tags are averaged.
    A single visible Tag remains valid, which is essential for hand occlusion.
    """
    _, anchor, _ = max(candidates, key=lambda item: item[2])
    anchor_rotation = Rotation.from_matrix(anchor[:3, :3])
    inliers = []
    for _, pose, _ in candidates:
        translation_error_m = np.linalg.norm(pose[:3, 3] - anchor[:3, 3])
        rotation_error_deg = float((anchor_rotation.inv() * Rotation.from_matrix(
            pose[:3, :3])).magnitude() * 180.0 / np.pi)
        if translation_error_m <= 0.025 and rotation_error_deg <= 3.0:
            inliers.append(pose)
    # The anchor necessarily passes its own zero-error check, but retain this
    # guard so a future gate change cannot emit an empty trajectory frame.
    if not inliers:
        inliers = [anchor]
    pose, spread_mm, spread_deg = average_camera_pose(inliers)
    return pose, len(inliers), spread_mm, spread_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="从固定桌面 Tag 直接导出米制相机轨迹（离线）")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--tag-map", default=None,
                        help="可选的固定桌面多 Tag 地图 JSON；提供后任一已映射 Tag 可用于轨迹")
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--trajectory-name", default="camera_trajectory_fixed_tag.csv")
    parser.add_argument("--min-visible-ratio", type=float, default=0.95,
                        help="低于该比例则导出后明确报错；默认要求 Tag 可见至少 95%%")
    args = parser.parse_args()
    if not 0.0 < args.min_visible_ratio <= 1.0:
        parser.error("--min-visible-ratio 必须在 (0, 1] 内")
    session = pathlib.Path(args.session_dir).expanduser().resolve()
    output_name = pathlib.Path(args.trajectory_name)
    if output_name.name != args.trajectory_name or output_name.suffix != ".csv":
        parser.error("--trajectory-name 必须是 session-dir 内的单个 .csv 文件名")
    output = session / output_name
    video = session / "raw_video.mp4"
    timestamps_path = preferred_frame_timestamps_path(session)
    if output.exists():
        raise SystemExit(f"为保护已有轨迹，拒绝覆盖：{output}")
    if not video.is_file() or not timestamps_path.is_file():
        raise SystemExit("session-dir 中缺少 raw_video.mp4 或 frame_timestamps.csv")
    timestamps, timestamps_path = load_preferred_frame_times(session)
    print("camera_frame_timestamp_source:", timestamps_path)
    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text(encoding="utf-8"))
    sizes = config["marker_size_map"]
    tag_map_path = None if args.tag_map is None else pathlib.Path(args.tag_map).expanduser().resolve()
    if tag_map_path is None:
        known_tags = {
            args.tag_id: {"size_m": float(sizes.get(args.tag_id, sizes.get(str(args.tag_id), sizes["default"]))),
                          "T_world_tag": np.eye(4)},
        }
        world_frame = f"ArUco Tag {args.tag_id}"
        route = "fixed_tag_direct_formal_candidate"
    else:
        if not tag_map_path.is_file():
            raise SystemExit(f"找不到 --tag-map：{tag_map_path}")
        tag_map = json.loads(tag_map_path.read_text(encoding="utf-8"))
        if tag_map.get("schema") != "am_umi_world_tag_map_v1":
            raise SystemExit("--tag-map 不是 am_umi_world_tag_map_v1 文件")
        known_tags = {}
        for raw_id, item in tag_map.get("tags", {}).items():
            tag_id = int(raw_id)
            transform = np.asarray(item["T_world_tag"], dtype=float)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise SystemExit(f"Tag {tag_id} 的 T_world_tag 无效")
            known_tags[tag_id] = {"size_m": float(item["size_m"]), "T_world_tag": transform}
        if not known_tags:
            raise SystemExit("--tag-map 中没有可用 Tag")
        root_id = int(tag_map["world_tag_id"])
        world_frame = f"multi-Tag tabletop map (root Tag {root_id})"
        route = "fixed_multi_tag_direct_formal_candidate"
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config["aruco_dict"]["predefined"])
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    camera, distortion = load_intrinsics(pathlib.Path(args.intrinsics).expanduser().resolve())
    rows, visible = [], 0
    visible_by_tag = {tag_id: 0 for tag_id in known_tags}
    multi_tag_frames = 0
    max_translation_spread_mm = 0.0
    max_rotation_spread_deg = 0.0
    rejected_candidate_count = 0
    total_candidate_count = 0
    print("FIXED_TAG_CAMERA_TRAJECTORY_EXPORT_STARTED", flush=True)
    if tag_map_path is not None:
        print("multi_tag_map:", tag_map_path)
        print("mapped_tag_ids:", sorted(known_tags))
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index >= len(timestamps):
                break
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            candidates: list[tuple[int, np.ndarray, float]] = []
            if ids is not None:
                seen_in_frame = set()
                for marker_index, raw_id in enumerate(ids.reshape(-1)):
                    tag_id = int(raw_id)
                    if tag_id not in known_tags or tag_id in seen_in_frame:
                        continue
                    seen_in_frame.add(tag_id)
                    tag_to_camera = pose_from_marker(
                        corners[marker_index], known_tags[tag_id]["size_m"], camera, distortion)
                    if tag_to_camera is not None:
                        # T_world_camera = T_world_tag @ inv(T_camera_tag).
                        candidates.append((
                            tag_id,
                            known_tags[tag_id]["T_world_tag"] @ np.linalg.inv(tag_to_camera),
                            marker_area_px2(corners[marker_index]),
                        ))
                        visible_by_tag[tag_id] += 1
            if not candidates:
                rows.append((index, timestamps[index], "1", "true", "false", (0.0,) * 7))
            else:
                visible += 1
                pose, inlier_count, translation_spread_mm, rotation_spread_deg = robust_average_camera_pose(candidates)
                if len(candidates) > 1:
                    multi_tag_frames += 1
                total_candidate_count += len(candidates)
                rejected_candidate_count += len(candidates) - inlier_count
                max_translation_spread_mm = max(max_translation_spread_mm, translation_spread_mm)
                max_rotation_spread_deg = max(max_rotation_spread_deg, rotation_spread_deg)
                rows.append((index, timestamps[index], "2", "false", "false",
                             tuple(float(value) for value in (*pose[:3, 3],
                                                               *Rotation.from_matrix(pose[:3, :3]).as_quat()))))
            if (index + 1) % 750 == 0:
                print(f"已处理 {index + 1} 帧；可定位 {visible} 帧；各 Tag: {visible_by_tag}", flush=True)
    if len(rows) != len(timestamps):
        raise RuntimeError(f"视频帧数 {len(rows)} 与时间戳数 {len(timestamps)} 不一致")
    ratio = visible / len(rows)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame_idx", "timestamp", "state", "is_lost", "is_keyframe",
                         "x", "y", "z", "q_x", "q_y", "q_z", "q_w"])
        for index, timestamp, state, lost, keyframe, values in rows:
            writer.writerow([index, f"{timestamp:.6f}", state, lost, keyframe,
                             *[f"{value:.9f}" for value in values]])
    report = output.with_suffix(".fixed_tag_report.json")
    report.write_text(json.dumps({
        "schema": "am_umi_fixed_tag_camera_trajectory_v1",
        "pipeline_route": route,
        "session": str(session), "trajectory": str(output),
        "world_frame": world_frame,
        "tag_map": None if tag_map_path is None else str(tag_map_path),
        "visible_by_tag": visible_by_tag,
        "multi_tag_frames": multi_tag_frames,
        "max_multi_tag_translation_spread_mm": max_translation_spread_mm,
        "max_multi_tag_rotation_spread_deg": max_rotation_spread_deg,
        "total_tag_pose_candidates": total_candidate_count,
        "rejected_tag_pose_candidates": rejected_candidate_count,
        "rejected_tag_pose_ratio": (0.0 if total_candidate_count == 0
                                     else rejected_candidate_count / total_candidate_count),
        "frame_count": len(rows), "visible_frame_count": visible,
        "visible_ratio": ratio,
        "meets_min_visible_ratio": ratio >= args.min_visible_ratio,
        "note": "Direct metric PnP trajectory. It is valid only where the fixed table Tag is visible; it is not a generic SLAM map.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FIXED_TAG_CAMERA_TRAJECTORY_EXPORT_OK")
    print(f"frames: {len(rows)}; tag_visible: {visible} ({ratio:.3%})")
    print("trajectory:", output)
    print("report:", report)
    if ratio < args.min_visible_ratio:
        raise SystemExit("Tag 可见比例不足；轨迹已保存供诊断，但不要用于正式数据")


if __name__ == "__main__":
    main()
