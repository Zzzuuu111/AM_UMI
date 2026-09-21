"""Build one rigid tabletop coordinate system from co-visible fixed ArUco tags.

This is deliberately separate from SLAM and IMU calibration.  It estimates the
fixed transforms between world Tags (for example 13, 14, 15) from a short
recording in which all tags are visible together.  The resulting JSON lets a
later trajectory exporter use whichever world Tag is visible in a frame.
"""

from __future__ import annotations

import argparse
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

from imu_work.export_fixed_tag_camera_trajectory import load_intrinsics


def parse_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--tag-ids 应为逗号分隔的整数，例如 13,14,15") from exc
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(item < 0 for item in ids):
        raise argparse.ArgumentTypeError("--tag-ids 至少需要两个互不相同的非负编号")
    return ids


def tag_to_camera(corner: np.ndarray, size_m: float, camera: np.ndarray,
                  distortion: np.ndarray) -> np.ndarray | None:
    half = size_m / 2.0
    object_points = np.array([
        [-half, half, 0.0], [half, half, 0.0],
        [half, -half, 0.0], [-half, -half, 0.0],
    ], dtype=float)
    image_points = np.asarray(corner, dtype=float).reshape(4, 1, 2)
    image_points = cv2.fisheye.undistortPoints(image_points, camera, distortion, P=camera)
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or not np.isfinite(rvec).all() or tvec[2, 0] <= 0:
        return None
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rvec.ravel()).as_matrix()
    transform[:3, 3] = tvec.reshape(3)
    return transform


def average_transform(samples: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    translations = np.stack([item[:3, 3] for item in samples])
    rotations = Rotation.from_matrix(np.stack([item[:3, :3] for item in samples]))
    average_rotation = rotations.mean()
    average_translation = translations.mean(axis=0)
    translation_residual_mm = np.linalg.norm(translations - average_translation, axis=1) * 1000.0
    rotation_residual_deg = (average_rotation.inv() * rotations).magnitude() * 180.0 / np.pi
    transform = np.eye(4)
    transform[:3, :3] = average_rotation.as_matrix()
    transform[:3, 3] = average_translation
    return transform, float(np.median(translation_residual_mm)), float(np.median(rotation_residual_deg))


def main() -> None:
    parser = argparse.ArgumentParser(description="从同屏固定 Tag 建立桌面多 Tag 坐标地图")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--tag-ids", type=parse_ids, default=parse_ids("13,14,15"))
    parser.add_argument("--world-tag-id", type=int, default=13)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--min-pair-frames", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.world_tag_id not in args.tag_ids:
        parser.error("--world-tag-id 必须包含在 --tag-ids 中")
    if args.min_pair_frames < 5:
        parser.error("--min-pair-frames 至少为 5")

    session = pathlib.Path(args.session_dir).expanduser().resolve()
    video = session / "raw_video.mp4"
    output = pathlib.Path(args.out).expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频：{video}")
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有 Tag 地图：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text(encoding="utf-8"))
    sizes = config["marker_size_map"]
    tag_sizes = {
        tag_id: float(sizes.get(tag_id, sizes.get(str(tag_id), sizes["default"])))
        for tag_id in args.tag_ids
    }
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, config["aruco_dict"]["predefined"])
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                if hasattr(cv2.aruco, "ArucoDetector") else None)
    camera, distortion = load_intrinsics(pathlib.Path(args.intrinsics).expanduser().resolve())
    pair_samples: dict[int, list[np.ndarray]] = {
        tag_id: [] for tag_id in args.tag_ids if tag_id != args.world_tag_id
    }
    decoded = 0
    print("WORLD_TAG_MAP_CALIBRATION_STARTED", flush=True)
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            decoded += 1
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
            if ids is None:
                continue
            poses = {}
            for index, raw_id in enumerate(ids.reshape(-1)):
                tag_id = int(raw_id)
                if tag_id in tag_sizes and tag_id not in poses:
                    pose = tag_to_camera(corners[index], tag_sizes[tag_id], camera, distortion)
                    if pose is not None:
                        poses[tag_id] = pose
            if args.world_tag_id not in poses:
                continue
            world_from_camera = np.linalg.inv(poses[args.world_tag_id])
            for tag_id in pair_samples:
                if tag_id in poses:
                    pair_samples[tag_id].append(world_from_camera @ poses[tag_id])
            if decoded % 300 == 0:
                counts = {tag_id: len(samples) for tag_id, samples in pair_samples.items()}
                print(f"已处理 {decoded} 帧；与 Tag {args.world_tag_id} 同屏样本: {counts}", flush=True)

    missing = {tag_id: len(samples) for tag_id, samples in pair_samples.items()
               if len(samples) < args.min_pair_frames}
    if missing:
        raise SystemExit(
            f"同屏样本不足：{missing}。请录制时让 Tag {args.world_tag_id} 与每张其他固定 Tag 同时清晰可见。")

    tags = {
        str(args.world_tag_id): {
            "size_m": tag_sizes[args.world_tag_id],
            "T_world_tag": np.eye(4).tolist(),
            "pair_frames": decoded,
        }
    }
    for tag_id, samples in pair_samples.items():
        transform, median_translation_mm, median_rotation_deg = average_transform(samples)
        tags[str(tag_id)] = {
            "size_m": tag_sizes[tag_id],
            "T_world_tag": transform.tolist(),
            "pair_frames": len(samples),
            "median_translation_residual_mm": median_translation_mm,
            "median_rotation_residual_deg": median_rotation_deg,
        }
    result = {
        "schema": "am_umi_world_tag_map_v1",
        "purpose": "fixed tabletop Tags for multi-Tag visual trajectory recovery",
        "session": str(session),
        "world_tag_id": args.world_tag_id,
        "tag_ids": args.tag_ids,
        "aruco_dictionary": config["aruco_dict"]["predefined"],
        "tags": tags,
        "decoded_frames": decoded,
        "warning": "Do not move any mapped Tag after this file is created. Rebuild the map if a Tag moves.",
    }
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WORLD_TAG_MAP_CALIBRATION_OK")
    print("decoded_frames:", decoded)
    for tag_id, item in tags.items():
        print(f"Tag {tag_id}: pair_frames={item['pair_frames']}")
    print("saved:", output)


if __name__ == "__main__":
    main()
