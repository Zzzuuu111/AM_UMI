#!/usr/bin/env python3
"""Read-only ArUco visibility/size check on a video or live camera.

Detection does not need the marker's physical size. Pose estimation does:
after changing a printed marker, update its size in aruco_config.yaml and
rebuild any world-tag map that uses the old print.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "calibration/shared_tags/aruco_config.yaml"


def parse_ids(value: str) -> list[int]:
    ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ids or any(marker_id < 0 for marker_id in ids):
        raise argparse.ArgumentTypeError("--ids 需要逗号分隔的非负编号，例如 13,14")
    return sorted(set(ids))


def longest_missing_s(times: list[float], found: list[bool], frame_period_s: float) -> float:
    longest = 0.0
    start = None
    for timestamp, visible in zip(times, found):
        if not visible and start is None:
            start = timestamp
        elif visible and start is not None:
            longest = max(longest, timestamp - start)
            start = None
    if start is not None:
        longest = max(longest, times[-1] - start + frame_period_s)
    return longest


def build_detector(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dictionary_name = config["aruco_dict"]["predefined"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                if hasattr(cv2.aruco, "ArucoDetector") else None)
    return config, dictionary_name, dictionary, parameters, detector


def detect(gray, dictionary, parameters, detector):
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    found = {}
    if ids is not None:
        for marker_id, points in zip(ids.flatten(), corners):
            points = points.reshape(4, 2)
            edge_lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
            found[int(marker_id)] = {
                "edge_px": float(edge_lengths.mean()),
                "corners": points,
            }
    return found


def draw_preview(image, detected, target_ids, counts, analyzed_frames, preview_width):
    view = image.copy()
    panel_height = 52 + 34 * len(target_ids)
    overlay = view.copy()
    cv2.rectangle(overlay, (8, 8), (410, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, view, 0.35, 0, view)
    cv2.putText(view, "ArUco live check  |  q / Esc: stop", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    for row, marker_id in enumerate(target_ids):
        item = detected.get(marker_id)
        color = (40, 220, 40) if item is not None else (40, 40, 230)
        status = f"ID {marker_id}: {item['edge_px']:.0f} px" if item is not None else f"ID {marker_id}: not found"
        status += f"  ({counts[marker_id]}/{analyzed_frames})"
        cv2.putText(view, status, (20, 70 + 34 * row),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
        if item is not None:
            polygon = np.rint(item["corners"]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(view, [polygon], True, color, 4)
            x, y = polygon[0, 0]
            cv2.putText(view, str(marker_id), (int(x), max(25, int(y) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    if view.shape[1] > preview_width:
        scale = preview_width / view.shape[1]
        view = cv2.resize(view, (preview_width, round(view.shape[0] * scale)))
    return view


def main() -> None:
    parser = argparse.ArgumentParser(description="只读检查较小 ArUco Tag 的实际识别率")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="已有视频，例如 raw_video.mp4")
    source.add_argument("--camera", help="相机路径，例如 /dev/v4l/by-id/...-video-index0")
    parser.add_argument("--ids", type=parse_ids, default=[13], help="目标 Tag ID，默认 13")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="实时相机采样秒数；加 --preview 时可设 0，按 q 结束")
    parser.add_argument("--stride", type=int, default=1, help="每隔多少帧分析一帧，默认 1")
    parser.add_argument("--preview", action="store_true",
                        help="显示实时检测窗口，绿色=识别到，红色=本帧未识别")
    parser.add_argument("--preview-width", type=int, default=1280,
                        help="预览窗口最大宽度，默认 1280 像素")
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--printed-size-mm", type=float,
                        help="本次实物 Tag 黑色方形的边长；仅记录，不改变检测器")
    parser.add_argument("--output", type=Path, help="可选：另存 JSON 报告")
    args = parser.parse_args()
    if args.duration < 0 or args.stride < 1 or args.preview_width < 320:
        parser.error("--duration 不能小于 0；--stride 不小于 1；--preview-width 不小于 320")
    if min(args.camera_width, args.camera_height, args.camera_fps) <= 0:
        parser.error("相机分辨率和帧率必须大于 0")
    if args.camera is not None and args.duration == 0 and not args.preview:
        parser.error("实时相机 --duration 0 需要 --preview，以便按 q 退出")
    if args.preview and args.stride != 1:
        parser.error("实时预览请使用 --stride 1，以免跳帧显示旧状态")
    if args.printed_size_mm is not None and args.printed_size_mm <= 0:
        parser.error("--printed-size-mm 必须大于 0")
    if args.video is not None and not args.video.is_file():
        parser.error(f"视频不存在：{args.video}")
    if not args.config.is_file():
        parser.error(f"配置不存在：{args.config}")

    config, dictionary_name, dictionary, parameters, detector = build_detector(args.config)
    if args.camera is not None:
        capture = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
        source_name = args.camera
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        capture = cv2.VideoCapture(str(args.video))
        source_name = str(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频或相机：{source_name}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_period_s = args.stride / fps
    time_s: list[float] = []
    visible: dict[int, list[bool]] = {marker_id: [] for marker_id in args.ids}
    edge_px: dict[int, list[float]] = {marker_id: [] for marker_id in args.ids}
    any_visible: list[bool] = []
    counts = {marker_id: 0 for marker_id in args.ids}
    frame_index = 0
    start = time.monotonic()
    first_capture_time = None
    width = height = None
    window_name = "AM_UMI Tag 13-17 check"
    try:
        if args.preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            if args.camera is not None and args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            ok, image = capture.read()
            if not ok:
                break
            if frame_index % args.stride:
                frame_index += 1
                continue
            height, width = image.shape[:2]
            if args.camera is not None:
                captured_at = time.monotonic()
                if first_capture_time is None:
                    first_capture_time = captured_at
                timestamp = captured_at - first_capture_time
            else:
                timestamp = frame_index / fps
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = detect(gray, dictionary, parameters, detector)
            time_s.append(timestamp)
            any_visible.append(any(marker_id in detected for marker_id in args.ids))
            for marker_id in args.ids:
                is_visible = marker_id in detected
                visible[marker_id].append(is_visible)
                if is_visible:
                    counts[marker_id] += 1
                    edge_px[marker_id].append(detected[marker_id]["edge_px"])
            frame_index += 1
            if args.preview:
                view = draw_preview(image, detected, args.ids, counts, len(time_s),
                                    args.preview_width)
                cv2.imshow(window_name, view)
                wait_ms = max(1, round(1000 / fps)) if args.video is not None else 1
                key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), 27):
                    break
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        capture.release()
        if args.preview:
            cv2.destroyWindow(window_name)
    if not time_s:
        raise RuntimeError("没有读到可分析的视频帧")

    stats = {}
    configured_sizes = config.get("marker_size_map", {})
    for marker_id in args.ids:
        edges = edge_px[marker_id]
        stats[str(marker_id)] = {
            "configured_size_mm": float(configured_sizes.get(marker_id, configured_sizes.get("default", 0))) * 1000,
            "printed_size_mm": args.printed_size_mm,
            "detected_frames": sum(visible[marker_id]),
            "detection_ratio": sum(visible[marker_id]) / len(time_s),
            "longest_missing_s": longest_missing_s(time_s, visible[marker_id], frame_period_s),
            "edge_px_median": statistics.median(edges) if edges else None,
            "edge_px_p10": float(np.percentile(edges, 10)) if edges else None,
            "edge_px_min": min(edges) if edges else None,
        }
    report = {
        "schema": "am_umi_aruco_tag_size_check_v1",
        "safety": "read-only video/camera check; no robot connection or command",
        "source": source_name,
        "dictionary": dictionary_name,
        "resolution": [width, height],
        "source_fps": fps,
        "stride": args.stride,
        "sampled_frames": len(time_s),
        "sampled_duration_s": time_s[-1] - time_s[0],
        "target_ids": args.ids,
        "any_target_detection_ratio": sum(any_visible) / len(time_s),
        "longest_no_target_s": longest_missing_s(time_s, any_visible, frame_period_s),
        "markers": stats,
        "interpretation": "Detection is independent of physical size in the config. For a smaller printed tag, re-test at actual camera distance and update marker_size_map before pose estimation.",
    }
    if args.printed_size_mm is not None:
        changed_ids = [marker_id for marker_id in args.ids
                       if abs(stats[str(marker_id)]["configured_size_mm"] -
                              args.printed_size_mm) > 0.5]
        if changed_ids:
            report["size_config_warning"] = (
                f"Tag {changed_ids} 的打印边长与配置不一致；本检测率有效，"
                "但使用旧尺寸估计的三维位姿/尺度会错误。")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
        print(f"报告已保存：{args.output}")


if __name__ == "__main__":
    main()
