"""Check whether the fixed scene ArUco tag is usable in a hand-held UMI session.

This is a read-only quality gate for the recorded video.  It does not control
the robot or alter raw_video.mp4 / imu_raw.npz.  The separate JSON report uses
the host-clock frame_timestamps.csv timeline, not the nominal MP4 duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import av
import cv2
import yaml


def load_timestamps(path: pathlib.Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("frame_timestamps.csv is empty")
    return [float(row["relative_s"]) for row in rows]


def longest_missing_interval(times: list[float], detected: list[bool]) -> float:
    """Conservative gap estimate using the timestamps of sampled frames."""
    longest = 0.0
    gap_start = None
    for time_s, found in zip(times, detected):
        if not found and gap_start is None:
            gap_start = time_s
        elif found and gap_start is not None:
            longest = max(longest, time_s - gap_start)
            gap_start = None
    if gap_start is not None:
        longest = max(longest, times[-1] - gap_start)
    return longest


def visible_intervals(times: list[float], detected: list[bool]) -> list[dict]:
    """Return contiguous visible intervals in the sampled host-clock timeline."""
    intervals = []
    start = None
    for time_s, found in zip(times, detected):
        if found and start is None:
            start = time_s
        elif not found and start is not None:
            intervals.append({"start_s": start, "end_s": time_s,
                              "duration_s": time_s - start})
            start = None
    if start is not None:
        intervals.append({"start_s": start, "end_s": times[-1],
                          "duration_s": times[-1] - start})
    return intervals


def main():
    parser = argparse.ArgumentParser(
        description="检查手持 UMI 视频里固定场景 ArUco Tag 的可见率")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument(
        "--aruco-config",
        default="calibration/shared_tags/aruco_config.yaml",
        help="使用的 ArUco 字典和编号配置",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="每隔多少视频帧检查一次；1 最准确，建议首次保持默认",
    )
    parser.add_argument(
        "--target-tag-ids", default=None,
        help="仅将这些固定世界 Tag 计入连续可定位性，例如 13,14,15；不填则沿用任意 Tag",
    )
    parser.add_argument(
        "--report-name", default="tag_visibility_report.json",
        help="写入 session-dir 的独立报告文件名",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride 必须不小于 1")
    if args.target_tag_ids is None:
        target_tag_ids = None
    else:
        try:
            target_tag_ids = {int(item.strip()) for item in args.target_tag_ids.split(",")
                              if item.strip()}
        except ValueError:
            parser.error("--target-tag-ids 必须为逗号分隔的整数，例如 13,14,15")
        if not target_tag_ids or any(tag_id < 0 for tag_id in target_tag_ids):
            parser.error("--target-tag-ids 至少需要一个非负编号")

    session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
    video_path = session_dir / "raw_video.mp4"
    timestamp_path = session_dir / "frame_timestamps.csv"
    if not video_path.is_file() or not timestamp_path.is_file():
        raise RuntimeError("session-dir 中需要 raw_video.mp4 和 frame_timestamps.csv")

    config_path = pathlib.Path(args.aruco_config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dictionary_name = config["aruco_dict"]["predefined"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    # OpenCV 4.7+ exposes the detector as an object; older builds retain the
    # cv2.aruco.detectMarkers function.  AM_UMI currently has the new API.
    detector = (cv2.aruco.ArucoDetector(dictionary, parameters)
                if hasattr(cv2.aruco, "ArucoDetector") else None)

    timestamps = load_timestamps(timestamp_path)
    sampled_times: list[float] = []
    sampled_any_found: list[bool] = []
    sampled_target_found: list[bool] = []
    marker_counts: dict[int, int] = {}
    marker_visible: dict[int, list[bool]] = {}
    decoded_frames = 0
    analyzed_frames = 0

    print("TAG_VISIBILITY_CHECK_STARTED")
    print("video:", video_path)
    print("dictionary:", dictionary_name, "stride:", args.stride)
    print("target_tag_ids:", "any" if target_tag_ids is None else sorted(target_tag_ids))
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            decoded_frames += 1
            if frame_index % args.stride:
                continue
            if frame_index >= len(timestamps):
                raise RuntimeError("视频帧数多于 frame_timestamps.csv，不能可靠检查")
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, dictionary, parameters=parameters)
            ids_found = [] if ids is None else [int(marker_id[0]) for marker_id in ids]
            for marker_id in ids_found:
                marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
            # A marker first seen late in the video must be padded with its
            # preceding absent samples, otherwise its time-gap statistics
            # would be aligned to the start of the session incorrectly.
            for marker_id in ids_found:
                marker_visible.setdefault(marker_id, [False] * analyzed_frames)
            for marker_id, visible in marker_visible.items():
                visible.append(marker_id in ids_found)
            sampled_times.append(timestamps[frame_index])
            sampled_any_found.append(bool(corners))
            sampled_target_found.append(
                bool(corners) if target_tag_ids is None
                else bool(target_tag_ids.intersection(ids_found)))
            analyzed_frames += 1
            if analyzed_frames % 300 == 0:
                print(f"已检查 {analyzed_frames} 帧；当前检测到的 Tag 编号: {sorted(marker_counts)}", flush=True)

    if decoded_frames != len(timestamps):
        raise RuntimeError(
            f"视频 {decoded_frames} 帧、时间戳 {len(timestamps)} 行，二者不一致，停止生成报告")
    if not sampled_times:
        raise RuntimeError("没有可检查的视频帧")

    any_visible_frames = sum(sampled_any_found)
    any_visible_ratio = any_visible_frames / analyzed_frames
    target_visible_frames = sum(sampled_target_found)
    target_visible_ratio = target_visible_frames / analyzed_frames
    marker_stats = {
        str(marker_id): {
            "visible_frames": sum(visible),
            "visible_ratio": sum(visible) / analyzed_frames,
            "longest_sampled_missing_s": longest_missing_interval(sampled_times, visible),
            "visible_intervals_s": visible_intervals(sampled_times, visible),
        }
        for marker_id, visible in sorted(marker_visible.items())
    }
    report = {
        "kind": "handheld_umi_tag_visibility_report",
        "video": video_path.name,
        "timestamp_source": timestamp_path.name,
        "aruco_dictionary": dictionary_name,
        "stride": args.stride,
        "recorded_frames": decoded_frames,
        "analyzed_frames": analyzed_frames,
        "host_duration_s": timestamps[-1],
        "any_tag_visible_frames": any_visible_frames,
        "any_tag_visible_ratio": any_visible_ratio,
        "target_tag_ids": None if target_tag_ids is None else sorted(target_tag_ids),
        "target_tag_visible_frames": target_visible_frames,
        "target_tag_visible_ratio": target_visible_ratio,
        "marker_visible_frames": {str(key): value for key, value in sorted(marker_counts.items())},
        "marker_statistics": marker_stats,
        "longest_sampled_no_tag_s": longest_missing_interval(sampled_times, sampled_any_found),
        "longest_sampled_no_target_tag_s": longest_missing_interval(sampled_times, sampled_target_found),
        "note": "可见率按原始画面检测；相机-IMU 外参求解还需要后续的角速度/视觉旋转对齐。",
    }
    report_path = session_dir / args.report_name
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("TAG_VISIBILITY_CHECK_OK")
    print(f"frames: decoded={decoded_frames}, analyzed={analyzed_frames}")
    print(f"host_duration_s: {timestamps[-1]:.3f}")
    print(f"any_tag_visible_ratio: {any_visible_ratio:.3%}")
    print(f"target_tag_visible_ratio: {target_visible_ratio:.3%}")
    print("marker_visible_frames:", dict(sorted(marker_counts.items())))
    for marker_id, stats in marker_stats.items():
        print("marker", marker_id,
              f"visible_ratio={stats['visible_ratio']:.3%}",
              f"longest_missing_s={stats['longest_sampled_missing_s']:.3f}")
    print(f"longest_sampled_no_tag_s: {report['longest_sampled_no_tag_s']:.3f}")
    if target_tag_ids is not None:
        print(f"longest_sampled_no_target_tag_s: "
              f"{report['longest_sampled_no_target_tag_s']:.3f}")
    print("report:", report_path)


if __name__ == "__main__":
    main()
