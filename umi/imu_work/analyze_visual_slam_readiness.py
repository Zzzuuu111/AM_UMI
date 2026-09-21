"""Offline visual-front-end diagnostic for a recorded handheld session."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import av
import cv2
import numpy as np


def frame_times(path: pathlib.Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as f:
        return np.asarray([float(row["relative_s"]) for row in csv.DictReader(f)])


def contiguous_ranges(values: list[float], max_gap: float = 1.6) -> list[list[float]]:
    if not values:
        return []
    result, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value - previous > max_gap:
            result.append([round(start, 3), round(previous, 3)])
            start = value
        previous = value
    result.append([round(start, 3), round(previous, 3)])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="分析手持视频是否适合 ORB-SLAM3 视觉跟踪")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--sample-every", type=int, default=15,
                        help="每隔多少帧分析一次；15 对应约每 0.5 秒")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.sample_every <= 0:
        parser.error("--sample-every 必须为正数")

    session = pathlib.Path(args.session_dir).expanduser().resolve()
    video = session / "raw_video.mp4"
    times_path = session / "frame_timestamps.csv"
    if not video.is_file() or not times_path.is_file():
        raise SystemExit("session-dir 中缺少 raw_video.mp4 或 frame_timestamps.csv")
    out = (pathlib.Path(args.out).expanduser().resolve() if args.out else
           session / "visual_slam_readiness_report.json")
    if out.exists():
        raise SystemExit(f"为保护已有报告，拒绝覆盖：{out}")

    times = frame_times(times_path)
    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8,
                         fastThreshold=7)
    rows, low_feature_times, blurry_times = [], [], []
    print("VISUAL_SLAM_READINESS_ANALYSIS_STARTED", flush=True)
    with av.open(str(video)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index >= len(times):
                break
            if frame_index % args.sample_every:
                continue
            gray = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
            keypoints = orb.detect(gray, None)
            feature_count = len(keypoints)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            row = {"frame": frame_index, "time_s": float(times[frame_index]),
                   "orb_features": feature_count, "laplacian_variance": sharpness}
            rows.append(row)
            if feature_count < 250:
                low_feature_times.append(float(times[frame_index]))
            if sharpness < 20.0:
                blurry_times.append(float(times[frame_index]))
            if len(rows) % 60 == 0:
                print(f"已分析 {len(rows)} 个时间采样点", flush=True)
    if not rows:
        raise RuntimeError("未能从视频读取任何帧")
    feature = np.asarray([x["orb_features"] for x in rows])
    sharpness = np.asarray([x["laplacian_variance"] for x in rows])
    report = {
        "schema": "am_umi_visual_slam_readiness_v1",
        "session": str(session), "sample_count": len(rows),
        "orb_feature_count": {"min": int(feature.min()), "median": float(np.median(feature)),
                              "p10": float(np.quantile(feature, .1)), "max": int(feature.max()),
                              "below_250_ratio": float(np.mean(feature < 250))},
        "sharpness_laplacian_variance": {"min": float(sharpness.min()), "median": float(np.median(sharpness)),
                                           "p10": float(np.quantile(sharpness, .1)), "max": float(sharpness.max()),
                                           "below_20_ratio": float(np.mean(sharpness < 20.0))},
        "low_feature_periods_s": contiguous_ranges(low_feature_times),
        "blur_periods_s": contiguous_ranges(blurry_times),
        "samples": rows,
        "note": "ORB counts and sharpness alone are diagnostic signals, not a SLAM pass/fail guarantee.",
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VISUAL_SLAM_READINESS_ANALYSIS_OK")
    print(f"ORB features: median={np.median(feature):.0f}, p10={np.quantile(feature, .1):.0f}, min={feature.min()}")
    print(f"sharpness: median={np.median(sharpness):.1f}, p10={np.quantile(sharpness, .1):.1f}, min={sharpness.min():.1f}")
    print(f"low-feature samples: {np.mean(feature < 250):.1%}; blurry samples: {np.mean(sharpness < 20):.1%}")
    print("saved:", out)


if __name__ == "__main__":
    main()
