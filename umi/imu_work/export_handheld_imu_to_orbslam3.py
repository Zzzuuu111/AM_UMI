"""Export a recorded EMEET/JY901B session to the JSON consumed by UMI ORB-SLAM3.

The exporter preserves the real camera host-clock timestamps in the ``CORI``
stream and shifts IMU timestamps using the calibrated camera--IMU offset.  A
patched local ORB-SLAM3 reader consumes that CORI stream as per-frame times.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import load_preferred_frame_times, preferred_frame_timestamps_path


def load_frame_times(path: pathlib.Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("frame_timestamps.csv is empty")
    return np.asarray([float(row["relative_s"]) for row in rows], dtype=np.float64)


def keep_first_of_equal_runs(values: np.ndarray) -> np.ndarray:
    """Select the first packet carrying each newly refreshed sensor value."""
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    return np.r_[True, np.any(np.diff(values, axis=0) != 0, axis=1)]


def main():
    parser = argparse.ArgumentParser(description="将 JY901B 原始会话导出为 ORB-SLAM3 IMU JSON")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--camera-imu-calibration", required=True)
    parser.add_argument("--accel-time-calibration", default=None,
                        help="可选：含 accel_time_offset_for_camera_s 的独立加速度延迟标定")
    parser.add_argument("--accel-calibration", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--out", default=None,
                        help="默认写入 session-dir/imu_data.json；已有文件会被保护")
    parser.add_argument("--collapse-repeated-imu", action="store_true",
                        help="诊断选项：每段完全相同的连续数值只保留首包，按真实刷新率导出")
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
    output_path = (pathlib.Path(args.out).expanduser().resolve()
                   if args.out else session_dir / "imu_data.json")
    if output_path.exists():
        raise SystemExit(f"为保护已有导出，拒绝覆盖：{output_path}")
    imu_path = session_dir / "imu_raw.npz"
    timestamp_path = preferred_frame_timestamps_path(session_dir)
    if not imu_path.is_file() or not timestamp_path.is_file():
        raise RuntimeError("session-dir 中需要 imu_raw.npz 与 frame_timestamps.csv")

    camera_imu_path = pathlib.Path(args.camera_imu_calibration).expanduser().resolve()
    camera_imu = json.loads(camera_imu_path.read_text(encoding="utf-8"))
    offset_s = float(camera_imu["imu_time_offset_for_camera_s"])
    accel_time_path = (pathlib.Path(args.accel_time_calibration).expanduser().resolve()
                       if args.accel_time_calibration else camera_imu_path)
    accel_time = json.loads(accel_time_path.read_text(encoding="utf-8"))
    accel_offset_s = float(accel_time.get("accel_time_offset_for_camera_s", offset_s))
    accel_path = pathlib.Path(args.accel_calibration).expanduser().resolve()
    accel_calibration = json.loads(accel_path.read_text(encoding="utf-8"))
    # calib_accel.py v1 writes the concise key "bias" with unit m/s².
    accel_bias = np.asarray(
        accel_calibration.get("bias_mps2", accel_calibration.get("bias")),
        dtype=np.float64)
    accel_scale = np.asarray(accel_calibration["scale"], dtype=np.float64)
    gyro_path = pathlib.Path(args.gyro_bias).expanduser().resolve()
    gyro_bias = np.asarray(json.loads(gyro_path.read_text(encoding="utf-8"))["bias_deg_s"],
                           dtype=np.float64)
    if any(value.shape != (3,) for value in (accel_bias, accel_scale, gyro_bias)):
        raise RuntimeError("加速度或陀螺标定文件应包含三个轴的数值")

    frame_times, timestamp_path = load_preferred_frame_times(session_dir)
    raw = np.load(imu_path)
    required = {"t_accel_monotonic_s", "accel_raw", "t_gyro_monotonic_s", "gyro_raw",
                "accel_scale_g_per_lsb", "gyro_scale_deg_s_per_lsb",
                "recording_start_monotonic_ns"}
    missing = required - set(raw.files)
    if missing:
        raise RuntimeError(f"imu_raw.npz 缺少字段：{sorted(missing)}")
    start_s = float(raw["recording_start_monotonic_ns"]) / 1e9
    accel_times = np.asarray(raw["t_accel_monotonic_s"], dtype=np.float64) - start_s
    gyro_times = np.asarray(raw["t_gyro_monotonic_s"], dtype=np.float64) - start_s
    accel_raw = np.asarray(raw["accel_raw"], dtype=np.float64)
    gyro_raw = np.asarray(raw["gyro_raw"], dtype=np.float64)
    accel_mps2 = accel_raw * float(raw["accel_scale_g_per_lsb"]) * 9.80665
    accel_mps2 = (accel_mps2 - accel_bias) / accel_scale
    gyro_rad_s = np.deg2rad(gyro_raw * float(raw["gyro_scale_deg_s_per_lsb"]) - gyro_bias)

    # ORB-SLAM3's UMI reader advances ACCL/GYRO by one common array index.
    # Resample acceleration to gyro timestamps, which are the IMU integration
    # grid, while retaining JY901B's calibrated body coordinates.
    accel_order = np.argsort(accel_times)
    gyro_order = np.argsort(gyro_times)
    accel_times, accel_mps2 = accel_times[accel_order], accel_mps2[accel_order]
    gyro_times, gyro_rad_s = gyro_times[gyro_order], gyro_rad_s[gyro_order]
    raw_accel_count, raw_gyro_count = len(accel_times), len(gyro_times)
    if args.collapse_repeated_imu:
        accel_keep = keep_first_of_equal_runs(accel_raw[accel_order])
        gyro_keep = keep_first_of_equal_runs(gyro_raw[gyro_order])
        accel_times, accel_mps2 = accel_times[accel_keep], accel_mps2[accel_keep]
        gyro_times, gyro_rad_s = gyro_times[gyro_keep], gyro_rad_s[gyro_keep]
    # Calibration convention: camera t compares IMU at t + offset.  Express
    # the IMU host timestamp on the camera timeline by t_camera = t_imu-offset.
    imu_camera_timeline_s = gyro_times - offset_s
    # Accelerometer firmware can have a different low-pass delay from gyro.
    # For a gyro sample placed at camera time t, query acceleration at the
    # host time t + accel_offset_s, then export both values on gyro's common
    # integration timestamp grid required by UMI's ORB-SLAM3 reader.
    accel_query_times = imu_camera_timeline_s + accel_offset_s
    valid = ((accel_query_times >= accel_times[0]) & (accel_query_times <= accel_times[-1]))
    gyro_times, gyro_rad_s = gyro_times[valid], gyro_rad_s[valid]
    imu_camera_timeline_s, accel_query_times = imu_camera_timeline_s[valid], accel_query_times[valid]
    accel_on_gyro = np.column_stack([
        np.interp(accel_query_times, accel_times, accel_mps2[:, axis]) for axis in range(3)
    ])
    # The serial reader starts during the preview, often many seconds before
    # R begins video recording.  Do not feed that unrelated stationary prefix
    # to ORB-SLAM3's first frame.  Keep only the calibrated camera interval
    # plus the recorder's short post-video IMU tail.
    keep = ((imu_camera_timeline_s >= 0.0) &
            (imu_camera_timeline_s <= frame_times[-1] + 3.0))
    imu_camera_timeline_s = imu_camera_timeline_s[keep]
    gyro_rad_s = gyro_rad_s[keep]
    accel_on_gyro = accel_on_gyro[keep]
    if len(imu_camera_timeline_s) < 100:
        raise RuntimeError("裁剪到相机时间范围后 IMU 样本不足")
    to_samples = lambda times, values: [
        {"cts": float(time_s * 1000.0), "value": [float(item) for item in value]}
        for time_s, value in zip(times, values)
    ]
    payload = {
        "1": {"streams": {
            "ACCL": {"samples": to_samples(imu_camera_timeline_s, accel_on_gyro)},
            "GYRO": {"samples": to_samples(imu_camera_timeline_s, gyro_rad_s)},
            # The patched reader only requires cts; zeros keep this schema
            # compatible with the original GoPro telemetry extractor.
            "CORI": {"samples": [
                {"cts": float(time_s * 1000.0), "value": [0.0, 0.0, 0.0]}
                for time_s in frame_times
            ]},
        }},
        "am_umi_metadata": {
            "source": "JY901B host-clock session",
            "frame_timestamp_source": timestamp_path.name,
            "imu_time_offset_for_camera_s": offset_s,
            "accel_time_offset_for_camera_s": accel_offset_s,
            "offset_definition": "camera t compares IMU at t + offset",
            "units": {"ACCL": "m/s^2", "GYRO": "rad/s", "cts": "ms"},
            "accel_calibration": str(accel_path),
            "gyro_bias": str(gyro_path),
            "camera_imu_calibration": str(camera_imu_path),
            "accel_time_calibration": str(accel_time_path),
            "consecutive_equal_runs_collapsed": args.collapse_repeated_imu,
            "raw_accel_packets": raw_accel_count,
            "raw_gyro_packets": raw_gyro_count,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print("ORB_SLAM3_IMU_EXPORT_OK")
    print("output:", output_path)
    print(f"camera_frames: {len(frame_times)}; camera_duration_s: {frame_times[-1]:.3f}")
    print("frame_timestamp_source:", timestamp_path)
    print(f"imu_samples: {len(imu_camera_timeline_s)}; imu_rate_hz: "
          f"{(len(imu_camera_timeline_s)-1)/(imu_camera_timeline_s[-1]-imu_camera_timeline_s[0]):.2f}")
    print(f"applied_imu_time_offset_for_camera_s: {offset_s:+.4f}")
    print(f"applied_accel_time_offset_for_camera_s: {accel_offset_s:+.4f}")
    if args.collapse_repeated_imu:
        print(f"repeated IMU packets collapsed: accel {raw_accel_count}->{len(accel_times)}, "
              f"gyro {raw_gyro_count}->{len(gyro_times)}")


if __name__ == "__main__":
    main()
