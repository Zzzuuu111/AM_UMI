"""Prepare a separate offline test copy, preserving every original session file."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from imu_work.uvc_payload_timing import align_session_frame_timestamps, load_preferred_frame_times


def main():
    p = argparse.ArgumentParser(description='复制现有采集数据并生成严格 UVC 时间轴，不修改原录制')
    p.add_argument('--session-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--allow-timing-outliers', action='store_true',
                   help='仅离线诊断：保留并标注超过 50 ms 的接收差值，不代表精确对齐已通过')
    calibration = ROOT / 'calibration/handheld_gripper_camera'
    p.add_argument('--camera-imu-calibration', default=str(
        calibration / 'camera_imu/emeet_jy901b_rotation_time_v4_uvc_source.json'))
    p.add_argument('--accel-time-calibration', default=None,
                   help='可选：单独的加速度延迟/平移候选；只影响新副本的 IMU 导出')
    p.add_argument('--accel-calibration', default=str(calibration / 'imu/jy901b_accel_v1.json'))
    p.add_argument('--gyro-bias', default=str(calibration / 'imu/jy901b_gyro_bias_v1.json'))
    p.add_argument('--collapse-repeated-imu', action='store_true',
                   help='仅诊断：折叠 JY901B 连续重复值，按实际数据刷新率导出')
    args = p.parse_args()
    source, target = Path(args.session_dir).resolve(), Path(args.output_dir).resolve()
    if target.exists():
        p.error(f'拒绝覆盖: {target}')
    names = ['raw_video.mp4', 'imu_raw.npz', 'frame_timestamps.csv', 'raw_uvc_payload_headers.bin', 'metadata.json']
    for name in names:
        if not (source / name).is_file():
            p.error(f'缺少输入: {source / name}')
    target.mkdir(parents=True)
    for name in names:
        shutil.copy2(source / name, target / name)
    timing = align_session_frame_timestamps(target)
    times, _ = load_preferred_frame_times(target)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise RuntimeError('重建时间轴未通过严格递增检查')
    if timing['associations_over_50ms'] and not args.allow_timing_outliers:
        raise RuntimeError('时间匹配存在大于 50 ms 的偏差；保留副本供检查，不继续回放')
    camera_imu = Path(args.camera_imu_calibration).resolve()
    accel_time = Path(args.accel_time_calibration).resolve() if args.accel_time_calibration else None
    accel_calibration = Path(args.accel_calibration).resolve()
    gyro_bias = Path(args.gyro_bias).resolve()
    for candidate in (camera_imu, accel_calibration, gyro_bias, *([accel_time] if accel_time else [])):
        if not candidate.is_file():
            p.error(f'缺少标定输入: {candidate}')
    cmd = [sys.executable, str(ROOT / 'imu_work/export_handheld_imu_to_orbslam3.py'),
           '--session-dir', str(target), '--out', str(target / 'imu_data_strict_uvc.json'),
           '--camera-imu-calibration', str(camera_imu),
           '--accel-calibration', str(accel_calibration), '--gyro-bias', str(gyro_bias)]
    if accel_time:
        cmd.extend(['--accel-time-calibration', str(accel_time)])
    if args.collapse_repeated_imu:
        cmd.append('--collapse-repeated-imu')
    subprocess.run(cmd, check=True, cwd=ROOT)
    report = dict(source_session=str(source), prepared_session=str(target), copied_raw_files=names,
                  calibration_sources=dict(camera_imu=str(camera_imu),
                      accel_time=str(accel_time) if accel_time else None,
                      accel=str(accel_calibration), gyro_bias=str(gyro_bias)),
                  calibration_files_modified=False, original_session_modified=False, timing=timing,
                  diagnostic_timing_outliers_allowed=args.allow_timing_outliers,
                  timing_outliers_present=bool(timing['associations_over_50ms']),
                  repeated_imu_packets_collapsed=args.collapse_repeated_imu)
    (target / 'preparation_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('独立复测数据副本已准备；原视频、IMU、标定均未改动。', flush=True)


if __name__ == '__main__':
    main()
