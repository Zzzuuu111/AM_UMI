"""Record a long stationary JY901B sequence for Allan-variance analysis."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import serial

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.record_sync_session import parse_reads_precise


def main() -> None:
    parser = argparse.ArgumentParser(description="录制 JY901B 长静止数据，用于 Allan 方差分析")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--seconds", type=float, default=1800.0,
                        help="建议 1800–3600 秒；最少 1200 秒")
    parser.add_argument("--read-timeout-ms", type=float, default=2.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.baud <= 0 or args.seconds < 1200 or args.read_timeout_ms <= 0:
        parser.error("baud/timeout 必须为正，Allan 录制至少 1200 秒")
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"为保护已有数据，拒绝覆盖：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("ALLAN_RECORDING_STARTED")
    print(f"端口：{args.port} @ {args.baud}；时长：{args.seconds / 60:.1f} 分钟")
    print("请让已组装的相机—夹爪—IMU 整体完全静止；不要触碰装置或 USB 线。")
    chunks = []
    start_ns = time.monotonic_ns()
    next_report = 60.0
    try:
        with serial.Serial(args.port, args.baud,
                           timeout=args.read_timeout_ms / 1000.0) as device:
            while True:
                elapsed = (time.monotonic_ns() - start_ns) / 1e9
                if elapsed >= args.seconds:
                    break
                request = max(1, min(int(device.in_waiting), 256))
                before = time.monotonic_ns()
                chunk = device.read(request)
                after = time.monotonic_ns()
                if chunk:
                    chunks.append((before / 1e9, after / 1e9, bytes(chunk)))
                if elapsed >= next_report:
                    print(f"已静止录制 {elapsed / 60:.0f}/{args.seconds / 60:.0f} 分钟", flush=True)
                    next_report += 60.0
    except KeyboardInterrupt:
        raise SystemExit("已提前取消；不保存不完整的 Allan 录制，请重新开始。")

    t_a, accel, t_g, gyro = parse_reads_precise(chunks, args.baud)
    np.savez_compressed(
        out,
        t_accel_monotonic_s=t_a,
        accel_raw=accel,
        t_gyro_monotonic_s=t_g,
        gyro_raw=gyro,
        accel_scale_g_per_lsb=16.0 / 32768.0,
        gyro_scale_deg_s_per_lsb=2000.0 / 32768.0,
        timestamp_method="continuous_uart_8n1_packet_center_v1",
        recording_start_monotonic_ns=start_ns,
        recording_stop_monotonic_ns=time.monotonic_ns(),
    )
    print("ALLAN_RECORDING_OK")
    print("saved:", out)
    print(f"samples: accel={len(accel)}, gyro={len(gyro)}")
    print(f"duration_s: {t_g[-1] - t_g[0]:.2f}; gyro_rate_hz: {(len(t_g)-1)/(t_g[-1]-t_g[0]):.2f}")


if __name__ == "__main__":
    main()
