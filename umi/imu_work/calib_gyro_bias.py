"""Measure JY901B gyroscope bias after the final hand-held rig is assembled.

Keep the camera, gripper, IMU, and cable completely still for the whole
measurement.  This writes a calibration record; it does not reconfigure the
IMU firmware.

Example:
  python imu_work/calib_gyro_bias.py \
    --port /dev/ttyUSB0 --baud 460800 --seconds 10 \
    --out calibration/handheld_gripper_camera/imu/jy901b_gyro_bias_v1.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time

import numpy as np
import serial


GYRO_SCALE_DPS = 2000.0 / 32768.0


def parse_gyro_packets(buffer: bytes) -> list[tuple[int, int, int]]:
    """Extract checksum-valid 0x55 0x52 JY901B angular-rate packets."""
    packets = []
    index = 0
    while index <= len(buffer) - 11:
        if buffer[index] == 0x55 and buffer[index + 1] == 0x52:
            packet = buffer[index:index + 11]
            if (sum(packet[:10]) & 0xFF) == packet[10]:
                packets.append(tuple(
                    int.from_bytes(packet[offset:offset + 2], "little", signed=True)
                    for offset in (2, 4, 6)))
                index += 11
                continue
        index += 1
    return packets


def main():
    parser = argparse.ArgumentParser(
        description="Measure static JY901B gyroscope bias for the final hand-held rig.")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out", required=True,
                        help="new JSON output path; existing files are protected")
    args = parser.parse_args()
    if args.seconds < 3:
        parser.error("--seconds must be at least 3 seconds")

    output_path = pathlib.Path(args.out).expanduser()
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing calibration: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Keep the complete hand-held rig fully still; do not touch its cable.")
    print(f"Measuring gyroscope bias for {args.seconds:.1f} seconds ...", flush=True)
    chunks: list[bytes] = []
    with serial.Serial(args.port, args.baud, timeout=0.05) as device:
        start = time.monotonic()
        while time.monotonic() - start < args.seconds:
            chunk = device.read(2048)
            if chunk:
                chunks.append(chunk)

    # Join all reads before parsing so a packet split at a serial-read boundary
    # is not silently discarded.
    raw = np.asarray(parse_gyro_packets(b"".join(chunks)), dtype=np.float64)
    if len(raw) < 100:
        raise SystemExit(
            f"Only received {len(raw)} valid gyro samples. Check port/baud and retry.")

    gyro_dps = raw * GYRO_SCALE_DPS
    mean = gyro_dps.mean(axis=0)
    std = gyro_dps.std(axis=0)
    norm = np.linalg.norm(gyro_dps, axis=1)
    p99 = float(np.percentile(norm, 99))
    result = {
        "schema": "am_umi_jy901b_gyro_bias_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "port": args.port,
        "baud": args.baud,
        "duration_seconds": args.seconds,
        "sample_count": int(len(raw)),
        "gyro_scale_deg_s_per_lsb": GYRO_SCALE_DPS,
        "bias_deg_s": mean.tolist(),
        "std_deg_s": std.tolist(),
        "norm_p99_deg_s": p99,
        "correction": "gyro_corrected_deg_s = gyro_raw_deg_s - bias_deg_s",
        "note": "Measured while the final camera-gripper-IMU rig was stationary.",
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    print("GYRO_BIAS_CALIBRATION_OK")
    print("samples:", len(raw))
    print("bias_deg_s:", "[" + ", ".join(f"{item:+.4f}" for item in mean) + "]")
    print("std_deg_s:", "[" + ", ".join(f"{item:.4f}" for item in std) + "]")
    print(f"norm_p99_deg_s: {p99:.4f}")
    if p99 > 5.0:
        print("WARNING: gyro movement/noise is high; keep the rig still and repeat.")
    print("saved:", output_path)


if __name__ == "__main__":
    main()
