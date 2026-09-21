"""Identify the signed IMU gyro axis corresponding to each camera rotation.

This is a coarse, practical verification after the camera and IMU are mounted
as one rigid hand-held assembly.  It is not a substitute for the later
camera-IMU rotational extrinsic calibration, which estimates an arbitrary 3D
rotation from visual and IMU motion.

The three prompted moves use the OpenCV camera convention: +X image-right,
+Y image-down, +Z forward through the lens.
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


def record_motion(device, seconds: float) -> np.ndarray:
    chunks = []
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        chunk = device.read(2048)
        if chunk:
            chunks.append(chunk)
    raw = np.asarray(parse_gyro_packets(b"".join(chunks)), dtype=np.float64)
    if len(raw) < 40:
        raise RuntimeError(f"only {len(raw)} valid gyro samples received")
    return raw * GYRO_SCALE_DPS


def main():
    parser = argparse.ArgumentParser(
        description="核验已固定相机与 IMU 的粗略坐标轴对应关系。")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--seconds-per-axis", type=float, default=2.0)
    parser.add_argument("--gyro-bias", required=True,
                        help="由 calib_gyro_bias.py 生成的陀螺零偏 JSON")
    parser.add_argument("--out", required=True,
                        help="新的 JSON 输出路径；同名文件会被保护，不会覆盖")
    args = parser.parse_args()
    if args.seconds_per_axis < 1.0:
        parser.error("--seconds-per-axis must be at least 1 second")

    bias_path = pathlib.Path(args.gyro_bias).expanduser()
    bias = np.asarray(json.loads(bias_path.read_text(encoding="utf-8"))["bias_deg_s"],
                      dtype=np.float64)
    if bias.shape != (3,):
        raise SystemExit(f"{bias_path} 中的 bias_deg_s 格式不正确")
    output_path = pathlib.Path(args.out).expanduser()
    if output_path.exists():
        raise SystemExit(f"为保护已有标定，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    instructions = [
        (
            "相机 +X（画面向右）方向的旋转",
            "让镜头朝向桌面/场景；测量期间平稳地让镜头向下俯。"
        ),
        (
            "相机 +Y（画面向下）方向的旋转",
            "让镜头朝向桌面/场景；测量期间平稳地让镜头向右转。"
        ),
        (
            "相机 +Z（沿镜头前方）方向的旋转",
            "让镜头朝向桌面/场景；测量期间平稳地让画面顺时针旋转。"
        ),
    ]
    records = []
    with serial.Serial(args.port, args.baud, timeout=0.05) as device:
        for camera_axis, instruction in instructions:
            input(
                f"\n[{camera_axis}] 请把整套手持夹爪拿稳，摆到舒适的起始姿态后按回车。\n"
                f"  {instruction}\n"
                f"  回车后请连续转动 {args.seconds_per_axis:.1f} 秒：")
            print("  正在采集，请开始平稳转动……", flush=True)
            gyro_dps = record_motion(device, args.seconds_per_axis) - bias
            # Median reduces the effect of the first/last few samples while
            # retaining sign for a deliberately steady rotation.
            response = np.median(gyro_dps, axis=0)
            dominant = int(np.argmax(np.abs(response)))
            strength = float(abs(response[dominant]))
            records.append({
                "camera_axis": camera_axis,
                "median_gyro_deg_s": response.tolist(),
                "dominant_imu_axis": "XYZ"[dominant],
                "dominant_sign": "+" if response[dominant] >= 0 else "-",
                "dominant_speed_deg_s": strength,
            })
            print(
                f"  IMU 响应 [X,Y,Z] = [{response[0]:+.1f}, {response[1]:+.1f}, {response[2]:+.1f}] °/s"
                f" → 对应 IMU {('+' if response[dominant] >= 0 else '-')}{'XYZ'[dominant]} 轴")

    axes = [record["dominant_imu_axis"] for record in records]
    strengths = [record["dominant_speed_deg_s"] for record in records]
    unique_axes = len(set(axes)) == 3
    sufficient_motion = min(strengths) >= 10.0
    result = {
        "schema": "am_umi_camera_imu_axis_check_v1",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "port": args.port,
        "baud": args.baud,
        "gyro_bias_source": str(bias_path),
        "seconds_per_axis": args.seconds_per_axis,
        "camera_convention": "+X image-right, +Y image-down, +Z forward through lens",
        "measurements": records,
        "passed_distinct_axes": unique_axes,
        "passed_min_speed": sufficient_motion,
        "note": "Coarse signed-axis check only; perform visual-IMU extrinsic calibration later.",
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("\n相机-IMU 坐标轴核验通过" if unique_axes and sufficient_motion
          else "\n相机-IMU 坐标轴核验需要复查")
    print("三个相机方向是否对应三个不同 IMU 轴：", "是" if unique_axes else "否")
    print("每次主响应是否至少 10°/s：", "是" if sufficient_motion else "否")
    if not unique_axes:
        print("警告：有方向重复对应同一 IMU 轴；请用更干净的单轴转动重新测量。")
    if not sufficient_motion:
        print("警告：至少一次转动过慢；请更清楚、平稳地重新做三次转动。")
    print("已保存：", output_path)


if __name__ == "__main__":
    main()
