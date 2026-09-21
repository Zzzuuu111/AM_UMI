#!/usr/bin/env python3
"""Temporarily set JY901B bandwidth to 256 Hz and measure fresh samples.

No SAVE command is sent. The persistent configuration is therefore left
untouched, and unplugging/replugging the IMU restores the saved settings.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import serial


PACKET_SIZE = 11
UNLOCK = bytes((0xFF, 0xAA, 0x69, 0x88, 0xB5))
SET_BANDWIDTH_256HZ = bytes((0xFF, 0xAA, 0x1F, 0x00, 0x00))


def valid_packets(data: bytes):
    index = 0
    while index + PACKET_SIZE <= len(data):
        if data[index] != 0x55:
            index += 1
            continue
        packet = data[index:index + PACKET_SIZE]
        if (sum(packet[:10]) & 0xFF) == packet[10]:
            yield packet
            index += PACKET_SIZE
        else:
            index += 1


def capture(port: serial.Serial, seconds: float) -> bytes:
    port.reset_input_buffer()
    deadline = time.monotonic() + seconds
    chunks = []
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


def analyze(data: bytes, seconds: float):
    packets = list(valid_packets(data))
    counts = Counter(packet[1] for packet in packets)
    result = {}
    for packet_type, label in ((0x51, "accel"), (0x52, "gyro")):
        vectors = [packet[2:8] for packet in packets if packet[1] == packet_type]
        changes = sum(current != previous
                      for previous, current in zip(vectors, vectors[1:]))
        result[label] = {
            "packets": len(vectors),
            "packet_rate_hz": len(vectors) / seconds,
            "changed_vectors": changes,
            "changed_vector_rate_hz": changes / seconds,
            "unchanged_ratio": (
                1.0 - changes / (len(vectors) - 1)
                if len(vectors) > 1 else 1.0),
        }
    return counts, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="临时设置 JY901B 256 Hz 带宽并检查真实数据更新率")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument(
        "--apply-transient-256hz",
        action="store_true",
        help="必须显式给出；只写运行内存，不发送保存命令")
    args = parser.parse_args()

    if not args.apply_transient_256hz:
        parser.error("未修改设备。若确认临时测试，请添加 --apply-transient-256hz")
    if args.seconds <= 0:
        parser.error("--seconds 必须大于 0")

    print("JY901B_TRANSIENT_BANDWIDTH_TEST_STARTED")
    print(f"串口: {args.port} @ {args.baud}")
    print("即将临时设置带宽为 256 Hz；不会发送保存命令，也不会修改波特率或回传率。")
    print("拔插 IMU 后会恢复设备中原先保存的带宽设置。")

    with serial.Serial(args.port, args.baud, timeout=0.05) as port:
        passive = capture(port, 1.0)
        if not any(True for _ in valid_packets(passive)):
            raise RuntimeError("未收到合法数据包；未向设备发送任何写命令")

        port.write(UNLOCK)
        port.flush()
        time.sleep(0.25)
        port.write(SET_BANDWIDTH_256HZ)
        port.flush()
        time.sleep(0.3)

        input("请连续、平缓地转动整套手持装置；准备好后按回车开始测量：")
        started = time.monotonic()
        data = capture(port, args.seconds)
        elapsed = time.monotonic() - started

    counts, result = analyze(data, elapsed)
    print("JY901B_TRANSIENT_BANDWIDTH_WRITE_SENT")
    print("有效包:", {f"0x{kind:02x}": count for kind, count in counts.items()})
    for label in ("accel", "gyro"):
        item = result[label]
        print(
            f"{label}: packets={item['packets']} "
            f"packet_rate_hz={item['packet_rate_hz']:.2f} "
            f"changed_vector_rate_hz={item['changed_vector_rate_hz']:.2f} "
            f"unchanged_ratio={item['unchanged_ratio']:.3%}")
    print("说明：changed_vector_rate_hz 是动态测试中的新数值更新率下限，"
          "不是厂商保证的内部采样率。")
    print("本次未保存配置；测试后请拔插 IMU，恢复原先持久化设置。")


if __name__ == "__main__":
    main()
