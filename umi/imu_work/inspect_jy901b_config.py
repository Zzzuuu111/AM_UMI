#!/usr/bin/env python3
"""Read JY901B configuration registers without modifying the device.

The script uses the WitMotion normal-protocol read command (FF AA 27 ...).
It never sends unlock, register-write, save, reset, or baud-change commands.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import serial


PACKET_SIZE = 11

RATE_NAMES = {
    0x01: "0.2 Hz",
    0x02: "0.5 Hz",
    0x03: "1 Hz",
    0x04: "2 Hz",
    0x05: "5 Hz",
    0x06: "10 Hz",
    0x07: "20 Hz",
    0x08: "50 Hz",
    0x09: "100 Hz",
    0x0A: "125 Hz（仅部分型号）",
    0x0B: "200 Hz",
    0x0C: "单次输出",
    0x0D: "不自动输出",
}

BAUD_NAMES = {
    0x01: "4800",
    0x02: "9600",
    0x03: "19200",
    0x04: "38400",
    0x05: "57600",
    0x06: "115200",
    0x07: "230400",
    0x08: "460800",
    0x09: "921600",
}

BANDWIDTH_NAMES = {
    0x00: "256 Hz",
    0x01: "184 Hz",
    0x02: "94 Hz",
    0x03: "44 Hz",
    0x04: "21 Hz",
    0x05: "10 Hz",
    0x06: "5 Hz",
}


def valid_packets(data: bytes):
    """Yield checksum-valid normal-protocol packets from an arbitrary stream."""
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


def receive_for(port: serial.Serial, seconds: float) -> bytes:
    deadline = time.monotonic() + seconds
    chunks = []
    while time.monotonic() < deadline:
        chunk = port.read(4096)
        if chunk:
            chunks.append(chunk)
    return b"".join(chunks)


def read_four_registers(port: serial.Serial, start_register: int):
    port.reset_input_buffer()
    command = bytes((0xFF, 0xAA, 0x27,
                     start_register & 0xFF,
                     (start_register >> 8) & 0xFF))
    port.write(command)
    port.flush()
    response = receive_for(port, 0.8)
    read_packets = [packet for packet in valid_packets(response)
                    if packet[1] == 0x71]
    if not read_packets:
        return None, response
    packet = read_packets[-1]
    values = [int.from_bytes(packet[offset:offset + 2], "little")
              for offset in range(2, 10, 2)]
    return values, response


def fmt(value: int, names: dict[int, str]) -> str:
    return f"0x{value:04x} ({names.get(value, '未知/此型号自定义')})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="只读检查 JY901B 输出内容、回传率、波特率和带宽")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=460800)
    args = parser.parse_args()

    print("JY901B_CONFIG_READ_ONLY_STARTED")
    print(f"串口: {args.port} @ {args.baud}")
    print("安全模式：不会解锁、写寄存器、保存或复位设备。")

    with serial.Serial(args.port, args.baud, timeout=0.05) as port:
        passive = receive_for(port, 1.0)
        packet_counts = Counter(packet[1] for packet in valid_packets(passive))
        print("被动数据包计数:",
              {f"0x{kind:02x}": count for kind, count in packet_counts.items()})
        if not packet_counts:
            raise RuntimeError("未收到合法的 0x55 数据包；请检查串口和波特率")

        basic, basic_raw = read_four_registers(port, 0x02)
        bandwidth, bandwidth_raw = read_four_registers(port, 0x1F)

    if basic is None or bandwidth is None:
        print("JY901B_CONFIG_READ_RESPONSE_MISSING")
        print("此设备正常输出数据，但没有返回 0x71 读寄存器响应。")
        print(f"basic_response_bytes: {len(basic_raw)}")
        print(f"bandwidth_response_bytes: {len(bandwidth_raw)}")
        print("没有对设备作任何修改；可能需要用官方上位机读取配置。")
        return

    output_mask, output_rate, configured_baud = basic[:3]
    bandwidth_value = bandwidth[0]
    print("JY901B_CONFIG_READ_ONLY_OK")
    print(f"输出掩码 RSW: 0x{output_mask:04x}")
    print(f"回传率 RRATE: {fmt(output_rate, RATE_NAMES)}")
    print(f"串口速率 BAUD: {fmt(configured_baud, BAUD_NAMES)}")
    print(f"滤波带宽 BANDWIDTH: {fmt(bandwidth_value, BANDWIDTH_NAMES)}")
    print("注意：回传率表示包发送频率，不等于每包都含有新的传感器采样。")


if __name__ == "__main__":
    main()
