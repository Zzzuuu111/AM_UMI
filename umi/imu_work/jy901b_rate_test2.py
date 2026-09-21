"""
Comprehensive JY901B rate-register test at 115200:
  - capture ack after each write
  - both byte orders x 200/100/50 Hz
  - measure real output rate after each attempt
No save is sent (safe: unsaved writes revert on power cycle).

Run: python imu_work/jy901b_rate_test2.py
"""
import time
from collections import Counter

import serial


def unlock(ser):
    ser.write(bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
    ser.flush()
    time.sleep(0.2)
    ser.read(64)


def count_packets(buf):
    types = Counter()
    i = 0
    while i < len(buf) - 11:
        if buf[i] == 0x55 and buf[i + 1] in (0x50, 0x51, 0x52):
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:
                types[hex(pkt[1])] += 1
                i += 11
                continue
        i += 1
    return types


def capture(ser, seconds):
    ser.reset_input_buffer()
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(256)
        if chunk:
            buf += chunk
    return buf


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

    for order_name, hi_first in (("低字节在前", False), ("高字节在前", True)):
        for hz in (200, 100, 50):
            val_lo, val_hi = (hz & 0xFF), (hz >> 8)
            if hi_first:
                frame = bytes([0xFF, 0xAA, 0x03, val_hi, val_lo])
            else:
                frame = bytes([0xFF, 0xAA, 0x03, val_lo, val_hi])
            unlock(ser)
            ser.write(frame)
            ser.flush()
            time.sleep(0.25)
            ack = capture(ser, 0.8)
            buf = capture(ser, 2.0)
            types = count_packets(buf)
            rate = types.most_common(1)[0][1] / 2.0 if types else 0.0
            print(f"写 {hz}Hz {order_name} {frame.hex()}")
            print(f"  ack: {ack.hex()[:100]}")
            print(f"  实测: {dict(types)} ≈ {rate:.0f} Hz/类型")
    ser.close()
    print("\n全部试完。若所有实测仍 ≈10 Hz，说明此板速率寄存器不可通过串口修改。")
    print("后续选项：1) 用维特官方 Windows 上位机改  2) 换标准 JY901 模块")


if __name__ == "__main__":
    main()
