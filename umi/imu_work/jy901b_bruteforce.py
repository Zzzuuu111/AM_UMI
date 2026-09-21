"""
Brute-force probe for the rate register on a JY901-style clone.
Writes 50Hz (0x0032) to candidate registers with unlock, measures real
output rate after each. No save -> safe (reverts on power cycle).

Run: python imu_work/jy901b_bruteforce.py
"""
import time
from collections import Counter

import serial

CANDIDATES = [0x03, 0x05, 0x0C, 0x0D, 0x14, 0x1E, 0x21, 0x23, 0x2C]


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


def measure(ser, seconds=2.0):
    ser.reset_input_buffer()
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(512)
        if chunk:
            buf += chunk
    types = count_packets(buf)
    return types.most_common(1)[0][1] / seconds if types else 0.0, types


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    base, types = measure(ser, 2)
    print(f"基准速率 ≈ {base:.1f} Hz/类型  {dict(types)}")
    for reg in CANDIDATES:
        unlock(ser)
        ser.write(bytes([0xFF, 0xAA, reg, 0x32, 0x00]))  # value=50Hz
        ser.flush()
        time.sleep(0.25)
        rate, types = measure(ser, 2)
        mark = "  <-- 变了!" if rate > base * 1.5 else ""
        print(f"写 0x{reg:02X}=0x0032: 实测 ≈ {rate:.1f} Hz/类型{mark}")
    ser.close()
    print("\n若全部无变化：串口改不了速率（固件锁定或寄存器非标准）。")
    print("下一步建议：找板上的小按键/配置焊盘，或换维特官方正品 JY901。")


if __name__ == "__main__":
    main()
