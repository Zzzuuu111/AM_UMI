"""
JY901B rate register systematic test with unlock before every write.
Tries 200/100/50/20 Hz; measures real output rate after each.
If none apply immediately, does save + tells user to power-cycle.

Run: python imu_work/jy901b_rate_test.py
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
        if buf[i] == 0x55 and buf[i + 1] in (0x50, 0x51, 0x52, 0x53, 0x54):
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:
                types[hex(pkt[1])] += 1
                i += 11
                continue
        i += 1
    return types


def measure(ser, seconds=3.0):
    ser.reset_input_buffer()
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(1024)
        if chunk:
            buf += chunk
    types = count_packets(buf)
    if not types:
        return 0.0, types
    return types.most_common(1)[0][1] / seconds, types


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    rate0, types0 = measure(ser, 2)
    print(f"当前速率 ≈ {rate0:.0f} Hz/类型  {dict(types0)}")

    for hz, val in ((200, 0xC8), (100, 0x64), (50, 0x32), (20, 0x14)):
        unlock(ser)
        ser.write(bytes([0xFF, 0xAA, 0x03, val & 0xFF, val >> 8]))
        ser.flush()
        time.sleep(0.3)
        rate, types = measure(ser, 3)
        print(f"写速率={hz}Hz(0x{val:02x}): 实测 ≈ {rate:.0f} Hz/类型  {dict(types)}")
        if rate > hz * 0.7:
            print(f"  ✅ {hz}Hz 生效")
            break
    else:
        print("全部未即时生效 → 执行保存，试'重启后生效'")
        unlock(ser)
        ser.write(bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]))
        ser.flush()
        time.sleep(0.3)
        print("已发送保存。请拔插一次 USB，然后运行：")
        print("  python imu_work/read_imu_serial.py --scan --wit --seconds 3")
        print("若 115200 段每类型包数明显 >30（如 ≥150），说明速率在重启后生效并已保存。")
    ser.close()


if __name__ == "__main__":
    main()
