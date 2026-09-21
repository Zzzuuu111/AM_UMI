"""
JY901B final config, following WitMotion official procedure:
  unlock (FF AA 69 88 B5) + 200ms delay before EVERY write group.

Current module state: 115200 baud (in RAM), rate 10Hz, all types output.

This script (run at 115200):
  1. unlock -> write rate=200Hz -> unlock -> write mask=0x07 -> unlock -> save
  2. verify packet rate/types at 115200
  3. instruct power-cycle to test persistence

Run: python imu_work/jy901b_final_config.py
"""
import time
from collections import Counter

import serial


def unlock(ser):
    ser.write(bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))
    ser.flush()
    time.sleep(0.2)
    ser.read(64)  # drain


def write_reg(ser, reg, val, name):
    unlock(ser)
    frame = bytes([0xFF, 0xAA, reg, val & 0xFF, val >> 8])
    ser.write(frame)
    ser.flush()
    time.sleep(0.2)
    print(f"  解锁+写 {name}: {frame.hex()}")


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


def read_for(ser, seconds):
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(1024)
        if chunk:
            buf += chunk
    return buf


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    print("== 官方时序配置（当前 115200）==")
    write_reg(ser, 0x03, 0x00C8, "回传速率=200Hz")
    write_reg(ser, 0x02, 0x0007, "输出掩码=时间+加速度+角速度")
    unlock(ser)
    ser.write(bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]))  # 保存
    ser.flush()
    time.sleep(0.2)
    print("  解锁+保存: ffaa000000")

    print("\n== 验证 ==")
    buf = read_for(ser, 3)
    types = count_packets(buf)
    print(f"  3 秒包统计: {dict(types)}")
    if types:
        top = types.most_common(1)[0][1] / 3.0
        print(f"  每类型速率 ≈ {top:.0f} Hz")
        if top > 150:
            print("  ✅ 200Hz 生效")
        else:
            print("  ⚠️ 速率仍低——把输出发给助手")
    else:
        print("  ❌ 无合法包——波特率可能变了，把输出发给助手")

    print("\n== 持久化验证（手动）==")
    print("  1. 拔插一次 USB")
    print("  2. python imu_work/read_imu_serial.py --scan --wit --seconds 2")
    print("  3. 看 115200 段是否仍有高频包")
    ser.close()


if __name__ == "__main__":
    main()
