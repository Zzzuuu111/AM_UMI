"""
Re-configure JY901B at 115200 baud (current state) with behavioral verification.

Steps:
  1. write rate=200Hz (0x03=0xC8), count packets 2s -> expect ~200 of each type
     if not, fall back: 100Hz, 50Hz, 20Hz
  2. write output mask=0x07 (time+accel+gyro), verify 0x53/0x54 disappear
  3. save (0x00=0x0000) AT THE CURRENT BAUD, then instruct power-cycle test

Run: python imu_work/jy901b_config2.py
"""
import time
from collections import Counter

import serial


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


def write_cmd(ser, reg, val, tag):
    ser.reset_input_buffer()
    frame = bytes([0xFF, 0xAA, reg, val & 0xFF, val >> 8])
    ser.write(frame)
    ser.flush()
    time.sleep(0.3)
    ack = ser.read(64)
    print(f"  {tag}: sent {frame.hex()} ack={ack.hex()[:48] or '(空)'}")
    return ack


def rate_of(types, seconds):
    if not types:
        return 0.0
    t = types.most_common(1)[0][1]
    return t / seconds


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
    print("== 当前 115200 状态确认 ==")
    t0 = read_for(ser, 2)
    types = count_packets(t0)
    print(f"  初始: {dict(types)} -> ~{rate_of(types, 2):.0f} Hz/类型")

    print("\n== 尝试设置回传速率 ==")
    target = None
    for hz, val in ((200, 0xC8), (100, 0x64), (50, 0x32), (20, 0x14)):
        write_cmd(ser, 0x03, val, f"速率={hz}Hz")
        buf = read_for(ser, 2)
        types = count_packets(buf)
        r = rate_of(types, 2)
        print(f"    验证: {dict(types)} -> ~{r:.0f} Hz/类型")
        if r > hz * 0.7:
            target = hz
            print(f"  ✅ 速率 {hz}Hz 生效")
            break
    if target is None:
        print("  ❌ 速率寄存器写不动（0x03 全部尝试失败）——把输出发给助手")
        ser.close()
        return

    print("\n== 设置输出掩码 = 时间+加速度+角速度 (0x07) ==")
    write_cmd(ser, 0x02, 0x07, "掩码=0x07")
    buf = read_for(ser, 2)
    types = count_packets(buf)
    print(f"  验证: {dict(types)}")
    if "0x53" not in types and "0x54" not in types:
        print("  ✅ 掩码生效（角度/磁场已关闭）")
    else:
        print("  ⚠️ 掩码未生效（0x53/0x54 仍在输出）")

    print("\n== 保存（当前波特率下发送）==")
    write_cmd(ser, 0x00, 0x00, "保存")

    print("\n== 下一步（手动）==")
    print("  1. 拔插一次 USB")
    print("  2. 运行: python imu_work/read_imu_serial.py --scan --wit --seconds 2")
    print("  3. 若 115200 段仍有包且频率高 -> 配置已持久化 ✅")
    ser.close()


if __name__ == "__main__":
    main()
