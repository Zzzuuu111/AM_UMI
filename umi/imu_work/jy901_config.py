"""
Configure a WitMotion/JY901-style IMU module for SLAM use.

Changes (written over serial, then saved):
  reg 0x02 = 0x0007 : output mask = time(0x01) + accel(0x02) + gyro(0x04)
  reg 0x03 = 0x00C8 : report rate = 200 Hz
  reg 0x04 = 0x0006 : baud rate = 115200
  reg 0x00 = save

Usage:
  python imu_work/jy901_config.py            # configure, then verify at 115200
  python imu_work/jy901_config.py --reset    # restore 9600/10Hz defaults? (no-op
                                             # placeholder; JY901 restores by
                                             # shorting pads or 'ff aa 20 00 00')
"""
import argparse
import time

import serial


def send(ser, reg, val):
    frame = bytes([0xFF, 0xAA, reg, val & 0xFF, (val >> 8) & 0xFF])
    ser.write(frame)
    ser.flush()
    time.sleep(0.15)
    ack = ser.read(20)
    return frame, ack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--verify-only", action="store_true",
                    help="just count packets at 115200 without writing config")
    args = ap.parse_args()

    if not args.verify_only:
        ser = serial.Serial(args.port, 9600, timeout=1)
        print("== 在 9600 下发送配置 ==")
        for reg, val, name in [(0x02, 0x0007, "输出内容=时间+加速度+角速度"),
                               (0x03, 0x00C8, "回传速率=200Hz"),
                               (0x04, 0x0006, "波特率=115200")]:
            frame, ack = send(ser, reg, val)
            print(f"  写 {name}: {frame.hex()}  ack={ack.hex() or '(无)'}")
        frame, ack = send(ser, 0x00, 0x0000)
        print(f"  保存配置: {frame.hex()}  ack={ack.hex() or '(无)'}")
        ser.close()
        time.sleep(0.3)

    # verify at 115200
    print("\n== 用 115200 重开并验证 ==")
    try:
        ser = serial.Serial(args.port, 115200, timeout=1)
    except Exception as e:
        print(f"打开失败: {e}\n若失败：拔插一次 USB 再试；波特率未变则模块仍是 9600。")
        return
    t0 = time.time()
    buf = b""
    while time.time() - t0 < 3:
        chunk = ser.read(1024)
        if chunk:
            buf += chunk
    ser.close()

    print(f"3 秒收到 {len(buf)} 字节")
    if not buf:
        print("没有数据：配置可能未生效。拔插一次 USB，再用 9600 跑 --scan 检查。")
        return
    # count 0x55 packets by type
    from collections import Counter
    types = Counter()
    ok = 0
    i = 0
    while i < len(buf) - 11:
        if buf[i] == 0x55 and buf[i + 1] in (0x50, 0x51, 0x52, 0x53, 0x54, 0x59):
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:
                types[hex(pkt[1])] += 1
                ok += 1
                i += 11
                continue
        i += 1
    print(f"合法包 {ok} 个, 各类型: {dict(types)}")
    rate = ok / 3.0
    print(f"总包率 ≈ {rate:.0f} 包/秒")
    if types.get("0x51", 0) > 150:
        print("✅ 加速度包率 ≥150 包/3s → 200Hz 配置生效")
    else:
        print("⚠️ 包率偏低，可能需要重试配置或检查接线")


if __name__ == "__main__":
    main()
