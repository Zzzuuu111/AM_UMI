"""
Record JY901B data at 460800/200Hz with host timestamps, and run a shake
test: verify the accelerometer contains real linear acceleration (|a| should
spike well above 1g during shaking) and the gyro sees large angular rates.

Usage:
  python imu_work/record_jy901b.py --seconds 6 --out imu_work/shake_test.npz
  # keep the module still for 2s, shake hard for 2s, still for 2s
"""
import argparse
import time

import numpy as np
import serial

ACCEL_SCALE = 16.0 / 32768.0      # g per LSB (±16g range)
GYRO_SCALE = 2000.0 / 32768.0     # deg/s per LSB (±2000deg/s range)


def parse_packets(buf):
    """Return list of (type, x, y, z) for valid 0x51/0x52 packets."""
    out = []
    i = 0
    n = len(buf)
    while i < n - 11:
        if buf[i] == 0x55 and buf[i + 1] in (0x51, 0x52):
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:
                x = int.from_bytes(pkt[2:4], "little", signed=True)
                y = int.from_bytes(pkt[4:6], "little", signed=True)
                z = int.from_bytes(pkt[6:8], "little", signed=True)
                out.append((pkt[1], x, y, z))
                i += 11
                continue
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--out", default="imu_work/shake_test.npz")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    print(f"录制 {args.seconds}s ... 前2s静止 → 中间用力摇晃 → 后2s静止")
    chunks = []  # (t_before, t_after, bytes)
    t0 = time.time()
    last_sec = -1

    def phase(sec):
        total = args.seconds
        still_len = min(2.0, total / 3)
        if sec < still_len:
            return "静止……"
        if sec < total - still_len:
            return ">>> 用力摇晃！<<<"
        return "静止……"

    while time.time() - t0 < args.seconds:
        t_before = time.time()
        chunk = ser.read(2048)
        t_after = time.time()
        if chunk:
            chunks.append((t_before, t_after, chunk))
        sec = int(time.time() - t0)
        if sec != last_sec and sec <= args.seconds:
            last_sec = sec
            bar = "#" * sec + "-" * max(int(args.seconds) - sec, 0)
            print(f"\r[{bar}] {sec}s/{args.seconds:.0f}s  {phase(sec)}", end="", flush=True)
    print()
    ser.close()

    total = sum(len(c) for _, _, c in chunks)
    print(f"收到 {total} 字节, {len(chunks)} 个块")

    # per-byte arrival time: distribute each read's bytes uniformly over its
    # own [t_before, t_after] interval (self-calibrating)
    times = []
    pkts = []
    for t_before, t_after, chunk in chunks:
        n = len(chunk)
        span = max(t_after - t_before, 1e-6)
        i = 0
        while i < n - 11:
            if chunk[i] == 0x55 and chunk[i + 1] in (0x51, 0x52):
                pkt = chunk[i:i + 11]
                if sum(pkt[:10]) & 0xFF == pkt[10]:
                    t_pkt = t_before + span * (i / n)
                    x = int.from_bytes(pkt[2:4], "little", signed=True)
                    y = int.from_bytes(pkt[4:6], "little", signed=True)
                    z = int.from_bytes(pkt[6:8], "little", signed=True)
                    times.append(t_pkt)
                    pkts.append((pkt[1], x, y, z))
                    i += 11
                    continue
            i += 1

    if not pkts:
        print("没有解析到任何包！")
        return

    t = np.array(times)
    types = np.array([p[0] for p in pkts])
    acc = np.array([p[1:] for p in pkts if p[0] == 0x51], dtype=np.float64)
    gyr = np.array([p[1:] for p in pkts if p[0] == 0x52], dtype=np.float64)

    print(f"包总数 {len(pkts)} (acc {len(acc)}, gyro {len(gyr)})")
    print(f"每类型速率 ≈ {len(acc)/(t[-1]-t[0]):.1f} Hz")
    dt = np.diff(t)
    print(f"时间戳间隔: 中位 {np.median(dt)*1000:.2f} ms, 最大 {dt.max()*1000:.2f} ms")

    acc_g = acc * ACCEL_SCALE            # g
    acc_mag = np.linalg.norm(acc_g, axis=1)
    gyr_dps = gyr * GYRO_SCALE           # deg/s
    print(f"\n加速度幅度: 静止均值 {acc_mag.mean():.2f}g, 最大 {acc_mag.max():.2f}g")
    print(f"角速度峰值: {np.abs(gyr_dps).max():.0f} deg/s")

    if acc_mag.max() > 1.3:
        print("✅ 摇晃时加速度超过 1.3g → 含真实线性加速度，可用于 SLAM")
    else:
        print("❌ 加速度幅度没超过 1.3g → 疑似被滤波（线性加速度被吃掉），需换原始输出模式")

    if np.abs(gyr_dps).max() > 100:
        print("✅ 角速度峰值 >100°/s，陀螺仪动态响应正常")

    np.savez_compressed(args.out, t=t, types=types,
                        accel=acc, gyro=gyr,
                        accel_scale=ACCEL_SCALE, gyro_scale=GYRO_SCALE)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
