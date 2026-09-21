"""
Six-position accelerometer calibration for the JY901B.

Place the module still on each of its 6 faces for ~5s each, following the
prompts. The script fits per-axis scale + bias so that |a| = 9.81 m/s^2 in
every orientation.

Usage:
  python imu_work/calib_accel.py --out imu_work/accel_calib.json
"""
import argparse
import json
import time

import numpy as np
import serial

FACES = [
    ("Z 轴朝上（正面朝上平放）", np.array([0.0, 0.0, 9.81])),
    ("Z 轴朝下（翻面平放）",     np.array([0.0, 0.0, -9.81])),
    ("X 轴朝上（侧立，长边朝上）", np.array([9.81, 0.0, 0.0])),
    ("X 轴朝下（侧立，长边朝下）", np.array([-9.81, 0.0, 0.0])),
    ("Y 轴朝上（另一侧立）",     np.array([0.0, 9.81, 0.0])),
    ("Y 轴朝下（另一侧立，朝下）", np.array([0.0, -9.81, 0.0])),
]
SETTLE = 2.0
MEASURE = 3.0


def countdown(seconds, label):
    for k in range(int(seconds), 0, -1):
        print(f"\r    {label} {k}s ...", end="", flush=True)
        time.sleep(1)
    print(f"\r    {label} 开始            ", flush=True)


def read_means(port, baud):
    ser = serial.Serial(port, baud, timeout=0.05)
    buf = b""
    t0 = time.time()
    last = -1
    while time.time() - t0 < MEASURE:
        chunk = ser.read(2048)
        if chunk:
            buf += chunk
        sec = int(time.time() - t0)
        if sec != last and sec < MEASURE:
            last = sec
            print(f"\r    测量中 {sec+1}/{int(MEASURE)}s", end="", flush=True)
    print()
    ser.close()
    acc = []
    i = 0
    n = len(buf)
    while i < n - 11:
        if buf[i] == 0x55 and buf[i + 1] == 0x51:
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:
                x = int.from_bytes(pkt[2:4], "little", signed=True)
                y = int.from_bytes(pkt[4:6], "little", signed=True)
                z = int.from_bytes(pkt[6:8], "little", signed=True)
                acc.append([x, y, z])
                i += 11
                continue
        i += 1
    if not acc:
        raise RuntimeError("没读到数据，检查端口/波特率")
    a = np.array(acc, dtype=np.float64) * 16.0 / 32768.0  # g
    return a.mean(axis=0), len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--out", default="imu_work/accel_calib.json")
    args = ap.parse_args()

    measured = []
    for idx, (name, g_true) in enumerate(FACES, 1):
        input(f"\n>>> [{idx}/6] 拿着夹爪把模块摆成【{name}】，靠在桌面/书堆上保持完全静止，然后按回车")
        countdown(SETTLE, "静置")
        mean_g, n = read_means(args.port, args.baud)
        measured.append(mean_g * 9.80665)  # m/s^2
        print(f"    实测 (m/s²): [{mean_g[0]*9.80665:+.2f}, "
              f"{mean_g[1]*9.80665:+.2f}, {mean_g[2]*9.80665:+.2f}]  "
              f"({n} 样本)")

    M = np.array(measured)
    # Magnitude-only fit: orientations need NOT be exact. Find per-axis
    # scale s and bias b minimizing sum(|(m_k - b)/s| - 9.81)^2 over poses.
    from scipy.optimize import least_squares

    def resid(p):
        s = p[:3]
        b = p[3:]
        out = []
        for m in M:
            a = (m - b) / s
            out.append(np.linalg.norm(a) - 9.81)
        return np.array(out)

    x0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    sol = least_squares(resid, x0, bounds=([0.8, 0.8, 0.8, -5, -5, -5],
                                           [1.2, 1.2, 1.2, 5, 5, 5]))
    scales = sol.x[:3].tolist()
    biases = sol.x[3:].tolist()
    # residual check
    resid = []
    for m in M:
        a = (m - np.array(biases)) / np.array(scales)
        resid.append(np.linalg.norm(a) - 9.81)
    print("\n== 标定结果 ==")
    print(f"  scale : {['%.5f' % s for s in scales]}  (理想 1.0000)")
    print(f"  bias  : {['%.3f' % b for b in biases]} m/s²  (理想 0)")
    print(f"  各姿态模长残差: {['%+.3f' % r for r in resid]} m/s² (应 <0.2)")
    calib = {"scale": scales, "bias": biases, "unit": "m/s2",
             "raw_scale_g": 16.0 / 32768.0}
    with open(args.out, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"已保存: {args.out}")
    print("校正公式: a_true = (raw*16/32768*9.80665 - bias) / scale")


if __name__ == "__main__":
    main()
