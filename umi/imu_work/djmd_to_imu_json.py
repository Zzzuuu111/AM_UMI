"""
Generate gopro_slam imu_data.json directly from the video's own djmd stream
(fused attitude quaternions, natively synchronized to the video).

  GYRO = differentiate quaternions: w = 2 q_dot * q^-1
  ACCL = synthetic gravity in body frame: R^-1 * [0,0,-9.81]
  CORI = quaternions (cts only, matching LoadTelemetry expectations)

No JY901B, no cross-correlation alignment, no tail padding needed: djmd
covers the whole video by construction and its timestamps are video-native.

Use for DEMO relocalization runs (map already has scale; real IMU is not
needed and its interaction with the fork's relocalization path is buggy).

Usage:
  python imu_work/djmd_to_imu_json.py -i <video.mp4> -o <dir>/imu_data.json
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from extract_dji_imu import _parse_fields  # noqa: E402


def read_varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not (b & 0x80):
            break
        s += 7
    return r, i


def parse_fields(buf):
    out = []
    i = 0
    while i < len(buf):
        try:
            tag, i = read_varint(buf, i)
        except IndexError:
            break
        fn, wt = tag >> 3, tag & 7
        if fn == 0:
            break
        if wt == 0:
            v, i = read_varint(buf, i)
            out.append((fn, wt, v))
        elif wt == 1:
            out.append((fn, wt, buf[i:i + 8]))
            i += 8
        elif wt == 2:
            ln, i = read_varint(buf, i)
            out.append((fn, wt, buf[i:i + ln]))
            i += ln
        elif wt == 5:
            out.append((fn, wt, buf[i:i + 4]))
            i += 4
        else:
            break
    return out


def extract_quats(video_path):
    sizes_out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_packets", "-select_streams", "2",
         "-show_entries", "packet=size", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True)
    sizes = [int(x) for x in sizes_out.stdout.split()]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video_path),
         "-map", "0:2", "-c", "copy", "-f", "data", "-"],
        capture_output=True)
    dj = raw.stdout
    if len(dj) != sum(sizes):
        raise RuntimeError("djmd 提取长度不符")
    pkts = []
    i = 0
    for s in sizes:
        pkts.append(dj[i:i + s])
        i += s

    quats = []
    first_ts = None
    for pkt in pkts[1:]:
        fm = [v for fn, wt, v in parse_fields(pkt) if fn == 3]
        if not fm:
            continue
        fields = parse_fields(fm[0])
        t_us = 0
        hdr = [v for fn, wt, v in fields if fn == 1]
        if hdr:
            hf = parse_fields(hdr[0])
            tv = [v for fn, wt, v in hf if fn == 2]
            if tv:
                t_us = tv[0]
        if first_ts is None:
            first_ts = t_us
        imu = [v for fn, wt, v in fields if fn == 3]
        if not imu:
            continue
        att = [v for fn, wt, v in parse_fields(imu[0]) if fn == 2]
        if not att:
            continue
        qs = []
        for fn, wt, v in parse_fields(att[0]):
            if fn == 3:
                d = {a: struct.unpack("<f", c)[0] for a, b, c in parse_fields(v) if b == 5}
                qs.append([d.get(2), d.get(3), d.get(4), d.get(1)])  # x,y,z,w
        n = len(qs)
        for j, q in enumerate(qs):
            quats.append((t_us + j * 1_000_000 / 59.94 / n, q))
    q = np.array([x[1] for x in quats])
    t = np.array([(x[0] - first_ts) / 1e6 for x in quats])
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return t, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    t, q = extract_quats(args.input)
    print(f"djmd 四元数: {len(q)} 个, 时长 {t[-1]:.2f}s")

    def qc(x):
        return np.array([-x[0], -x[1], -x[2], x[3]])

    def qm(a, b):
        return np.array([
            a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
            a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
            a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
            a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]])

    def quat_to_rot(x):
        x0, y0, z0, w0 = x
        return np.array([
            [1-2*(y0*y0+z0*z0), 2*(x0*y0-z0*w0), 2*(x0*z0+y0*w0)],
            [2*(x0*y0+z0*w0), 1-2*(x0*x0+z0*z0), 2*(y0*z0-x0*w0)],
            [2*(x0*z0-y0*w0), 2*(y0*z0+x0*w0), 1-2*(x0*x0+y0*y0)]])

    q_dot = np.gradient(q, t, axis=0)
    w = np.array([2 * qm(q_dot[k], qc(q[k]))[:3] for k in range(len(q))])
    g_world = np.array([0.0, 0.0, -9.81])
    a = np.array([quat_to_rot(q[k]) @ g_world for k in range(len(q))])

    cts = (t * 1000.0)
    accl = [{"value": [float(x), float(y), float(z)], "cts": float(c)}
            for (x, y, z), c in zip(a, cts)]
    gyro = [{"value": [float(x), float(y), float(z)], "cts": float(c)}
            for (x, y, z), c in zip(w, cts)]
    cori = [{"cts": float(c)} for c in cts]
    result = {"1": {"streams": {
        "ACCL": {"samples": accl},
        "GYRO": {"samples": gyro},
        "CORI": {"samples": cori}}}}
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"已导出: {out}  ({len(accl)} 样本 @ {len(accl)/t[-1]:.0f} Hz)")


if __name__ == "__main__":
    main()
