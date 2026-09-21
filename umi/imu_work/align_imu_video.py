"""
Align a JY901B recording with an Osmo Action 4 video and export
gopro_slam-format imu_data.json.

Method: the camera's djmd stream contains fused attitude quaternions at
~1000 Hz, natively synchronized to the video. Differentiate them to get a
reference angular velocity |w_video(t_video)|. Cross-correlate its magnitude
with the JY901B gyro magnitude |w_imu(t_host)| to find the time offset
dt = t_video - t_host. Then export ACCL/GYRO with video-relative cts (ms).

Usage:
  python imu_work/align_imu_video.py \
      --imu imu_work/sync_session.npz \
      --video "/path/to/DJI_xxx.MP4" \
      --out imu_work/imu_data.json
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


def extract_video_omega(video_path):
    """Return (t_vid_s, w_mag_rads) from the djmd stream of an OA4 MP4."""
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
        raise RuntimeError(f"djmd 提取长度不符 {len(dj)} != {sum(sizes)}")
    pkts = []
    i = 0
    for s in sizes:
        pkts.append(dj[i:i + s])
        i += s

    quats = []      # (t_us_dev, x, y, z, w)
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

    def qc(q_):
        return np.array([-q_[0], -q_[1], -q_[2], q_[3]])

    def qm(a, b):
        return np.array([
            a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
            a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
            a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
            a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]])

    qd = np.gradient(q, t, axis=0)
    w = np.array([2 * qm(qd[k], qc(q[k]))[:3] for k in range(len(q))])
    wmag = np.linalg.norm(w, axis=1)
    return t, wmag, t[-1]


def cross_correlate(t_ref, a_ref, t_imu, a_imu, max_lag_s=20.0):
    """Both series live on different clocks. Normalize each to its own
    start, resample to 1000Hz, then correlate the shapes. Returns
    dt = t_ref - t_imu (seconds) and the normalized peak."""
    fs = 1000.0
    dur_ref = t_ref[-1] - t_ref[0]
    dur_imu = t_imu[-1] - t_imu[0]
    if dur_ref < 5 or dur_imu < 5:
        raise RuntimeError("两路数据时长不足 5 秒，检查录制")
    t_ref_n = t_ref - t_ref[0]
    t_imu_n = t_imu - t_imu[0]
    g_ref = np.arange(0, dur_ref, 1 / fs)
    g_imu = np.arange(0, dur_imu, 1 / fs)
    a_ref_r = np.interp(g_ref, t_ref_n, a_ref)
    a_imu_r = np.interp(g_imu, t_imu_n, a_imu)
    a_ref_r = a_ref_r - a_ref_r.mean()
    a_imu_r = a_imu_r - a_imu_r.mean()
    a_ref_r /= (a_ref_r.std() + 1e-9)
    a_imu_r /= (a_imu_r.std() + 1e-9)
    # numpy convention: c[k] = sum a[n+k] v[n], k in [-(len(v)-1), len(a)-1]
    corr = np.correlate(a_ref_r, a_imu_r, mode="full")
    k = np.arange(len(corr)) - (len(a_imu_r) - 1)
    sel = np.abs(k) <= max_lag_s * fs
    kk = k[sel]
    cc = corr[sel]
    best = int(np.argmax(cc))
    dt = kk[best] / fs  # t_ref - t_imu
    return dt, cc[best] / len(g_ref), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="imu_work/imu_data.json")
    ap.add_argument("--accel-calib", default=None,
                    help="accel_calib.json from calib_accel.py (optional)")
    args = ap.parse_args()

    d = np.load(args.imu)
    t_a, accel_raw = d["t_a"], d["accel"]
    t_g, gyro_raw = d["t_g"], d["gyro"]
    acc_scale = float(d["accel_scale"])
    gyr_scale = float(d["gyro_scale"])

    accel_calib = None
    if args.accel_calib:
        with open(args.accel_calib) as f:
            accel_calib = json.load(f)
        print(f"应用加速度标定: scale={accel_calib['scale']} "
              f"bias={accel_calib['bias']}")

    t_ref, w_ref, t_video_end = extract_video_omega(args.video)
    print(f"视频参考角速度: {len(t_ref)} 样本, 时长 {t_video_end:.2f}s")

    w_imu = np.linalg.norm(gyro_raw.astype(np.float64) * gyr_scale, axis=1) * np.pi / 180.0
    print(f"IMU 角速度: {len(t_g)} 样本, 时长 {t_g[-1]-t_g[0]:.2f}s")

    dt, corr_peak, grid = cross_correlate(t_ref, w_ref, t_g, w_imu)
    print(f"\n时间偏移 Δ = t_video - t_host = {dt*1000:+.1f} ms")
    print(f"互相关峰: {corr_peak:.3f}  (>0.5 对齐可靠; <0.3 需重录)")

    # ---- export imu_data.json ----
    # cts = video-relative ms = (t_host - imu_start) + dt, dt = t_video - t_imu(normalized)
    cts_a = (t_a - t_a[0] + dt) * 1000.0
    cts_g = (t_g - t_g[0] + dt) * 1000.0
    dur_ms = t_video_end * 1000.0
    sel_a = (cts_a >= 0) & (cts_a <= dur_ms)
    sel_g = (cts_g >= 0) & (cts_g <= dur_ms)
    accel_mps2 = accel_raw.astype(np.float64) * acc_scale * 9.80665
    if accel_calib is not None:
        s = np.array(accel_calib["scale"])
        b = np.array(accel_calib["bias"])
        # fit model: measured = s * true + b  ->  true = (measured - b) / s
        accel_mps2 = (accel_mps2 - b) / s
    gyro_rads = gyro_raw.astype(np.float64) * gyr_scale * np.pi / 180.0

    accl = [{"value": [float(x), float(y), float(z)], "cts": float(c)}
            for (x, y, z), c in zip(accel_mps2[sel_a], cts_a[sel_a])]
    gyro = [{"value": [float(x), float(y), float(z)], "cts": float(c)}
            for (x, y, z), c in zip(gyro_rads[sel_g], cts_g[sel_g])]
    cori = [{"cts": float(c)} for c in cts_a[sel_a]]

    # ---- pad tail to full video duration (gopro_slam reads IMU with no
    # bounds check; the IMU stream MUST cover the whole video) ----
    rate_hz = 201.0
    step_ms = 1000.0 / rate_hz
    end_covered = max(cts_a[sel_a].max(), cts_g[sel_g].max())
    if end_covered < dur_ms - step_ms:
        pad_ms = end_covered - dur_ms
        print(f"\n⚠️ IMU 只覆盖视频的 [0, {end_covered/1000:.2f}s]，"
              f"尾部 {abs(pad_ms)/1000:.2f}s 用最后一个样本填充")
        print("   （正式录制请让 IMU 先于视频开始、晚于视频结束，留 ≥5s 余量）")
        pad_ts = np.arange(end_covered + step_ms, dur_ms + 1e-6, step_ms)
        last_a = accel_mps2[sel_a][-1].tolist()
        last_g = gyro_rads[sel_g][-1].tolist()
        for c in pad_ts:
            accl.append({"value": last_a, "cts": float(c)})
            gyro.append({"value": last_g, "cts": float(c)})
            cori.append({"cts": float(c)})

    result = {"1": {"streams": {
        "ACCL": {"samples": accl},
        "GYRO": {"samples": gyro},
        "CORI": {"samples": cori},
    }}}
    out = pathlib.Path(args.out)
    with open(out, "w") as f:
        json.dump(result, f)
    print(f"\n已导出: {out}")
    print(f"  ACCL {len(accl)} 样本 ({len(accl)/t_video_end:.0f} Hz), "
          f"GYRO {len(gyro)} 样本 ({len(gyro)/t_video_end:.0f} Hz)")
    print(f"  时间偏移记录: {dt*1000:+.1f} ms")


if __name__ == "__main__":
    main()
