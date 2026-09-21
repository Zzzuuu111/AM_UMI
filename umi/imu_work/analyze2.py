"""
Deep-dive: decode the DJI dbgi gyro/accel streams.

Hypothesis under test: the gyro block is a continuous 8192 Hz stream of
6-byte records [ts24(3B)][seq(1B)][val16(2B)], cut mid-record at packet
boundaries. The accel block is 4 such streams multiplexed (stride 24).

Verification: cross-correlate decoded val16 streams against angular
velocity derived from the Gyroflow CSV quaternions (independent source).
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from extract_dji_imu import _parse_fields, _get_fields  # noqa: E402


def get_blocks(pkt):
    body = _get_fields(_parse_fields(pkt), 1)
    body = _parse_fields(body[0])
    ts = _get_fields(body, 1)[0] if _get_fields(body, 1) else b""
    g200 = g201 = a2 = a3 = b""
    gb = _get_fields(body, 4)
    if gb:
        g = _parse_fields(gb[0])
        g200 = _get_fields(g, 200)[0] if _get_fields(g, 200) else b""
        g201 = _get_fields(g, 201)[0] if _get_fields(g, 201) else b""
    ab = _get_fields(body, 5)
    if ab:
        a = _parse_fields(ab[0])
        a2 = _get_fields(a, 2)[0] if _get_fields(a, 2) else b""
        a3 = _get_fields(a, 3)[0] if _get_fields(a, 3) else b""
    return ts, g200, g201, a2, a3


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


def parse_ts(ts):
    out = {}
    i = 0
    while i < len(ts):
        tag, i = read_varint(ts, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = read_varint(ts, i)
            out[fn] = v
        else:
            break
    return out


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    return np.array([
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ])


def main():
    db = pathlib.Path(__file__).parent / "dbgi_0009.bin"
    buf = db.read_bytes()
    packets = _get_fields(_parse_fields(buf), 2)
    n = len(packets)

    # ---- packet timestamps (us) ----
    t_us = np.array([parse_ts(get_blocks(p)[0]).get(2, 0) for p in packets], dtype=np.int64)

    # ---- concatenate gyro blocks ----
    gyro_stream = b"".join(get_blocks(p)[2] for p in packets)
    print(f"gyro concat len: {len(gyro_stream)}, packets: {n}")
    print(f"packet dt median: {np.median(np.diff(t_us))} us")

    # ---- find record phase: [ts24][seq][val16], 6B, seq increments +1 ----
    best = None
    for off in range(6):
        k = np.arange(off + 3, len(gyro_stream) - 6 + 1, 6)
        seq = np.frombuffer(gyro_stream, dtype=np.uint8)[k]
        # longest run of consecutive +1 (mod 256)
        d = (np.diff(seq.astype(np.int64)) == 1).astype(int)
        run = best_run = 0
        for x in d:
            run = run + 1 if x else 0
            best_run = max(best_run, run)
        if best is None or best_run > best[0]:
            best = (best_run, off)
    print(f"best seq-run: {best[0]} consecutive, offset {best[1]} (expect ~{len(gyro_stream)//6 - 2})")

    off = best[1]
    idx = np.arange(off, len(gyro_stream) - 6 + 1, 6)
    ts24 = np.array([int.from_bytes(gyro_stream[i:i + 3], "little") for i in idx], dtype=np.int64)
    seq = np.frombuffer(gyro_stream, dtype=np.uint8)[idx + 3].astype(np.int64)
    val16 = np.array([int.from_bytes(gyro_stream[i + 4:i + 6], "little", signed=True)
                      for i in idx], dtype=np.int64)

    # ---- ts24 steps (mod 2^24) should be multiples of 2^19 ----
    step = (np.diff(ts24) + (1 << 24)) % (1 << 24)
    mult = np.round(step / (1 << 19)).astype(np.int64)
    vals, cnts = np.unique(mult, return_counts=True)
    order = np.argsort(-cnts)
    print("\nts24 step multiples of 2^19:")
    for v, c in zip(vals[order][:8], cnts[order][:8]):
        print(f"  {v}x : {c} ({100*c/len(mult):.1f}%)")

    # implied sample rate from ts24: steps of 2^19 per sample at 2^32 Hz clock
    # -> 8192 Hz. Check against packet timestamps:
    total_us = t_us[-1] - t_us[0]
    rate_from_packets = len(val16) / (total_us / 1e6)
    print(f"\nsamples: {len(val16)}, packet-based rate: {rate_from_packets:.1f} Hz")

    # per-sample time (uniform 8192 Hz approximation)
    t_s = np.arange(len(val16)) / 8192.0

    # ---- reference: CSV quaternions -> angular velocity ----
    csv_path = pathlib.Path("/home/zzzjh/Osmo Action 4/Osma Action 4 Wide.csv")
    df = pd.read_csv(csv_path)
    tc = df["timestamp_ms"].values.astype(float) / 1000.0
    q = df[["org_quat_x", "org_quat_y", "org_quat_z", "org_quat_w"]].values.astype(float)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    q_dot = np.gradient(q, tc, axis=0)
    w = np.array([2 * quat_mul(q_dot[k], quat_conj(q[k]))[:3] for k in range(len(q))])
    print(f"\nCSV ref: {len(q)} rows, t=[{tc[0]:.3f},{tc[-1]:.3f}]s")
    print("CSV quat start:", np.round(q[0], 4), " end:", np.round(q[-1], 4))
    qrel = quat_mul(q[-1], quat_conj(q[0]))
    angle = 2 * np.arccos(np.clip(qrel[3], -1, 1)) * 180 / np.pi
    print(f"total rotation magnitude: {angle:.2f} deg, axis: {np.round(qrel[:3]/np.linalg.norm(qrel[:3]), 3)}")

    # rotation axis: which body axis dominates?
    for ax in range(3):
        print(f"  CSV |w_{'xyz'[ax]}| mean {np.abs(w[:, ax]).mean():.4f} rad/s, peak {np.abs(w[:, ax]).max():.2f} rad/s")

    # ---- correlate val16 vs CSV omega ----
    # resample val16 to CSV times (nearest)
    ti = np.searchsorted(t_s, tc, side="left").clip(0, len(val16) - 1)
    v_res = val16[ti]
    print("\nval16 vs CSV omega correlation (raw val16 resampled):")
    for ax in range(3):
        c = np.corrcoef(v_res, w[:, ax])[0, 1]
        cn = np.corrcoef(v_res, -w[:, ax])[0, 1]
        print(f"  axis {'xyz'[ax]}: corr {c:+.3f} / {-cn:+.3f}")

    # ---- decimation tests: maybe val16 interleaves axes ----
    for m in (2, 3, 4):
        for p in range(m):
            sub = val16[p::m]
            ts = t_s[p::m]
            ti2 = np.searchsorted(ts, tc, side="left").clip(0, len(sub) - 1)
            s_res = sub[ti2]
            bestc = max(abs(np.corrcoef(s_res, w[:, ax])[0, 1]) for ax in range(3))
            if bestc > 0.5:
                print(f"  decim m={m} phase={p}: best |corr| {bestc:.3f}")

    # ---- save extracted series for further work ----
    np.savez_compressed(
        pathlib.Path(__file__).parent / "gyro_stream_0009.npz",
        t_s=t_s, val16=val16, ts24=ts24, seq=seq, tc=tc, w=w, q=q, t_us=t_us)
    print("\nsaved gyro_stream_0009.npz")


if __name__ == "__main__":
    main()
