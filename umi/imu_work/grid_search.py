"""
Grid-search: which byte layout of the dbgi gyro/accel blocks encodes angular
velocity? Oracle = CSV-quaternion-derived omega(t).

For every (stride, phase, offset, width, byteorder) candidate we extract a
sub-stream, resample it to the CSV timeline and report the best |correlation|
with any of the 3 omega axes.
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
    g201 = a3 = b""
    gb = _get_fields(body, 4)
    if gb:
        g = _parse_fields(gb[0])
        g201 = _get_fields(g, 201)[0] if _get_fields(g, 201) else b""
    ab = _get_fields(body, 5)
    if ab:
        a = _parse_fields(ab[0])
        a3 = _get_fields(a, 3)[0] if _get_fields(a, 3) else b""
    return g201, a3


def decode(sub, width, order):
    n = len(sub) // width
    sub = sub[: n * width]
    a = np.frombuffer(sub, dtype=np.uint8).reshape(-1, width)
    if order == "BE":
        a = a[:, ::-1]
    if width == 1:
        return a[:, 0].astype(np.float64)
    if width == 2:
        v = a[:, 0].astype(np.int32) | (a[:, 1].astype(np.int32) << 8)
        return np.where(v >= 32768, v - 65536, v).astype(np.float64)
    if width == 3:
        v = (a[:, 0].astype(np.int64) | (a[:, 1].astype(np.int64) << 8)
             | (a[:, 2].astype(np.int64) << 16))
        return np.where(v >= (1 << 23), v - (1 << 24), v).astype(np.float64)
    if width == 4:
        v = (a[:, 0].astype(np.int64) | (a[:, 1].astype(np.int64) << 8)
             | (a[:, 2].astype(np.int64) << 16) | (a[:, 3].astype(np.int64) << 24))
        return v.astype(np.float64)
    raise ValueError(width)


def search(name, block, tc, w, g_b, w_int, t_total):
    results = []
    for s in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24):
        if len(block) < s * 8:
            continue
        for p in range(s):
            for (off, width, order) in ((0, 1, "LE"), (0, 2, "LE"), (0, 2, "BE"),
                                        (1, 2, "LE"), (1, 2, "BE"), (2, 2, "LE"),
                                        (2, 2, "BE"), (0, 3, "LE"), (0, 3, "BE"),
                                        (0, 4, "LE"), (0, 4, "BE")):
                if off + width > s:
                    continue
                sub = block[p::s]
                if len(sub) // width < 100:
                    continue
                v = decode(sub, width, order)
                # decimate to ~1 kHz before low-pass (speed)
                k = max(len(v) // len(tc), 1)
                vd = v[::k]
                if len(vd) < 16:
                    continue
                win = max(len(vd) // 64, 3)
                kernel = np.ones(win) / win
                v_lp = np.convolve(vd - vd.mean(), kernel, mode="same")
                td = np.linspace(0, t_total, len(vd))
                ti = np.searchsorted(td, tc, side="left").clip(0, len(v_lp) - 1)
                vr = v_lp[ti]
                vr = vr - vr.mean()
                c_gyro = max(abs(np.corrcoef(vr, w[:, ax])[0, 1]) for ax in range(3))
                c_angle = max(abs(np.corrcoef(vr, w_int[:, ax])[0, 1]) for ax in range(3))
                c_acc = max(abs(np.corrcoef(vr, g_b[:, ax])[0, 1]) for ax in range(3))
                results.append((max(c_gyro, c_angle, c_acc), c_gyro, c_angle, c_acc,
                                s, p, off, width, order, len(v)))
    results.sort(reverse=True)
    print(f"\n== {name}: top-15 layout candidates (LP 0.2s) ==")
    print("  best |corr|: gyro / angle / accel-gravity")
    for r in results[:15]:
        best, cg, ca, cc, s, p, off, width, order, n = r
        print(f"  best={best:.3f} (gyro {cg:+.2f}, angle {ca:+.2f}, grav {cc:+.2f})"
              f"  stride={s:2d} ph={p} off={off} w={width} {order} n={n}")
    return results[0]


def main():
    db = pathlib.Path(__file__).parent / "dbgi_0009.bin"
    buf = db.read_bytes()
    packets = _get_fields(_parse_fields(buf), 2)

    csv_path = pathlib.Path("/home/zzzjh/Osmo Action 4/Osma Action 4 Wide.csv")
    df = pd.read_csv(csv_path)
    tc = df["timestamp_ms"].values.astype(float) / 1000.0
    q = df[["org_quat_x", "org_quat_y", "org_quat_z", "org_quat_w"]].values.astype(float)
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]

    def qc(q_):
        return np.array([-q_[0], -q_[1], -q_[2], q_[3]])

    def qm(a, b):
        return np.array([a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
                         a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
                         a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
                         a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]])

    def quat_to_rot(q_):
        x, y, z, w = q_
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    q_dot = np.gradient(q, tc, axis=0)
    w = np.array([2 * qm(q_dot[k], qc(q[k]))[:3] for k in range(len(q))])
    w_int = np.cumsum(w, axis=0) * np.gradient(tc)[:, None]
    # gravity in body frame: world->body rotation
    g_world = np.array([0.0, 0.0, -9.81])
    g_b = np.array([quat_to_rot(q[k]) @ g_world for k in range(len(q))])
    t_total = tc[-1] - tc[0]

    gyro_stream = b"".join(get_blocks(p)[0] for p in packets)
    accel_stream = b"".join(get_blocks(p)[1] for p in packets)
    print(f"gyro concat {len(gyro_stream)}B, accel concat {len(accel_stream)}B, t={t_total:.3f}s")

    search("GYRO block (field 4)", gyro_stream, tc, w, g_b, w_int, t_total)
    search("ACCEL block (field 5)", accel_stream, tc, w, g_b, w_int, t_total)


if __name__ == "__main__":
    main()
