"""
Analyze DJI Osmo Action 4 dbgi stream structure (imu_work/ exploratory script).

Steps:
  1. Parse protobuf: field2 packets -> {timestamp varint, gyro meta, gyro raw,
     accel meta, accel raw}.
  2. Decode timestamp varints, check monotonicity / deltas.
  3. Detect the sequence-counter stride in gyro raw (incrementing byte at
     fixed stride).
  4. Test candidate sample layouts (int24/int16/float32, 1 or 3 axes) for
     physical plausibility against the known motion of 0009 (90 deg rotation).
"""
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from extract_dji_imu import _parse_fields, _get_fields  # noqa: E402


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


def parse_ts(ts):
    """Decode the timestamp field bytes: field2 varint = ts, field3 varint = idx."""
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


def detect_counter_stride(raw, max_stride=16):
    """Find stride s and phase p such that raw[p::s] is an incrementing
    sequence for a long run. Returns (stride, phase, run_len, start_value)."""
    b = np.frombuffer(raw, dtype=np.uint8)
    best = None
    for s in range(2, max_stride + 1):
        for p in range(s):
            seq = b[p::s].astype(np.int64)
            # count length of longest run where seq[k+1] == seq[k]+1
            run = best_run = 1
            for k in range(1, len(seq)):
                if seq[k] == seq[k - 1] + 1:
                    run += 1
                else:
                    best_run = max(best_run, run)
                    run = 1
            best_run = max(best_run, run)
            if best is None or best_run > best[2]:
                best = (s, p, best_run, int(seq[0]))
    return best


def main():
    dbfile = pathlib.Path(__file__).parent / "dbgi_0009.bin"
    buf = dbfile.read_bytes()
    top = _parse_fields(buf)
    packets = _get_fields(top, 2)
    print(f"packets: {len(packets)}")

    # ---- 1. timestamps ----
    tsvals = []
    for pkt in packets:
        ts, *_ = get_blocks(pkt)
        tsvals.append(parse_ts(ts))
    f2 = [t.get(2, 0) for t in tsvals]
    f3 = [t.get(3, 0) for t in tsvals]
    f2 = np.array(f2, dtype=np.int64)
    f3 = np.array(f3, dtype=np.int64)
    print("ts field2 first 10:", f2[:10])
    print("ts field2 last 3 :", f2[-3:])
    d = np.diff(f2)
    print("ts delta: min %.0f max %.0f median %.0f" % (d.min(), d.max(), np.median(d)))
    print("ts field3 (idx) first/last:", f3[:5], f3[-3:], "monotonic:", bool((np.diff(f3) == 1).all()))
    # try interpreting delta in us / ns / 90kHz ticks
    for name, scale in [("us", 1e6), ("ns", 1e9), ("90kHz", 90000.0)]:
        print(f"  delta in {name}: median {np.median(d)/scale*1e3:.3f} ms")
    # forward vs backward: check whether f2 monotonic increasing or decreasing
    inc = bool((np.diff(f2) > 0).all())
    dec = bool((np.diff(f2) < 0).all())
    print("f2 strictly increasing:", inc, "| strictly decreasing:", dec)

    # ---- 2. counter stride in gyro raw ----
    p0 = get_blocks(packets[0])
    print("\ngyro meta 19B:", p0[1].hex())
    print("accel meta 289B head:", p0[3][:40].hex())
    for pi in [0, 1, 100, 505]:
        g = get_blocks(packets[pi])[2]
        s, ph, run, v0 = detect_counter_stride(g)
        print(f"pkt {pi}: gyro raw len={len(g)} counter stride={s} phase={ph} run={run} start={v0}")

    # ---- 3. accel raw: counter stride + constant runs ----
    for pi in [0, 100]:
        a = get_blocks(packets[pi])[4]
        s, ph, run, v0 = detect_counter_stride(a, max_stride=24)
        print(f"pkt {pi}: accel raw len={len(a)} counter stride={s} phase={ph} run={run} start={v0}")
    a0 = get_blocks(packets[0])[4]
    b = np.frombuffer(a0, dtype=np.uint8)
    # most common byte value
    vals, cnts = np.unique(b, return_counts=True)
    top = np.argsort(-cnts)[:5]
    print("accel pkt0 top bytes:", [(hex(int(vals[i])), int(cnts[i])) for i in top])


if __name__ == "__main__":
    main()
