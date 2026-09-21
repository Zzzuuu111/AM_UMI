"""
Parse the djmd stream (dvtm_ac203 protobuf) and dump the full field tree of
the first few FrameMeta messages, including the IMU frame meta
(DeviceAttitude quaternion array).
"""
import pathlib
import sys

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


def parse_fields(buf):
    out = []
    i = 0
    while i < len(buf):
        try:
            tag, i = read_varint(buf, i)
        except IndexError:
            break
        fn = tag >> 3
        wt = tag & 7
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


def describe(buf, indent=0, max_depth=4, label=""):
    pad = "  " * indent
    print(f"{pad}{label} ({len(buf)}B):")
    if indent >= max_depth:
        return
    for fn, wt, v in parse_fields(buf):
        if wt == 2 and len(v) > 2:
            # try: nested message vs blob
            sub = parse_fields(v)
            if sub and all(s[0] > 0 for s in sub):
                print(f"{pad}  field {fn} (len {len(v)}):")
                describe(v, indent + 2, max_depth, "")
                continue
        if wt == 2:
            if len(v) <= 24 and all(32 <= b < 127 for b in v):
                print(f"{pad}  field {fn}: str {v.decode()!r}")
            else:
                print(f"{pad}  field {fn}: bytes {v[:16].hex()}{'...' if len(v) > 16 else ''} (len {len(v)})")
        elif wt == 5:
            import struct
            print(f"{pad}  field {fn}: float {struct.unpack('<f', v)[0]:.6g}")
        else:
            print(f"{pad}  field {fn}: varint {v}")


def main():
    db = pathlib.Path(__file__).parent / "djmd_0009.bin"
    buf = db.read_bytes()
    print(f"djmd total: {len(buf)}B")

    # split top-level samples: each MP4 sample = ProductMeta{1:clip?,2:stream?,3:frame}
    # parse whole buffer as repeated length-delimited top-level messages
    samples = []
    i = 0
    while i < len(buf):
        ln, i = read_varint(buf, i)
        samples.append(buf[i:i + ln])
        i += ln
    print(f"top-level samples: {len(samples)}")
    print(f"sample0 len: {len(samples[0])}, sample1 len: {len(samples[1])}")

    print("\n=== sample 0 (clip meta) ===")
    describe(samples[0], 0, 3, "ProductMeta")

    print("\n=== sample 1 (first frame) ===")
    describe(samples[1], 0, 4, "ProductMeta(frame)")

    # count quaternions in frame metas across all samples
    n_quat = []
    for s in samples[1:20]:
        f3 = [v for fn, wt, v in parse_fields(s) if fn == 3]
        if not f3:
            continue
        frame = parse_fields(f3[0])
        imu = [v for fn, wt, v in frame if fn == 3]
        nq = 0
        if imu:
            imu_f = parse_fields(imu[0])
            att = [v for fn, wt, v in imu_f if fn == 2]
            if att:
                att_f = parse_fields(att[0])
                qs = [v for fn, wt, v in att_f if fn == 3]
                if qs:
                    nq = len(parse_fields(qs[0]))
        n_quat.append(nq)
    print(f"\nquaternions per frame (samples 1-20): {n_quat}")


if __name__ == "__main__":
    main()
