"""Inspect raw V4L2 UVCH (UVC payload header metadata) captured by v4l2-ctl.

The metadata stream consists of concatenated Linux ``uvc_meta_buf`` blocks:
host-reception timestamp (u64 ns), USB SOF (u16), followed by an exact UVC
payload header.  This utility reports whether the camera actually supplies
PTS and SCR fields.  It never opens a camera or changes camera controls.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics


UVC_STREAM_PTS = 1 << 2
UVC_STREAM_SCR = 1 << 3


def median_delta(values: list[int], scale: float = 1.0):
    if len(values) < 2:
        return None
    values = sorted(values)
    delta = [b - a for a, b in zip(values, values[1:]) if b > a]
    return None if not delta else statistics.median(delta) * scale


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 EMEET UVC UVCH 元数据中的 PTS/SCR")
    parser.add_argument("--input", required=True, help="v4l2-ctl --stream-to 生成的 UVCH 二进制文件")
    args = parser.parse_args()
    path = pathlib.Path(args.input).expanduser().resolve()
    data = path.read_bytes()
    offset, invalid = 0, 0
    host_ns, pts, scr_stc, sof = [], [], [], []
    while offset + 12 <= len(data):
        timestamp_ns = int.from_bytes(data[offset:offset + 8], "little")
        usb_sof = int.from_bytes(data[offset + 8:offset + 10], "little")
        header_len = data[offset + 10]
        if not 2 <= header_len <= 12 or offset + 10 + header_len > len(data):
            invalid += 1
            offset += 1
            continue
        header = data[offset + 10:offset + 10 + header_len]
        flags = header[1]
        host_ns.append(timestamp_ns)
        sof.append(usb_sof)
        cursor = 2
        if flags & UVC_STREAM_PTS:
            if cursor + 4 <= len(header):
                pts.append(int.from_bytes(header[cursor:cursor + 4], "little"))
            cursor += 4
        if flags & UVC_STREAM_SCR:
            if cursor + 6 <= len(header):
                scr_stc.append(int.from_bytes(header[cursor:cursor + 4], "little"))
            cursor += 6
        offset += 10 + header_len
    print("UVC_PAYLOAD_METADATA_INSPECTION")
    print("input:", path)
    print("bytes:", len(data), "parsed_blocks:", len(host_ns), "resync_bytes:", invalid)
    print("host_reception_timestamp_blocks:", len(host_ns))
    print("PTS_blocks:", len(pts), "present:", "YES" if pts else "NO")
    print("SCR_blocks:", len(scr_stc), "present:", "YES" if scr_stc else "NO")
    host_delta_ms = median_delta(host_ns, 1e-6)
    if host_delta_ms is not None:
        print(f"host_timestamp_delta_median_ms: {host_delta_ms:.3f}")
    pts_delta = median_delta(pts)
    if pts_delta is not None:
        print(f"PTS_delta_median_ticks: {pts_delta:.1f}")
    scr_delta = median_delta(scr_stc)
    if scr_delta is not None:
        print(f"SCR_STC_delta_median_ticks: {scr_delta:.1f}")
    if not host_ns:
        raise SystemExit("未解析出 UVCH 块：检查 video2 与 video3 是否在同一时间启动流")
    if not pts and not scr_stc:
        print("RESULT: metadata node works, but this camera did not expose device PTS/SCR in this capture.")
    elif scr_stc:
        print("RESULT: camera exposes SCR source-clock data; it can be used for improved device-to-host timing analysis.")
    else:
        print("RESULT: camera exposes PTS but not SCR; useful, but source-clock-to-host alignment is weaker.")


if __name__ == "__main__":
    main()
