"""
Record a JY901B sync session for later alignment with a camera video.

Guided workflow with countdown prompts:
  0-5s:   prepare; press the camera record button at the prompt
  5s-…:   recording (move the camera: rotate + translate)
  last 10s: press stop on the camera, IMU keeps recording (margin)

Ctrl+C stops the IMU recording early and still saves the data.

Usage:
  python imu_work/record_sync_session.py --seconds 90 --out imu_work/sync_xxx.npz
"""
import argparse
import time

import numpy as np
import serial

ACCEL_SCALE = 16.0 / 32768.0      # g per LSB
GYRO_SCALE = 2000.0 / 32768.0     # deg/s per LSB


def parse_reads(reads):
    """Assign per-packet host timestamps: distribute each read's bytes
    uniformly over its own [t_before, t_after] interval."""
    accel = []
    gyro = []
    for t_before, t_after, chunk in reads:
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
                    if pkt[1] == 0x51:
                        accel.append((t_pkt, x, y, z))
                    else:
                        gyro.append((t_pkt, x, y, z))
                    i += 11
                    continue
            i += 1
    if not accel:
        raise RuntimeError("没有解析到数据（检查波特率/接线）")
    t_a = np.array([a[0] for a in accel])
    a = np.array([a[1:] for a in accel], dtype=np.int16)
    t_g = np.array([g[0] for g in gyro])
    g = np.array([g[1:] for g in gyro], dtype=np.int16)
    return t_a, a, t_g, g


def parse_reads_precise_with_device_time(reads, baud):
    """Decode a continuous JY901B byte stream with UART-rate packet times.

    ``serial.read`` boundaries are unrelated to JY901B packet boundaries.  The
    older parser treated each read independently, which could lose a packet
    split across two reads and gave every packet a timestamp spread over the
    reader timeout.  Here the receive-end time and UART's 8N1 byte time give a
    consistent sub-millisecond timestamp for the centre of every complete
    packet.  This is still a host-clock estimate, not hardware triggering.
    """
    if baud <= 0:
        raise ValueError("baud 必须为正数")
    byte_s = 10.0 / float(baud)  # 8N1: start + 8 payload + stop bits
    packet_bytes = bytearray()
    packet_times = []
    accel, gyro = [], []
    accel_host, gyro_host = [], []
    accel_segment, gyro_segment = [], []
    # JY901B can emit a 0x50 RTC packet before every 0x51/0x52/0x53 group.
    # Its calendar fields are sometimes unset (year/month/day = zero), so we
    # only require its time-of-day millisecond counter to be monotonic within a
    # capture.  It is still a device-side sample clock, independent of USB
    # delivery batching on the host.
    device_clock_host, device_clock_s, device_clock_segment = [], [], []
    last_device_s = None
    device_day_offset_s = 0.0
    current_device_s = None
    current_segment = 0
    previous_end = None
    for _, t_after, chunk in reads:
        if not chunk:
            continue
        # Bytes in a received chunk arrived serially immediately before the
        # host received its final byte.  Clamp only pathological overlaps from
        # a coarse host clock; normal gaps intentionally remain gaps.
        first_time = float(t_after) - (len(chunk) - 1) * byte_s
        if previous_end is not None:
            first_time = max(first_time, previous_end + byte_s)
        byte_times = [first_time + index * byte_s for index in range(len(chunk))]
        previous_end = byte_times[-1]
        packet_bytes.extend(chunk)
        packet_times.extend(byte_times)
        while len(packet_bytes) >= 11:
            if packet_bytes[0] != 0x55 or packet_bytes[1] not in (0x50, 0x51, 0x52, 0x53):
                del packet_bytes[0]
                del packet_times[0]
                continue
            pkt = packet_bytes[:11]
            if (sum(pkt[:10]) & 0xFF) != pkt[10]:
                del packet_bytes[0]
                del packet_times[0]
                continue
            # Mid-packet time is less biased than receive-end time by roughly
            # half a UART packet.  Sensor internal output latency is calibrated
            # later against the camera through the time-offset solve.
            t_pkt = packet_times[5]
            packet_type = pkt[1]
            if packet_type == 0x50:
                raw_s = (pkt[5] * 3600.0 + pkt[6] * 60.0 + pkt[7]
                         + int.from_bytes(pkt[8:10], "little") / 1000.0)
                previous_device_s = current_device_s
                if last_device_s is not None and raw_s < last_device_s - 1.0:
                    # A time-of-day rollover; captures here are minutes, but
                    # this makes long static recordings safe as well.
                    device_day_offset_s += 24.0 * 3600.0
                current_device_s = raw_s + device_day_offset_s
                # A serial device may contain one stale packet when opened.
                # It cannot be part of a 200 Hz device clock if the next 0x50
                # value is seconds or minutes away.  Keep segments separate
                # and later retain the longest coherent one.
                if (previous_device_s is not None
                        and abs(current_device_s - previous_device_s) > 1.0):
                    current_segment += 1
                last_device_s = raw_s
                device_clock_s.append(current_device_s)
                device_clock_host.append(t_pkt)
                device_clock_segment.append(current_segment)
            elif packet_type in (0x51, 0x52):
                xyz = tuple(int.from_bytes(pkt[index:index + 2], "little", signed=True)
                            for index in (2, 4, 6))
                # If time output was not enabled, preserve the prior UART
                # timestamp behavior instead of silently inventing a clock.
                sample_time = current_device_s if current_device_s is not None else t_pkt
                target, raw_host = ((accel, accel_host) if packet_type == 0x51
                                    else (gyro, gyro_host))
                target.append((sample_time, *xyz))
                raw_host.append(t_pkt)
                (accel_segment if packet_type == 0x51 else gyro_segment).append(current_segment)
            del packet_bytes[:11]
            del packet_times[:11]
    if not accel or not gyro:
        raise RuntimeError("没有解析到完整的 JY901B 加速度/陀螺包")
    # Select the longest coherent device-clock segment.  This removes a stale
    # packet left in the USB serial driver before a fresh recording begins.
    if device_clock_s:
        counts = {}
        for segment in accel_segment + gyro_segment:
            counts[segment] = counts.get(segment, 0) + 1
        keep_segment = max(counts, key=counts.get) if counts else current_segment
        accel_keep = [i for i, segment in enumerate(accel_segment) if segment == keep_segment]
        gyro_keep = [i for i, segment in enumerate(gyro_segment) if segment == keep_segment]
        marker_keep = [i for i, segment in enumerate(device_clock_segment)
                       if segment == keep_segment]
    else:
        accel_keep = list(range(len(accel)))
        gyro_keep = list(range(len(gyro)))
        marker_keep = []
    t_a_raw = np.asarray([accel[i][0] for i in accel_keep], dtype=np.float64)
    a = np.asarray([accel[i][1:] for i in accel_keep], dtype=np.int16)
    t_g_raw = np.asarray([gyro[i][0] for i in gyro_keep], dtype=np.float64)
    g = np.asarray([gyro[i][1:] for i in gyro_keep], dtype=np.int16)
    t_a_host = np.asarray([accel_host[i] for i in accel_keep], dtype=np.float64)
    t_g_host = np.asarray([gyro_host[i] for i in gyro_keep], dtype=np.float64)
    device_clock_s = np.asarray(device_clock_s, dtype=np.float64)
    device_clock_host = np.asarray(device_clock_host, dtype=np.float64)
    if marker_keep:
        device_clock_s = device_clock_s[marker_keep]
        device_clock_host = device_clock_host[marker_keep]
    uses_device_clock = len(device_clock_s) >= 2 and current_device_s is not None
    if uses_device_clock:
        # Fit the device clock onto host monotonic time.  The fit absorbs USB
        # receive jitter while retaining JY901B's uniform 5 ms sample clock.
        x0 = float(np.mean(device_clock_s))
        y0 = float(np.mean(device_clock_host))
        x = device_clock_s - x0
        y = device_clock_host - y0
        denom = float(np.dot(x, x))
        scale = float(np.dot(x, y) / denom) if denom > 0 else 1.0
        t_a = y0 + scale * (t_a_raw - x0)
        t_g = y0 + scale * (t_g_raw - x0)
        t_a_device, t_g_device = t_a_raw, t_g_raw
    else:
        scale = 1.0
        t_a, t_g = t_a_raw, t_g_raw
        t_a_device = np.empty(0, dtype=np.float64)
        t_g_device = np.empty(0, dtype=np.float64)
    return {
        "t_accel_monotonic_s": t_a,
        "accel_raw": a,
        "t_gyro_monotonic_s": t_g,
        "gyro_raw": g,
        "t_accel_uart_host_s": t_a_host,
        "t_gyro_uart_host_s": t_g_host,
        "t_accel_device_s": t_a_device,
        "t_gyro_device_s": t_g_device,
        "uses_device_clock": uses_device_clock,
        "device_clock_to_host_scale": scale,
        "device_time_packet_count": len(device_clock_s),
    }


def parse_reads_precise(reads, baud):
    """Compatibility wrapper returning the four arrays used by old tools."""
    parsed = parse_reads_precise_with_device_time(reads, baud)
    return (parsed["t_accel_monotonic_s"], parsed["accel_raw"],
            parsed["t_gyro_monotonic_s"], parsed["gyro_raw"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--out", default="imu_work/sync_session.npz")
    args = ap.parse_args()

    start_hint = min(5.0, args.seconds * 0.1)
    stop_hint = args.seconds - 10.0

    ser = serial.Serial(args.port, args.baud, timeout=0.05)
    reads = []
    t0 = time.time()
    t_wall0 = time.strftime("%H:%M:%S")
    last_sec = -1
    print(f"[{t_wall0}] IMU 录制启动，计划 {args.seconds:.0f}s（随时可 Ctrl+C 提前结束）")
    print(f"  第 {start_hint:.0f}s 起提示按相机开始录像；倒数 10s 提示停止录像\n")
    interrupted = False
    try:
        while time.time() - t0 < args.seconds:
            t_before = time.time()
            chunk = ser.read(2048)
            t_after = time.time()
            if chunk:
                reads.append((t_before, t_after, chunk))
            sec = int(time.time() - t0)
            if sec != last_sec and sec <= args.seconds:
                last_sec = sec
                remain = int(args.seconds - sec)
                if sec == int(start_hint):
                    hint = ">>> 现在按下相机开始录像！ <<<"
                elif sec == int(stop_hint):
                    hint = ">>> 现在按下相机停止录像！ <<<"
                elif sec < start_hint:
                    hint = "准备中……（提示出现时按相机录像键）"
                elif sec < stop_hint:
                    hint = "录制中：拿着相机转动+平移"
                else:
                    hint = "收尾余量（视频应已停止）"
                bar = "#" * min(sec, 60) + "-" * max(60 - sec, 0)
                print(f"\r[{bar}] {sec:3d}s/{args.seconds:.0f}s 剩{remain:3d}s  {hint}", end="", flush=True)
        print()
    except KeyboardInterrupt:
        interrupted = True
        print("\n提前终止，正在保存已录数据……")
    ser.close()
    t_wall1 = time.strftime("%H:%M:%S")
    total_bytes = sum(len(c) for _, _, c in reads)
    dur = time.time() - t0
    print(f"[{t_wall1}] 结束，实际录制 {dur:.1f}s，{total_bytes} 字节"
          + ("（提前终止）" if interrupted else ""))

    if not reads:
        print("❌ 什么都没录到")
        return
    t_a, a, t_g, g = parse_reads(reads)
    print(f"accel {len(a)} 包 ({len(a)/(t_a[-1]-t_a[0]):.1f} Hz), gyro {len(g)} 包")
    dt = np.diff(t_a) * 1000
    print(f"ACCL 时间戳间隔: 中位 {np.median(dt):.2f}ms, 最大 {dt.max():.2f}ms"
          + ("  ✓" if 3 < np.median(dt) < 8 else "  ⚠️ 间隔异常，检查串口稳定性"))
    np.savez_compressed(args.out, t_a=t_a, accel=a, t_g=t_g, gyro=g,
                        accel_scale=ACCEL_SCALE, gyro_scale=GYRO_SCALE,
                        wall_start=t_wall0, wall_end=t_wall1)
    print(f"已保存: {args.out}")
    print("提示：视频时长必须短于本次实际录制时长（IMU 先开后停），"
          "否则需要尾部填充或重录。")


if __name__ == "__main__":
    main()
