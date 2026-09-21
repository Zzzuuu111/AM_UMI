"""
Read raw data from a CP2102/CH340 USB IMU board on /dev/ttyUSB0.

Modes:
  --scan            try common baud rates, report what each one emits
  --baud 115200     dump raw hex at one baud rate
  --wit             parse WitMotion/JY61 style 0x55 packets (accel/gyro/angle)

Needs: pip install pyserial
Run on the machine where the board is plugged in (needs dialout permission).
"""
import argparse
import sys
import time


def open_port(port, baud):
    try:
        import serial
    except ImportError:
        sys.exit("缺少 pyserial：先运行  pip install pyserial")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except Exception as e:
        sys.exit(f"打开 {port} 失败：{e}\n"
                 "常见原因：节点不存在（拔插一次）/ 用户不在 dialout 组（需注销重登）")
    return ser


def read_for(ser, seconds):
    buf = b""
    t0 = time.time()
    while time.time() - t0 < seconds:
        chunk = ser.read(1024)
        if chunk:
            buf += chunk
    return buf


def describe(buf):
    """Quick summary of what a byte stream looks like."""
    if not buf:
        return "无数据输出"
    n55 = sum(1 for i in range(len(buf) - 1) if buf[i] == 0x55)
    printable = sum(1 for b in buf if 32 <= b < 127)
    info = f"{len(buf)} 字节, 0x55 出现 {n55} 次"
    if printable > len(buf) * 0.5:
        info += ", 大量可打印字符（可能是 ESP32 启动日志，非 IMU 数据）"
    return info


def parse_wit(buf):
    """Parse common 11-byte WitMotion/JY901 packets.

    In particular, 0x50 is the device real-time-clock packet.  It must be
    accepted here rather than treated as noise, otherwise the scan cannot tell
    whether the user enabled the IMU's time output.
    """
    seen = {"50": "设备时间", "51": "加速度", "52": "角速度", "53": "角度", "54": "磁场"}
    counts = {}
    i = 0
    while i < len(buf) - 11:
        if buf[i] == 0x55 and buf[i + 1] in (0x50, 0x51, 0x52, 0x53, 0x54):
            pkt = buf[i:i + 11]
            if sum(pkt[:10]) & 0xFF == pkt[10]:  # checksum ok
                t = f"{pkt[1]:02x}"
                counts[t] = counts.get(t, 0) + 1
                if counts[t] <= 2:
                    import struct
                    x, y, z = struct.unpack("<hhh", pkt[2:8])
                    if t == "50":
                        # 0x55 0x50 YY MM DD hh mm ss msL msH checksum
                        ms = int.from_bytes(pkt[8:10], "little")
                        val = (f"20{pkt[2]:02d}-{pkt[3]:02d}-{pkt[4]:02d} "
                               f"{pkt[5]:02d}:{pkt[6]:02d}:{pkt[7]:02d}.{ms:03d}")
                    elif t == "51":
                        val = f"ax={x/32768*16:.2f}g ay={y/32768*16:.2f}g az={z/32768*16:.2f}g"
                    elif t == "52":
                        val = f"wx={x/32768*2000:.1f}°/s wy={y/32768*2000:.1f}°/s wz={z/32768*2000:.1f}°/s"
                    else:
                        val = f"x={x/32768*180:.1f}° y={y/32768*180:.1f}° z={z/32768*180:.1f}°"
                    print(f"  [{seen[t]}] {val}")
                i += 11
                continue
        i += 1
    if not counts:
        print("  未发现合法的 0x55 包（不是 WitMotion/JY61 协议，或波特率不对）")
    else:
        print("  各类型包数：", counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--scan", action="store_true", help="尝试常见波特率")
    ap.add_argument("--wit", action="store_true", help="按 WitMotion/JY61 解析")
    args = ap.parse_args()

    if args.scan:
        for baud in (9600, 115200, 460800, 921600):
            print(f"--- 波特率 {baud} ---")
            try:
                ser = open_port(args.port, baud)
                buf = read_for(ser, args.seconds)
                ser.close()
                print(f"  {describe(buf)}")
                if args.wit and buf:
                    parse_wit(buf)
            except SystemExit as e:
                print(f"  {e}")
    else:
        baud = args.baud or 115200
        ser = open_port(args.port, baud)
        buf = read_for(ser, args.seconds)
        ser.close()
        print(f"波特率 {baud}: {describe(buf)}")
        if args.wit:
            parse_wit(buf)
        elif buf:
            print("原始字节（前 160 字节）：")
            for k in range(0, min(160, len(buf)), 16):
                chunk = buf[k:k + 16]
                print("  " + " ".join(f"{b:02x}" for b in chunk))


if __name__ == "__main__":
    main()
