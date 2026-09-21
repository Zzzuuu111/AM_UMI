"""
Capture the module's ack/response bytes right after a rate write.

JY901-family modules reply to accepted writes with 0x55 0x61 (or similar)
frames; clones may silently reject. This dumps everything received for 1s
after unlock+write so we can see what the module actually says.

Run: python imu_work/jy901b_ack.py
"""
import time

import serial


def main():
    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

    def dump(tag, seconds=1.0):
        ser.reset_input_buffer()
        t0 = time.time()
        buf = b""
        while time.time() - t0 < seconds:
            chunk = ser.read(256)
            if chunk:
                buf += chunk
        print(f"  {tag}: {buf.hex()}")
        return buf

    dump("静默 1s 的数据流(参考)")
    ser.write(bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))  # unlock
    ser.flush()
    time.sleep(0.2)
    ser.read(64)
    ser.write(bytes([0xFF, 0xAA, 0x03, 0xC8, 0x00]))  # rate=200
    ser.flush()
    time.sleep(0.2)
    dump("解锁+写速率后的 1s 应答")

    ser.write(bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]))  # unlock again
    ser.flush()
    time.sleep(0.2)
    ser.read(64)
    ser.write(bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]))  # save
    ser.flush()
    time.sleep(0.2)
    dump("解锁+保存后的 1s 应答")
    ser.close()
    print("\n解读要点：找 '55 61' 开头的确认帧；若数据流在写指令后立刻出现"
          "乱码或停顿，说明模块处理了指令。")


if __name__ == "__main__":
    main()
