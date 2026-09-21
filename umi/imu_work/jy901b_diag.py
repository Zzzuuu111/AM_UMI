"""
Diagnose JY901B register write/read: find correct byte order for config.

Sequence (all at 9600):
  1. drain buffer
  2. write rate=200Hz low-byte-first:  FF AA 03 C8 00  -> dump ack
  3. read back reg 0x03:              FF AA 27 03 00  -> dump ack
  4. write rate=200Hz high-byte-first: FF AA 03 00 C8 -> dump ack
  5. read back reg 0x03 again -> dump ack
  6. same test for baud reg 0x04 (value 0x0006=115200), low-first only
  7. report: which writes were acknowledged (55 61 ...) and read-back values

Run: python imu_work/jy901b_diag.py
"""
import time
import serial


def txrx(ser, frame, tag):
    ser.reset_input_buffer()
    ser.write(frame)
    ser.flush()
    time.sleep(0.4)
    ack = ser.read(64)
    print(f"  {tag}: sent {frame.hex():<12} ack {ack.hex() or '(空)'}")
    return ack


def read_reg(ser, reg, tag):
    return txrx(ser, bytes([0xFF, 0xAA, 0x27, reg, 0x00]), tag)


def main():
    ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
    print("== 字节序与寄存器测试 ==")
    txrx(ser, bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]), "解锁(可能无效)")
    read_reg(ser, 0x03, "读 0x03 初始值")
    txrx(ser, bytes([0xFF, 0xAA, 0x03, 0xC8, 0x00]), "写 0x03=0x00C8 低字节在前")
    read_reg(ser, 0x03, "读回 0x03")
    txrx(ser, bytes([0xFF, 0xAA, 0x03, 0x00, 0xC8]), "写 0x03=0x00C8 高字节在前")
    read_reg(ser, 0x03, "读回 0x03")
    txrx(ser, bytes([0xFF, 0xAA, 0x04, 0x06, 0x00]), "写 0x04=0x0006(115200) 低字节在前")
    read_reg(ser, 0x04, "读回 0x04")
    txrx(ser, bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]), "保存")
    ser.close()
    print("\n解读:")
    print("  ack 含 '55 61' = 模块确认写入; 含 '55 71' = 读回应答")
    print("  读回 0x03 若变为 00 c8 = 写入成功且低/高字节序可用")
    print("  把上面完整输出贴给助手分析")


if __name__ == "__main__":
    main()
