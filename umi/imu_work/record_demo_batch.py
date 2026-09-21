"""
Batch demo recorder for AM2UMI data collection.

Records N demos back-to-back. Per demo:
  0-5s:   countdown, get ready (camera pointing at the desk, tag 13 in view)
  5s:     prompt to press the camera RECORD button
  ...:    do the task (hand-held gripper)
  when done: press camera STOP, then press ENTER in this terminal
          -> the IMU segment is saved immediately, next demo starts in 3s

Safety cap: if ENTER is never pressed, the segment auto-ends at --seconds.
Ctrl+C at any time saves the current segment and exits the whole batch.
Each demo is saved as its own npz: imu_work/demo_05.npz, demo_06.npz, ...

Usage:
  python imu_work/record_demo_batch.py --count 10 --start-index 5 \
      --seconds 60 --out-prefix imu_work/demo_
"""
import argparse
import os
import sys
import termios
import time
import tty

import numpy as np
import serial

from record_sync_session import ACCEL_SCALE, GYRO_SCALE, parse_reads


def _kbhit():
    """Non-blocking check for a keypress. Returns True if a key is waiting."""
    import select
    return select.select([sys.stdin], [], [], 0.0)[0]


def _getch():
    """Read one character (terminal is in cbreak mode)."""
    return os.read(sys.stdin.fileno(), 1)


def read_key(timeout_s):
    """Wait up to timeout_s for a keypress. Returns the char or None."""
    import select
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if ready:
        return os.read(sys.stdin.fileno(), 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--count", type=int, required=True, help="number of demos to record")
    ap.add_argument("--start-index", type=int, default=1,
                    help="first demo number; files are named demo_XX.npz")
    ap.add_argument("--seconds", type=float, default=90.0,
                    help="per-demo safety cap (seconds); ENTER ends the demo early")
    ap.add_argument("--out-prefix", default="imu_work/demo_",
                    help="output path prefix; file is <prefix><idx>.npz")
    ap.add_argument("--gap", type=float, default=5.0,
                    help="seconds of rest between demos")
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0.05)

    old_tc = termios.tcgetattr(sys.stdin.fileno())
    tty.setcbreak(sys.stdin.fileno())

    interrupted = False
    try:
        for demo_idx in range(args.start_index, args.start_index + args.count):
            out_path = f"{args.out_prefix}{demo_idx:02d}.npz"
            print()
            print(f"===== Demo {demo_idx - args.start_index + 1}/{args.count} =====")
            print("准备阶段：拿好夹爪，相机对准桌面（tag 13 入画）")
            start_hint = 5.0
            t0 = time.time()
            t_wall0 = time.strftime("%H:%M:%S")
            reads = []
            state = "countdown"
            last_sec = -1
            print(f"[{t_wall0}] IMU 录制中（本条安全上限 {args.seconds:.0f}s）")
            while True:
                t_before = time.time()
                chunk = ser.read(2048)
                t_after = time.time()
                if chunk:
                    reads.append((t_before, t_after, chunk))
                elapsed = time.time() - t0
                sec = int(elapsed)

                if state == "countdown" and sec >= int(start_hint):
                    state = "recording"
                    print(f">>> [{sec:3d}s] 现在按下相机开始录像！ <<<")

                if sec != last_sec:
                    last_sec = sec
                    if state == "countdown":
                        hint = f"准备中…… {int(start_hint - elapsed) + 1}s 后提示开录"
                    else:
                        remain = max(0, int(args.seconds - elapsed))
                        hint = f"录制中（任务做完→停相机→回到这里按 Enter）；剩余上限 {remain}s"
                    print(f"\r[{sec:3d}s] {hint}    ", end="", flush=True)

                if elapsed >= args.seconds:
                    print("\n>>> 达到安全上限，自动结束本条。")
                    break

                key = read_key(0.2)
                if key is not None:
                    if key in (b'\r', b'\n', b' '):
                        if state == "recording":
                            print(f"\n>>> [{elapsed:.0f}s] 收到 Enter，结束本条。")
                            break
                        else:
                            print("（还在倒计时，稍等）")
                    elif key in (b'q', b'Q', b'\x03'):
                        raise KeyboardInterrupt

            # save this demo's segment
            t_wall1 = time.strftime("%H:%M:%S")
            dur = time.time() - t0
            if not reads:
                print("⚠️ 本条没有录到数据，跳过保存")
                continue
            t_a, a, t_g, g = parse_reads(reads)
            np.savez_compressed(out_path, t_a=t_a, accel=a, t_g=t_g, gyro=g,
                                accel_scale=ACCEL_SCALE, gyro_scale=GYRO_SCALE,
                                wall_start=t_wall0, wall_end=t_wall1,
                                demo_index=demo_idx)
            print(f"✅ 已保存 {out_path}（实际 {dur:.1f}s，"
                  f"accel {len(a)} 包 @ {len(a) / max(t_a[-1] - t_a[0], 1e-6):.0f}Hz）")
            print(f"   记住本条的视频文件名（DJI_...MP4）！")

            # gap before next demo
            print(f"休息 {args.gap:.0f}s 后开始下一条……")
            for s in range(int(args.gap), 0, -1):
                if read_key(1.0) is not None:
                    break
                print(f"\r  {s}s", end="", flush=True)
            print()
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n已中断批量录制。")
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tc)
        ser.close()

    print("完成。请在相机里按时间顺序记下各条视频文件名，与 demo_XX.npz 一一对应。")


if __name__ == "__main__":
    main()
