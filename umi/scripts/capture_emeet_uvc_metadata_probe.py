"""Capture a short EMEET video+UVCH metadata timing probe without terminal jobs.

It starts MJPEG video streaming on the UVC video node, simultaneously captures
the paired UVCH metadata node with v4l2-ctl, then invokes the local inspector.
No camera controls are changed.  Output is intentionally an explicit new
directory so raw metadata can be retained for timing diagnostics.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="采集并检查 EMEET UVC PTS/SCR 元数据")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-device", default="/dev/video2")
    parser.add_argument("--metadata-device", default="/dev/video3")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.seconds <= 0 or args.fps <= 0:
        parser.error("--seconds 和 --fps 必须为正数")
    output = pathlib.Path(args.output_dir).expanduser().resolve()
    if output.exists():
        raise SystemExit(f"为保护已有探测结果，拒绝覆盖：{output}")
    output.mkdir(parents=True)
    uvch = output / "uvch.bin"
    ffmpeg_log = output / "ffmpeg.stderr.log"
    ffmpeg = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "warning",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
        "-i", args.video_device, "-t", str(args.seconds), "-f", "null", "-",
    ]
    metadata = [
        "timeout", f"{args.seconds + 2.0:.1f}s", "v4l2-ctl",
        f"--device={args.metadata_device}", "--stream-mmap=3", f"--stream-to={uvch}",
    ]
    print("UVC_METADATA_PROBE_STARTED")
    print("video:", args.video_device, "metadata:", args.metadata_device)
    with ffmpeg_log.open("wb") as log:
        video_process = subprocess.Popen(
            ffmpeg, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=log
        )
        try:
            # Let video streaming negotiate its MJPEG mode before metadata capture.
            time.sleep(0.5)
            result = subprocess.run(
                metadata, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, timeout=args.seconds + 5.0,
            )
        finally:
            try:
                video_process.wait(timeout=args.seconds + 3.0)
            except subprocess.TimeoutExpired:
                video_process.terminate()
                video_process.wait(timeout=3.0)
    (output / "v4l2_ctl.stdout.log").write_text(result.stdout or "", encoding="utf-8")
    print("metadata_capture_returncode:", result.returncode,
          "(124 is expected because timeout stops the metadata stream)")
    print("uvch_bytes:", uvch.stat().st_size if uvch.exists() else 0)
    inspect = pathlib.Path(__file__).with_name("inspect_uvc_payload_metadata.py")
    inspected = subprocess.run([sys.executable, str(inspect), "--input", str(uvch)])
    if inspected.returncode != 0:
        raise SystemExit(inspected.returncode)
    print("UVC_METADATA_PROBE_OK")
    print("output_dir:", output)


if __name__ == "__main__":
    main()
