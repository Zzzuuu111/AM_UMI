"""Open a UVC camera for live viewing only; no robot, IMU, or video recording.

Examples:
    python -u scripts/preview_umi_camera.py
    python -u scripts/preview_umi_camera.py --device /dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index1

Click the preview window, then press q or Esc to exit.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2


def open_camera(path: str, width: int, height: int, fps: float, pixel_format: str):
    resolved = os.path.realpath(path)
    camera = cv2.VideoCapture(resolved, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        return None
    if pixel_format:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, frame = camera.read()
    if not ok or frame is None:
        camera.release()
        return None
    return camera, resolved, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="stable /dev/v4l/by-id path; omit to find an EMEET camera")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pixel-format", default="MJPG")
    parser.add_argument("--display-width", type=int, default=960)
    args = parser.parse_args()
    if len(args.pixel_format) != 4:
        parser.error("--pixel-format must be a four-character code, e.g. MJPG")

    candidates = [args.device] if args.device else sorted(
        glob.glob("/dev/v4l/by-id/*EMEET*-video-index*"))
    if not candidates:
        raise SystemExit("No EMEET /dev/v4l/by-id camera found; pass --device explicitly.")

    opened = None
    for candidate in candidates:
        opened = open_camera(candidate, args.width, args.height, args.fps, args.pixel_format)
        if opened is not None:
            device = candidate
            break
    if opened is None:
        raise SystemExit("Could not read a frame from any candidate: " + ", ".join(candidates))
    camera, resolved, frame = opened
    source_height, source_width = frame.shape[:2]
    print("CAMERA_PREVIEW_OPEN")
    print("device:", device)
    print("resolved:", resolved)
    print(f"frame: {source_width}x{source_height}; requested: {args.width}x{args.height}@{args.fps:g}")
    print("Click the preview window. Press q or Esc to exit.")

    window_name = "UMI EMEET camera preview"
    count = 0
    interval_start = time.monotonic()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera stopped returning frames")
            count += 1
            height, width = frame.shape[:2]
            scale = min(1.0, args.display_width / width)
            display = (cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                       if scale < 1.0 else frame)
            elapsed = time.monotonic() - interval_start
            if elapsed >= 1.0:
                measured_fps = count / elapsed
                print(f"live_fps: {measured_fps:.1f}", flush=True)
                count = 0
                interval_start = time.monotonic()
            cv2.putText(display, "q / Esc: exit", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("CAMERA_PREVIEW_CLOSED")


if __name__ == "__main__":
    main()
