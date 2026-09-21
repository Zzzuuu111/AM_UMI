"""
Record a calibration video from a UVC camera with live charuco detection feedback.

The preview shows EXACTLY what the calibrator will see:
    - green dots: detected charuco corners
    - colored tag borders: detected markers
    - corner count text, green when >= min_corners

Workflow:
    1. aim the board until the corner count turns green (>=14)
    2. press SPACE to start recording
    3. sweep poses (each pose: hold still until green, then move)
    4. auto-stop after --duration, or press 'q' to quit anytime

Usage (umi env):
    python scripts/record_uvc_calib.py --device /dev/video2 --duration 40 -o deploy_cam_calib.mp4
"""
import argparse
import sys
import time

import cv2
import numpy as np

ROOT_DIR = __file__.rsplit('/', 2)[0]
sys.path.insert(0, ROOT_DIR)

from umi.common.cv_util import get_charuco_board  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Record UVC calibration video with live detection")
    parser.add_argument("--device", default="/dev/video2", help="V4L2 device path")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=40.0, help="Record duration in seconds")
    parser.add_argument("--min_corners", type=int, default=14, help="Green threshold for corner count")
    parser.add_argument("-o", "--output", default="deploy_cam_calib.mp4")
    args = parser.parse_args()

    board = get_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open {args.device}. "
                         f"Is another program using it? Run: sudo pkill -f 'ffmpeg|ffplay'")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened {args.device}: {width}x{height}")
    print(f"Aim the board until 'corners' turns green, then press SPACE to record {args.duration:.0f}s.")
    print("q = quit | SPACE = start/stop recording")

    writer = None
    recording = False
    record_start = 0.0
    n_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed, stopping.")
            break

        overlay = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # live detection
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        n_corners = 0 if charuco_ids is None else len(charuco_ids)
        n_markers = 0 if marker_ids is None else len(marker_ids)

        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        if charuco_ids is not None and len(charuco_ids) > 0:
            for pt in charuco_corners.reshape(-1, 2):
                cv2.circle(overlay, tuple(map(int, pt)), 4, (0, 200, 0), -1)

        # sharpness of the board area (upper 2/3, excluding gripper strip)
        roi = cv2.cvtColor(overlay[int(0.08 * height): int(0.55 * height),
                                   int(0.05 * width): int(0.95 * width)], cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()

        good = n_corners >= args.min_corners
        c_color = (0, 200, 0) if good else (0, 0, 255)
        s_color = (0, 200, 0) if sharpness >= 150 else (0, 0, 255)

        cv2.putText(overlay, f"corners {n_corners}  markers {n_markers}  {'GOOD' if good else 'need >=%d' % args.min_corners}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, c_color, 3)
        cv2.putText(overlay, f"sharpness {sharpness:5.0f}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, s_color, 3)

        if recording:
            elapsed = time.time() - record_start
            remaining = max(0.0, args.duration - elapsed)
            cv2.putText(overlay, f"REC {n_frames}  remaining {remaining:4.0f}s",
                        (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            writer.write(frame)
            n_frames += 1
            if elapsed >= args.duration:
                recording = False
                writer.release()
                writer = None
                print(f"Recording done: {n_frames} frames -> {args.output}")
        else:
            cv2.putText(overlay, "SPACE to start recording", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)

        cv2.imshow("calib capture (SPACE=record, q=quit)", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == 32:  # SPACE
            if recording:
                recording = False
                writer.release()
                writer = None
                print(f"Recording stopped early: {n_frames} frames -> {args.output}")
            else:
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (width, height))
                if not writer.isOpened():
                    print(f"Failed to open output {args.output}")
                    writer = None
                else:
                    recording = True
                    record_start = time.time()
                    n_frames = 0
                    print(f"Recording started ({args.duration:.0f}s) -> {args.output}")

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
