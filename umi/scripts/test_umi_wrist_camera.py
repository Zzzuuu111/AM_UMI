"""Verify the AM2Pro wrist EMEET camera without moving the robot."""

import argparse
import time

import cv2


DEFAULT_DEVICE = (
    "/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open wrist camera: {args.device}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

        frames = 0
        first_shape = None
        start = time.monotonic()
        deadline = start + 3.0
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("camera opened but did not return a frame")
            first_shape = frame.shape
            frames += 1
        elapsed = time.monotonic() - start
        print("WRIST_CAMERA_CAPTURE_OK")
        print("device:", args.device)
        print("frame_shape:", first_shape)
        print(f"measured_fps: {frames / elapsed:.1f}")
        print("reported_width:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("reported_height:", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"reported_fps: {cap.get(cv2.CAP_PROP_FPS):.1f}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
