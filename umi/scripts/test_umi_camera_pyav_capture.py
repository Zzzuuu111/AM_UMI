"""Measure EMEET V4L2 capture with PyAV, including BGR conversion.

This separates OpenCV's VideoCapture behavior from the UVC device and lets us
choose the reliable reader for the hand-held recorder.
"""

import argparse
import os
import time

import av


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    resolved = os.path.realpath(args.device)
    options = {
        "input_format": "mjpeg",
        "video_size": f"{args.width}x{args.height}",
        "framerate": str(args.fps),
    }
    print("Opening PyAV V4L2 capture:", resolved, flush=True)
    container = av.open(resolved, format="v4l2", options=options)
    count = 0
    first_shape = None
    start = time.monotonic()
    try:
        for frame in container.decode(video=0):
            # The recorder needs BGR input for its preview and H.264 writer;
            # include this conversion in the measured rate.
            image = frame.to_ndarray(format="bgr24")
            if first_shape is None:
                first_shape = tuple(image.shape)
            count += 1
            if time.monotonic() - start >= args.seconds:
                break
    finally:
        container.close()
    elapsed = time.monotonic() - start
    print("PYAV_CAMERA_CAPTURE_OK")
    print("frame_shape:", first_shape)
    print("frames:", count)
    print(f"measured_fps: {count / elapsed:.2f}")


if __name__ == "__main__":
    main()
