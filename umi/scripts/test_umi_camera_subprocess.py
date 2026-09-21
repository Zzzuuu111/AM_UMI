"""Diagnose whether the wrist camera opens inside a plain Python child process."""

import multiprocessing as mp
import os
import time

import cv2


DEVICE = "/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0"


def worker(device, result_queue):
    path = os.path.realpath(device)
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            result_queue.put((False, f"cannot open {path}"))
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frames = 0
        frame_shape = None
        start = time.monotonic()
        while time.monotonic() - start < 3.0:
            ok, frame = cap.read()
            if not ok or frame is None:
                result_queue.put((False, "camera stopped returning frames"))
                return
            frames += 1
            frame_shape = tuple(frame.shape)
        elapsed = time.monotonic() - start
        result_queue.put((True, (frame_shape, frames / elapsed)))
    finally:
        cap.release()


def main():
    result_queue = mp.Queue()
    process = mp.Process(target=worker, args=(DEVICE, result_queue))
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise RuntimeError("camera child process did not finish within 10 seconds")
    if result_queue.empty():
        raise RuntimeError(f"camera child exited without a result (exit={process.exitcode})")
    ok, detail = result_queue.get()
    if not ok:
        raise RuntimeError(detail)
    print("CAMERA_SUBPROCESS_OPEN_OK")
    print("device:", os.path.realpath(DEVICE))
    print("frame_shape:", detail[0])
    print(f"measured_fps: {detail[1]:.1f}")


if __name__ == "__main__":
    main()
