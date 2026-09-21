#!/usr/bin/env python3
"""Replay one AM_UMI recorded session into OpenVINS ROS 2 topics.

This deliberately reads only existing files.  It does not open UVC devices,
serial ports, or robot-control software.  Camera stamps come from the strict
UVC source-clock CSV and IMU stamps come from the already exported JY901B
camera timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import time

import cv2
import rclpy
from builtin_interfaces.msg import Time
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, Imu


CAMERA_TOPIC = "/am_umi/camera0/image_raw"
IMU_TOPIC = "/am_umi/imu0"
EPOCH_S = 1_700_000_000


def stamp(relative_s: float) -> Time:
    ns = EPOCH_S * 1_000_000_000 + int(round(relative_s * 1_000_000_000))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def camera_times(path: pathlib.Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty timestamp CSV: {path}")
    times = [float(row["uvc_source_relative_s"]) for row in rows]
    if not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise RuntimeError("camera timestamps must be strictly increasing")
    return times


def imu_events(path: pathlib.Path) -> list[tuple[float, list[float], list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    streams = payload["1"]["streams"]
    accel = streams["ACCL"]["samples"]
    gyro = streams["GYRO"]["samples"]
    if len(accel) != len(gyro):
        raise RuntimeError("ACCL/GYRO sample count differs")
    out = []
    for a, g in zip(accel, gyro):
        ta, tg = float(a["cts"]) / 1000.0, float(g["cts"]) / 1000.0
        if abs(ta - tg) > 1e-5:
            raise RuntimeError("ACCL/GYRO timeline differs")
        out.append((tg, list(g["value"]), list(a["value"])))
        if len(out[-1][1]) != 3 or len(out[-1][2]) != 3 or not all(
            math.isfinite(v) for v in [tg, *out[-1][1], *out[-1][2]]
        ):
            raise RuntimeError("invalid IMU vector/time")
    if not out:
        raise RuntimeError("empty IMU stream")
    if any(b[0] <= a[0] for a, b in zip(out, out[1:])):
        raise RuntimeError("IMU timestamps must be strictly increasing")
    return out


def publish_imu(pub, t: float, gyro: list[float], accel: list[float]) -> None:
    msg = Imu()
    msg.header.stamp = stamp(t)
    msg.header.frame_id = "jy901b"
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
    pub.publish(msg)


def publish_image(pub, t: float, frame) -> None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    msg = Image()
    msg.header.stamp = stamp(t)
    msg.header.frame_id = "emeet_optical"
    msg.height, msg.width = gray.shape
    msg.encoding = "mono8"
    msg.is_bigendian = 0
    msg.step = msg.width
    msg.data = gray.tobytes()
    pub.publish(msg)


def sleep_recorded(dt_s: float, speed: float) -> None:
    # Large timestamp gaps are not expected in a valid session; cap protects
    # an accidental malformed CSV from making the test appear frozen.
    if dt_s > 0:
        time.sleep(min(dt_s / speed, 0.2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay AM_UMI session to OpenVINS ROS2 topics")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--imu-json", default="imu_data_strict_uvc.json")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="recorded-time replay multiplier; keep 1.0 for first validation")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="optional recorded-time limit; 0 replays the complete session")
    parser.add_argument("--flush-seconds", type=float, default=3.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")

    session = pathlib.Path(args.session_dir).resolve()
    csv_path = session / "frame_timestamps_uvc_source.csv"
    video_path = session / "raw_video.mp4"
    imu_path = pathlib.Path(args.imu_json)
    if not imu_path.is_absolute():
        imu_path = session / imu_path
    if not (csv_path.is_file() and video_path.is_file() and imu_path.is_file()):
        raise SystemExit("session needs raw_video.mp4, frame_timestamps_uvc_source.csv, and IMU JSON")

    times = camera_times(csv_path)
    imus = imu_events(imu_path)
    cori = json.loads(imu_path.read_text())["1"]["streams"]["CORI"]["samples"]
    if len(cori) != len(times) or any(abs(float(x["cts"]) / 1000 - t) > 1e-6 for x, t in zip(cori, times)):
        raise RuntimeError("CORI and UVC source timestamps disagree; re-export IMU first")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    rclpy.init()
    node = rclpy.create_node("am_umi_openvins_replay")
    qos = QoSProfile(depth=2000)
    pub_imu = node.create_publisher(Imu, IMU_TOPIC, qos)
    pub_cam = node.create_publisher(Image, CAMERA_TOPIC, qos)
    deadline = time.monotonic() + 30
    while not (pub_cam.get_subscription_count() and pub_imu.get_subscription_count()):
        if time.monotonic() > deadline:
            raise RuntimeError("OpenVINS camera/IMU subscribers not ready in 30 seconds")
        rclpy.spin_once(node, timeout_sec=0.1)
    print("OPENVINS_REPLAY_STARTED", flush=True)
    print(f"frames={len(times)} imu={len(imus)} speed={args.speed}", flush=True)

    imu_index = 0
    last_t = 0.0
    published_frames = 0
    try:
        for frame_index, camera_t in enumerate(times):
            if args.max_seconds > 0 and camera_t > args.max_seconds:
                break
            while imu_index < len(imus) and imus[imu_index][0] <= camera_t:
                imu_t, gyro, accel = imus[imu_index]
                sleep_recorded(imu_t - last_t, args.speed)
                publish_imu(pub_imu, imu_t, gyro, accel)
                last_t = imu_t
                imu_index += 1
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"video ended at frame {frame_index}, expected {len(times)}")
            if frame.shape[:2] != (1080, 1920):
                raise RuntimeError(f"unexpected source resolution: {frame.shape}")
            frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            sleep_recorded(camera_t - last_t, args.speed)
            publish_image(pub_cam, camera_t, frame)
            published_frames += 1
            last_t = camera_t
            if frame_index and frame_index % 300 == 0:
                print(f"replayed_frames={frame_index} imu_index={imu_index}", flush=True)
            rclpy.spin_once(node, timeout_sec=0.0)

        if published_frames:
            # A camera update is processed only after a later IMU message.
            tail_limit = times[published_frames - 1] + 0.1
            while imu_index < len(imus) and imus[imu_index][0] <= tail_limit:
                imu_t, gyro, accel = imus[imu_index]
                sleep_recorded(imu_t - last_t, args.speed)
                publish_imu(pub_imu, imu_t, gyro, accel)
                last_t = imu_t
                imu_index += 1
        time.sleep(args.flush_seconds)
        print(f"published_frames={published_frames} published_imu={imu_index}", flush=True)
        print("OPENVINS_REPLAY_FINISHED", flush=True)
    finally:
        cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
