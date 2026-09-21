#!/usr/bin/env python3
"""Render one UMI episode as a safe, offline review video.

The output combines the recorded policy image with TCP trajectory and gripper
telemetry.  This is a visual data-review aid, not a physical robot replay: the
script imports no motor/controller code and cannot command the AM2Pro.
"""

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import zarr
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs


def open_root(path):
    if path.is_dir():
        return zarr.open(str(path), mode="r"), None
    store = zarr.ZipStore(str(path), mode="r")
    return zarr.group(store=store), store


def draw_text(image, text, xy, scale=0.55, color=(225, 225, 225), thickness=1):
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def project_trace(points, rect, axes):
    x0, y0, x1, y1 = rect
    values = points[:, axes]
    lo = values.min(axis=0)
    hi = values.max(axis=0)
    span = np.maximum(hi - lo, 0.01)
    pad = span * 0.10
    lo -= pad
    hi += pad
    u = (values[:, 0] - lo[0]) / (hi[0] - lo[0])
    v = (values[:, 1] - lo[1]) / (hi[1] - lo[1])
    px = x0 + 8 + u * (x1 - x0 - 16)
    py = y1 - 8 - v * (y1 - y0 - 16)
    return np.rint(np.stack([px, py], axis=1)).astype(np.int32)


def draw_trace(canvas, points_px, current, rect, title):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (75, 75, 75), 1)
    draw_text(canvas, title, (x0 + 7, y0 + 18), 0.48, (190, 190, 190))
    if len(points_px) > 1:
        cv2.polylines(canvas, [points_px], False, (105, 105, 105), 1,
                      cv2.LINE_AA)
    if current > 0:
        cv2.polylines(canvas, [points_px[:current + 1]], False,
                      (70, 210, 245), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points_px[current]), 5, (60, 230, 80), -1,
               cv2.LINE_AA)


def project_trace_xyz(points, rect):
    """Project an equal-scale X/Y/Z trajectory into an isometric review view."""
    x0, y0, x1, y1 = rect
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    span = max(float(np.ptp(points, axis=0).max()), 0.01) * 1.20

    # Isometric projection: screen-right is +X/-Y and screen-up is a mix of
    # +Z and +X/+Y.  All coordinates use one shared metric span, unlike two
    # independent XY/XZ projections.
    def project(values):
        normalized = (np.asarray(values, dtype=np.float64) - center) / span
        projected = np.column_stack((
            0.70710678 * (normalized[:, 0] - normalized[:, 1]),
            normalized[:, 2] - 0.40824829 * (normalized[:, 0] + normalized[:, 1]),
        ))
        scale = min((x1 - x0 - 34) / 1.62, (y1 - y0 - 42) / 1.82)
        pixels = np.empty_like(projected)
        pixels[:, 0] = (x0 + x1) * 0.5 + projected[:, 0] * scale
        pixels[:, 1] = (y0 + y1) * 0.5 - projected[:, 1] * scale
        return np.rint(pixels).astype(np.int32)

    # Draw a faint cube and coordinate axes so the oblique view remains
    # legible as a metric X/Y/Z representation rather than a decorative path.
    cube = np.asarray([[x, y, z]
                       for x in (-0.5, 0.5)
                       for y in (-0.5, 0.5)
                       for z in (-0.5, 0.5)], dtype=np.float64) * span + center
    cube_px = project(cube)
    origin_and_axes = project(np.asarray([
        center,
        center + np.array([0.44 * span, 0.0, 0.0]),
        center + np.array([0.0, 0.44 * span, 0.0]),
        center + np.array([0.0, 0.0, 0.44 * span]),
    ]))
    return project(points), cube_px, origin_and_axes


def draw_trace_xyz(canvas, points_px, cube_px, axes_px, current, rect):
    x0, y0, x1, y1 = rect
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (75, 75, 75), 1)
    draw_text(canvas, "TCP XYZ (equal scale)", (x0 + 7, y0 + 18),
              0.40, (190, 190, 190))
    for index, point in enumerate(cube_px):
        for other_index in range(index + 1, len(cube_px)):
            # Cube vertices differ by exactly one logical axis.
            if bin(index ^ other_index).count("1") == 1:
                cv2.line(canvas, tuple(point), tuple(cube_px[other_index]),
                         (55, 55, 55), 1, cv2.LINE_AA)
    axis_colors = ((80, 120, 245), (100, 210, 110), (245, 180, 70))
    for point, label, color in zip(axes_px[1:], ("X", "Y", "Z"), axis_colors):
        cv2.arrowedLine(canvas, tuple(axes_px[0]), tuple(point), color, 1,
                        cv2.LINE_AA, tipLength=0.10)
        draw_text(canvas, label, (int(point[0]) + 3, int(point[1]) - 3), 0.42, color, 1)
    if len(points_px) > 1:
        cv2.polylines(canvas, [points_px], False, (105, 105, 105), 1,
                      cv2.LINE_AA)
    if current > 0:
        cv2.polylines(canvas, [points_px[:current + 1]], False,
                      (70, 210, 245), 2, cv2.LINE_AA)
    cv2.circle(canvas, tuple(points_px[current]), 5, (60, 230, 80), -1,
               cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="生成只读的 UMI episode 图像/轨迹同步回放视频；不会连接机械臂")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int,
                        help="只渲染前 N 帧，用于快速检查")
    parser.add_argument("--max-position-step-m", type=float, default=0.08)
    parser.add_argument("--max-rotation-step-rad", type=float, default=1.0)
    parser.add_argument("--trajectory-view", choices=("combined", "xyz3d", "orthographic"),
                        default="combined",
                        help=("轨迹面板：XYZ 三维 + XY/XZ 正投影（默认）、"
                              "仅三维视图或仅 XY/XZ 正投影"))
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须为正数")

    dataset = Path(args.dataset).expanduser().resolve()
    register_codecs()
    root, store = open_root(dataset)
    try:
        ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
        if args.episode < 0 or args.episode >= len(ends):
            parser.error(f"episode 必须在 0..{len(ends) - 1} 范围内")
        start = 0 if args.episode == 0 else int(ends[args.episode - 1])
        end = int(ends[args.episode])
        if args.max_frames is not None:
            if args.max_frames <= 0:
                parser.error("--max-frames 必须为正数")
            end = min(end, start + args.max_frames)
        data = root["data"]
        required = ("camera0_rgb", "robot0_eef_pos",
                    "robot0_eef_rot_axis_angle", "robot0_gripper_width")
        missing = [name for name in required if name not in data]
        if missing:
            raise RuntimeError("缺少数组：" + ", ".join(missing))

        pos = np.asarray(data["robot0_eef_pos"][start:end], dtype=np.float64)
        rot = np.asarray(data["robot0_eef_rot_axis_angle"][start:end], dtype=np.float64)
        width = np.asarray(data["robot0_gripper_width"][start:end, 0],
                           dtype=np.float64)
        if len(pos) == 0:
            raise RuntimeError("所选 episode 没有帧")
        pos_steps = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
        rotations = Rotation.from_rotvec(rot)
        rot_steps = np.r_[0.0, (rotations[1:] * rotations[:-1].inv()).magnitude()]

        if args.output:
            output = Path(args.output).expanduser().resolve()
        else:
            stem = dataset.name
            output = dataset.parent / f"{stem}.episode_{args.episode:03d}.offline_replay.mp4"
        if output.exists():
            raise FileExistsError(f"为保护已有文件，拒绝覆盖：{output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        canvas_w, canvas_h = 960, 540
        image_side = 540
        if args.trajectory_view == "combined":
            trajectory_rect = (562, 155, 770, 340)
            xy_rect = (778, 155, 953, 245)
            xz_rect = (778, 250, 953, 340)
        elif args.trajectory_view == "xyz3d":
            trajectory_rect = (562, 155, 953, 340)
        else:
            xy_rect = (562, 155, 755, 340)
            xz_rect = (764, 155, 953, 340)

        if args.trajectory_view in ("combined", "xyz3d"):
            xyz, xyz_cube, xyz_axes = project_trace_xyz(pos, trajectory_rect)
        if args.trajectory_view in ("combined", "orthographic"):
            xy = project_trace(pos, xy_rect, (0, 1))
            xz = project_trace(pos, xz_rect, (0, 2))
        rate = Fraction(str(args.fps)).limit_denominator(1000)
        container = av.open(str(output), mode="w")
        stream = container.add_stream("libx264", rate=rate)
        stream.width = canvas_w
        stream.height = canvas_h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        try:
            for i, source_idx in enumerate(range(start, end)):
                rgb = np.asarray(data["camera0_rgb"][source_idx])
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (image_side, image_side),
                                 interpolation=cv2.INTER_AREA)
                canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)
                canvas[:, :image_side] = bgr
                draw_text(canvas, "AM_UMI OFFLINE DATA REPLAY", (562, 28), 0.62,
                          (90, 220, 255), 2)
                draw_text(canvas, "NO ROBOT COMMANDS", (562, 52), 0.52,
                          (80, 190, 80), 1)
                draw_text(canvas, f"episode {args.episode}  frame {i + 1}/{len(pos)}",
                          (562, 82))
                draw_text(canvas, f"time {i / args.fps:7.3f} s", (562, 106))
                draw_text(canvas,
                          f"TCP xyz [{pos[i,0]:+.3f}, {pos[i,1]:+.3f}, {pos[i,2]:+.3f}] m",
                          (562, 130), 0.48)
                if args.trajectory_view in ("combined", "xyz3d"):
                    draw_trace_xyz(canvas, xyz, xyz_cube, xyz_axes, i, trajectory_rect)
                if args.trajectory_view in ("combined", "orthographic"):
                    draw_trace(canvas, xy, i, xy_rect, "TCP X-Y")
                    draw_trace(canvas, xz, i, xz_rect, "TCP X-Z")

                pos_bad = pos_steps[i] > args.max_position_step_m
                rot_bad = rot_steps[i] > args.max_rotation_step_rad
                draw_text(canvas, f"position step: {pos_steps[i] * 1000:6.2f} mm",
                          (562, 376), 0.52, (60, 60, 255) if pos_bad else (220, 220, 220),
                          2 if pos_bad else 1)
                draw_text(canvas, f"rotation step: {rot_steps[i]:6.3f} rad",
                          (562, 402), 0.52, (60, 60, 255) if rot_bad else (220, 220, 220),
                          2 if rot_bad else 1)
                draw_text(canvas, f"gripper width: {width[i] * 1000:6.1f} mm",
                          (562, 428), 0.52)
                bar_x0, bar_y0, bar_x1, bar_y1 = 562, 446, 940, 468
                cv2.rectangle(canvas, (bar_x0, bar_y0), (bar_x1, bar_y1),
                              (90, 90, 90), 1)
                ratio = float(np.clip(width[i] / 0.09, 0, 1))
                cv2.rectangle(canvas, (bar_x0 + 2, bar_y0 + 2),
                              (bar_x0 + 2 + int((bar_x1 - bar_x0 - 4) * ratio),
                               bar_y1 - 2), (225, 170, 55), -1)
                progress = (i + 1) / len(pos)
                cv2.rectangle(canvas, (562, 505), (940, 520), (75, 75, 75), 1)
                cv2.rectangle(canvas, (563, 506),
                              (563 + int(376 * progress), 519), (70, 210, 245), -1)

                frame = av.VideoFrame.from_ndarray(canvas, format="bgr24")
                frame.pts = i
                frame.time_base = Fraction(rate.denominator, rate.numerator)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    finally:
        if store is not None:
            store.close()

    print("AM_UMI_OFFLINE_DATASET_REPLAY_OK")
    print("dataset:", dataset)
    print(f"episode: {args.episode}; frames: {len(pos)}; fps: {args.fps:.3f}")
    print("output:", output)
    print("No action was sent to the robot.")


if __name__ == "__main__":
    main()
