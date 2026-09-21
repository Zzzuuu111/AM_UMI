"""Create the standard UMI bottom-of-image hand-gripper exclusion mask.

White pixels are ignored by the patched gopro_slam reader.  The trapezoid
matches the original UMI GoPro reader's draw_gripper_mask() geometry, so it
removes camera-rigid fingers and their ArUco markers from ORB features while
preserving the workspace above them.
"""

from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="生成手持夹爪区域的 ORB-SLAM3 遮罩")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--profile", choices=("umi_bottom_v1", "emeet_observed_v2"),
                        default="umi_bottom_v1",
                        help="umi_bottom_v1 为原 UMI 下方梯形；emeet_observed_v2 额外遮住 EMEET 画面中可见的固定夹爪两侧")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("宽和高必须为正数")
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"为保护已有遮罩，拒绝覆盖：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    h, w = args.height, args.width
    mask = np.zeros((h, w), np.uint8)
    # Same normalized geometry as UMI's mono_inertial_gopro_vi.cc.
    top_y = round((1.0 - 0.37) * h)
    mid_x = w / 2.0
    top_half = 0.25 * h / 2.0
    bottom_half = 1.4 * h / 2.0
    bottom = np.array([[round(mid_x - bottom_half), h - 1],
                       [round(mid_x - top_half), top_y],
                       [round(mid_x + top_half), top_y],
                       [round(mid_x + bottom_half), h - 1]], np.int32)
    cv2.fillPoly(mask, [bottom], 255)
    if args.profile == "emeet_observed_v2":
        # Envelope measured from the EMEET hand-held rig images.  The left and
        # right white housings, black fingers and their tags are rigidly tied
        # to the camera and must not become ORB world landmarks. Coordinates
        # are normalized so this profile also works at non-default resolution.
        def poly(points):
            return np.asarray([[round(x * w / 960), round(y * h / 540)]
                               for x, y in points], np.int32)
        left = poly([(115, 539), (115, 325), (205, 275), (235, 145),
                     (400, 145), (430, 270), (370, 539)])
        right = poly([(530, 539), (545, 270), (610, 145), (795, 145),
                      (825, 280), (950, 325), (950, 539)])
        cv2.fillPoly(mask, [left, right], 255)
    if not cv2.imwrite(str(out), mask):
        raise RuntimeError(f"无法写入：{out}")
    print("HANDHELD_GRIPPER_MASK_CREATED")
    print("output:", out)
    print(f"profile: {args.profile}; resolution: {w}x{h}; ignored_area_ratio: {np.mean(mask > 0):.1%}")
    print("白色区域会被 ORB-SLAM3 忽略；原始视频与标定文件没有被修改。")


if __name__ == "__main__":
    main()
