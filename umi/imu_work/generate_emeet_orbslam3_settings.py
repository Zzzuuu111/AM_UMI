"""Generate an EMEET/JY901B ORB-SLAM3 settings YAML from saved calibration."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="生成 EMEET + JY901B 专用 ORB-SLAM3 设置")
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--camera-imu-calibration", required=True)
    parser.add_argument("--translation-noise-calibration",
                        help="可选：由固定 Tag 高激励序列得到的平移外参/噪声 JSON")
    parser.add_argument("--translation-override-calibration",
                        help="可选：用联合加速度时间估计中的平移覆盖前者；噪声仍取前者")
    parser.add_argument("--out", required=True, help="新的 YAML 文件；已有文件会被保护")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("宽、高和 FPS 必须为正数")
    if not float(args.fps).is_integer():
        parser.error("当前 ORB-SLAM3 读取器要求 --fps 为整数")

    intrinsics_path = pathlib.Path(args.intrinsics).expanduser().resolve()
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    if intrinsics.get("intrinsic_type") != "FISHEYE":
        raise RuntimeError("仅支持 FISHEYE 内参 JSON")
    source_w, source_h = int(intrinsics["image_width"]), int(intrinsics["image_height"])
    scale_x, scale_y = args.width / source_w, args.height / source_h
    if abs(scale_x - scale_y) > 1e-9:
        raise RuntimeError("目标分辨率必须保持原始画面纵横比")
    value = intrinsics["intrinsics"]
    focal = float(value["focal_length"]) * scale_x
    cx = float(value["principal_pt_x"]) * scale_x
    cy = float(value["principal_pt_y"]) * scale_y

    calibration_path = pathlib.Path(args.camera_imu_calibration).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rotation_camera_imu = np.asarray(calibration["R_camera_imu"], dtype=np.float64)
    if rotation_camera_imu.shape != (3, 3):
        raise RuntimeError("R_camera_imu 必须为 3×3")
    # ORB-SLAM3's T_b_c1 maps camera coordinates into IMU/body coordinates.
    rotation_body_camera = rotation_camera_imu.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_body_camera
    translation_note = "translation: initialized to zero (not yet calibrated)"
    noise_gyro, noise_acc = 0.0030, 0.0300
    if args.translation_noise_calibration:
        tn_path = pathlib.Path(args.translation_noise_calibration).expanduser().resolve()
        tn = json.loads(tn_path.read_text(encoding="utf-8"))
        if not tn.get("provisional_plausibility_pass", False):
            raise RuntimeError("平移/噪声估计未通过基础合理性检查，拒绝写入 VIO 设置")
        r_camera_imu = np.asarray(tn["r_camera_to_imu_in_camera_m"], dtype=np.float64)
        if r_camera_imu.shape != (3,):
            raise RuntimeError("r_camera_to_imu_in_camera_m 必须为三维向量")
        # p_b = R_bc (p_c - r_ci), so t_bc = -R_bc r_ci.
        transform[:3, 3] = -rotation_body_camera @ r_camera_imu
        gyro_density = np.asarray(tn["gyro_noise_density_rad_s_sqrt_s"], dtype=np.float64)
        accel_density = np.asarray(tn["accel_noise_density_mps2_sqrt_s"], dtype=np.float64)
        if gyro_density.shape != (3,) or accel_density.shape != (3,):
            raise RuntimeError("噪声密度必须各包含三个坐标轴")
        # ORB-SLAM3 accepts one isotropic value; RMS preserves the total
        # measured variance without optimistically selecting a quiet axis.
        noise_gyro = float(np.sqrt(np.mean(gyro_density ** 2)))
        noise_acc = float(np.sqrt(np.mean(accel_density ** 2)))
        translation_note = f"translation/noise source: {tn_path} (provisional; validate independently)"
    if args.translation_override_calibration:
        override_path = pathlib.Path(args.translation_override_calibration).expanduser().resolve()
        override = json.loads(override_path.read_text(encoding="utf-8"))
        r_camera_imu = np.asarray(override["r_camera_to_imu_in_camera_m"], dtype=np.float64)
        if r_camera_imu.shape != (3,):
            raise RuntimeError("覆盖平移必须为三维向量")
        transform[:3, 3] = -rotation_body_camera @ r_camera_imu
        translation_note = f"translation source: {override_path}; noise source remains as selected above (provisional)"
    matrix_values = ", ".join(f"{item:.9f}" for item in transform.reshape(-1))

    output_path = pathlib.Path(args.out).expanduser().resolve()
    if output_path.exists():
        raise SystemExit(f"为保护已有设置，拒绝覆盖：{output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = f'''%YAML:1.0
# EMEET 1080P + JY901B, generated from AM_UMI calibration.
# Input is resized by gopro_slam to {args.width}x{args.height}; the patched
# reader uses per-frame host timestamps exported in imu_data.json/CORI.
File.version: "1.0"
Camera.type: "KannalaBrandt8"
Camera1.fx: {focal:.9f}
Camera1.fy: {focal:.9f}
Camera1.cx: {cx:.9f}
Camera1.cy: {cy:.9f}
Camera1.k1: {float(value['radial_distortion_1']):.12f}
Camera1.k2: {float(value['radial_distortion_2']):.12f}
Camera1.k3: {float(value['radial_distortion_3']):.12f}
Camera1.k4: {float(value['radial_distortion_4']):.12f}
Camera.width: {args.width}
Camera.height: {args.height}
Camera.fps: {int(args.fps)}
Camera.RGB: 0

# T_b_c1: camera -> JY901B/body. Rotation comes from the independently
# validated R_camera_imu. {translation_note}
IMU.T_b_c1: !!opencv-matrix
    rows: 4
    cols: 4
    dt: f
    data: [ {matrix_values} ]

# Static measured noise densities for JY901B at about 200 Hz (or conservative
# defaults when no translation/noise calibration JSON was supplied).
IMU.NoiseGyro: {noise_gyro:.9f}
IMU.NoiseAcc: {noise_acc:.9f}
IMU.GyroWalk: 5.0e-5
IMU.AccWalk: 0.0055
IMU.Frequency: 200.0

ORBextractor.nFeatures: 1500
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
System.thFarPoints: 20.0

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 1.0
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 1.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
'''
    output_path.write_text(text, encoding="utf-8")
    print("EMEET_ORB_SLAM3_SETTINGS_CREATED")
    print("output:", output_path)
    print(f"resolution: {args.width}x{args.height}; fps: {args.fps:.3f}")
    print("T_b_c1 rotation source:", calibration_path)
    print(translation_note)
    print(f"noise densities: gyro={noise_gyro:.6f} rad/s/sqrt(s); accel={noise_acc:.6f} m/s^2/sqrt(s)")


if __name__ == "__main__":
    main()
