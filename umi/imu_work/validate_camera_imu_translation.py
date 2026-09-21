"""Independently validate a fixed camera--IMU translation estimate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.estimate_camera_imu_translation_noise import (  # noqa: E402
    chunks, interp, intrinsics, load_times, robust_fit, skew, tag_poses,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="独立验证相机—IMU 平移外参")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--translation-calibration", required=True)
    parser.add_argument("--accel-time-calibration", default=None,
                        help="可选：含 accel_time_offset_for_camera_s 的联合估计 JSON")
    parser.add_argument("--rotation-time-calibration", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--accel-calibration", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    session = pathlib.Path(args.session_dir).expanduser().resolve()
    translation_path = pathlib.Path(args.translation_calibration).expanduser().resolve()
    rot_path = pathlib.Path(args.rotation_time_calibration).expanduser().resolve()
    accel_time_path = (pathlib.Path(args.accel_time_calibration).expanduser().resolve()
                       if args.accel_time_calibration else translation_path)
    accel_time = json.loads(accel_time_path.read_text(encoding="utf-8"))
    r_camera_imu = np.asarray(accel_time["r_camera_to_imu_in_camera_m"], float)
    if r_camera_imu.shape != (3,):
        raise RuntimeError("平移标定文件中的相机到 IMU 向量格式错误")
    rotcal = json.loads(rot_path.read_text(encoding="utf-8"))
    rci = np.asarray(rotcal["R_camera_imu"], float)
    offset = float(rotcal["imu_time_offset_for_camera_s"])
    accel_offset = float(accel_time.get("accel_time_offset_for_camera_s", offset))
    acccal = json.loads(pathlib.Path(args.accel_calibration).expanduser().read_text())
    abias = np.asarray(acccal.get("bias_mps2", acccal.get("bias")), float)
    ascale = np.asarray(acccal["scale"], float)
    gbias = np.asarray(json.loads(pathlib.Path(args.gyro_bias).expanduser().read_text())["bias_deg_s"], float)
    raw = np.load(session / "imu_raw.npz")
    start = float(raw["recording_start_monotonic_ns"]) / 1e9
    ta = np.asarray(raw["t_accel_monotonic_s"], float) - start
    tg = np.asarray(raw["t_gyro_monotonic_s"], float) - start
    acc = np.asarray(raw["accel_raw"], float) * float(raw["accel_scale_g_per_lsb"]) * 9.80665
    acc = (acc - abias) / ascale
    gyro = np.deg2rad(np.asarray(raw["gyro_raw"], float) * float(raw["gyro_scale_deg_s_per_lsb"]) - gbias)
    order = np.argsort(ta); ta, acc = ta[order], acc[order]
    order = np.argsort(tg); tg, gyro = tg[order], gyro[order]
    import yaml
    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text())
    sizes = config["marker_size_map"]
    size = float(sizes.get(args.tag_id, sizes.get(str(args.tag_id), sizes["default"])))
    times, rct, pwc = tag_poses(session / "raw_video.mp4", load_times(session / "frame_timestamps.csv"),
                                 args.tag_id, size, *intrinsics(args.intrinsics),
                                 config["aruco_dict"]["predefined"])
    rows, rhs = [], []
    for begin, end in chunks(times):
        t, r, pos = times[begin:end], rct[begin:end], pwc[begin:end]
        dt = float(np.median(np.diff(t)))
        window = max(5, min(len(t) // 2 * 2 - 1, 31))
        if window % 2 == 0: window -= 1
        if window < 5: continue
        from scipy.signal import savgol_filter
        from scipy.spatial.transform import Rotation
        acc_world = savgol_filter(pos, window, 3 if window >= 7 else 2, deriv=2, delta=dt, axis=0)
        omega = np.zeros((len(t), 3))
        for i in range(1, len(t) - 1):
            omega[i] = -(Rotation.from_matrix(r[i + 1]) * Rotation.from_matrix(r[i - 1]).inv()).as_rotvec() / (t[i + 1] - t[i - 1])
        omega[0], omega[-1] = omega[1], omega[-2]
        alpha = savgol_filter(omega, window, 3 if window >= 7 else 2, deriv=1, delta=dt, axis=0)
        query = t + accel_offset
        valid = (query >= ta[0]) & (query <= ta[-1])
        for i in np.flatnonzero(valid)[window // 2: -window // 2 or None]:
            f_camera = rci @ interp(np.array([query[i]]), ta, acc)[0]
            a_camera = r[i] @ acc_world[i]
            lever = skew(alpha[i]) + skew(omega[i]) @ skew(omega[i])
            # R_ct * g = a_camera + lever*r_ci - f_camera.
            rows.append(r[i])
            rhs.append(a_camera + lever @ r_camera_imu - f_camera)
    a, b = np.asarray(rows), np.asarray(rhs)
    if len(a) < 150:
        raise RuntimeError("独立会话有效样本不足")
    # Same robust per-sample fit helper, with a 3x3 gravity-only system.
    gravity, keep, residual = robust_fit(a, b)
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    gravity_norm = float(np.linalg.norm(gravity))
    passed = bool(rmse <= 0.20 and 8.5 <= gravity_norm <= 11.0 and keep.sum() >= 300)
    report = {
        "schema": "am_umi_camera_imu_translation_validation_v1",
        "session": str(session), "translation_calibration": str(translation_path),
        "rotation_time_calibration": str(rot_path),
        "accel_time_calibration": str(accel_time_path),
        "accel_time_offset_for_camera_s": accel_offset,
        "r_camera_to_imu_in_camera_m": r_camera_imu.tolist(),
        "gravity_in_tag_world_mps2": gravity.tolist(), "gravity_norm_mps2": gravity_norm,
        "fit_rmse_mps2": rmse, "inlier_samples": int(keep.sum()),
        "candidate_samples": int(len(keep)), "pass_recommendation": passed,
        "note": "Independent fixed-translation validation; a pass supports but does not guarantee full VIO stability.",
    }
    out = pathlib.Path(args.out).expanduser().resolve() if args.out else session / "camera_imu_translation_validation.json"
    if out.exists():
        raise SystemExit(f"为保护已有报告，拒绝覆盖：{out}")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("CAMERA_IMU_TRANSLATION_VALIDATION_OK")
    print("session:", session)
    print(f"accel_time_offset_s: {accel_offset:+.4f}")
    print(f"gravity_norm_mps2: {gravity_norm:.4f}; fit_rmse_mps2: {rmse:.4f}")
    print(f"samples: {keep.sum()} / {len(keep)}")
    print("pass_recommendation:", "YES" if passed else "NO")
    print("report:", out)


if __name__ == "__main__":
    main()
