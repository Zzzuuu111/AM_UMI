"""Jointly estimate camera--IMU translation and accelerometer time offset.

Gyro/camera timing is determined from rotation.  An external IMU can have a
different accelerometer filtering latency, so this tool estimates an explicit
accelerometer offset from metric fixed-Tag motion.  It is intentionally
validated on a separate session before being used for VIO export.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.estimate_camera_imu_translation_noise import (  # noqa: E402
    chunks, interp, intrinsics, load_times, robust_fit, skew, tag_poses,
)


def camera_motion_samples(times, rct, pwc):
    """Return camera-time samples independent of any IMU timing choice."""
    out_t, out_a, out_l, out_r = [], [], [], []
    for begin, end in chunks(times):
        t, r, pos = times[begin:end], rct[begin:end], pwc[begin:end]
        dt = float(np.median(np.diff(t)))
        window = max(5, min(len(t) // 2 * 2 - 1, 31))
        if window % 2 == 0:
            window -= 1
        if window < 5:
            continue
        acc_world = savgol_filter(pos, window, 3 if window >= 7 else 2,
                                  deriv=2, delta=dt, axis=0)
        omega = np.zeros((len(t), 3))
        for i in range(1, len(t) - 1):
            omega[i] = -(Rotation.from_matrix(r[i + 1]) *
                         Rotation.from_matrix(r[i - 1]).inv()).as_rotvec() / (t[i + 1] - t[i - 1])
        omega[0], omega[-1] = omega[1], omega[-2]
        alpha = savgol_filter(omega, window, 3 if window >= 7 else 2,
                              deriv=1, delta=dt, axis=0)
        for i in range(window // 2, len(t) - window // 2):
            out_t.append(t[i])
            out_a.append(r[i] @ acc_world[i])
            out_l.append(skew(alpha[i]) + skew(omega[i]) @ skew(omega[i]))
            out_r.append(r[i])
    return np.asarray(out_t), np.asarray(out_a), np.asarray(out_l), np.asarray(out_r)


def main() -> None:
    parser = argparse.ArgumentParser(description="离线估计加速度专用时间偏移与相机—IMU 平移")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--rotation-time-calibration", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--accel-calibration", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--min-offset-s", type=float, default=-0.120)
    parser.add_argument("--max-offset-s", type=float, default=0.080)
    parser.add_argument("--step-s", type=float, default=0.002)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.step_s <= 0 or args.min_offset_s >= args.max_offset_s:
        parser.error("偏移范围或步长无效")
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"为保护已有结果，拒绝覆盖：{out}")
    session = pathlib.Path(args.session_dir).expanduser().resolve()
    rot_path = pathlib.Path(args.rotation_time_calibration).expanduser().resolve()
    rot = json.loads(rot_path.read_text())
    rci = np.asarray(rot["R_camera_imu"], float)
    gyro_offset = float(rot["imu_time_offset_for_camera_s"])
    acccal = json.loads(pathlib.Path(args.accel_calibration).expanduser().read_text())
    bias = np.asarray(acccal.get("bias_mps2", acccal.get("bias")), float)
    scale = np.asarray(acccal["scale"], float)
    raw = np.load(session / "imu_raw.npz")
    start = float(raw["recording_start_monotonic_ns"]) / 1e9
    ta = np.asarray(raw["t_accel_monotonic_s"], float) - start
    acc = np.asarray(raw["accel_raw"], float) * float(raw["accel_scale_g_per_lsb"]) * 9.80665
    acc = (acc - bias) / scale
    order = np.argsort(ta); ta, acc = ta[order], acc[order]
    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text())
    sizes = config["marker_size_map"]
    size = float(sizes.get(args.tag_id, sizes.get(str(args.tag_id), sizes["default"])))
    times, rct, pwc = tag_poses(session / "raw_video.mp4", load_times(session / "frame_timestamps.csv"),
                                 args.tag_id, size, *intrinsics(args.intrinsics),
                                 config["aruco_dict"]["predefined"])
    t, camera_acc, lever, rotation = camera_motion_samples(times, rct, pwc)
    offsets = np.arange(args.min_offset_s, args.max_offset_s + args.step_s * .25, args.step_s)
    results = []
    print("ACCEL_TIME_OFFSET_SEARCH_STARTED")
    for offset in offsets:
        query = t + offset
        valid = (query >= ta[0]) & (query <= ta[-1])
        if valid.sum() < 300:
            continue
        f_camera = (rci @ interp(query[valid], ta, acc).T).T
        a = np.concatenate((lever[valid], -rotation[valid]), axis=2)
        b = f_camera - camera_acc[valid]
        solution, keep, residual = robust_fit(a, b)
        rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
        condition = float(np.linalg.cond(a[keep].reshape(-1, 6)))
        results.append({"offset_s": float(offset), "rmse_mps2": rmse,
                        "condition": condition, "inliers": int(keep.sum()),
                        "solution": solution})
    if not results:
        raise RuntimeError("没有可用的偏移候选")
    best = min(results, key=lambda x: x["rmse_mps2"])
    r_camera_imu = best["solution"][:3]
    gravity = best["solution"][3:]
    result = {
        "schema": "am_umi_camera_imu_accel_time_translation_v1",
        "session": str(session), "rotation_time_source": str(rot_path),
        "gyro_time_offset_for_camera_s": gyro_offset,
        "accel_time_offset_for_camera_s": best["offset_s"],
        "time_offset_definition": "for camera host time t, compare acceleration at t + accel_time_offset_for_camera_s",
        "r_camera_to_imu_in_camera_m": r_camera_imu.tolist(),
        "estimated_gravity_in_tag_world_mps2": gravity.tolist(),
        "gravity_norm_mps2": float(np.linalg.norm(gravity)),
        "fit_rmse_mps2": best["rmse_mps2"], "fit_condition_number": best["condition"],
        "inlier_samples": best["inliers"], "candidate_samples": int(len(t)),
        "offset_search": [{key: value for key, value in row.items() if key != "solution"} for row in results],
        "note": "Candidate acceleration-specific delay and translation. Validate independently before VIO export.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("ACCEL_TIME_OFFSET_SEARCH_OK")
    print(f"gyro_offset_s: {gyro_offset:+.4f}; accel_offset_s: {best['offset_s']:+.4f}")
    print("r_camera_to_imu_in_camera_m:", [round(float(x), 5) for x in r_camera_imu])
    print(f"gravity_norm_mps2: {np.linalg.norm(gravity):.4f}; rmse_mps2: {best['rmse_mps2']:.4f}")
    print("saved:", out)


if __name__ == "__main__":
    main()
