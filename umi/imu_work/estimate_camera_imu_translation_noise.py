"""Provisional camera--IMU translation and JY901B noise estimate from a fixed tag.

The camera pose is metric because the fixed ArUco marker has a known size.
Rigid-body acceleration then relates the camera origin, the IMU origin, and
gravity.  This is an offline estimate; validate it on another motion session
before enabling full VIO for production data.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

import av
import cv2
import numpy as np
import yaml
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from imu_work.uvc_payload_timing import load_preferred_frame_times


def load_times(path):
    # Preserve the legacy path-shaped API used by the independent validator,
    # but prefer the session's UVC PTS/SCR-derived time axis when available.
    times, _ = load_preferred_frame_times(pathlib.Path(path).parent)
    return times


def intrinsics(path):
    d = json.loads(pathlib.Path(path).read_text())
    x = d["intrinsics"]
    k = np.array([[x["focal_length"], 0, x["principal_pt_x"]],
                  [0, x["focal_length"], x["principal_pt_y"]], [0, 0, 1.]], float)
    dist = np.array([x["radial_distortion_1"], x["radial_distortion_2"],
                     x["radial_distortion_3"], x["radial_distortion_4"]], float).reshape(4, 1)
    return k, dist


def tag_poses(video, frame_times, tag_id, size, k, dist, dictionary_name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    h = size / 2
    obj = np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], float)
    ts, rs, ps = [], [], []
    print("TAG_METRIC_POSE_EXTRACTION_STARTED", flush=True)
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= len(frame_times):
                raise RuntimeError("视频帧数超过 frame_timestamps.csv")
            gray = cv2.cvtColor(fr.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            hit = np.flatnonzero(ids.reshape(-1) == tag_id)
            if len(hit) != 1:
                continue
            img = np.asarray(corners[int(hit[0])], float).reshape(4,1,2)
            und = cv2.fisheye.undistortPoints(img, k, dist, P=k)
            ok, rv, tv = cv2.solvePnP(obj, und, k, np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and np.isfinite(rv).all() and tv[2,0] > 0:
                rct = Rotation.from_rotvec(rv.ravel()).as_matrix()
                # Camera origin in the fixed Tag/world frame.
                pwc = -rct.T @ tv.reshape(3)
                ts.append(frame_times[i]); rs.append(rct); ps.append(pwc)
            if len(ts) and len(ts) % 750 == 0:
                print(f"已获得 {len(ts)} 帧固定 Tag 位姿", flush=True)
    if len(ts) < 300:
        raise RuntimeError("可用 Tag 位姿不足 300 帧")
    return np.asarray(ts), np.asarray(rs), np.asarray(ps)


def chunks(times, max_gap=0.10, min_count=35):
    starts = np.r_[0, np.flatnonzero(np.diff(times) > max_gap) + 1]
    ends = np.r_[starts[1:], len(times)]
    return [(a,b) for a,b in zip(starts, ends) if b-a >= min_count]


def skew(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])


def interp(t, tx, values):
    return np.column_stack([np.interp(t, tx, values[:,j]) for j in range(3)])


def robust_fit(a, b):
    """Fit one six-unknown linear system per 3D sample, robustly by sample."""
    keep = np.ones(len(b), dtype=bool)
    for _ in range(4):
        x, *_ = np.linalg.lstsq(a[keep].reshape(-1, a.shape[-1]),
                                b[keep].reshape(-1), rcond=None)
        residual = np.linalg.norm(a @ x - b, axis=1)
        threshold = max(np.quantile(residual[keep], .85), 0.15)
        keep = residual <= threshold
    x, *_ = np.linalg.lstsq(a[keep].reshape(-1, a.shape[-1]),
                            b[keep].reshape(-1), rcond=None)
    residual = np.linalg.norm(a @ x - b, axis=1)
    return x, keep, residual


def main():
    p = argparse.ArgumentParser(description="估计相机—IMU 平移外参和 JY901B VIO 噪声")
    p.add_argument("--session-dir", required=True)
    p.add_argument("--tag-id", type=int, default=13)
    p.add_argument("--intrinsics", required=True)
    p.add_argument("--rotation-time-calibration", required=True)
    p.add_argument("--accel-calibration", required=True)
    p.add_argument("--gyro-bias", required=True)
    p.add_argument("--aruco-config", default="calibration/shared_tags/aruco_config.yaml")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    session = pathlib.Path(args.session_dir).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"为保护已有结果，拒绝覆盖：{out}")
    out.parent.mkdir(parents=True, exist_ok=True)

    rotcal_path = pathlib.Path(args.rotation_time_calibration).expanduser().resolve()
    rotcal = json.loads(rotcal_path.read_text())
    rci = np.asarray(rotcal["R_camera_imu"], float)
    offset = float(rotcal["imu_time_offset_for_camera_s"])
    acccal_path = pathlib.Path(args.accel_calibration).expanduser().resolve()
    acccal = json.loads(acccal_path.read_text())
    abias = np.asarray(acccal.get("bias_mps2", acccal.get("bias")), float)
    ascale = np.asarray(acccal["scale"], float)
    gbias_path = pathlib.Path(args.gyro_bias).expanduser().resolve()
    gbias = np.asarray(json.loads(gbias_path.read_text())["bias_deg_s"], float)
    raw = np.load(session / "imu_raw.npz")
    start = float(raw["recording_start_monotonic_ns"]) / 1e9
    ta = np.asarray(raw["t_accel_monotonic_s"], float) - start
    tg = np.asarray(raw["t_gyro_monotonic_s"], float) - start
    acc = np.asarray(raw["accel_raw"], float) * float(raw["accel_scale_g_per_lsb"]) * 9.80665
    acc = (acc - abias) / ascale
    gyro = np.deg2rad(np.asarray(raw["gyro_raw"], float) * float(raw["gyro_scale_deg_s_per_lsb"]) - gbias)
    order = np.argsort(ta); ta, acc = ta[order], acc[order]
    order = np.argsort(tg); tg, gyro = tg[order], gyro[order]

    config = yaml.safe_load(pathlib.Path(args.aruco_config).expanduser().read_text())
    sizes = config["marker_size_map"]
    size = float(sizes.get(args.tag_id, sizes.get(str(args.tag_id), sizes["default"])))
    ft, frame_time_path = load_preferred_frame_times(session)
    print("camera_frame_timestamp_source:", frame_time_path)
    times, rct, pwc = tag_poses(session / "raw_video.mp4", ft, args.tag_id, size,
                                 *intrinsics(args.intrinsics), config["aruco_dict"]["predefined"])

    rows_a, rows_b = [], []
    used = 0
    for begin, end in chunks(times):
        t, r, pos = times[begin:end], rct[begin:end], pwc[begin:end]
        dt = float(np.median(np.diff(t)))
        window = min(len(t) // 2 * 2 - 1, 31)
        # 21 samples is about 0.7 s and suppresses PnP position jitter.
        window = max(5, min(window, 31))
        if window % 2 == 0: window -= 1
        if window < 5: continue
        acc_world = savgol_filter(pos, window, 3 if window >= 7 else 2, deriv=2, delta=dt, axis=0)
        omega = np.zeros((len(t),3))
        for i in range(1, len(t)-1):
            omega[i] = -(Rotation.from_matrix(r[i+1]) * Rotation.from_matrix(r[i-1]).inv()).as_rotvec() / (t[i+1]-t[i-1])
        omega[0], omega[-1] = omega[1], omega[-2]
        alpha = savgol_filter(omega, window, 3 if window >= 7 else 2, deriv=1, delta=dt, axis=0)
        query = t + offset
        valid = (query >= ta[0]) & (query <= ta[-1])
        for i in np.flatnonzero(valid)[window//2: -window//2 or None]:
            f_camera = rci @ interp(np.array([query[i]]), ta, acc)[0]
            a_camera = r[i] @ acc_world[i]
            lever = skew(alpha[i]) + skew(omega[i]) @ skew(omega[i])
            rows_a.append(np.hstack((lever, -r[i])))
            rows_b.append(f_camera - a_camera)
            used += 1
    if used < 150:
        raise RuntimeError(f"仅 {used} 个有效刚体加速度样本，无法估计平移")
    a, b = np.asarray(rows_a), np.asarray(rows_b)
    solution, keep, residual = robust_fit(a, b)
    r_camera_imu = solution[:3]
    gravity_world = solution[3:]
    # Noise density is estimated from the two instructed stationary windows.
    imu_camera_t = tg - offset
    stationary = ((imu_camera_t >= 0) & (imu_camera_t <= 5)) | ((imu_camera_t >= 110) & (imu_camera_t <= 120))
    dt_imu = float(np.median(np.diff(tg)))
    gyro_density = np.std(gyro[stationary], axis=0) * np.sqrt(dt_imu)
    accel_on_gyro = interp(tg[stationary], ta, acc)
    accel_density = np.std(accel_on_gyro, axis=0) * np.sqrt(dt_imu)
    condition = float(np.linalg.cond(a[keep].reshape(-1, a.shape[-1])))
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    plausible = bool(np.linalg.norm(r_camera_imu) <= .25 and 7.0 <= np.linalg.norm(gravity_world) <= 13.0 and condition < 1e5)
    result = {
        "schema": "am_umi_camera_imu_translation_noise_v1",
        "session": str(session), "fixed_tag_id": args.tag_id,
        "rotation_time_source": str(rotcal_path),
        "translation_name": "r_camera_to_imu_in_camera_m",
        "translation_definition": "vector from camera optical origin to IMU origin, expressed in camera coordinates",
        "r_camera_to_imu_in_camera_m": r_camera_imu.tolist(),
        "estimated_gravity_in_tag_world_mps2": gravity_world.tolist(),
        "gravity_norm_mps2": float(np.linalg.norm(gravity_world)),
        "fit_rmse_mps2": rmse, "fit_condition_number": condition,
        "inlier_samples": int(keep.sum()), "candidate_samples": int(len(keep)),
        "gyro_noise_density_rad_s_sqrt_s": gyro_density.tolist(),
        "accel_noise_density_mps2_sqrt_s": accel_density.tolist(),
        "provisional_plausibility_pass": plausible,
        "note": "Provisional estimate from one session; validate on an independent high-excitation session before changing production VIO settings.",
    }
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("CAMERA_IMU_TRANSLATION_NOISE_ESTIMATE_OK")
    print("tag_pose_frames:", len(times), "rigid_body_samples:", int(keep.sum()), "/", len(keep))
    print("r_camera_to_imu_in_camera_m:", [round(x,5) for x in r_camera_imu])
    print(f"gravity_norm_mps2: {np.linalg.norm(gravity_world):.4f}")
    print(f"fit_rmse_mps2: {rmse:.4f}; condition: {condition:.1f}")
    print("provisional_plausibility_pass:", "YES" if plausible else "NO")
    print("saved:", out)


if __name__ == "__main__":
    main()
