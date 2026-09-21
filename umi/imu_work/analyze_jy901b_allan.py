"""Estimate JY901B Allan-deviation curves and conservative VIO noise terms."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def overlapping_allan(values: np.ndarray, dt: float, clusters: np.ndarray) -> np.ndarray:
    """Overlapping Allan deviation for samples shaped (N, 3)."""
    cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    output = []
    n = len(values)
    for m in clusters:
        means = (cumulative[m:] - cumulative[:-m]) / m
        difference = means[m:] - means[:-m]
        output.append(np.sqrt(0.5 * np.mean(difference ** 2, axis=0)))
    return np.asarray(output)


def slope_fit(tau: np.ndarray, deviation: np.ndarray, target_slope: float,
              tolerance: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Select a log-log region near a theoretical Allan slope."""
    log_tau, log_dev = np.log(tau), np.log(deviation)
    slope = np.gradient(log_dev, log_tau)
    candidate = np.flatnonzero(np.abs(slope - target_slope) <= tolerance)
    if len(candidate) < 3:
        # Keep a physically meaningful default window if no clean region is
        # found; returned values remain diagnostic rather than authoritative.
        candidate = np.arange(2, min(8, len(tau))) if target_slope < 0 else np.arange(max(0, len(tau)-8), len(tau))
    return candidate, slope


def main() -> None:
    parser = argparse.ArgumentParser(description="计算 JY901B Allan 方差和 VIO 噪声候选值")
    parser.add_argument("--input", required=True)
    parser.add_argument("--accel-calibration", required=True)
    parser.add_argument("--gyro-bias", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = pathlib.Path(args.out).expanduser().resolve()
    if out.exists():
        raise SystemExit(f"为保护已有报告，拒绝覆盖：{out}")
    raw = np.load(pathlib.Path(args.input).expanduser().resolve())
    accel_cal = json.loads(pathlib.Path(args.accel_calibration).expanduser().read_text())
    gyro_bias = np.asarray(json.loads(pathlib.Path(args.gyro_bias).expanduser().read_text())["bias_deg_s"], float)
    accel_bias = np.asarray(accel_cal.get("bias_mps2", accel_cal.get("bias")), float)
    accel_scale = np.asarray(accel_cal["scale"], float)
    tg = np.asarray(raw["t_gyro_monotonic_s"], float)
    ta = np.asarray(raw["t_accel_monotonic_s"], float)
    gyro = np.deg2rad(np.asarray(raw["gyro_raw"], float) * float(raw["gyro_scale_deg_s_per_lsb"]) - gyro_bias)
    accel = np.asarray(raw["accel_raw"], float) * float(raw["accel_scale_g_per_lsb"]) * 9.80665
    accel = (accel - accel_bias) / accel_scale
    # Use gyro grid, resampling accelerometer only for a common Allan timeline.
    dt = float(np.median(np.diff(tg)))
    accel = np.column_stack([np.interp(tg, ta, accel[:, axis]) for axis in range(3)])
    n = len(tg)
    max_cluster = max(4, n // 12)  # at least twelve clusters in the longest tau
    clusters = np.unique(np.clip(np.round(np.geomspace(1, max_cluster, 50)).astype(int), 1, max_cluster))
    tau = clusters * dt
    print("ALLAN_ANALYSIS_STARTED")
    print(f"samples: {n}; sample_rate_hz: {1/dt:.3f}; tau range: {tau[0]:.4f}–{tau[-1]:.1f} s")
    gyro_dev = overlapping_allan(gyro, dt, clusters)
    accel_dev = overlapping_allan(accel, dt, clusters)
    white_indices, gyro_slope = slope_fit(tau, np.linalg.norm(gyro_dev, axis=1), -0.5)
    accel_white_indices, accel_slope = slope_fit(tau, np.linalg.norm(accel_dev, axis=1), -0.5)
    walk_indices, _ = slope_fit(tau, np.linalg.norm(gyro_dev, axis=1), +0.5, tolerance=0.25)
    accel_walk_indices, _ = slope_fit(tau, np.linalg.norm(accel_dev, axis=1), +0.5, tolerance=0.25)
    # Allan convention: white noise density N ≈ sigma(tau)*sqrt(tau).
    gyro_noise_axes = np.median(gyro_dev[white_indices] * np.sqrt(tau[white_indices, None]), axis=0)
    accel_noise_axes = np.median(accel_dev[accel_white_indices] * np.sqrt(tau[accel_white_indices, None]), axis=0)
    # Bias random walk: sigma(tau) ≈ K*sqrt(tau/3).  This is a provisional
    # continuous-time coefficient for tuning, not automatically written to SLAM YAML.
    gyro_walk_axes = np.median(gyro_dev[walk_indices] * np.sqrt(3.0 / tau[walk_indices, None]), axis=0)
    accel_walk_axes = np.median(accel_dev[accel_walk_indices] * np.sqrt(3.0 / tau[accel_walk_indices, None]), axis=0)
    report = {
        "schema": "am_umi_jy901b_allan_v1",
        "input": str(pathlib.Path(args.input).expanduser().resolve()),
        "sample_rate_hz": 1.0 / dt,
        "tau_s": tau.tolist(),
        "gyro_allan_deviation_rad_s": gyro_dev.tolist(),
        "accel_allan_deviation_mps2": accel_dev.tolist(),
        "gyro_white_noise_density_rad_s_sqrt_s_axes": gyro_noise_axes.tolist(),
        "accel_white_noise_density_mps2_sqrt_s_axes": accel_noise_axes.tolist(),
        "gyro_bias_random_walk_rad_s2_sqrt_s_axes": gyro_walk_axes.tolist(),
        "accel_bias_random_walk_mps3_sqrt_s_axes": accel_walk_axes.tolist(),
        "provisional_isotropic": {
            "gyro_noise": float(np.sqrt(np.mean(gyro_noise_axes ** 2))),
            "accel_noise": float(np.sqrt(np.mean(accel_noise_axes ** 2))),
            "gyro_walk": float(np.sqrt(np.mean(gyro_walk_axes ** 2))),
            "accel_walk": float(np.sqrt(np.mean(accel_walk_axes ** 2))),
        },
        "fit_windows": {"gyro_white_indices": white_indices.tolist(),
                        "accel_white_indices": accel_white_indices.tolist(),
                        "gyro_walk_indices": walk_indices.tolist(),
                        "accel_walk_indices": accel_walk_indices.tolist()},
        "note": "Allan-derived candidates require independent VIO validation; do not automatically replace formal ORB-SLAM3 settings.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    values = report["provisional_isotropic"]
    print("ALLAN_ANALYSIS_OK")
    print("provisional isotropic:", {key: f"{value:.7g}" for key, value in values.items()})
    print("saved:", out)


if __name__ == "__main__":
    main()
