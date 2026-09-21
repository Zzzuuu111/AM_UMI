"""Validate an accelerometer calibration using an independent still recording.

Keep the IMU completely still in an orientation that was not one of the six
calibration poses.  A valid calibration makes the corrected acceleration
magnitude close to gravity regardless of orientation.

Usage:
    python imu_work/verify_accel_calib.py \
        --input /tmp/jy901b_still.npz \
        --calib calibration/handheld_gripper_camera/imu/jy901b_accel_v1.json
"""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NPZ from record_jy901b.py")
    parser.add_argument("--calib", required=True, help="accelerometer calibration JSON")
    args = parser.parse_args()

    recording = np.load(args.input)
    with open(args.calib) as f:
        calibration = json.load(f)

    raw_g = recording["accel"].astype(np.float64) * float(recording["accel_scale"])
    raw_mps2 = raw_g * 9.80665
    scale = np.asarray(calibration["scale"], dtype=np.float64)
    bias = np.asarray(calibration["bias"], dtype=np.float64)
    corrected = (raw_mps2 - bias) / scale
    magnitude = np.linalg.norm(corrected, axis=1)

    print("ACCEL_CALIBRATION_VALIDATION")
    print(f"samples: {len(corrected)}")
    print("corrected_mean_mps2: [" + ", ".join(f"{v:+.3f}" for v in corrected.mean(axis=0)) + "]")
    print(f"gravity_mean_mps2: {magnitude.mean():.4f}")
    print(f"gravity_std_mps2: {magnitude.std():.4f}")
    print(f"gravity_range_mps2: [{magnitude.min():.4f}, {magnitude.max():.4f}]")
    error = abs(magnitude.mean() - 9.80665)
    if error < 0.2 and magnitude.std() < 0.1:
        print("ACCEL_CALIBRATION_VALIDATION_OK")
    else:
        print("ACCEL_CALIBRATION_VALIDATION_WARN")
        print("Keep the module still and retry in a different diagonal orientation; "
              "if it still fails, redo calibration with more stable poses.")


if __name__ == "__main__":
    main()
