"""
Convert Gyroflow-exported CSV (DJI Osmo Action 4) to gopro_slam's imu_data.json.

Gyroflow's CSV for DJI contains valid orientation quaternions but zeros for
raw gyro/accel. This script reconstructs the IMU data:

  GYRO: differentiate the orientation quaternion -> body-frame angular velocity.
  ACCL: synthesize as gravity rotated into the body frame: R^-1 @ [0,0,-9.81].
        (valid approximation for slow hand-held sweeps where linear
        acceleration is negligible)

Output format matches what cheng-chi/ORB_SLAM3 gopro_slam LoadTelemetry reads:
  {"1": {"streams": {"ACCL": {"samples": [{"value":[x,y,z], "cts": ms}...]},
                     "GYRO": {"samples": [...]},
                     "CORI": {"samples": [{"cts": ms}...]}}}}

Usage:
    python scripts/gyroflow_csv_to_imu_json.py -i <export.csv> -o <imu_data.json>
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# %%
import json
import pathlib
import click
import numpy as np
import pandas as pd


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    return np.array([
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ])


def quat_to_rot(q):
    """Unit quaternion (x,y,z,w) -> 3x3 rotation matrix (world->body)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@click.command()
@click.option('-i', '--input', required=True, help='Gyroflow exported CSV')
@click.option('-o', '--output', default=None, help='Output imu_data.json')
@click.option('-g', '--gravity', type=float, default=9.81, help='Gravity magnitude (m/s^2)')
def main(input, output, gravity):
    input = pathlib.Path(os.path.expanduser(input))
    assert input.is_file(), f"Input CSV not found: {input}"
    if output is None:
        output = input.with_name(input.stem + '_imu_data.json')
    output = pathlib.Path(os.path.expanduser(output))

    df = pd.read_csv(input)
    t = df['timestamp_ms'].values.astype(float) / 1000.0
    q = df[['org_quat_x', 'org_quat_y', 'org_quat_z', 'org_quat_w']].values.astype(float)

    # unwrap quaternion sign
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]

    # ---- GYRO: differentiate quaternion (body-frame angular velocity) ----
    q_dot = np.gradient(q, t, axis=0)
    gyro = np.array([2 * quat_mul(q_dot[k], quat_conj(q[k]))[:3]
                     for k in range(len(q))])

    # ---- ACCL: gravity rotated into body frame ----
    acc = np.zeros_like(gyro)
    g_world = np.array([0.0, 0.0, -gravity])
    for k in range(len(q)):
        R = quat_to_rot(q[k])          # world -> body
        acc[k] = R @ g_world

    # ---- build openicc-style json (cts in milliseconds) ----
    cts = (t * 1000.0)
    accl_samples = [{"value": [float(a[0]), float(a[1]), float(a[2])],
                     "cts": float(c)} for a, c in zip(acc, cts)]
    gyro_samples = [{"value": [float(g[0]), float(g[1]), float(g[2])],
                     "cts": float(c)} for g, c in zip(gyro, cts)]
    cori_samples = [{"cts": float(c)} for c in cts]

    result = {"1": {"streams": {
        "ACCL": {"samples": accl_samples},
        "GYRO": {"samples": gyro_samples},
        "CORI": {"samples": cori_samples},
    }}}

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(result, f)
    print(f"Saved -> {output}")
    print(f"  samples: {len(gyro_samples)}, duration: {t[-1]-t[0]:.2f}s")
    print(f"  gyro magnitude range: [{np.abs(gyro).max():.3f}] rad/s")
    print(f"  acc magnitude: mean={np.linalg.norm(acc,axis=1).mean():.3f} m/s^2 (should ~= gravity)")


if __name__ == "__main__":
    main()
