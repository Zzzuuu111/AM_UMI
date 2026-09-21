#!/usr/bin/env python
"""Batch2 demo quality report: SLAM trajectory + tag detection + gripper width."""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def analyze_demo(demo_dir: Path) -> dict:
    out = {'demo': demo_dir.name, 'video_s': None, 'frames': None}

    # ---- SLAM trajectory ----
    csv_path = demo_dir / 'camera_trajectory.csv'
    if not csv_path.exists():
        out['slam'] = 'MISSING'
        return out
    rows = list(csv.DictReader(csv_path.open()))
    n = len(rows)
    out['frames'] = n
    lost_flags = np.array([r['is_lost'] == 'true' for r in rows])
    n_lost = int(lost_flags.sum())
    lost_pct = 100.0 * n_lost / max(n, 1)
    out['lost_pct'] = lost_pct
    out['n_lost'] = n_lost
    # jumps = lost -> tracked transitions (each = one relocalization)
    jumps = int(np.sum((~lost_flags[1:]) & lost_flags[:-1]))
    out['jumps'] = jumps
    xyz = np.array([[float(r['x']), float(r['y']), float(r['z'])]
                    for r in rows if r['is_lost'] == 'false'])
    if len(xyz):
        lo, hi = xyz.min(0), xyz.max(0)
        out['x_range'] = (lo[0], hi[0])
        out['y_range'] = (lo[1], hi[1])
        out['z_range'] = (lo[2], hi[2])
        # speed / smoothness: max inter-frame jump of tracked poses
        d = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        out['max_jump_m'] = float(d.max()) if len(d) else 0.0
    else:
        out['x_range'] = (np.nan, np.nan)
        out['max_jump_m'] = np.nan

    # ---- tag detection ----
    pkl_path = demo_dir / 'tag_detection.pkl'
    if pkl_path.exists():
        dets = pickle.load(open(pkl_path, 'rb'))
        n_frames = len(dets)
        counts = {}
        widths_m = []
        for e in dets:
            td = e.get('tag_dict') or {}
            for tid in td:
                counts[tid] = counts.get(tid, 0) + 1
            if 0 in td and 1 in td:
                t0 = td[0]['tvec']
                t1 = td[1]['tvec']
                widths_m.append(float(np.linalg.norm(np.array(t0) - np.array(t1))))
        for tid in (0, 1, 13):
            out[f'tag{tid}_pct'] = 100.0 * counts.get(tid, 0) / max(n_frames, 1)
        if widths_m:
            out['width_min_m'] = float(np.min(widths_m))
            out['width_max_m'] = float(np.max(widths_m))
            out['width_n'] = len(widths_m)
        else:
            out['width_min_m'] = out['width_max_m'] = np.nan
            out['width_n'] = 0

    # ---- video duration ----
    imu_path = demo_dir / 'imu_data.json'
    if imu_path.exists():
        d = json.load(open(imu_path))
        try:
            samples = d['1']['streams']['GYRO']['samples']
            out['video_s'] = len(samples) / 200.0
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input_dir', required=True)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    root = Path(args.input_dir).expanduser().absolute()
    results = []
    for csv in sorted(root.glob('demo*/camera_trajectory.csv')):
        results.append(analyze_demo(csv.parent))

    hdr = ('demo', 'frames', 'lost%', 'jumps', 'x_range', 'max_jump_m',
           'tag0%', 'tag1%', 'tag13%', 'w_min', 'w_max', 'w_n')
    print(' | '.join(hdr))
    bad = []
    for r in results:
        if 'lost_pct' not in r:
            print(f"{r['demo']:10s} | SLAM MISSING")
            bad.append(r['demo'])
            continue
        xr = r.get('x_range', (np.nan, np.nan))
        print(
            f"{r['demo']:10s} | {r['frames']:5d} | {r['lost_pct']:5.1f} | "
            f"{r['jumps']:2d} | [{xr[0]:5.2f},{xr[1]:5.2f}] | "
            f"{r.get('max_jump_m', np.nan):5.3f} | "
            f"{r.get('tag0_pct', 0):5.1f} | {r.get('tag1_pct', 0):5.1f} | "
            f"{r.get('tag13_pct', 0):5.1f} | "
            f"{r.get('width_min_m', np.nan):5.3f} | {r.get('width_max_m', np.nan):5.3f} | "
            f"{r.get('width_n', 0):4d}")
        if r['lost_pct'] > 10 or r['jumps'] > 3 or r.get('tag0_pct', 0) < 90 \
                or r.get('tag1_pct', 0) < 90 or r.get('width_n', 0) < 50:
            bad.append(r['demo'])
    print()
    print(f"total demos: {len(results)}")
    print(f"flagged: {bad if bad else 'none'}")
    if args.verbose:
        for r in results:
            print(json.dumps(r, default=str))


if __name__ == '__main__':
    main()
