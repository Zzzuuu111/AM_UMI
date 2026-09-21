"""Independent diagnostic comparison: OpenVINS IMU state vs fixed-Tag camera pose.

Fits one rigid SE(3) transform, never rescales the reported metric error.
Tag PnP shares camera intrinsics, so it is a reference, not metrology ground truth.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def align_rigid(estimated, reference):
    x, y = estimated - estimated.mean(0), reference - reference.mean(0)
    u, singular, vt = np.linalg.svd(x.T @ y)
    fix = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ fix @ u.T
    translation = reference.mean(0) - rotation @ estimated.mean(0)
    scale_diagnostic = float(np.sum(singular * np.diag(fix)) / np.sum(x * x))
    return rotation, translation, scale_diagnostic


def main():
    p = argparse.ArgumentParser(description='用固定 Tag 参考轨迹检查 OpenVINS；只做刚体对齐，不调整尺度')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--rotation-calibration', required=True)
    p.add_argument('--translation-calibration', required=True)
    args = p.parse_args()
    run = Path(args.run_dir)
    out = run / 'fixed_tag_comparison.json'
    if out.exists():
        raise SystemExit(f'拒绝覆盖: {out}')
    states = np.loadtxt(run / 'state_estimate.txt', comments='#', ndmin=2)
    if len(states) < 3 or not np.isfinite(states).all():
        raise SystemExit('有效状态不足')
    with Path(args.reference).open() as f:
        all_rows = list(csv.DictReader(f))
    ref = [r for r in all_rows if r['is_lost'] == 'false']
    ref_t = np.array([float(r['timestamp']) for r in ref])
    est_t = states[:, 0] - 1_700_000_000
    all_camera_t = np.array([float(r['timestamp']) for r in all_rows])
    reference_duration_s = float(all_camera_t[-1] - all_camera_t[0])
    indices = np.clip(np.searchsorted(ref_t, est_t), 0, len(ref_t)-1)
    previous = np.maximum(indices - 1, 0)
    indices = np.where(abs(ref_t[previous]-est_t) < abs(ref_t[indices]-est_t), previous, indices)
    keep = abs(ref_t[indices]-est_t) < 0.00002
    states, indices = states[keep], indices[keep]
    if len(states) < 20:
        raise SystemExit('匹配时间戳不足；检查源时钟与导出数据')
    rci = np.array(json.loads(Path(args.rotation_calibration).read_text())['R_camera_imu'])
    pci = np.array(json.loads(Path(args.translation_calibration).read_text())['r_camera_to_imu_in_camera_m'])
    pic = -rci.T @ pci
    # OpenVINS saves JPL q_GtoI; scipy's Hamilton matrix with the SAME xyzw
    # components equals R_ItoG (transpose of OpenVINS quat_2_Rot).
    rgi = Rotation.from_quat(states[:, 1:5]).as_matrix()
    est_pos = states[:, 5:8] + np.einsum('nij,j->ni', rgi, pic)
    est_rot = rgi @ rci.T
    ref_pos = np.array([[float(ref[i][k]) for k in ('x','y','z')] for i in indices])
    ref_rot = Rotation.from_quat([[float(ref[i][k]) for k in ('q_x','q_y','q_z','q_w')] for i in indices]).as_matrix()
    r, t, scale = align_rigid(est_pos, ref_pos)
    error = np.linalg.norm(est_pos @ r.T + t - ref_pos, axis=1)
    angle = Rotation.from_matrix((r @ est_rot) @ np.transpose(ref_rot, (0,2,1))).magnitude() * 180 / np.pi
    report = dict(matched_states=len(states), reference='fixed Tag PnP; shares camera intrinsics, not independent metrology',
                  reference_duration_s=reference_duration_s,
                  alignment='one SE3, no scale correction',
                  position_rmse_m=float(np.sqrt(np.mean(error**2))), position_p95_m=float(np.percentile(error,95)),
                  position_max_m=float(error.max()), rotation_median_deg=float(np.median(angle)),
                  rotation_p95_deg=float(np.percentile(angle,95)),
                  diagnostic_sim3_scale_to_reference=scale,
                  matched_start_s=float(states[0,0]-1_700_000_000), matched_end_s=float(states[-1,0]-1_700_000_000),
                  estimated_camera_span_m=np.ptp(est_pos,axis=0).tolist(), reference_camera_span_m=np.ptp(ref_pos,axis=0).tolist(),
                  note='只报告误差，不以算法正常退出代替精度验收。')
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
