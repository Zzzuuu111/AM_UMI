"""Diagnostic phase errors; never modifies trajectories or rescales their errors."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from compare_openvins_fixed_tag import align_rigid


GUIDES = {
    'vio': [(0,5,'静止'), (5,20,'左右平移'), (20,35,'上下平移'), (35,50,'前后平移'),
            (50,65,'点头'), (65,80,'摇头'), (80,95,'滚转'), (95,110,'混合'), (110,121,'静止尾部')],
    'rotation': [(0,3,'静止'), (3,15,'点头'), (15,27,'摇头'), (27,39,'滚转'), (39,46,'静止尾部')],
}


def main():
    p = argparse.ArgumentParser(description='按采集提示时间段审计 VIO 误差；提示不代表实际动作完全遵守')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--rotation-calibration', required=True)
    p.add_argument('--translation-calibration', required=True)
    p.add_argument('--guide', choices=GUIDES, required=True)
    args = p.parse_args()
    out = Path(args.run_dir) / 'motion_phase_audit.json'
    if out.exists():
        raise SystemExit(f'拒绝覆盖: {out}')
    st = np.loadtxt(Path(args.run_dir) / 'state_estimate.txt', ndmin=2)
    with Path(args.reference).open() as f:
        refs = [r for r in csv.DictReader(f) if r['is_lost']=='false']
    rt = np.array([float(r['timestamp']) for r in refs])
    et = st[:,0]-1700000000
    i = np.clip(np.searchsorted(rt, et),0,len(rt)-1)
    prev = np.maximum(i-1,0)
    i = np.where(abs(rt[prev]-et)<abs(rt[i]-et),prev,i)
    valid = abs(rt[i]-et)<.00002
    st, et, i = st[valid],et[valid],i[valid]
    rci = np.array(json.loads(Path(args.rotation_calibration).read_text())['R_camera_imu'])
    pci = np.array(json.loads(Path(args.translation_calibration).read_text())['r_camera_to_imu_in_camera_m'])
    pic = -rci.T @ pci
    rgi = Rotation.from_quat(st[:,1:5]).as_matrix()
    est = st[:,5:8] + np.einsum('nij,j->ni',rgi,pic)
    ref = np.array([[float(refs[k][a]) for a in ('x','y','z')] for k in i])
    r,t,_ = align_rigid(est,ref)
    error = np.linalg.norm(est @ r.T + t-ref,axis=1)
    phases = []
    for begin,end,label in GUIDES[args.guide]:
        take = (et>=begin)&(et<end)
        n = int(take.sum())
        row = dict(planned_phase=label, interval_s=[begin,end], samples=n)
        if n >= 20:
            x,y = est[take],ref[take]
            radius_x = float(np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=1))))
            radius_y = float(np.sqrt(np.mean(np.sum((y-y.mean(0))**2,axis=1))))
            scale = align_rigid(x,y)[2] if radius_x>1e-6 else None
            row.update(position_rmse_using_global_se3_m=float(np.sqrt(np.mean(error[take]**2))),
                       position_p95_using_global_se3_m=float(np.percentile(error[take],95)),
                       diagnostic_local_sim3_scale=scale,
                       estimated_position_rms_radius_m=radius_x, reference_position_rms_radius_m=radius_y,
                       accel_bias_median_mps2=np.median(st[take,14:17],axis=0).tolist())
        phases.append(row)
    report = dict(run_dir=args.run_dir, reference=args.reference, guide=args.guide, phases=phases,
                  note='时间段来自录制提示，不保证实际动作相符。位置误差始终用整段 SE3 对齐，未缩放。局部尺度仅诊断；静止/小位移段数值不稳定，不能用作标定。')
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
