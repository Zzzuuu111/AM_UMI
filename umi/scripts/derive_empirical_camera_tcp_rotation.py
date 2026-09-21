#!/usr/bin/env python3
"""Derive a provisional camera->TCP rotation from supervised axis probes.

It combines: (1) fixed-Tag hand-held motion directions in world coordinates,
(2) the robot's reference TCP orientation, and (3) observed robot directions.
Only an offline candidate geometry JSON is written; no hardware is accessed.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umi.common.pose_util import pose_to_mat  # noqa: E402

STAGES = {"forward": (5., 11.), "up": (11., 17.), "left": (17., 23.)}

def unit(x, name):
    x = np.asarray(x, dtype=float); n = np.linalg.norm(x)
    if n < 1e-8: raise ValueError(f"{name} 为零向量")
    return x / n

def main():
    p = argparse.ArgumentParser(description="由实体轴向验证离线推导 provisional camera-TCP rotation。")
    p.add_argument("--axis-trajectory", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--accepted-translation-geometry", required=True)
    p.add_argument("--empirical-axis-map", required=True)
    p.add_argument("--fit-axes", default="forward,left",
                   help="参与刚体旋转拟合的语义轴，默认 forward,left；可选 forward,down,left")
    p.add_argument("--out", required=True)
    a = p.parse_args(); out = Path(a.out).expanduser().resolve()
    if out.exists(): p.error(f"refusing to overwrite existing output: {out}")
    rows=[]
    with Path(a.axis_trajectory).expanduser().open() as f:
        for r in csv.DictReader(f):
            if r['is_lost'].lower() != 'true':
                rows.append({k: float(r[k]) for k in ('timestamp','x','y','z','q_x','q_y','q_z','q_w')})
    base=[r for r in rows if 1 <= r['timestamp'] <= 4]
    if len(base)<5: raise ValueError('基准静止段不足')
    p0=np.median([[r['x'],r['y'],r['z']] for r in base],axis=0)
    r0=Rotation.from_quat([[r['q_x'],r['q_y'],r['q_z'],r['q_w']] for r in base]).mean().as_matrix()
    measured={}
    for name,(start,end) in STAGES.items():
        samples=np.asarray([[r['x'],r['y'],r['z']] for r in rows if start<=r['timestamp']<=end])
        delta=samples-p0; measured[name]=unit(delta[np.argmax(np.linalg.norm(delta,axis=1))],name)
    reference=json.loads(Path(a.reference).expanduser().read_text())
    R_base_tcp=pose_to_mat(np.asarray(reference['robot_state']['ActualTCPPose'],float))[:3,:3]
    direction_map=json.loads(Path(a.empirical_axis_map).expanduser().read_text())['semantic_mapping']
    # The probe established: old source forward physically moved up; old source
    # up physically moved down.  The semantic hand-held down source is -up.
    b_forward=unit(direction_map['handheld_forward']['unit_tcp_translation_base'],'base forward')
    b_down=unit(direction_map['handheld_down']['unit_tcp_translation_base'],'base down')
    b_left=unit(direction_map['handheld_left']['unit_tcp_translation_base'],'base left')
    source_by_axis={'forward':R_base_tcp.T@b_forward, 'down':R_base_tcp.T@b_down,
                    'left':R_base_tcp.T@b_left}
    target_by_axis={'forward':r0.T@measured['forward'], 'down':r0.T@(-measured['up']),
                    'left':r0.T@measured['left']}
    fit_axes=tuple(x.strip() for x in a.fit_axes.split(',') if x.strip())
    if len(fit_axes) < 2 or any(x not in source_by_axis for x in fit_axes):
        p.error("--fit-axes 必须是 forward/down/left 中至少两个，例如 forward,left")
    sources=np.stack([source_by_axis[name] for name in fit_axes])
    targets=np.stack([target_by_axis[name] for name in fit_axes])
    correction, rms = Rotation.align_vectors(targets, sources)
    per=[]
    for name, x in source_by_axis.items():
        y = target_by_axis[name]
        per.append({'axis':name, 'used_for_fit':name in fit_axes,
                    'residual_deg':float(np.degrees(np.arccos(np.clip(correction.apply(x)@y,-1,1))))})
    trans=json.loads(Path(a.accepted_translation_geometry).expanduser().read_text())
    pose=np.asarray(trans['pose_cam_tcp'],float); pose[3:]=correction.as_rotvec()
    data={
      'schema':'am_umi_camera_tcp_geometry_v1','status':'candidate',
      'transform_direction':'T_camera_tcp; pose_cam_tcp is translation[m] + rotation-vector[rad]',
      'pose_cam_tcp':pose.tolist(),'translation_method':'accepted fixed-point pivot translation from v2',
      'rotation_method':'empirical supervised robot axis probes; Kabsch fit of selected direction correspondences',
      'fit_axes':list(fit_axes),
      'source_translation_geometry':str(Path(a.accepted_translation_geometry).expanduser().resolve()),
      'source_axis_trajectory':str(Path(a.axis_trajectory).expanduser().resolve()),
      'source_empirical_axis_map':str(Path(a.empirical_axis_map).expanduser().resolve()),
      'fit_rms_vector_error':float(rms),'per_axis_residual_deg':per,
      'note':'Provisional diagnostic candidate only. Must pass offline replayability and supervised physical dry-run before acceptance.'
    }
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print('EMPIRICAL_CAMERA_TCP_ROTATION_CANDIDATE_OK')
    print('rotation rotvec rad:',pose[3:].tolist())
    print('per-axis residual deg:',[(x['axis'],round(x['residual_deg'],2)) for x in per])
    print('saved:',out)
if __name__=='__main__': main()
