#!/usr/bin/env python3
"""Collect held AM2Pro hand-eye samples at safe, preplanned poses.

This is for a heavy wrist tool: the controller holds each pose while the
camera records a stable Tag frame and the *actual* joint state is saved.
No gripper command is sent.  Robot motion requires explicit --execute.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import av
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umi.common.cv_util import detect_localize_aruco_tags, parse_aruco_config, parse_fisheye_intrinsics  # noqa: E402
from umi.common.pose_util import mat_to_pose  # noqa: E402
from umi.real_world.am2pro_interpolation_controller import AM2ProInterpolationController  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402

URDF_JOINTS = ("right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
               "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll")

def state(robot):
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        value=robot.get_state()
        if value is not None: return value
        time.sleep(.05)
    raise RuntimeError("控制器未返回关节状态")

def settle(robot, target, duration, tolerance):
    print("  目标关节角:", np.round(target,1), flush=True)
    robot.servoJ(target, duration=duration)
    until=time.monotonic()+duration
    latest=state(robot)
    while time.monotonic()<until:
        time.sleep(.1); latest=robot.get_state() or latest
    actual=np.asarray(latest['ActualQ'][:6],float); error=actual-target
    print("  实际关节角:",np.round(actual,1),"误差:",np.round(error,2),flush=True)
    return latest, float(np.max(np.abs(error))) <= tolerance

def frame_after(frames, count=20):
    frame=None
    for _ in range(count): frame=next(frames)
    return frame.to_ndarray(format='rgb24')

def main():
    p=argparse.ArgumentParser(description='AM2Pro 通扭矩保持的自动手眼采样；加 --execute 才会移动。')
    p.add_argument('--output',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--axis-plan',required=True,help='含 forward/up/left 的已离线通过轴向计划')
    p.add_argument('--robot-config',default='example/eval_robots_config.yaml')
    p.add_argument('--camera-device',default='/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0')
    p.add_argument('--intrinsics',default='calibration/robot_wrist_camera/intrinsics/emeet_wrist_1920x1080_30fps_fisheye_v1.json')
    p.add_argument('--aruco-yaml',default='calibration/shared_tags/aruco_config.yaml')
    p.add_argument('--tag-id',type=int,default=13)
    p.add_argument('--duration-s',type=float,default=10.)
    p.add_argument('--hold-s',type=float,default=2.)
    p.add_argument('--prepare-wait-s',type=float,default=8.)
    p.add_argument('--tolerance-deg',type=float,default=3.)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    out=Path(a.output).expanduser().resolve()
    if out.exists(): p.error(f'拒绝覆盖已有输出: {out}')
    if min(a.duration_s,a.hold_s,a.prepare_wait_s,a.tolerance_deg)<=0: p.error('时间和容差必须为正')
    ref=json.loads(Path(a.reference).expanduser().read_text())
    if ref.get('tcp_frame')!='right_tcp': p.error('reference 必须是 right_tcp')
    q_ref=np.asarray(ref['robot_state']['ActualQ'][:6],float)
    plan=json.loads(Path(a.axis_plan).expanduser().read_text())
    axes=plan.get('axes',{})
    for name in ('forward','up','left'):
        if not axes.get(name,{}).get('accepted_for_low_speed_physical_test'): p.error(f'轴 {name} 未获离线批准')
    # Three Cartesian test endpoints plus modest wrist orientation changes.
    targets=[('V9 基准位',q_ref),
             ('轴向姿态：forward',np.asarray(axes['forward']['planned_joints_deg'],float)),
             ('轴向姿态：down（旧字段 up）',np.asarray(axes['up']['planned_joints_deg'],float)),
             ('轴向姿态：left',np.asarray(axes['left']['planned_joints_deg'],float))]
    for label,yaw,roll in [('腕部偏航 +12°',12.,0.),('腕部偏航 -12°',-12.,0.),('腕部滚转 +15°',0.,15.)]:
        q=q_ref.copy(); q[4]+=yaw; q[5]+=roll; targets.append((label,q))
    print('AM2PRO_TORQUE_HELD_HAND_EYE_COLLECTION')
    print('动作总览：回 V9 基准位后等待 %.0f 秒；依次在 7 个低速姿态停住并自动检查 Tag %d。'%(a.prepare_wait_s,a.tag_id))
    print('不会发送夹爪开合命令。若任何姿态接近桌面或 Tag 不可见，请 Ctrl+C；程序会回 V9 基准位。')
    if not a.execute:
        print('仅预览：加 --execute 才会通扭矩和移动机械臂。')
        for i,(label,q) in enumerate(targets,1): print(f'  {i}. {label}:',np.round(q,1))
        return
    cfg=yaml.safe_load(Path(a.robot_config).read_text())['robots'][0]
    intr=parse_fisheye_intrinsics(json.loads(Path(a.intrinsics).read_text()))
    aruco=parse_aruco_config(yaml.safe_load(Path(a.aruco_yaml).read_text()))
    kin=create_kinematics_backend('ros2_dh',str(ROOT/'alohamini2pro_right_arm_kinematics.urdf'),URDF_JOINTS,'right_Fixed_Jaw')
    samples=[]; camera=None
    with AM2ProInterpolationController(robot_usb_port=cfg.get('robot_usb_port','/dev/ttyACM0'),frequency=50,
        receive_latency=cfg.get('robot_obs_latency',.005),server_python=cfg.get('robot_python'),ik_backend=cfg.get('ik_backend','ros2_dh'),
        flip_joints=cfg.get('joint_flip',[]),gripper_width_min=cfg.get('gripper_width_min',0.),gripper_width_max=cfg.get('gripper_width_max',.074),
        gripper_servo_closed=cfg.get('gripper_servo_closed',5.3),gripper_servo_open=cfg.get('gripper_servo_open',93.6),verbose=True) as robot:
      try:
        print('[1/7] 正在回 V9 基准位…',flush=True); _,ok=settle(robot,q_ref,a.duration_s,a.tolerance_deg)
        if not ok: raise RuntimeError('V9 基准位未收敛，未开始采集')
        print(f'已到基准位；请观察，{a.prepare_wait_s:.0f} 秒后开始。',flush=True); time.sleep(a.prepare_wait_s)
        camera=av.open(str(Path(a.camera_device).resolve()),format='v4l2',options={'input_format':'mjpeg','video_size':'1920x1080','framerate':'30'})
        frames=camera.decode(video=0); _=frame_after(frames)
        for index,(label,target) in enumerate(targets,1):
            print(f'[{index}/{len(targets)}] {label}',flush=True)
            latest,ok=settle(robot,target,a.duration_s,a.tolerance_deg)
            if not ok: print('  跳过：关节未收敛。',flush=True); continue
            time.sleep(a.hold_s)
            image=frame_after(frames)
            tags=detect_localize_aruco_tags(image,aruco['aruco_dict'],aruco['marker_size_map'],intr)
            if a.tag_id not in tags: print(f'  跳过：未检测到 Tag {a.tag_id}。',flush=True); continue
            latest=state(robot); q=np.asarray(latest['ActualQ'][:6],float)
            samples.append({'img':image,'tcp_pose':mat_to_pose(kin.forward_kinematics(q)),
                            'joint_deg':np.r_[q[:5],-q[5]],'capture_monotonic_s':time.monotonic(),'tag_id':a.tag_id,'label':label})
            pickle.dump(samples,out.open('wb')); print(f'  已保存样本 {len(samples)}。',flush=True)
      except KeyboardInterrupt: print('采集被中断。',flush=True)
      finally:
        if camera is not None: camera.close()
        print('正在安全返回 V9 基准位…',flush=True); robot.servoJ(q_ref,duration=a.duration_s); time.sleep(a.duration_s+.2)
    if len(samples)<5: raise RuntimeError(f'仅保存 {len(samples)} 个有效样本；至少需要 5 个，不能用于标定')
    print('TORQUE_HELD_HAND_EYE_COLLECTION_OK'); print('samples:',len(samples)); print('output:',out)
if __name__=='__main__': main()
