"""
Move the AM2Pro arm to a pose / read its current pose, WITHOUT a SpaceMouse.

Used for initial positioning before running eval_real.py --no_spacemouse:
the eval script holds the pose the arm is already at, so use this tool first
to drive the arm to the task start pose.

Usage (umi env):
    # read current TCP pose (base frame, x,y,z + rx,ry,rz axis-angle)
    python scripts/am2pro_move_to.py -rc example/eval_robots_config.yaml --read

    # move to an absolute TCP pose over --duration seconds
    python scripts/am2pro_move_to.py -rc example/eval_robots_config.yaml \
        --pose 0.45,0.0,0.25,0,0,0 --duration 3

    # relative move (+2cm in z)
    python scripts/am2pro_move_to.py -rc example/eval_robots_config.yaml --move 0,0,0.02

    # set gripper width (0 = closed, 0.09 = open)
    python scripts/am2pro_move_to.py -rc example/eval_robots_config.yaml --gripper 0.03
"""
import time

import click
import numpy as np
import yaml

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.am2pro_interpolation_controller import (
    AM2ProInterpolationController)


@click.command()
@click.option('-rc', '--robot_config', required=True, help='eval_robots_config.yaml')
@click.option('--pose', default=None, help='Absolute TCP pose: x,y,z,rx,ry,rz')
@click.option('--move', default=None, help='Relative delta: x,y,z,rx,ry,rz')
@click.option('--gripper', default=None, type=float, help='Gripper width [0, 0.09]')
@click.option('--ik-backend', default=None, help="IK backend override: placo | ros2_dh")
@click.option('--duration', default=2.0, type=float, help='Move duration in seconds')
@click.option('--read', is_flag=True, default=False, help='Just print the current pose and exit')
def main(robot_config, pose, move, gripper, duration, read, ik_backend):
    cfg = yaml.safe_load(open(robot_config))
    rc = cfg['robots'][0]
    assert rc['robot_type'] == 'am2pro', f"robot[0] must be am2pro, got {rc['robot_type']}"

    with AM2ProInterpolationController(
        robot_usb_port=rc.get('robot_usb_port', '/dev/ttyACM0'),
        frequency=50,
        receive_latency=rc.get('robot_obs_latency', 0.005),
        server_python=rc.get('robot_python'),
        ik_backend=ik_backend or rc.get('ik_backend', 'ros2_dh'),
        urdf_path=rc.get('urdf_path'),
        flip_joints=rc.get('joint_flip', []),
        joint_model_candidate=rc.get('joint_model_candidate'),
        gripper_width_min=rc.get('gripper_width_min', 0.0),
        gripper_width_max=rc.get('gripper_width_max', 0.09),
        gripper_servo_closed=rc.get('gripper_servo_closed', 97.1),
        gripper_servo_open=rc.get('gripper_servo_open', 1.4),
        verbose=True,
    ) as robot:
        # the server's first state sample arrives a beat after startup;
        # poll briefly instead of assuming it is already available
        state = None
        for _ in range(100):
            state = robot.get_state()
            if state is not None:
                break
            time.sleep(0.05)
        if state is None:
            raise SystemExit("No state received from the robot server within 5s. "
                             "Check /dev/ttyACM0 and rerun.")

        cur_pose = state['ActualTCPPose']
        print(f"current TCP pose : {np.round(cur_pose, 4)}")
        print(f"current joints    : {np.round(state['ActualQ'][:6], 2)}")

        if read:
            return

        if pose is not None:
            target = np.array([float(x) for x in pose.split(',')])
            if target.shape == (3,):
                target = np.concatenate([target, np.zeros(3)])
            assert target.shape == (6,), "pose must be x,y,z[,rx,ry,rz]"
        elif move is not None:
            delta = np.array([float(x) for x in move.split(',')])
            if delta.shape == (3,):
                delta = np.concatenate([delta, np.zeros(3)])
            assert delta.shape == (6,), "move must be x,y,z[,rx,ry,rz]"
            target = cur_pose + delta
        else:
            raise SystemExit("Need one of --pose / --move / --read")

        if gripper is not None:
            robot.schedule_gripper(float(gripper), target_time=time.time() + duration)
        print(f"moving to        : {np.round(target, 4)} over {duration}s ...")
        # send the move FIRST, then record the ACTUAL trajectory (10 Hz)
        robot.servoL(target, duration=duration)
        samples = []
        t0 = time.time()
        while time.time() - t0 < duration + 1.0:
            s = robot.get_state()
            if s is not None:
                samples.append(np.concatenate([
                    s['ActualTCPPose'][:6], s['ActualQ'][:6]]))
            time.sleep(0.1)

        traj = np.array(samples)
        print(f"\n--- 实际轨迹({len(traj)} 个采样点,含命令发出前 ---")
        if len(traj) >= 3:
            a, b = traj[0, :3], traj[-1, :3]
            line_dir = (b - a)
            n = np.linalg.norm(line_dir)
            line_dir = line_dir / max(n, 1e-9)
            perp = np.linalg.norm(np.cross(traj[:, :3] - a, line_dir), axis=1) * 1000
            print(f"起点 {np.round(a,3)} → 终点 {np.round(b,3)}")
            print(f"最大横向偏离(垂直直线) = {perp.max():.1f} mm   "
                  f"xy 平面最大偏离 = {np.abs(traj[:,:2]-(a[:2]+np.outer(np.linspace(0,1,len(traj)),(b-a)[:2]))).sum(axis=1).max()*1000:.1f} mm")
            print(f"x 范围 [{traj[:,0].min():.3f}, {traj[:,0].max():.3f}]  "
                  f"y 范围 [{traj[:,1].min():.3f}, {traj[:,1].max():.3f}]  "
                  f"z 范围 [{traj[:,2].min():.3f}, {traj[:,2].max():.3f}]")
            print(f"z 采样轨迹: {np.round(traj[::max(1,len(traj)//15), 2], 3)}")
            j0, j1 = traj[0, 6:], traj[-1, 6:]
            print(f"关节角: 起点 {np.round(j0,1)}")
            print(f"        终点 {np.round(j1,1)}")
            print(f"        增量 {np.round(j1-j0,1)}")
            print(f"        各关节活动范围: {np.round(traj[:,6:].max(axis=0)-traj[:,6:].min(axis=0),1)}")
        time.sleep(duration + 1.0)

        state = robot.get_state()
        actual = state['ActualTCPPose']
        err_pos = np.linalg.norm(actual[:3] - target[:3]) * 1000
        print(f"final TCP pose   : {np.round(actual, 4)}")
        print(f"final joints     : {np.round(state['ActualQ'][:6], 2)}")
        print(f"position error   : {err_pos:.1f} mm  (per-axis mm: "
              f"{(np.abs(actual[:3] - target[:3]) * 1000).round(1)})")
        rot_err = np.abs(actual[3:] - target[3:]).round(3)
        print(f"rotation error   : {rot_err}")


if __name__ == "__main__":
    main()
