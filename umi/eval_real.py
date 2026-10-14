"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import select
import sys
import time
from collections import deque
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import (
    Spacemouse, DummySpacemouse, KeyboardSpacemouse)
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.real_world.policy_tcp_frame_adapter import (
    load_right_tcp_to_demo_tcp, right_obs_to_demo_obs,
    demo_actions_to_right_actions)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_eval_key_events(key_counter, poll_window=True):
    """Collect keys from pynput, the OpenCV window, and terminal stdin.

    Some Linux desktop sessions do not let pynput receive global keyboard
    events. The terminal fallback deliberately requires Enter, making it a
    dependable way to start (c), stop (s), or exit (x) a live evaluation.
    """
    events = key_counter.get_press_events()

    if poll_window:
        cv_key = cv2.pollKey()
        if cv_key >= 0:
            char = chr(cv_key & 0xFF).lower()
            if char.isprintable():
                events.append(KeyCode(char=char))

    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        readable = []
    if readable:
        for char in sys.stdin.readline().lower():
            if char.isprintable() and not char.isspace():
                events.append(KeyCode(char=char))
    return events


def hold_am2pro_at_current_pose(robots):
    """Cancel queued Cartesian motion and hold the measured arm/gripper pose."""
    states = [robot.get_state() for robot in robots]
    if any(state is None for state in states):
        raise RuntimeError('cannot hold position: AM2Pro state is unavailable')
    for robot, state in zip(robots, states):
        robot.servoL(np.asarray(state['ActualTCPPose'], dtype=np.float64),
                     duration=0.1)
        robot.schedule_gripper(float(state['gripper_position']),
                               target_time=time.time() + 0.1)


def move_am2pro_to_joint_reference(robot, reference_q, min_duration=8.0,
                                  max_joint_speed_deg_s=10.0,
                                  tolerance_deg=4.0,
                                  require_confirmation=True,
                                  reference_pose=None,
                                  tcp_position_tolerance_m=0.005,
                                  tcp_rotation_tolerance_deg=1.5):
    """Reach a saved right_tcp joint pose and verify the measured result."""
    state = robot.get_state()
    if state is None:
        raise RuntimeError('cannot preposition: AM2Pro state is unavailable')
    current_q = np.asarray(state['ActualQ'][:6], dtype=float)
    if current_q.shape != (6,) or not np.all(np.isfinite(current_q)):
        raise RuntimeError('cannot preposition: invalid current joint readback')
    max_delta = float(np.max(np.abs(reference_q - current_q)))
    duration = max(min_duration, max_delta / max_joint_speed_deg_s)
    print('REFERENCE PREPOSITION: current joints:', np.round(current_q, 1))
    print('REFERENCE PREPOSITION: target joints :', np.round(reference_q, 1))
    print(f'REFERENCE PREPOSITION: duration {duration:.1f}s; '
          f'maximum average joint speed {max_joint_speed_deg_s:.1f} deg/s')
    if require_confirmation and not click.confirm(
            '工作区和移动路径已清空，可以低速移动到保存的起始基准吗？', default=False):
        raise click.Abort()
    robot.servoJ(reference_q, duration=duration)
    motion_done_at = time.time() + duration
    deadline = motion_done_at + 5.0
    latest = state

    def reference_errors(measured):
        joints = np.asarray(measured.get('ActualQ', [])[:6], dtype=float)
        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            return np.inf, np.inf, np.inf
        joint_error = float(np.max(np.abs(joints - reference_q)))
        if reference_pose is None:
            return joint_error, 0.0, 0.0
        actual_pose = np.asarray(measured.get('ActualTCPPose', []), dtype=float)
        if actual_pose.shape != (6,) or not np.all(np.isfinite(actual_pose)):
            return joint_error, np.inf, np.inf
        position_error = float(np.linalg.norm(actual_pose[:3] - reference_pose[:3]))
        rotation_error = float(np.rad2deg(np.linalg.norm((
            st.Rotation.from_rotvec(actual_pose[3:]) *
            st.Rotation.from_rotvec(reference_pose[3:]).inv()
        ).as_rotvec())))
        return joint_error, position_error, rotation_error

    def reached(measured):
        joint_error, position_error, rotation_error = reference_errors(measured)
        return (joint_error <= tolerance_deg and
                position_error <= tcp_position_tolerance_m and
                rotation_error <= tcp_rotation_tolerance_deg)

    def wait_for_settle(motion_done_at, deadline, latest):
        while time.time() < deadline:
            time.sleep(0.1)
            candidate = robot.get_state()
            if candidate is None:
                continue
            latest = candidate
            if time.time() >= motion_done_at and reached(latest):
                break
        return latest

    latest = wait_for_settle(motion_done_at, deadline, latest)
    joint_error, position_error, rotation_error = reference_errors(latest)
    # The elbow can settle about one degree short under gravity even with its
    # verified P=36 setting.  Make one slow, bounded correction using measured
    # joint error, then check the same TCP tolerance before starting policy.
    if (not reached(latest) and np.isfinite(position_error) and
            joint_error <= 2.0 and position_error <= 0.020 and
            rotation_error <= 3.0):
        measured_q = np.asarray(latest['ActualQ'][:6], dtype=float)
        correction = np.clip(reference_q - measured_q, -1.5, 1.5)
        corrected_target = reference_q + correction
        correction_duration = max(3.0, float(np.max(np.abs(
            corrected_target - measured_q))) / max_joint_speed_deg_s)
        print('REFERENCE PREPOSITION: small joint correction:',
              np.round(correction, 2),
              f'over {correction_duration:.1f}s')
        robot.servoJ(corrected_target, duration=correction_duration)
        motion_done_at = time.time() + correction_duration
        latest = wait_for_settle(motion_done_at, motion_done_at + 5.0, latest)
    final_q = np.asarray(latest['ActualQ'][:6], dtype=float)
    final_error, final_position_error, final_rotation_error = reference_errors(latest)
    print('REFERENCE PREPOSITION: readback:', np.round(final_q, 1))
    if reference_pose is not None:
        print('REFERENCE PREPOSITION: TCP error {:.1f} mm / {:.2f} deg'.format(
            final_position_error * 1000, final_rotation_error))
    if (final_error > tolerance_deg or
            final_position_error > tcp_position_tolerance_m or
            final_rotation_error > tcp_rotation_tolerance_deg):
        raise RuntimeError('reference preposition did not converge: '
                           f'max joint error {final_error:.1f} deg, '
                           f'TCP error {final_position_error * 1000:.1f} mm / '
                           f'{final_rotation_error:.2f} deg')
    print(f'REFERENCE PREPOSITION reached; max joint error {final_error:.2f} deg.')


def move_am2pro_gripper_to_reference(robot, reference_width,
                                    max_speed_m_s=0.02,
                                    tolerance_m=0.003):
    """Settle the gripper at the saved start width before policy control."""
    state = robot.get_state()
    if state is None:
        raise RuntimeError('cannot preposition gripper: state is unavailable')
    current_width = float(state['gripper_position'])
    if not np.isfinite(current_width):
        raise RuntimeError('cannot preposition gripper: invalid width readback')
    print('REFERENCE GRIPPER: current {:.1f} mm; target {:.1f} mm'.format(
        current_width * 1000, reference_width * 1000))
    step_period = 0.1
    max_step = max_speed_m_s * step_period
    while abs(reference_width - current_width) > max_step:
        current_width += np.sign(reference_width - current_width) * max_step
        robot.schedule_gripper(float(current_width), target_time=time.time() + step_period)
        time.sleep(step_period)
    robot.schedule_gripper(float(reference_width), target_time=time.time() + step_period)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        time.sleep(step_period)
        state = robot.get_state()
        if state is None:
            continue
        measured_width = float(state['gripper_position'])
        if np.isfinite(measured_width) and abs(measured_width - reference_width) <= tolerance_m:
            print('REFERENCE GRIPPER reached; readback {:.1f} mm.'.format(
                measured_width * 1000))
            return
    if state is not None:
        measured_width = float(state['gripper_position'])
        if np.isfinite(measured_width):
            robot.schedule_gripper(measured_width, target_time=time.time() + step_period)
    raise RuntimeError('reference gripper did not converge to {:.1f} mm'.format(
        reference_width * 1000))


def stream_am2pro_policy_waypoints(env, poses, prediction_indices, dt,
                                  robot_action_latency, cycle_end, deadline,
                                  stop_requested):
    """Dispatch predicted waypoints until the next inference cycle starts.

    AM2Pro's schedule_waypoint resets its current interpolator on each call.
    Sending a future batch at once would retain only the final waypoint.
    """
    dispatch_start = time.time()
    sent_poses = []
    sent_timestamps = []
    sent_indices = []
    stop_reason = None
    for step, (pose, prediction_index) in enumerate(zip(poses, prediction_indices)):
        precise_wait(min(dispatch_start + step * dt, cycle_end, deadline),
                     time_func=time.time)
        if time.time() + dt >= deadline:
            stop_reason = 'max_duration'
            break
        if time.time() >= cycle_end:
            break
        if stop_requested():
            stop_reason = 'manual'
            break
        # The controller subtracts robot_action_latency in env.exec_actions.
        # Keep a full dt for each limited Cartesian step after that offset.
        target_time = time.time() + dt + robot_action_latency
        env.exec_actions(actions=np.asarray(pose)[None],
                         timestamps=np.asarray([target_time]),
                         compensate_latency=True)
        sent_poses.append(pose)
        sent_timestamps.append(target_time)
        sent_indices.append(prediction_index)
    return (np.asarray(sent_poses), np.asarray(sent_timestamps),
            sent_indices, stop_reason)


def queue_am2pro_policy_trajectory(env, poses, prediction_indices, timestamps,
                                  cycle_end, deadline, stop_requested):
    """Queue the full future horizon, then replace its remainder next cycle."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    keep = (timestamps > time.time() + 0.05) & (timestamps < deadline)
    poses = np.asarray(poses)[keep]
    indices = np.asarray(prediction_indices)[keep].tolist()
    timestamps = timestamps[keep]
    if not len(poses):
        hold_am2pro_at_current_pose(env.robots)
        if stop_requested():
            return poses, timestamps, indices, 'manual', time.time()
        print('No executable waypoint; holding and replanning next cycle.')
        return poses, timestamps, indices, None, time.time()
    poses, timestamps, kept_indices = env.exec_am2pro_policy_trajectory(
        actions=poses, timestamps=timestamps, compensate_latency=True)
    indices = [indices[int(i)] for i in kept_indices]
    queued_at = time.time()
    stop_reason = None
    while time.time() < min(cycle_end, deadline):
        if stop_requested():
            stop_reason = 'manual'
            break
        precise_wait(min(time.time() + 0.05, cycle_end, deadline),
                     time_func=time.time)
    if stop_reason is None and time.time() >= deadline:
        stop_reason = 'max_duration'
    return poses, timestamps, indices, stop_reason, queued_at

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            # AM2Pro has no sphere keys: skip pairs without a configured sphere
            this_r = robots_config[this_robot_idx].get('sphere_radius', 0) or 0
            that_r = robots_config[that_robot_idx].get('sphere_radius', 0) or 0
            if this_r <= 0 or that_r <= 0:
                continue
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


def limit_pose_sequence(target_poses, current_poses, max_pos_step, max_rot_step,
        current_gripper_widths=None, max_gripper_step=None):
    """Limit absolute Cartesian policy targets to a smooth, safe sequence.

    The policy predicts absolute poses.  For real AM2Pro evaluation we should
    not allow a visual mismatch to turn into a discontinuous Cartesian jump.
    Each waypoint is therefore limited relative to the preceding accepted pose.
    """
    limited = np.array(target_poses, dtype=np.float64, copy=True)
    n_robots = len(current_poses)
    previous = np.asarray(current_poses, dtype=np.float64).copy()
    previous_grippers = None
    if current_gripper_widths is not None:
        previous_grippers = np.asarray(current_gripper_widths, dtype=np.float64).copy()
        assert len(previous_grippers) == n_robots

    for waypoint in limited:
        for robot_idx in range(n_robots):
            offset = 7 * robot_idx
            target = waypoint[offset:offset + 6]
            reference = previous[robot_idx]

            delta_pos = target[:3] - reference[:3]
            delta_pos_norm = np.linalg.norm(delta_pos)
            if delta_pos_norm > max_pos_step:
                target[:3] = reference[:3] + delta_pos * (max_pos_step / delta_pos_norm)

            reference_rot = st.Rotation.from_rotvec(reference[3:])
            target_rot = st.Rotation.from_rotvec(target[3:])
            delta_rot = (target_rot * reference_rot.inv()).as_rotvec()
            delta_rot_norm = np.linalg.norm(delta_rot)
            if delta_rot_norm > max_rot_step:
                delta_rot *= max_rot_step / delta_rot_norm
                target[3:] = (st.Rotation.from_rotvec(delta_rot) * reference_rot).as_rotvec()

            previous[robot_idx] = target

            # The diffusion policy predicts an absolute gripper width.  Limit
            # it relative to the last accepted command as well, so a bad frame
            # cannot make the AM2Pro gripper snap open or shut in one cycle.
            if previous_grippers is not None and max_gripper_step is not None:
                gripper_idx = offset + 6
                target_gripper = np.clip(waypoint[gripper_idx], 0.0, 0.09)
                delta_gripper = target_gripper - previous_grippers[robot_idx]
                if abs(delta_gripper) > max_gripper_step:
                    target_gripper = previous_grippers[robot_idx] + np.sign(delta_gripper) * max_gripper_step
                waypoint[gripper_idx] = target_gripper
                previous_grippers[robot_idx] = target_gripper
    return limited


def smooth_policy_action_horizon(target_poses):
    """Reduce single-step reversals inside a predicted action chunk."""
    original = np.asarray(target_poses, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] % 7:
        raise ValueError('policy target poses must have shape (steps, 7 * robots)')
    smoothed = original.copy()
    if len(original) < 3:
        return smoothed
    for robot_idx in range(original.shape[1] // 7):
        offset = robot_idx * 7
        for column in (*range(offset, offset + 3), offset + 6):
            values = original[:, column]
            smoothed[1:-1, column] = (
                0.25 * values[:-2] + 0.5 * values[1:-1] +
                0.25 * values[2:])
            smoothed[0, column] = 0.75 * values[0] + 0.25 * values[1]
            smoothed[-1, column] = 0.75 * values[-1] + 0.25 * values[-2]
        rotations = st.Rotation.from_rotvec(original[:, offset + 3:offset + 6])
        for step in range(len(original)):
            if step == 0:
                indices, weights = [0, 1], [0.75, 0.25]
            elif step == len(original) - 1:
                indices, weights = [step - 1, step], [0.25, 0.75]
            else:
                indices, weights = [step - 1, step, step + 1], [0.25, 0.5, 0.25]
            smoothed[step, offset + 3:offset + 6] = (
                rotations[indices].mean(weights=weights).as_rotvec())
    return smoothed


def skip_passed_policy_waypoints(poses, timestamps, indices, current_pose,
                                 recent_displacement, tolerance_m=0.001,
                                 allow_return=False):
    """Skip a new horizon's points already behind a moving TCP.

    Only apply the phase guard when the overall policy path agrees with the
    measured motion. A genuine policy reversal must remain possible.
    """
    poses = np.asarray(poses)
    timestamps = np.asarray(timestamps)
    indices = list(indices)
    if len(poses) < 2:
        return poses, timestamps, indices, []
    motion = np.asarray(recent_displacement, dtype=float)[:3]
    path = poses[-1, :3] - poses[0, :3]
    motion_norm = np.linalg.norm(motion)
    path_norm = np.linalg.norm(path)
    if motion_norm < 0.001 or path_norm < 0.001 or np.dot(
            motion, path) < 0.5 * motion_norm * path_norm:
        return poses, timestamps, indices, []
    forward = motion / motion_norm
    progress = (poses[:, :3] - np.asarray(current_pose)[:3]) @ forward
    ahead = np.flatnonzero(progress >= -tolerance_m)
    if not len(ahead):
        if allow_return and np.linalg.norm(poses[-1, :3] - np.asarray(current_pose)[:3]) >= 0.005:
            return poses, timestamps, indices, []
        return poses[:0], timestamps[:0], [], indices
    first = int(ahead[0])
    return poses[first:], timestamps[first:], indices[first:], indices[:first]


def policy_direction_vote(poses, current_pose, motion_direction,
                          min_goal_distance_m=0.005):
    """Vote on the new horizon's goal relative to the recent robot motion."""
    poses = np.asarray(poses)
    motion = np.asarray(motion_direction, dtype=float)[:3]
    if not len(poses) or np.linalg.norm(motion) < 0.001:
        return 0
    goal = poses[-1, :3] - np.asarray(current_pose, dtype=float)[:3]
    goal_norm = np.linalg.norm(goal)
    if goal_norm < min_goal_distance_m:
        return 0
    cosine = float(np.dot(goal, motion) / (goal_norm * np.linalg.norm(motion)))
    return 1 if cosine >= 0.5 else -1 if cosine <= -0.5 else 0

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--num-inference-steps', '--num_inference_steps', default=None,
              type=click.IntRange(1, 100),
              help='DDIM denoising iterations per prediction; default uses the checkpoint setting.')
@click.option('--policy-sampling-seed', type=click.IntRange(0, 2147483647),
              default=None,
              help='Reuse this diffusion noise seed at every inference for consistent replanning.')
@click.option('--smooth-action-horizon', is_flag=True, default=False,
              help='Smooth adjacent predicted poses before speed limiting.')
@click.option('--max_duration', '-md', default=2000000, type=float,
              help='Maximum policy episode duration in seconds; 0 disables the time limit.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--max_pos_speed', default=0.06, type=float,
              help='AM2Pro Cartesian policy speed limit in m/s.')
@click.option('--max_rot_speed', default=0.5, type=float,
              help='AM2Pro Cartesian policy rotation limit in rad/s.')
@click.option('--max_gripper_speed', default=0.2, type=float,
              help='Maximum UMI gripper-width change in m/s.')
@click.option('--max_policy_waypoints', default=1, type=int,
              help='Legacy single-point limit; use --stream-until-next-inference for continuous AM2Pro execution.')
@click.option('--stream-until-next-inference', is_flag=True, default=False,
              help='AM2Pro: execute predicted points at control frequency until the next inference cycle, then replan.')
@click.option('--queue-policy-horizon', is_flag=True, default=False,
              help='AM2Pro: queue the full continuous prediction and replace its unexecuted suffix at the next inference.')
@click.option('--policy-execution-latency', default=0.15,
              type=click.FloatRange(min=0.0, max=0.5),
              help='AM2Pro queued mode: provisional arm tracking allowance in seconds; only future points beyond this margin are selected.')
@click.option('--policy-right-tcp-to-demo-tcp', type=click.Path(exists=True, dir_okay=False),
              default=None, help='Transform from robot right_tcp to the demonstration fixed-jaw tip.')
@click.option('--allow-candidate-policy-tcp-transform', is_flag=True, default=False,
              help='Explicitly permit the provisional TCP calibration for a supervised AM2Pro trial.')
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--no_spacemouse', is_flag=True, default=False,
              help="Run without a SpaceMouse: robot holds its current pose, press 'c' to start the policy.")
@click.option('--terminal-controls', is_flag=True, default=False,
              help='Use c/s/x followed by Enter in the terminal, without an OpenCV window.')
@click.option('--start-reference', type=click.Path(exists=True, dir_okay=False),
              default=None, help='Saved right_tcp reference.json for initial positioning.')
@click.option('--move-to-reference', is_flag=True, default=False,
              help='Move AM2Pro joints to --start-reference and verify readback before accepting c.')
@click.option('--auto-confirm-reference', is_flag=True, default=False,
              help='For a supervised single run, begin the reference move without a y prompt; still verify readback.')
@click.option('--preflight-only', is_flag=True, default=False,
              help='Initialize camera, robot, and policy; run one inference, then exit without actions.')
@click.option('--auto-start-policy', is_flag=True, default=False,
              help='Start one guarded, timed policy episode automatically, then exit (returns to start unless --hold-on-stop is set).')
@click.option('--save-policy-trace', is_flag=True, default=False,
              help='Save each prediction, submitted waypoint and controller readback for offline diagnosis.')
@click.option('--hold-on-stop', is_flag=True, default=False,
              help='AM2Pro: cancel queued motion and hold the measured pose after a policy episode.')
@click.option('--single-run-return-to-reference', is_flag=True, default=False,
              help='After confirmed preposition, start one policy episode; on manual stop or timeout, hold, return slowly to the saved joint reference, and exit.')
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, num_inference_steps, policy_sampling_seed,
    smooth_action_horizon, max_duration,
    frequency, max_pos_speed, max_rot_speed, max_gripper_speed,
    max_policy_waypoints, stream_until_next_inference, queue_policy_horizon,
    policy_execution_latency,
    policy_right_tcp_to_demo_tcp, allow_candidate_policy_tcp_transform,
    command_latency,
    no_mirror, sim_fov, camera_intrinsics, mirror_swap,
    no_spacemouse, terminal_controls, start_reference, move_to_reference,
    auto_confirm_reference,
    preflight_only, auto_start_policy, save_policy_trace,
    hold_on_stop, single_run_return_to_reference):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    if max_duration < 0:
        raise click.BadParameter('--max_duration must be nonnegative')
    if max_duration == 0:
        max_duration = float('inf')
    if max_gripper_speed <= 0:
        raise click.BadParameter('--max_gripper_speed must be positive')
    if max_policy_waypoints < 1:
        raise click.BadParameter('--max_policy_waypoints must be at least 1')
    if stream_until_next_inference and queue_policy_horizon:
        raise click.UsageError('choose either streaming or queued policy trajectory')
    if preflight_only and auto_start_policy:
        raise click.UsageError('--preflight-only and --auto-start-policy cannot be used together')
    if preflight_only and move_to_reference:
        raise click.UsageError('--preflight-only cannot move to a reference')
    if move_to_reference and start_reference is None:
        raise click.UsageError('--move-to-reference requires --start-reference')
    if auto_confirm_reference and not (move_to_reference and single_run_return_to_reference):
        raise click.UsageError(
            '--auto-confirm-reference requires --move-to-reference and '
            '--single-run-return-to-reference')
    if single_run_return_to_reference:
        if not (move_to_reference and hold_on_stop and
                (stream_until_next_inference or queue_policy_horizon)):
            raise click.UsageError(
                '--single-run-return-to-reference requires --move-to-reference, '
                '--hold-on-stop, and a continuous policy trajectory mode')
        if auto_start_policy or preflight_only:
            raise click.UsageError(
                '--single-run-return-to-reference cannot use --auto-start-policy '
                'or --preflight-only')
    if auto_start_policy:
        if stream_until_next_inference or queue_policy_horizon:
            raise click.UsageError('--auto-start-policy does not support continuous streaming')
        if max_duration > 5.0:
            raise click.UsageError('--auto-start-policy requires --max_duration <= 5 seconds')
        if max_pos_speed > 0.03 or max_rot_speed > 0.15:
            raise click.UsageError(
                '--auto-start-policy is guarded at --max_pos_speed <= 0.03 '
                'and --max_rot_speed <= 0.15')
        if max_gripper_speed > 0.02 or max_policy_waypoints != 1:
            raise click.UsageError(
                '--auto-start-policy requires --max_gripper_speed <= 0.02 '
                'and --max_policy_waypoints 1')

    # BimanualUmiEnv expects the parent to exist.  Make fresh evaluation
    # output locations usable without a separate shell setup command.
    output_path = pathlib.Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = str(output_path)
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    robots_config = robot_config_data['robots']
    right_to_demo = None
    if policy_right_tcp_to_demo_tcp is not None:
        if len(robots_config) != 1 or robots_config[0]['robot_type'] != 'am2pro':
            raise click.UsageError('policy TCP transform supports one AM2Pro arm')
        try:
            right_to_demo, transform_status = load_right_tcp_to_demo_tcp(
                policy_right_tcp_to_demo_tcp,
                allow_candidate=allow_candidate_policy_tcp_transform)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if transform_status == 'candidate' and not preflight_only and max_duration > 5.0:
            if (not single_run_return_to_reference or not move_to_reference or
                    not hold_on_stop or max_pos_speed > 0.03 or
                    max_rot_speed > 0.15 or max_gripper_speed > 0.02):
                raise click.UsageError(
                    'candidate TCP transform for a full trial requires '
                    '--single-run-return-to-reference, --move-to-reference, '
                    '--hold-on-stop and low policy speed limits')
        print('Policy TCP transform:', policy_right_tcp_to_demo_tcp, transform_status)
    elif allow_candidate_policy_tcp_transform:
        raise click.UsageError('--allow-candidate-policy-tcp-transform requires --policy-right-tcp-to-demo-tcp')
    if (stream_until_next_inference or queue_policy_horizon) and (
            len(robots_config) != 1 or robots_config[0]['robot_type'] != 'am2pro'):
        raise click.UsageError('continuous policy execution requires one AM2Pro arm')
    if (not (stream_until_next_inference or queue_policy_horizon) and
            max_policy_waypoints > 1 and
            len(robots_config) == 1 and robots_config[0]['robot_type'] == 'am2pro'):
        raise click.UsageError('AM2Pro needs a continuous policy mode for multiple waypoints')
    reference_q = None
    reference_pose = None
    reference_gripper_width = None
    if move_to_reference:
        if len(robots_config) != 1 or robots_config[0]['robot_type'] != 'am2pro':
            raise click.UsageError('--move-to-reference currently requires one AM2Pro arm')
        with open(start_reference, encoding='utf-8') as reference_file:
            reference_data = json.load(reference_file)
        if reference_data.get('tcp_frame') != 'right_tcp':
            raise click.UsageError('--start-reference must use right_tcp')
        reference_q = np.asarray(
            reference_data.get('robot_state', {}).get('ActualQ', [])[:6],
            dtype=float)
        if reference_q.shape != (6,) or not np.all(np.isfinite(reference_q)):
            raise click.UsageError('--start-reference lacks six valid joint values')
        reference_pose = np.asarray(
            reference_data.get('robot_state', {}).get('ActualTCPPose', []),
            dtype=float)
        if reference_pose.shape != (6,) or not np.all(np.isfinite(reference_pose)):
            raise click.UsageError('--start-reference lacks a valid 6D TCP pose')
        reference_gripper_width = float(
            reference_data.get('robot_state', {}).get('gripper_position', np.nan))
        if not np.isfinite(reference_gripper_width) or not 0 <= reference_gripper_width <= max_gripper_width:
            raise click.UsageError('--start-reference lacks a valid gripper width')
    if hold_on_stop and not all(
            rc['robot_type'].startswith('am2pro') for rc in robots_config):
        raise click.UsageError('--hold-on-stop currently requires AM2Pro robots')
    grippers_config = robot_config_data.get('grippers')
    if grippers_config is None:
        if all(rc['robot_type'].startswith('am2pro') for rc in robots_config):
            grippers_config = [{} for _ in robots_config]
        else:
            raise ValueError('robot config lacks grippers')
    # A single-arm AM2Pro setup has no left-to-right transform.
    if 'tx_left_right' in robot_config_data:
        tx_robot1_robot0 = np.array(robot_config_data['tx_left_right'])
    elif len(robots_config) == 1:
        tx_robot1_robot0 = np.eye(4)
    else:
        raise ValueError('multi-arm robot config lacks tx_left_right')

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    # Keep tensor storages memory-mapped.  The checkpoint also contains a
    # training optimizer, which is not needed for deployment and should not
    # be paged into RAM just to run evaluation.
    print('Opening checkpoint metadata ...')
    payload = torch.load(
        ckpt_path, map_location='cpu', pickle_module=dill, mmap=True)
    cfg = payload['cfg']
    if num_inference_steps is None:
        num_inference_steps = int(cfg.policy.num_inference_steps)
    # config layout varies between image/timm workspaces
    model_name = cfg.policy.obs_encoder.get('model_name', None)
    if model_name is None and 'rgb_model' in cfg.policy.obs_encoder:
        model_name = cfg.policy.obs_encoder.rgb_model.get('name', None)
    print("model_name:", model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    print("num_inference_steps:", num_inference_steps)
    print("policy_sampling_seed:", policy_sampling_seed)
    print("smooth_action_horizon:", smooth_action_horizon)
    camera_down_sample_steps = int(cfg.task.shape_meta.obs.camera0_rgb.down_sample_steps)
    robot_down_sample_steps = int(cfg.task.shape_meta.obs.robot0_eef_pos.down_sample_steps)
    gripper_down_sample_steps = int(cfg.task.shape_meta.obs.robot0_gripper_width.down_sample_steps)
    print("observation downsample steps: camera={}, robot={}, gripper={}".format(
        camera_down_sample_steps, robot_down_sample_steps, gripper_down_sample_steps))
    # Without a SpaceMouse, drive the arm with the keyboard (hold keys to
    # move, release to stop) so positioning can happen inside the eval window.
    if no_spacemouse:
        sm_class = KeyboardSpacemouse
        print("\n[Keyboard control] arrows=前后左右  W/S=上/下  Q/E=roll  A/D=yaw  "
              "N/M=夹爪关/开  C=交给策略  S=停  X键=退出\n"
              "若按键未被全局监听捕获，可在终端输入 c / s / x 后按回车。\n")
    else:
        sm_class = Spacemouse
    with SharedMemoryManager() as shm_manager:
        with sm_class(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                # eval_real displays the processed observation below with
                # cv2.imshow; the camera's separate visual shared-memory
                # stream is unnecessary and can stall this EMEET UVC camera.
                enable_multi_cam_vis=False,
                # latency
                camera_obs_latency=0.17,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                camera_down_sample_steps=camera_down_sample_steps,
                robot_down_sample_steps=robot_down_sample_steps,
                gripper_down_sample_steps=gripper_down_sample_steps,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # AM2Pro 150-degree wrist camera
                camera_match_name=robot_config_data.get('camera_match_name', None),
                camera_device_path=robot_config_data.get('camera_device_path', None),
                camera_settings=robot_config_data.get('camera_settings', None),
                camera_crop=robot_config_data.get('camera_crop', None),
                # action
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # Create only the network required for deployment.  Instantiating
            # the training workspace also creates a duplicate EMA model and a
            # large AdamW optimizer, which is unnecessary and can stall a
            # laptop before inference has even started.
            state_name = 'ema_model' if cfg.training.use_ema else 'model'
            print(f'Constructing deployment policy ({state_name}) ...')
            policy = hydra.utils.instantiate(cfg.policy)
            print('Restoring deployment model weights ...')
            policy.load_state_dict(payload['state_dicts'][state_name], strict=True)
            del payload
            print('Deployment model weights restored.')

            policy.num_inference_steps = num_inference_steps
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            print('Moving policy to CUDA ...')
            device = torch.device('cuda')
            policy.eval().to(device)
            print('Policy is on CUDA.')

            print("Warming up policy inference")
            print('Getting warmup observation ...')
            obs = env.get_obs()
            print('Warmup observation ready.')
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            print('Running warmup inference ...')
            policy_obs, policy_start_pose = (right_obs_to_demo_obs(
                obs, episode_start_pose, right_to_demo)
                if right_to_demo is not None else (obs, episode_start_pose))
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=policy_obs, shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=policy_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, policy_obs, action_pose_repr)
                if right_to_demo is not None:
                    action = demo_actions_to_right_actions(action, right_to_demo)
                assert action.shape[-1] == 7 * len(robots_config)
                del result
            torch.cuda.synchronize()
            print('Warmup inference complete.')

            if preflight_only:
                print('AM2PRO_FORMAL_PREFLIGHT_OK')
                print('No policy action was sent to the robot.')
                return

            if move_to_reference:
                move_am2pro_to_joint_reference(
                    env.robots[0], reference_q,
                    require_confirmation=not auto_confirm_reference,
                    reference_pose=reference_pose)
                move_am2pro_gripper_to_reference(
                    env.robots[0], reference_gripper_width)

            print('Ready!')
            auto_start_pending = auto_start_policy or single_run_return_to_reference
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                # AM2Pro: gripper state lives in robot ring buffer (gripper_position),
                # no separate gripper controller; fall back to robot state.
                gripper_target_pos = np.zeros(len(gripper_states), dtype=np.float64)
                for gs_idx, gs in enumerate(gripper_states):
                    if gs is not None:
                        gripper_target_pos[gs_idx] = float(gs['gripper_position'])
                    else:
                        # AM2Pro: gripper in meters is in the robot ring buffer
                        rs = robot_states[gs_idx]
                        gripper_target_pos[gs_idx] = float(rs.get('gripper_position', 0.0))
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # Do not enter the interactive observation/display loop
                    # for an explicitly requested automatic test.  This keeps
                    # the motion test independent of desktop key handling.
                    if auto_start_pending:
                        print('Auto-starting one policy episode ...')
                        auto_start_pending = False
                        break

                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    if not terminal_controls:
                        cv2.imshow('default', vis_img[...,::-1])
                    press_events = get_eval_key_events(
                        key_counter, poll_window=not terminal_controls)
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke in (Key.esc, KeyCode(char='x')):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif (key_stroke == KeyCode(char='m')
                              and match_replay_buffer is not None):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                if env.grippers[robot_idx] is not None:
                                    env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                else:
                                    # AM2Pro: gripper integrated in robot
                                    env.robots[robot_idx].schedule_gripper(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            # AM2Pro has no height_threshold key: -inf disables
                            height_threshold=robots_config[robot_idx].get(
                                'height_threshold', -np.inf))
                    
                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                episode_start_pose = None
                policy_motion_held = False
                policy_ended_normally = False
                trace_episode_id = env.replay_buffer.n_episodes
                trace_file = None
                if save_policy_trace:
                    trace_path = output_path.joinpath(
                        f'policy_trace_episode_{trace_episode_id:03d}.jsonl')
                    trace_file = trace_path.open('w', encoding='utf-8')
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    waypoint_mode = ('queued continuous horizon' if queue_policy_horizon
                                     else 'stream until next inference'
                                     if stream_until_next_inference
                                     else f'{max_policy_waypoints} waypoint/cycle')
                    print("Started! policy limiter: {:.3f} m/s, {:.3f} rad/s, {}".format(
                        max_pos_speed, max_rot_speed, waypoint_mode))
                    iter_idx = 0
                    perv_target_pose = None
                    direction_votes = deque(maxlen=3)
                    last_motion_direction = np.zeros(3)
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        policy_obs, policy_start_pose = (right_obs_to_demo_obs(
                            obs, episode_start_pose, right_to_demo)
                            if right_to_demo is not None else (obs, episode_start_pose))

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=policy_obs, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=policy_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            if policy_sampling_seed is None:
                                result = policy.predict_action(obs_dict)
                            else:
                                sampling_generator = torch.Generator(
                                    device=device).manual_seed(policy_sampling_seed)
                                result = policy.predict_action(
                                    obs_dict, generator=sampling_generator)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, policy_obs, action_pose_repr)
                            if right_to_demo is not None:
                                action = demo_actions_to_right_actions(action, right_to_demo)
                            print('Inference latency:', time.time() - s)
                            inference_seconds = time.time() - s
                        
                        # convert policy action to env actions
                        raw_target_poses = action.copy()
                        this_target_poses = (smooth_policy_action_horizon(action)
                                             if smooth_action_horizon else action)
                        smoothed_target_poses = this_target_poses.copy()
                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        current_poses = np.asarray(episode_start_pose, dtype=np.float64)
                        for robot_id in range(len(robots_config)):
                            current_poses[robot_id] = np.concatenate([
                                obs[f'robot{robot_id}_eef_pos'],
                                obs[f'robot{robot_id}_eef_rot_axis_angle']
                            ], axis=-1)[-1]
                        this_target_poses = limit_pose_sequence(
                            target_poses=this_target_poses,
                            current_poses=current_poses,
                            max_pos_step=max_pos_speed * dt,
                            max_rot_step=max_rot_speed * dt,
                            current_gripper_widths=np.array([
                                obs[f'robot{robot_id}_gripper_width'][-1, 0]
                                for robot_id in range(len(robots_config))
                            ], dtype=np.float64),
                            max_gripper_step=max_gripper_speed * dt)
                        limited_target_poses = this_target_poses.copy()
                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    # AM2Pro has no height_threshold key: -inf disables
                                    height_threshold=robots_config[robot_idx].get(
                                        'height_threshold', -np.inf)
                                )
                            
                            # solve collison between two robots
                            solve_sphere_collision(
                                ee_poses=target_pose.reshape([len(robots_config), -1]),
                                robots_config=robots_config
                            )

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = (policy_execution_latency
                                               if queue_policy_horizon else 0.01)
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            submitted_prediction_indices = [len(action) - 1]
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            submitted_prediction_indices = np.flatnonzero(is_new).tolist()
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        if not (stream_until_next_inference or queue_policy_horizon):
                            this_target_poses = this_target_poses[:max_policy_waypoints]
                            action_timestamps = action_timestamps[:max_policy_waypoints]
                            submitted_prediction_indices = submitted_prediction_indices[:max_policy_waypoints]

                        # Do not submit a new waypoint after a timed test has
                        # expired.  Checking only later in the loop could let
                        # one additional policy action through after timeout.
                        if time.time() - eval_t_start > max_duration:
                            print("Max Duration reached before action submission.")
                            if hold_on_stop:
                                hold_am2pro_at_current_pose(env.robots)
                                policy_motion_held = True
                            env.end_episode()
                            policy_ended_normally = True
                            break

                        stream_stop_reason = None
                        queue_sent_at = None
                        phase_skipped_indices = []
                        direction_vote = 0
                        return_confirmed = False
                        if stream_until_next_inference or queue_policy_horizon:
                            # Re-anchor the selected future chunk to fresh
                            # robot readback.  Earlier predicted points were
                            # skipped due to observation/inference latency.
                            latest_state = env.robots[0].get_state()
                            if latest_state is None:
                                raise RuntimeError('AM2Pro state unavailable before policy chunk')
                            if queue_policy_horizon:
                                # The observation-anchored limiter above is
                                # stale by the time this cycle is queued.
                                # Select future raw poses, then constrain them
                                # from the latest measured TCP instead.
                                this_target_poses = smoothed_target_poses[
                                    submitted_prediction_indices].copy()
                                recent_states = env.robots[0].get_state(k=6)
                                recent_displacement = np.zeros(3)
                                if (recent_states is not None and
                                        len(recent_states['ActualTCPPose']) >= 2):
                                    recent_displacement = (
                                        recent_states['ActualTCPPose'][-1, :3] -
                                        recent_states['ActualTCPPose'][0, :3])
                                if np.linalg.norm(recent_displacement) >= 0.001:
                                    last_motion_direction = recent_displacement
                                direction_vote = policy_direction_vote(
                                    this_target_poses, latest_state['ActualTCPPose'],
                                    last_motion_direction)
                                direction_votes.append(direction_vote)
                                return_confirmed = (
                                    direction_vote == -1 and
                                    len(direction_votes) == 3 and
                                    direction_votes.count(-1) >= 2)
                                (this_target_poses, action_timestamps,
                                 submitted_prediction_indices,
                                 phase_skipped_indices) = skip_passed_policy_waypoints(
                                    this_target_poses, action_timestamps,
                                    submitted_prediction_indices,
                                    latest_state['ActualTCPPose'],
                                    recent_displacement,
                                    allow_return=return_confirmed)
                                if direction_vote == -1 and not return_confirmed:
                                    # Wait for two of three fresh observations
                                    # to agree before reversing robot motion.
                                    phase_skipped_indices.extend(submitted_prediction_indices)
                                    this_target_poses = this_target_poses[:0]
                                    action_timestamps = action_timestamps[:0]
                                    submitted_prediction_indices = []
                            this_target_poses = limit_pose_sequence(
                                target_poses=this_target_poses,
                                current_poses=np.asarray(
                                    latest_state['ActualTCPPose'], dtype=np.float64)[None],
                                max_pos_step=max_pos_speed * dt,
                                max_rot_step=max_rot_speed * dt,
                                current_gripper_widths=np.asarray(
                                    [latest_state['gripper_position']], dtype=np.float64),
                                max_gripper_step=max_gripper_speed * dt)
                            if queue_policy_horizon:
                                for target_pose in this_target_poses:
                                    solve_table_collision(
                                        ee_pose=target_pose[:6],
                                        gripper_width=target_pose[6],
                                        height_threshold=robots_config[0].get(
                                            'height_threshold', -np.inf))

                            def stop_requested():
                                events = get_eval_key_events(
                                    key_counter, poll_window=not terminal_controls)
                                return (KeyCode(char='s') in events or
                                        KeyCode(char='x') in events)

                            cycle_end_wall = time.time() + (t_cycle_end - time.monotonic())
                            if queue_policy_horizon:
                                (this_target_poses, action_timestamps,
                                 submitted_prediction_indices,
                                 stream_stop_reason,
                                 queue_sent_at) = queue_am2pro_policy_trajectory(
                                    env=env, poses=this_target_poses,
                                    prediction_indices=submitted_prediction_indices,
                                    timestamps=action_timestamps,
                                    cycle_end=cycle_end_wall,
                                    deadline=eval_t_start + max_duration,
                                    stop_requested=stop_requested)
                            else:
                                (this_target_poses, action_timestamps,
                                 submitted_prediction_indices,
                                 stream_stop_reason) = stream_am2pro_policy_waypoints(
                                    env=env, poses=this_target_poses,
                                    prediction_indices=submitted_prediction_indices,
                                    dt=dt,
                                    robot_action_latency=robots_config[0].get(
                                        'robot_action_latency', 0.0),
                                    cycle_end=cycle_end_wall,
                                    deadline=eval_t_start + max_duration,
                                    stop_requested=stop_requested)
                        else:
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                        if trace_file is not None:
                            trace_record = {
                                'schema': 'am_umi_policy_cycle_trace_v1',
                                'cycle_index': iter_idx // steps_per_inference,
                                'wall_time': time.time(),
                                'num_inference_steps': num_inference_steps,
                                'policy_sampling_seed': policy_sampling_seed,
                                'policy_right_tcp_to_demo_tcp': policy_right_tcp_to_demo_tcp,
                                'policy_tcp_transform_status': (
                                    transform_status if right_to_demo is not None else None),
                                'smooth_action_horizon': smooth_action_horizon,
                                'queue_policy_horizon': queue_policy_horizon,
                                'queue_sent_at': queue_sent_at,
                                'policy_execution_latency_s': action_exec_latency,
                                'phase_skipped_indices': phase_skipped_indices,
                                'direction_vote': direction_vote,
                                'direction_votes': list(direction_votes),
                                'return_confirmed': return_confirmed,
                                'observation_timestamp': float(obs_timestamps[-1]),
                                'observation_timestamps': obs_timestamps.tolist(),
                                'inference_seconds': float(inference_seconds),
                                'current_pose_6d': current_poses.tolist(),
                                'current_gripper_width_m': [float(obs[f'robot{robot_id}_gripper_width'][-1, 0])
                                                            for robot_id in range(len(robots_config))],
                                'raw_target_poses_7d': raw_target_poses.tolist(),
                                'smoothed_target_poses_7d': smoothed_target_poses.tolist(),
                                'limited_target_poses_7d': limited_target_poses.tolist(),
                                'submitted_target_poses_7d': this_target_poses.tolist(),
                                'submitted_timestamps': action_timestamps.tolist(),
                                'submitted_prediction_indices': submitted_prediction_indices,
                            }
                            trace_file.write(json.dumps(trace_record, ensure_ascii=False) + '\n')
                            trace_file.flush()
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        if stream_stop_reason is not None:
                            print('Policy chunk stopped:', stream_stop_reason)
                            if hold_on_stop:
                                hold_am2pro_at_current_pose(env.robots)
                                policy_motion_held = True
                            env.end_episode()
                            policy_ended_normally = True
                            break

                        stop_episode = False
                        if not auto_start_policy:
                            # The regular interactive evaluator shows its
                            # camera frame and accepts the manual stop key.
                            # The automatic, timed smoke test deliberately
                            # skips GUI calls: they can block under some
                            # OpenCV/Linux desktop combinations.
                            episode_id = env.replay_buffer.n_episodes
                            obs_left_img = obs['camera0_rgb'][-1]
                            obs_right_img = obs['camera0_rgb'][-1]
                            vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            if not terminal_controls:
                                cv2.imshow('default', vis_img[...,::-1])

                            press_events = get_eval_key_events(
                                key_counter, poll_window=not terminal_controls)
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    # Stop episode
                                    # Hand control back to human
                                    print('Stopped.')
                                    stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            if hold_on_stop:
                                hold_am2pro_at_current_pose(env.robots)
                                policy_motion_held = True
                            env.end_episode()
                            policy_ended_normally = True
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    if hold_on_stop:
                        hold_am2pro_at_current_pose(env.robots)
                        policy_motion_held = True
                    env.end_episode()
                finally:
                    try:
                        if hold_on_stop and not policy_motion_held:
                            # A stopped or failed task must not execute a
                            # previously queued waypoint or return trajectory.
                            hold_am2pro_at_current_pose(env.robots)
                            policy_motion_held = True
                        if policy_motion_held:
                            print('Policy stopped; AM2Pro holding the measured pose and gripper width.')
                    finally:
                        if trace_file is not None:
                            trace_file.close()
                            readback_path = output_path.joinpath(
                                f'policy_robot_readback_episode_{trace_episode_id:03d}.npz')
                            np.savez_compressed(readback_path,
                                                **env.robots[0].get_all_state())
                            print('Policy trace:', trace_path)
                            print('Robot readback:', readback_path)
                
                print("Stopped.")

                if single_run_return_to_reference:
                    if policy_ended_normally:
                        print('Returning slowly to the saved v6 joint reference ...')
                        try:
                            move_am2pro_to_joint_reference(
                                env.robots[0], reference_q,
                                require_confirmation=False,
                                reference_pose=reference_pose)
                        except BaseException:
                            hold_am2pro_at_current_pose(env.robots)
                            raise
                        print('Returned to the saved v6 joint reference.')
                    else:
                        print('Policy interrupted by an exception or Ctrl+C; '
                              'holding position without automatic return.')
                    print('Single policy episode finished; exiting.')
                    break

                if not hold_on_stop and episode_start_pose is not None:
                    # The short empty-gripper diagnostic can still return to
                    # its start pose when --hold-on-stop is not requested.
                    print("Returning to start pose ...")
                    for robot_idx in range(len(robots_config)):
                        env.robots[robot_idx].servoL(
                            episode_start_pose[robot_idx], duration=2.0)
                    time.sleep(2.5)
                    print("Returned to start pose.")

                if auto_start_policy:
                    print('Automatic policy episode finished; exiting.')
                    break



# %%
if __name__ == '__main__':
    main()
