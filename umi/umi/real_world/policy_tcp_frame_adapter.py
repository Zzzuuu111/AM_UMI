"""Coordinate boundary between fixed-jaw-tip policy data and robot right_tcp."""

import json

import numpy as np

from umi.common.pose_util import mat_to_pose, pose_to_mat


def load_right_tcp_to_demo_tcp(path, allow_candidate=False):
    with open(path, encoding='utf-8') as stream:
        data = json.load(stream)
    if (data.get('schema') != 'am_umi_tcp_frame_transform_v1' or
            data.get('parent_frame') != 'right_tcp' or
            data.get('child_frame') != 'fixed_jaw_inner_front_tip'):
        raise ValueError('policy TCP transform has unexpected schema or frames')
    status = data.get('status')
    if status != 'accepted' and not (status == 'candidate' and allow_candidate):
        raise ValueError('policy TCP transform is not accepted; candidate requires explicit opt-in')
    pose = np.asarray(data.get('pose_parent_child'), dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError('policy TCP transform must contain six finite pose values')
    return pose_to_mat(pose), status


def right_obs_to_demo_obs(env_obs, episode_start_pose, right_to_demo):
    """Copy one-arm observations and start pose into the policy's TCP frame."""
    pos = np.asarray(env_obs['robot0_eef_pos'])
    rot = np.asarray(env_obs['robot0_eef_rot_axis_angle'])
    fixed_poses = mat_to_pose(pose_to_mat(np.concatenate((pos, rot), axis=-1)) @ right_to_demo)
    fixed_obs = dict(env_obs)
    fixed_obs['robot0_eef_pos'] = fixed_poses[..., :3]
    fixed_obs['robot0_eef_rot_axis_angle'] = fixed_poses[..., 3:]
    fixed_start = [mat_to_pose(pose_to_mat(episode_start_pose[0]) @ right_to_demo)]
    return fixed_obs, fixed_start


def demo_actions_to_right_actions(demo_actions, right_to_demo):
    """Convert absolute fixed-tip targets to right_tcp; keep gripper width."""
    actions = np.asarray(demo_actions)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError('expected one-arm action horizon with 7 columns')
    right_actions = actions.copy()
    right_actions[:, :6] = mat_to_pose(
        pose_to_mat(actions[:, :6]) @ np.linalg.inv(right_to_demo))
    return right_actions
