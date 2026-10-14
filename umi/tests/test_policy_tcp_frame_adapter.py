"""Offline frame checks for the V-jaw policy boundary."""

import json
import tempfile
import unittest

import numpy as np

from umi.common.pose_util import pose_to_mat
from umi.real_world.policy_tcp_frame_adapter import (
    demo_actions_to_right_actions, load_right_tcp_to_demo_tcp,
    right_obs_to_demo_obs)


class PolicyTcpFrameAdapterTest(unittest.TestCase):
    def test_relative_policy_action_matches_replay_conjugation(self):
        right_to_demo = pose_to_mat(np.array([-.022, -.011, -.004, 2.12, 2.21, .10]))
        start = np.array([.35, -.12, .21, .1, -.2, .4])
        obs = {
            'robot0_eef_pos': np.tile(start[:3], (2, 1)),
            'robot0_eef_rot_axis_angle': np.tile(start[3:], (2, 1)),
            'robot0_gripper_width': np.array([[.04], [.04]]),
        }
        policy_obs, policy_start = right_obs_to_demo_obs(obs, [start], right_to_demo)
        np.testing.assert_allclose(
            pose_to_mat(policy_start[0]), pose_to_mat(start) @ right_to_demo, atol=1e-12)
        np.testing.assert_allclose(obs['robot0_eef_pos'][0], start[:3])
        self.assertIs(policy_obs['robot0_gripper_width'], obs['robot0_gripper_width'])

        relative = pose_to_mat(np.array([.02, -.01, .03, .1, .02, -.03]))
        fixed_target = pose_to_mat(policy_start[0]) @ relative
        from umi.common.pose_util import mat_to_pose
        target = np.r_[mat_to_pose(fixed_target), .025][None]
        right_target = demo_actions_to_right_actions(target, right_to_demo)
        expected = pose_to_mat(start) @ right_to_demo @ relative @ np.linalg.inv(right_to_demo)
        np.testing.assert_allclose(pose_to_mat(right_target[0, :6]), expected, atol=1e-12)
        self.assertEqual(right_target[0, 6], .025)

    def test_candidate_requires_explicit_opt_in(self):
        data = {
            'schema': 'am_umi_tcp_frame_transform_v1', 'status': 'candidate',
            'parent_frame': 'right_tcp', 'child_frame': 'fixed_jaw_inner_front_tip',
            'pose_parent_child': [0, 0, 0, 0, 0, 0],
        }
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as stream:
            json.dump(data, stream)
            stream.flush()
            with self.assertRaisesRegex(ValueError, 'explicit opt-in'):
                load_right_tcp_to_demo_tcp(stream.name)
            matrix, status = load_right_tcp_to_demo_tcp(stream.name, allow_candidate=True)
            np.testing.assert_allclose(matrix, np.eye(4))
            self.assertEqual(status, 'candidate')


if __name__ == '__main__':
    unittest.main()
