"""Offline checks for replaceable AM2Pro policy trajectories (no devices)."""

import unittest
import time
from collections import deque

import numpy as np

from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.real_world.am2pro_controller_server import schedule_policy_waypoint
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from eval_real import (skip_passed_policy_waypoints,
                       queue_am2pro_policy_trajectory, policy_direction_vote)
from unittest.mock import patch


class PolicyTrajectoryTest(unittest.TestCase):
    def test_three_fresh_horizons_use_two_of_three_direction_vote(self):
        votes = deque(maxlen=3)
        for goal, expected in ((0.02, 1), (-0.02, -1), (-0.03, -1)):
            poses = np.zeros((2, 7))
            poses[:, 0] = goal
            vote = policy_direction_vote(poses, np.zeros(6), [0.003, 0, 0])
            votes.append(vote)
            self.assertEqual(vote, expected)
        self.assertEqual(list(votes), [1, -1, -1])
        self.assertGreaterEqual(votes.count(-1), 2)
        poses[:, 0] = 0.002
        self.assertEqual(policy_direction_vote(poses, np.zeros(6), [0.003, 0, 0]), 0)

    def test_skip_passed_points_only_when_policy_continues_forward(self):
        poses = np.zeros((8, 7))
        poses[:, 0] = np.arange(8) * 0.002
        times = np.arange(8) * 0.1
        current = np.array([0.007, 0, 0, 0, 0, 0])
        selected, selected_times, indices, skipped = skip_passed_policy_waypoints(
            poses, times, list(range(8)), current, [0.003, 0, 0])
        self.assertEqual(indices, [3, 4, 5, 6, 7])
        self.assertEqual(skipped, [0, 1, 2])
        np.testing.assert_array_equal(selected[:, 0], poses[3:, 0])
        np.testing.assert_array_equal(selected_times, times[3:])
        reverse, _, reverse_indices, reverse_skipped = skip_passed_policy_waypoints(
            poses[::-1], times, list(range(8)), current, [0.003, 0, 0])
        np.testing.assert_array_equal(reverse, poses[::-1])
        self.assertEqual(reverse_indices, list(range(8)))
        self.assertEqual(reverse_skipped, [])

    def test_skip_passed_points_returns_empty_when_whole_horizon_is_behind(self):
        poses = np.zeros((3, 7))
        poses[:, 0] = [0.001, 0.002, 0.003]
        selected, _, indices, skipped = skip_passed_policy_waypoints(
            poses, np.arange(3) * 0.1, [5, 6, 7],
            [0.010, 0, 0, 0, 0, 0], [0.003, 0, 0])
        self.assertEqual(len(selected), 0)
        self.assertEqual(indices, [])
        self.assertEqual(skipped, [5, 6, 7])

    def test_consistent_return_can_use_horizon_behind_current_pose(self):
        poses = np.zeros((3, 7))
        poses[:, 0] = [0.001, 0.003, 0.005]
        current = [0.012, 0, 0, 0, 0, 0]
        selected, _, indices, skipped = skip_passed_policy_waypoints(
            poses, np.arange(3) * .1, [5, 6, 7], current,
            [0.003, 0, 0], allow_return=True)
        np.testing.assert_array_equal(selected, poses)
        self.assertEqual(indices, [5, 6, 7])
        self.assertEqual(skipped, [])

    def test_empty_horizon_holds_and_replans_without_ending_episode(self):
        class Env:
            robots = [object()]

        with patch('eval_real.hold_am2pro_at_current_pose') as hold:
            poses, times, indices, reason, _ = queue_am2pro_policy_trajectory(
                Env(), np.empty((0, 7)), [], [],
                time.time() + 0.1, time.time() + 1.0, lambda: False)
        self.assertEqual(len(poses), 0)
        self.assertEqual(len(times), 0)
        self.assertEqual(indices, [])
        self.assertIsNone(reason)
        hold.assert_called_once()

    def test_append_then_replace_keeps_position_continuous_and_speed_bounded(self):
        start = np.zeros(6)
        path = PoseTrajectoryInterpolator(times=[0.0], poses=[start])
        last = 0.0
        for i in range(1, 17):
            target = np.array([0.002 * i, 0, 0, 0, 0, 0])
            path, last, replaces, accepted = schedule_policy_waypoint(
                path, last, target, i * 0.1, 0.01, 0.03, 0.15)
            self.assertTrue(accepted)
            self.assertFalse(replaces)
        self.assertEqual(len(path.times), 17)
        handoff = 0.51
        before = path(handoff)
        new = np.array([0.004, 0.003, 0, 0, 0, 0])
        path, last, replaces, accepted = schedule_policy_waypoint(
            path, last, new, 0.62, handoff, 0.03, 0.15)
        self.assertTrue(accepted)
        self.assertTrue(replaces)
        np.testing.assert_allclose(path(handoff), before, atol=1e-10)
        self.assertLess(path.times[-1], 1.6)
        samples = path(np.linspace(handoff, path.times[-1], 100))[:, :3]
        speed = np.linalg.norm(np.diff(samples, axis=0), axis=1) / (
            (path.times[-1] - handoff) / 99)
        self.assertLessEqual(float(speed.max()), 0.030001)

    def test_past_point_does_not_change_trajectory(self):
        path = PoseTrajectoryInterpolator(times=[0.0], poses=[np.zeros(6)])
        updated, last, replaces, accepted = schedule_policy_waypoint(
            path, 0.0, np.ones(6), 0.05, 0.1, 0.03, 0.15)
        self.assertIs(updated, path)
        self.assertEqual(last, 0.0)
        self.assertFalse(replaces)
        self.assertFalse(accepted)

    def test_client_queues_ordered_arm_and_gripper_targets(self):
        class Robot:
            def __init__(self):
                self.calls = []

            def schedule_policy_waypoint(self, pose, gripper_width, target_time):
                self.calls.append((np.asarray(pose), gripper_width, target_time))

        class Accumulator:
            def __init__(self):
                self.recorded = None

            def put(self, actions, timestamps):
                self.recorded = (actions.copy(), timestamps.copy())

        env = BimanualUmiEnv.__new__(BimanualUmiEnv)
        robot = Robot()
        env.robots = [robot]
        env.robots_config = [{'robot_type': 'am2pro',
                              'robot_action_latency': 0.02}]
        env.action_accumulator = Accumulator()
        actions = np.zeros((16, 7))
        actions[:, 0] = np.arange(16) * 0.002
        actions[:, 6] = np.arange(16) * 0.001
        timestamps = time.time() + 0.3 + np.arange(16) * 0.1
        sent_actions, sent_times, kept = env.exec_am2pro_policy_trajectory(
            actions, timestamps)
        self.assertEqual(len(robot.calls), 16)
        np.testing.assert_allclose(sent_actions, actions)
        np.testing.assert_allclose(sent_times, timestamps)
        np.testing.assert_array_equal(kept, np.arange(16))
        for index, (pose, width, target_time) in enumerate(robot.calls):
            self.assertAlmostEqual(pose[0], actions[index, 0])
            self.assertAlmostEqual(width, actions[index, 6])
            self.assertAlmostEqual(target_time,
                                   timestamps[index] - 0.02, places=5)
        self.assertIsNotNone(env.action_accumulator.recorded)


if __name__ == '__main__':
    unittest.main()
