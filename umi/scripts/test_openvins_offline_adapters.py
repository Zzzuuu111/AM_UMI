"""No ROS/hardware needed: test timestamp and rigid-alignment adapter contracts."""
import importlib.util
import contextlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mocks = {n: types.ModuleType(n) for n in ('rclpy', 'rclpy.qos', 'builtin_interfaces', 'builtin_interfaces.msg', 'sensor_msgs', 'sensor_msgs.msg')}
        mocks['rclpy.qos'].QoSProfile = object
        mocks['builtin_interfaces.msg'].Time = types.SimpleNamespace
        mocks['sensor_msgs.msg'].Image = object
        mocks['sensor_msgs.msg'].Imu = object
        with patch.dict(sys.modules, mocks):
            cls.replay = load('replay_test', ROOT / 'third_party/open_vins/am_umi_replay.py')
        cls.compare = load('compare_test', ROOT / 'imu_work/compare_openvins_fixed_tag.py')
        cls.exporter = load('exporter_test', ROOT / 'imu_work/export_handheld_imu_to_orbslam3.py')

    def test_source_column_not_receive_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'frames.csv'
            f.write_text('relative_s,uvc_source_relative_s\n0,0\n0.07,0.033\n')
            self.assertEqual(self.replay.camera_times(f), [0, .033])
            f.write_text('relative_s,uvc_source_relative_s\n0,0\n0.07,0\n')
            with self.assertRaises(RuntimeError):
                self.replay.camera_times(f)

    def test_stamp_precision_and_negative_lead_in(self):
        s = self.replay.stamp(.123456789)
        self.assertEqual((s.sec, s.nanosec), (1700000000, 123456789))
        s = self.replay.stamp(-.01)
        self.assertEqual((s.sec, s.nanosec), (1699999999, 990000000))

    def test_imu_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / 'imu.json'
            payload = {'1': {'streams': {'ACCL': {'samples': [{'cts': 10, 'value': [0,0,9.8]}]},
                                        'GYRO': {'samples': [{'cts': 10, 'value': [0,0,0]}]}}}}
            f.write_text(json.dumps(payload))
            self.assertEqual(self.replay.imu_events(f)[0][0], .01)
            payload['1']['streams']['GYRO']['samples'][0]['cts'] = 20
            f.write_text(json.dumps(payload))
            with self.assertRaises(RuntimeError):
                self.replay.imu_events(f)

    def test_metric_alignment_does_not_rescale(self):
        x = np.random.default_rng(11).normal(size=(100,3))
        r = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
        y = x @ r.T + [1,2,3]
        fitted, trans, scale = self.compare.align_rigid(x,y)
        np.testing.assert_allclose(x @ fitted.T + trans, y, atol=1e-12)
        self.assertAlmostEqual(scale, 1)
        fitted, trans, scale = self.compare.align_rigid(x*2,y)
        self.assertAlmostEqual(scale, .5)
        self.assertGreater(np.linalg.norm(x*2 @ fitted.T + trans-y), 1)

    def test_global_ordered_uvc_matching(self):
        timing = load('timing_test', ROOT / 'imu_work/uvc_payload_timing.py')
        rng = np.random.default_rng(8)
        for _ in range(20):
            events = np.arange(8, dtype=np.int64) * 34_000_000
            receive = np.sort(rng.choice(220_000_000, size=5, replace=False))
            matched = timing._strict_monotonic_event_indices(events, receive)
            cost = np.sum(((events[matched]-receive)/1e6)**2)
            optimal = min(np.sum(((events[list(c)]-receive)/1e6)**2) for c in itertools.combinations(range(8),5))
            self.assertAlmostEqual(cost, optimal)
            self.assertTrue(np.all(np.diff(matched)>0))

    def test_repeated_imu_packets_keep_first_of_each_run(self):
        values = np.array([[1,2,3], [1,2,3], [4,2,3], [4,2,3], [4,2,4]])
        np.testing.assert_array_equal(
            self.exporter.keep_first_of_equal_runs(values),
            [True, False, True, False, True])
        np.testing.assert_array_equal(
            self.exporter.keep_first_of_equal_runs(np.empty((0, 3))), [])

    def test_phase_audit_keeps_metric_scale_error(self):
        with patch.dict(sys.modules, {'compare_openvins_fixed_tag': self.compare}):
            audit = load('phase_audit_test', ROOT / 'imu_work/audit_openvins_motion_phases.py')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            xyz = np.random.default_rng(31).normal(size=(100, 3)) * .01
            times = np.linspace(4, 14, len(xyz))
            state = np.zeros((len(xyz), 17))
            state[:, 0] = 1700000000 + times
            state[:, 4] = 1
            state[:, 5:8] = xyz
            np.savetxt(base / 'state_estimate.txt', state)
            reference = base / 'reference.csv'
            reference.write_text('timestamp,is_lost,x,y,z\n' + ''.join(
                f'{t},false,{x},{y},{z}\n' for t, (x,y,z) in zip(times, xyz*2)))
            (base / 'rotation.json').write_text(json.dumps({'R_camera_imu': np.eye(3).tolist()}))
            (base / 'translation.json').write_text(json.dumps({'r_camera_to_imu_in_camera_m': [0,0,0]}))
            argv = ['audit', '--run-dir', tmp, '--reference', str(reference),
                    '--rotation-calibration', str(base / 'rotation.json'),
                    '--translation-calibration', str(base / 'translation.json'), '--guide', 'rotation']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                audit.main()
                with self.assertRaises(SystemExit):
                    audit.main()  # Existing reports must not be overwritten.
            phase = json.loads((base / 'motion_phase_audit.json').read_text())['phases'][1]
            self.assertEqual(phase['samples'], 100)
            self.assertAlmostEqual(phase['diagnostic_local_sim3_scale'], 2)
            self.assertGreater(phase['position_rmse_using_global_se3_m'], .01)


if __name__ == '__main__':
    unittest.main()
