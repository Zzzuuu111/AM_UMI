#!/usr/bin/env python3
"""Small offline regression tests for the fixed-Tag trajectory path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


refiner = load_module(
    "fixed_tag_refiner_test",
    ROOT / "imu_work/refine_fixed_tag_camera_trajectory.py")
planner = load_module(
    "fixed_tag_planner_test",
    ROOT / "scripts_slam_pipeline/06_generate_dataset_plan.py")


class FixedTagTrajectoryTest(unittest.TestCase):
    def test_short_gap_and_spike_are_filled_but_long_gap_remains_lost(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "raw.csv"
            output = directory / "refined.csv"
            count = 32
            timestamps = np.arange(count) / 30.0
            positions = np.column_stack((0.1 * timestamps,
                                         np.zeros(count),
                                         np.ones(count)))
            quaternions = Rotation.from_euler(
                "z", (0.2 * timestamps)[:, None]).as_quat()
            lost = np.zeros(count, dtype=bool)
            lost[4:6] = True
            lost[16:23] = True
            positions[10, 0] += 0.2
            frame = pd.DataFrame({
                "frame_idx": np.arange(count),
                "timestamp": timestamps,
                "state": np.where(lost, 1, 2),
                "is_lost": lost,
                "is_keyframe": False,
                "x": positions[:, 0], "y": positions[:, 1], "z": positions[:, 2],
                "q_x": quaternions[:, 0], "q_y": quaternions[:, 1],
                "q_z": quaternions[:, 2], "q_w": quaternions[:, 3],
            })
            frame.loc[lost, refiner.POSE_COLUMNS] = 0.0
            frame.to_csv(source, index=False)
            original = source.read_bytes()
            report = refiner.run(argparse.Namespace(
                input=str(source), output=str(output), max_gap_frames=3,
                smooth_window=1, max_position_interpolation_error_m=0.03,
                max_rotation_interpolation_error_deg=10.0))
            result = pd.read_csv(output)
            result_lost = planner.trajectory_lost_mask(result)

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(report["rejected_spike_frames"], 1)
            self.assertTrue((~result_lost[4:6]).all())
            self.assertFalse(result_lost[10])
            self.assertTrue(result_lost[16:23].all())
            self.assertAlmostEqual(result.loc[10, "x"], 0.1 * timestamps[10])

    def test_fixed_tag_provenance_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "trajectory.csv"
            csv_path.write_text("frame_idx\n0\n", encoding="utf-8")
            with self.assertRaises(Exception):
                planner.require_fixed_tag_provenance(csv_path)
            report_path = csv_path.with_suffix(".fixed_tag_report.json")
            report_path.write_text(json.dumps({
                "schema": "am_umi_fixed_tag_camera_trajectory_v1"
            }), encoding="utf-8")
            self.assertEqual(
                planner.require_fixed_tag_provenance(csv_path), report_path)

    def test_fixed_tag_default_lost_limit_is_five_percent(self):
        self.assertEqual(
            planner.max_allowed_lost_frames(3542, "fixed_tag", None), 178)
        self.assertEqual(
            planner.max_allowed_lost_frames(3542, "slam", None), 10)
        self.assertEqual(
            planner.max_allowed_lost_frames(3542, "fixed_tag", 7), 7)


if __name__ == "__main__":
    unittest.main()
