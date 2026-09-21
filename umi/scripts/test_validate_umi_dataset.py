#!/usr/bin/env python3
import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_umi_dataset_test", ROOT / "scripts/validate_umi_dataset.py")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def args_for(dataset):
    return argparse.Namespace(
        dataset=str(dataset), fps=30.0, min_episode_frames=16,
        image_samples=16, action_tolerance=1e-5,
        gripper_min_m=-0.002, gripper_max_m=0.10,
        max_position_step_m=0.08, max_rotation_step_rad=1.0,
        max_gripper_step_m=0.03, min_image_std=8.0,
        min_image_range=32, max_image_clip_fraction=0.85,
        max_bad_image_fraction=0.05)


def make_dataset(path, *, bad_action=False, flat_images=False, frames=20):
    root = zarr.group(store=zarr.DirectoryStore(str(path)))
    data = root.create_group("data")
    meta = root.create_group("meta")
    meta.array("episode_ends", np.array([frames], dtype=np.int64))
    pos = np.zeros((frames, 3), dtype=np.float32)
    pos[:, 0] = np.arange(frames) * 0.001
    rot = np.zeros((frames, 3), dtype=np.float32)
    width = np.full((frames, 1), 0.05, dtype=np.float32)
    action = np.concatenate([pos, rot, width], axis=1)
    if bad_action:
        action[3, 0] += 0.1
    data.array("robot0_eef_pos", pos)
    data.array("robot0_eef_rot_axis_angle", rot)
    data.array("robot0_gripper_width", width)
    data.array("action", action)
    if flat_images:
        images = np.full((frames, 16, 16, 3), 127, dtype=np.uint8)
    else:
        y, x = np.indices((16, 16))
        image = np.stack([x * 16, y * 16, (x + y) * 8], axis=-1).astype(np.uint8)
        images = np.repeat(image[None], frames, axis=0)
    data.array("camera0_rgb", images)


class DatasetQualityGateTest(unittest.TestCase):
    def test_valid_dataset_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid.zarr"
            make_dataset(path)
            report = VALIDATOR.validate(args_for(path))
            self.assertFalse(any(x["level"] == "FAIL" for x in report["issues"]))

    def test_action_mismatch_and_flat_images_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.zarr"
            make_dataset(path, bad_action=True, flat_images=True)
            report = VALIDATOR.validate(args_for(path))
            codes = {x["code"] for x in report["issues"] if x["level"] == "FAIL"}
            self.assertIn("action_state_mismatch", codes)
            self.assertIn("flat_images", codes)

    def test_short_episode_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "short.zarr"
            make_dataset(path, frames=8)
            report = VALIDATOR.validate(args_for(path))
            codes = {x["code"] for x in report["issues"] if x["level"] == "FAIL"}
            self.assertIn("episode_too_short", codes)


if __name__ == "__main__":
    unittest.main()
