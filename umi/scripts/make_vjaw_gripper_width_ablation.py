#!/usr/bin/env python3
"""Copy a training ZipStore and relabel only its gripper widths for an ablation.

This creates synthetic labels; it does not calibrate the physical gripper.
No camera or robot device is opened.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import zarr


def relabel_widths(widths, close_threshold_m, open_offset_m, max_width_m):
    widths = np.asarray(widths, dtype=np.float32)
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise ValueError('source widths must be finite and nonnegative')
    return np.where(widths <= close_threshold_m, 0.0,
                    np.minimum(widths + open_offset_m, max_width_m)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--close-threshold-mm', type=float, default=5.0)
    parser.add_argument('--open-offset-mm', type=float, default=5.0)
    parser.add_argument('--max-width-mm', type=float, default=84.0)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest = output.with_name(output.name + '.synthetic_labels.json')
    if not source.is_file() or output.suffix != '.zip':
        parser.error('source must exist and output must end in .zip')
    if output.exists() or manifest.exists() or source == output:
        parser.error('refusing to overwrite the source or an existing experiment')
    if not (0 <= args.close_threshold_mm < args.max_width_mm and
            0 < args.open_offset_mm < args.max_width_mm):
        parser.error('invalid width thresholds')
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix='vjaw_width_ablation_', dir=output.parent) as temp:
        directory = zarr.DirectoryStore(str(Path(temp) / 'dataset.zarr'))
        with zarr.ZipStore(str(source), mode='r') as source_store:
            zarr.copy_store(source_store, directory, if_exists='raise')
        root = zarr.open(directory, mode='r+')
        width_array = root['data/robot0_gripper_width']
        action_array = root['data/action']
        widths = np.asarray(width_array[:], dtype=np.float32).reshape(-1)
        actions = np.asarray(action_array[:], dtype=np.float32)
        if actions.shape != (len(widths), 7) or not np.array_equal(actions[:, 6], widths):
            raise ValueError('source action width and state width do not match exactly')
        modified = relabel_widths(
            widths, args.close_threshold_mm / 1000,
            args.open_offset_mm / 1000, args.max_width_mm / 1000)
        width_array[:] = modified[:, None]
        actions[:, 6] = modified
        action_array[:] = actions
        if not np.array_equal(width_array[:].reshape(-1), action_array[:, 6]):
            raise RuntimeError('modified action and observation widths differ')
        episodes = len(root['meta/episode_ends'])
        zip_path = Path(temp) / 'dataset.zarr.zip'
        with zarr.ZipStore(str(zip_path), mode='w') as output_store:
            zarr.copy_store(directory, output_store, if_exists='raise')
        # Check the immutable archive before making it visible to training.
        with zarr.ZipStore(str(zip_path), mode='r') as check_store:
            check = zarr.open(check_store, mode='r')
            if not np.array_equal(check['data/robot0_gripper_width'][:].reshape(-1),
                                  check['data/action'][:, 6]):
                raise RuntimeError('archive verification failed')
        os.replace(zip_path, output)

    report = {
        'schema': 'am_umi_synthetic_gripper_width_ablation_v1',
        'warning': 'synthetic labels for an offline training experiment; not physical calibration',
        'source': str(source), 'output': str(output),
        'episodes': episodes, 'frames': int(len(widths)),
        'close_threshold_mm': args.close_threshold_mm,
        'open_offset_mm': args.open_offset_mm,
        'max_width_mm': args.max_width_mm,
        'source_range_mm': [float(widths.min() * 1000), float(widths.max() * 1000)],
        'output_range_mm': [float(modified.min() * 1000), float(modified.max() * 1000)],
        'closed_frames': int(np.count_nonzero(modified == 0)),
        'changed_frames': int(np.count_nonzero(modified != widths)),
        'modified_arrays': ['data/action[:, 6]', 'data/robot0_gripper_width'],
    }
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('VJAW_SYNTHETIC_GRIPPER_DATASET_OK')
    print(output)
    print(manifest)


if __name__ == '__main__':
    main()
