#!/usr/bin/env python3
"""Copy a merged V-jaw dataset and replace only its gripper width labels.

The source training dataset is never modified. All position, rotation, image,
and episode arrays are copied byte-for-byte at the Zarr value level. No robot
or camera is opened.
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from scripts.convert_handheld_fixed_tag_demo import interpolate_widths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='existing merged .zarr.zip')
    parser.add_argument('--output', required=True, help='new .zarr.zip; must not exist')
    parser.add_argument('--calibration', required=True, help='signed gripper calibration JSON')
    parser.add_argument('--smooth-window-frames', type=int, default=9)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    calibration_path = Path(args.calibration).expanduser().resolve()
    manifest_path = source.with_name(source.name + '.manifest.json')
    if not source.is_file() or not manifest_path.is_file():
        parser.error('source dataset or its merge manifest is missing')
    if output.exists() or output.with_name(output.name + '.manifest.json').exists():
        parser.error(f'refusing to overwrite output or manifest: {output}')
    if not output.name.endswith('.zarr.zip'):
        parser.error('output must end in .zarr.zip')
    partial_dir = output.with_name(output.name + '.partial.dir')
    partial_zip = output.with_name(output.name + '.partial')
    if partial_dir.exists() or partial_zip.exists():
        parser.error('partial output already exists; inspect before retrying')
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
    if calibration.get('width_measurement') != 'relative_tag_signed_hinge_rotation_deg':
        parser.error('calibration is not a signed V-jaw mapping')
    max_width = float(calibration['physical_gripper_range_m'][1])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    episodes = manifest['source_sessions']
    register_codecs()
    # Compute every label before copying images, so missing detections or
    # mismatched frame counts fail without creating a large partial dataset.
    labels = []
    summary = []
    for episode in episodes:
        session = Path(episode['source']).parent
        detection_path = session / 'tag_detection_vjaw_branch_fixed.pkl'
        with detection_path.open('rb') as file:
            detections = pickle.load(file)
        expected = int(episode['episode_ends'][-1])
        if len(detections) != expected:
            raise ValueError(f'{session}: {len(detections)} detections != {expected} frames')
        widths, report = interpolate_widths(
            detections, calibration, max_width, None, args.smooth_window_frames)
        labels.append(widths)
        summary.append({
            'session': str(session),
            'frames': expected,
            'first_30_median_mm': float(np.median(widths[:30]) * 1000),
            'last_30_median_mm': float(np.median(widths[-30:]) * 1000),
            'maximum_mm': float(np.max(widths) * 1000),
            'dual_tag_ratio': report['dual_tag_ratio'],
            'episode_zero_offset_deg': report['episode_zero_offset_deg'],
            'episode_start_valid_frames': report['episode_start_valid_frames'],
            'gripper_label_quality': (
                'low_dual_tag_visibility' if report['dual_tag_ratio'] < 0.5 else 'candidate'),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    input_store = zarr.ZipStore(str(source), mode='r')
    directory_store = zarr.DirectoryStore(str(partial_dir))
    try:
        zarr.copy_store(input_store, directory_store)
    finally:
        input_store.close()
        directory_store.close()
    root = zarr.open_group(str(partial_dir), mode='r+')
    ends = np.asarray(root['meta/episode_ends'][:], dtype=int)
    if len(ends) != len(labels) or np.any(np.diff(np.r_[0, ends]) != [len(x) for x in labels]):
        raise ValueError('copied dataset episode boundaries do not match source manifest')
    starts = np.r_[0, ends[:-1]]
    width_array = root['data/robot0_gripper_width']
    action_array = root['data/action']
    if width_array.shape != (int(ends[-1]), 1) or action_array.shape[1] != 7:
        raise ValueError('unexpected gripper or action array shape')
    for start, end, widths in zip(starts, ends, labels):
        width_array[int(start):int(end), 0] = widths
        action = np.asarray(action_array[int(start):int(end)])
        action[:, 6] = widths
        action_array[int(start):int(end)] = action
    zip_store = zarr.ZipStore(str(partial_zip), mode='w')
    try:
        zarr.copy_store(zarr.DirectoryStore(str(partial_dir)), zip_store)
    finally:
        zip_store.close()
    os.replace(partial_zip, output)
    shutil.rmtree(partial_dir)
    report = {
        'schema': 'am_umi_signed_gripper_relabel_v1',
        'offline_only': True,
        'source_dataset': str(source),
        'output': str(output),
        'calibration': str(calibration_path),
        'modified_arrays': ['data/robot0_gripper_width', 'data/action[:,6]'],
        'episodes': summary,
        'low_visibility_sessions': [
            x['session'] for x in summary if x['gripper_label_quality'] == 'low_dual_tag_visibility'],
    }
    output.with_name(output.name + '.manifest.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('SIGNED_VJAW_GRIPPER_RELABEL_OK')
    print('episodes:', len(labels), 'frames:', int(ends[-1]))
    print('output:', output)


if __name__ == '__main__':
    main()
