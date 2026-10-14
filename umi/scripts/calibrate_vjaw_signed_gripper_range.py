#!/usr/bin/env python3
"""Calibrate a signed V-jaw angle from labeled closed/open video segments.

Offline only. The input must be the branch-fixed Tag detections corresponding
to a video whose selected start/end segments show a closed jaw and whose open
segment shows the measured maximum opening.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umi.common.cv_util import get_signed_relative_tag_rotation_deg  # noqa: E402


def relative_rotations(detections, fixed_id, moving_id):
    rotations = []
    for frame in detections:
        tags = frame.get('tag_dict', {})
        if fixed_id not in tags or moving_id not in tags:
            rotations.append(None)
            continue
        fixed, _ = cv2.Rodrigues(np.asarray(tags[fixed_id]['rvec'], dtype=float))
        moving, _ = cv2.Rodrigues(np.asarray(tags[moving_id]['rvec'], dtype=float))
        rotations.append(Rotation.from_matrix(fixed.T @ moving))
    return rotations


def median_rotation(rotations, start, stop, label):
    vectors = [r.as_rotvec() for r in rotations[start:stop] if r is not None]
    if len(vectors) < 10:
        raise ValueError(f'{label} has fewer than 10 valid dual-Tag frames')
    return Rotation.from_rotvec(np.median(vectors, axis=0)), len(vectors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixed-tag-id', type=int, default=0)
    parser.add_argument('--moving-tag-id', type=int, default=1)
    parser.add_argument('--closed-start-stop', type=int, required=True)
    parser.add_argument('--open-start', type=int, required=True)
    parser.add_argument('--open-stop', type=int, required=True)
    parser.add_argument('--closed-end-start', type=int, required=True)
    parser.add_argument('--max-opening-m', type=float, required=True)
    parser.add_argument('--episode-start-closed-frames', type=int, default=30)
    parser.add_argument('--episode-start-min-valid-frames', type=int, default=4)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f'refusing to overwrite {output}')
    with source.open('rb') as file:
        detections = pickle.load(file)
    count = len(detections)
    if not (10 <= args.closed_start_stop <= args.open_start < args.open_stop
            <= args.closed_end_start < count - 10):
        parser.error('invalid or overlapping closed/open frame ranges')
    if args.max_opening_m <= 0:
        parser.error('max-opening-m must be positive')
    rotations = relative_rotations(detections, args.fixed_tag_id, args.moving_tag_id)
    closed, closed_count = median_rotation(
        rotations, 0, args.closed_start_stop, 'closed start')
    opened, open_count = median_rotation(
        rotations, args.open_start, args.open_stop, 'open')
    end, end_count = median_rotation(
        rotations, args.closed_end_start, count, 'closed end')
    delta = (closed.inv() * opened).as_rotvec()
    open_angle = float(np.rad2deg(np.linalg.norm(delta)))
    if open_angle < 5 or open_angle > 120:
        raise ValueError(f'implausible open angle {open_angle:.2f} deg')
    axis = delta / np.linalg.norm(delta)
    signed_angles = np.asarray([
        get_signed_relative_tag_rotation_deg(
            frame.get('tag_dict', {}), args.fixed_tag_id, args.moving_tag_id,
            closed.as_rotvec(), axis)
        for frame in detections], dtype=object)
    signed_angles = np.array([np.nan if x is None else float(x) for x in signed_angles])
    end_error = float(np.rad2deg(np.linalg.norm((closed.inv() * end).as_rotvec())))
    result = {
        'schema_version': 1,
        'status': 'candidate',
        'gripper_model': 'v_jaw_one_moving_tag_pair',
        'width_measurement': 'relative_tag_signed_hinge_rotation_deg',
        'fixed_jaw_tag_id': args.fixed_tag_id,
        'moving_jaw_tag_id': args.moving_tag_id,
        'closed_relative_rotvec': closed.as_rotvec().tolist(),
        'hinge_axis_closed_frame': axis.tolist(),
        'open_angle_deg': open_angle,
        'episode_start_closed_frames': args.episode_start_closed_frames,
        'episode_start_min_valid_frames': args.episode_start_min_valid_frames,
        'physical_gripper_range_m': [0.0, args.max_opening_m],
        'min_width': 0.0,
        'max_width': args.max_opening_m,
        'source_tag_detection': str(source),
        'labeled_frame_ranges': {
            'closed_start': [0, args.closed_start_stop],
            'open': [args.open_start, args.open_stop],
            'closed_end': [args.closed_end_start, count],
        },
        'diagnostics': {
            'valid_frames_per_segment': [closed_count, open_count, end_count],
            'closed_end_orientation_error_deg': end_error,
            'signed_angle_percentiles_deg': np.nanpercentile(
                signed_angles, [0, 1, 50, 99, 100]).tolist(),
        },
        'notes': [
            'Closed and open frame ranges must be visually checked against the raw video.',
            'Physical intermediate widths assume a linear angle-to-width mapping and need ruler verification.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('SIGNED_VJAW_CALIBRATION_CANDIDATE_OK')
    print('open_angle_deg:', round(open_angle, 3))
    print('closed_end_orientation_error_deg:', round(end_error, 3))
    print('saved:', output)


if __name__ == '__main__':
    main()
