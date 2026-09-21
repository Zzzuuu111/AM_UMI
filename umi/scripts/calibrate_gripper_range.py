# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
import collections
import pickle
import json
import numpy as np
from umi.common.cv_util import get_dual_tag_gripper_width


# %%
@click.command()
@click.option('-i', '--input', required=True, help='Tag detection pkl')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-t', '--tag_det_threshold', type=float, default=0.8)
@click.option('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag")
@click.option('--low-percentile', type=float, default=1.0, show_default=True,
    help='robust closed-end percentile of dual-tag separation')
@click.option('--high-percentile', type=float, default=99.0, show_default=True,
    help='robust open-end percentile of dual-tag separation')
def main(input, output, tag_det_threshold, nominal_z, low_percentile, high_percentile):
    tag_detection_results = pickle.load(open(input, 'rb'))
    
    # identify gripper hardware id
    n_frames = len(tag_detection_results)
    tag_counts = collections.defaultdict(lambda: 0)
    for frame in tag_detection_results:
        for key in frame['tag_dict'].keys():
            tag_counts[key] += 1
    tag_stats = collections.defaultdict(lambda: 0.0)
    for k, v in tag_counts.items():
        tag_stats[k] = v / n_frames
    
    max_tag_id = np.max(list(tag_stats.keys()))
    tag_per_gripper = 6
    max_gripper_id = max_tag_id // tag_per_gripper
    
    gripper_prob_map = dict()
    for gripper_id in range(max_gripper_id+1):
        left_id = gripper_id * tag_per_gripper
        right_id = left_id + 1
        left_prob = tag_stats[left_id]
        right_prob = tag_stats[right_id]
        gripper_prob = min(left_prob, right_prob)
        if gripper_prob <= 0:
            continue
        gripper_prob_map[gripper_id] = gripper_prob
    if len(gripper_prob_map) == 0:
        print("No grippers detected!")
        exit(1)
    
    gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[1])
    gripper_id = gripper_probs[-1][0]
    gripper_prob = gripper_probs[-1][1]
    print(f"Detected gripper id: {gripper_id} with probability {gripper_prob}")
    if gripper_prob < tag_det_threshold:
        print(f"Detection rate {gripper_prob} < {tag_det_threshold} threshold.")
        exit(1)
        
    # run calibration
    left_id = gripper_id * tag_per_gripper
    right_id = left_id + 1

    # Only use frames where *both* finger tags are visible.  The original UMI
    # helper combines a signed two-tag width with a positive one-tag fallback;
    # that is useful for sparse legacy data but invalid for an endpoint range.
    raw_tag_separations = list()
    for i, dt in enumerate(tag_detection_results):
        tag_dict = dt['tag_dict']
        width = get_dual_tag_gripper_width(
            tag_dict, left_id, right_id, nominal_z=nominal_z)
        if width is not None:
            raw_tag_separations.append(width)

    raw_tag_separations = np.asarray(raw_tag_separations, dtype=float)
    if len(raw_tag_separations) < 20:
        raise click.ClickException(
            'Need at least 20 valid frames with both finger tags visible.')
    if not 0 <= low_percentile < high_percentile <= 100:
        raise click.ClickException('Require 0 <= low-percentile < high-percentile <= 100.')

    raw_closed, raw_open = np.percentile(
        raw_tag_separations, [low_percentile, high_percentile])
    opening_span = float(raw_open - raw_closed)
    if opening_span <= 0.005:
        raise click.ClickException(
            'Observed opening span is <= 5 mm. Repeat the video while fully '
            'closing and fully opening the gripper.')

    result = {
        'schema_version': 2,
        'width_measurement': 'dual_tag_abs_camera_x_m',
        'gripper_id': gripper_id,
        'left_finger_tag_id': left_id,
        'right_finger_tag_id': right_id,
        'nominal_tag_depth_m': nominal_z,
        'valid_dual_tag_frames': int(len(raw_tag_separations)),
        'raw_tag_separation_percentiles_m': {
            'low_percentile': low_percentile,
            'low_value': float(raw_closed),
            'high_percentile': high_percentile,
            'high_value': float(raw_open),
        },
        # Raw tag-centre separation has a fixed mounting offset.  The closed
        # endpoint is therefore mapped to zero jaw opening.  The measured
        # finger travel provides the opening span without guessing a robot
        # gripper specification.
        'aruco_measured_width': [float(raw_closed), float(raw_open)],
        'aruco_actual_width': [0.0, opening_span],
        # Kept for older readers.  These are actual jaw-opening endpoints,
        # never signed raw tag coordinates.
        'min_width': 0.0,
        'max_width': opening_span,
    }
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    print('GRIPPER_RANGE_CALIBRATION_OK')
    print(f'dual_tag_frames: {len(raw_tag_separations)}')
    print(f'raw_tag_separation_m: closed={raw_closed:.5f}, open={raw_open:.5f}')
    print(f'calibrated_opening_range_m: [0.00000, {opening_span:.5f}]')
    print(f'saved: {output}')

# %%
if __name__ == "__main__":
    main()
