"""Estimate handheld camera-to-TCP translation from a fixed-point pivot video.

During the recording, the physical pinch point of the gripper is kept on one
stationary dot in the Tag-13 world.  The camera may be moved/rotated around
that dot.  If T_world_camera changes over the video, the unknown TCP position
in the camera frame and the unknown fixed world dot can be jointly recovered
by least squares:

    p_world = R_world_camera @ p_camera_tcp + t_world_camera

The recording needs only the existing ArUco detection pickle.  It does not
move a robot and does not use IMU data.
"""
import json
import os
import pickle
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import click
import numpy as np
from scipy.spatial.transform import Rotation


def tag_to_camera_matrix(tag):
    """Return T_camera_tag from an ArUco rvec/tvec entry."""
    out = np.eye(4)
    out[:3, :3] = Rotation.from_rotvec(
        np.asarray(tag['rvec'], dtype=float)).as_matrix()
    out[:3, 3] = np.asarray(tag['tvec'], dtype=float)
    return out


def solve_pivot(camera_poses, inlier_threshold_m=0.008, max_rounds=4):
    """Solve p_camera_tcp and the fixed pivot point; robustly reject outliers."""
    selected = np.ones(len(camera_poses), dtype=bool)
    solution = None
    residuals = None
    for _ in range(max_rounds):
        rows = []
        rhs = []
        for use, tx_world_camera in zip(selected, camera_poses):
            if not use:
                continue
            rows.append(np.concatenate([tx_world_camera[:3, :3], -np.eye(3)], axis=1))
            rhs.append(-tx_world_camera[:3, 3])
        if len(rows) < 6:
            raise ValueError('Too few inlier pivot poses after outlier rejection.')
        matrix = np.concatenate(rows, axis=0)
        vector = np.concatenate(rhs, axis=0)
        solution, _, _, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
        p_camera_tcp = solution[:3]
        p_world = solution[3:]
        residuals = np.array([
            np.linalg.norm(tx[:3, :3] @ p_camera_tcp + tx[:3, 3] - p_world)
            for tx in camera_poses
        ])
        new_selected = residuals <= inlier_threshold_m
        if np.array_equal(new_selected, selected):
            break
        selected = new_selected

    if solution is None:
        raise ValueError('No pivot solution was produced.')
    if selected.sum() < 12:
        raise ValueError(
            f'Only {selected.sum()} inlier frames remain; keep the jaw centre on '
            'the same dot more steadily and record again.')
    return solution[:3], solution[3:], residuals, selected, singular_values


@click.command()
@click.option('--input', 'input_path', required=True, type=click.Path(exists=True, dir_okay=False),
              help='tag_detection.pkl from the pivot recording')
@click.option('--output', required=True, type=click.Path(dir_okay=False),
              help='new camera-to-TCP geometry JSON; refuses to overwrite')
@click.option('--world-tag-id', default=13, show_default=True, type=int,
              help='fixed world Tag used as the world coordinate frame')
@click.option('--left-finger-tag-id', default=0, show_default=True, type=int)
@click.option('--right-finger-tag-id', default=1, show_default=True, type=int)
@click.option('--inlier-threshold-mm', default=8.0, show_default=True, type=float,
              help='maximum fixed-point residual retained as an inlier')
@click.option('--min-inlier-ratio', default=0.70, show_default=True, type=float,
              help='minimum fraction of frames that must obey the fixed-pivot model')
@click.option('--rotation-vector', default='0,0,0', show_default=True,
              help='assumed camera-to-TCP axis-angle in rad, formatted rx,ry,rz')
def main(input_path, output, world_tag_id, left_finger_tag_id, right_finger_tag_id,
         inlier_threshold_mm, min_inlier_ratio, rotation_vector):
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        raise click.ClickException(f'Refusing to overwrite existing output: {output}')
    try:
        assumed_rotation = np.array([float(x) for x in rotation_vector.split(',')])
    except ValueError as exc:
        raise click.ClickException('--rotation-vector must be rx,ry,rz') from exc
    if assumed_rotation.shape != (3,) or not np.isfinite(assumed_rotation).all():
        raise click.ClickException('--rotation-vector must contain three finite values')
    if inlier_threshold_mm <= 0:
        raise click.ClickException('--inlier-threshold-mm must be positive')
    if not 0 < min_inlier_ratio <= 1:
        raise click.ClickException('--min-inlier-ratio must be in (0, 1]')

    with open(input_path, 'rb') as f:
        detections = pickle.load(f)
    camera_poses = []
    tag_pair_frames = 0
    for frame in detections:
        tags = frame.get('tag_dict', {})
        if world_tag_id not in tags:
            continue
        if left_finger_tag_id in tags and right_finger_tag_id in tags:
            tag_pair_frames += 1
        # ArUco supplies T_camera_tag; Tag 13 defines the fixed world frame.
        camera_poses.append(np.linalg.inv(tag_to_camera_matrix(tags[world_tag_id])))

    if len(camera_poses) < 20:
        raise click.ClickException(
            f'Only {len(camera_poses)} frames see world Tag {world_tag_id}; need at least 20.')
    try:
        p_camera_tcp, p_world, residuals, inliers, singular_values = solve_pivot(
            camera_poses, inlier_threshold_m=inlier_threshold_mm / 1000.0)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    inlier_ratio = float(inliers.mean())
    if inlier_ratio < min_inlier_ratio:
        raise click.ClickException(
            'Fixed-point pivot rejected: only {:.1%} of frames are within '
            '{:.1f} mm (need at least {:.1%}). The jaw centre slid away from '
            'the point; record again with a more stable pivot.'.format(
                inlier_ratio, inlier_threshold_mm, min_inlier_ratio))

    inlier_residuals = residuals[inliers]
    # The translational pivot determines the TCP position.  The rig's camera
    # and tool axes are assumed aligned unless a measured rotation is supplied
    # explicitly.  Keeping this explicit prevents silently inventing an
    # orientation from a finger tag mounted in an arbitrary direction.
    pose_cam_tcp = np.concatenate([p_camera_tcp, assumed_rotation])
    result = {
        'schema': 'am_umi_camera_tcp_geometry_v1',
        'status': 'accepted',
        'transform_direction': 'camera_to_tcp',
        'pose_cam_tcp': pose_cam_tcp.tolist(),
        'translation_method': 'fixed_world_tag_pivot',
        'rotation_method': 'explicit_assumption',
        'assumed_rotation_vector_rad': assumed_rotation.tolist(),
        'world_tag_id': world_tag_id,
        'left_finger_tag_id': left_finger_tag_id,
        'right_finger_tag_id': right_finger_tag_id,
        'input': os.path.abspath(os.path.expanduser(input_path)),
        'quality': {
            'world_tag_frames': len(camera_poses),
            'both_finger_tag_frames': tag_pair_frames,
            'pivot_inlier_frames': int(inliers.sum()),
            'pivot_inlier_ratio': inlier_ratio,
            'pivot_residual_rmse_m': float(np.sqrt(np.mean(inlier_residuals ** 2))),
            'pivot_residual_p95_m': float(np.percentile(inlier_residuals, 95)),
            'linear_system_singular_values': singular_values.tolist(),
        },
        'note': (
            'Translation was estimated with the jaw-centre held at one fixed '
            'physical point. Rotation is an explicit mounting assumption and '
            'must be verified before formal dataset generation.'
        ),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print('HANDHELD_CAMERA_TCP_PIVOT_CALIBRATION_OK')
    print(f'world_tag_frames: {len(camera_poses)}; both_finger_tag_frames: {tag_pair_frames}')
    print(f'pivot_inliers: {inliers.sum()}/{len(inliers)}')
    print('pose_cam_tcp:', np.array2string(pose_cam_tcp, precision=5))
    print('pivot_rmse_mm: {:.3f}; p95_mm: {:.3f}'.format(
        result['quality']['pivot_residual_rmse_m'] * 1000,
        result['quality']['pivot_residual_p95_m'] * 1000))
    print(f'saved: {output}')


if __name__ == '__main__':
    main()
