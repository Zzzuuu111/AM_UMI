"""Independently validate a camera-to-TCP pivot geometry JSON.

Unlike the calibration tool, this script never fits a new TCP translation. It
uses the supplied geometry to project the same TCP point into the world for
every frame of a fresh fixed-point pivot recording. A good calibration keeps
those projected world points tightly clustered.
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

from scripts.calibrate_handheld_camera_tcp_pivot import tag_to_camera_matrix


@click.command()
@click.option('--input', 'input_path', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--geometry', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--output', required=True, type=click.Path(dir_okay=False))
@click.option('--world-tag-id', default=13, show_default=True, type=int)
@click.option('--inlier-threshold-mm', default=8.0, show_default=True, type=float)
@click.option('--min-inlier-ratio', default=0.70, show_default=True, type=float)
def main(input_path, geometry, output, world_tag_id, inlier_threshold_mm,
         min_inlier_ratio):
    output = os.path.abspath(os.path.expanduser(output))
    if os.path.exists(output):
        raise click.ClickException(f'Refusing to overwrite existing output: {output}')
    if inlier_threshold_mm <= 0 or not 0 < min_inlier_ratio <= 1:
        raise click.ClickException('Invalid inlier threshold or ratio.')
    with open(geometry, encoding='utf-8') as f:
        geometry_data = json.load(f)
    if geometry_data.get('schema') != 'am_umi_camera_tcp_geometry_v1':
        raise click.ClickException('Unknown geometry schema.')
    if geometry_data.get('status') not in (None, 'accepted'):
        raise click.ClickException('Geometry was rejected and cannot be validated.')
    pose = np.asarray(geometry_data.get('pose_cam_tcp'), dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise click.ClickException('geometry pose_cam_tcp must contain six finite values.')
    tcp_in_camera = pose[:3]

    with open(input_path, 'rb') as f:
        detections = pickle.load(f)
    world_tcp = []
    for frame in detections:
        tags = frame.get('tag_dict', {})
        if world_tag_id not in tags:
            continue
        tx_world_camera = np.linalg.inv(tag_to_camera_matrix(tags[world_tag_id]))
        world_tcp.append(tx_world_camera[:3, :3] @ tcp_in_camera + tx_world_camera[:3, 3])
    world_tcp = np.asarray(world_tcp)
    if len(world_tcp) < 20:
        raise click.ClickException(f'Only {len(world_tcp)} world-tag frames; need at least 20.')

    center = np.median(world_tcp, axis=0)
    residuals = np.linalg.norm(world_tcp - center, axis=1)
    threshold_m = inlier_threshold_mm / 1000.0
    inliers = residuals <= threshold_m
    ratio = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inliers.any() else float('inf')
    result = {
        'status': 'accepted' if ratio >= min_inlier_ratio else 'rejected_low_pivot_inlier_ratio',
        'geometry': os.path.abspath(os.path.expanduser(geometry)),
        'input': os.path.abspath(os.path.expanduser(input_path)),
        'world_tag_id': world_tag_id,
        'world_tcp_median_m': center.tolist(),
        'world_tag_frames': int(len(world_tcp)),
        'inlier_threshold_mm': inlier_threshold_mm,
        'inlier_frames': int(inliers.sum()),
        'inlier_ratio': ratio,
        'inlier_rmse_mm': rmse * 1000,
        'residual_p95_mm': float(np.percentile(residuals, 95) * 1000),
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('HANDHELD_CAMERA_TCP_PIVOT_VALIDATION_' + ('OK' if result['status'] == 'accepted' else 'REJECTED'))
    print('frames: {}; inliers: {}/{} ({:.1%})'.format(
        len(world_tcp), inliers.sum(), len(world_tcp), ratio))
    print('inlier_rmse_mm: {:.3f}; all_frame_p95_mm: {:.3f}'.format(
        result['inlier_rmse_mm'], result['residual_p95_mm']))
    print(f'saved: {output}')
    if result['status'] != 'accepted':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
