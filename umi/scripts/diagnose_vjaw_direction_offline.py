#!/usr/bin/env python3
"""Compare a live policy trace with similar recorded demo windows offline.

Reads a checkpoint, demo zip, and saved policy trace. No camera or robot code
is instantiated.
"""

import argparse
import json
import os
import pathlib
import sys

import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import diffusion_policy.workspace.train_diffusion_transformer_timm_workspace  # noqa: E402,F401
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep  # noqa: E402
from umi.common.pose_util import pose_to_mat, mat_to_pose10d  # noqa: E402
from umi.real_world.real_inference_util import get_real_umi_obs_dict  # noqa: E402


def vector_mm(array):
    return np.round(np.asarray(array) * 1000, 2).tolist()


def cosine(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else None


def demo_window(group, index, shape_meta):
    indices = np.array([index - 3, index], dtype=int)
    env_obs = {
        'camera0_rgb': np.asarray(group['data/camera0_rgb'].oindex[indices]),
        'robot0_eef_pos': np.asarray(group['data/robot0_eef_pos'].oindex[indices]),
        'robot0_eef_rot_axis_angle': np.asarray(
            group['data/robot0_eef_rot_axis_angle'].oindex[indices]),
        'robot0_gripper_width': np.asarray(
            group['data/robot0_gripper_width'].oindex[indices]),
    }
    start_pose = np.asarray(group['data/robot0_demo_start_pose'][index])
    observation = get_real_umi_obs_dict(
        env_obs, shape_meta=shape_meta, obs_pose_repr='relative',
        episode_start_pose=[start_pose])
    action_indices = index + np.arange(16) * 3
    recorded = np.asarray(group['data/action'].oindex[action_indices])
    base_pose = pose_to_mat(np.concatenate([
        env_obs['robot0_eef_pos'][-1],
        env_obs['robot0_eef_rot_axis_angle'][-1]]))
    relative = convert_pose_mat_rep(
        pose_to_mat(recorded[:, :6]), base_pose_mat=base_pose,
        pose_rep='relative', backward=False)
    target = np.concatenate([
        mat_to_pose10d(relative), recorded[:, 6:7]], axis=-1)
    return observation, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--sample-indices', required=True,
                        help='Comma-separated absolute frame indices in the demo zip')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    if output.exists():
        parser.error(f'refusing to overwrite {output}')
    indices = [int(x) for x in args.sample_indices.split(',')]

    with pathlib.Path(args.trace).open() as source:
        cycles = [json.loads(line) for line in source]
    if not cycles:
        parser.error('empty live trace')
    live = []
    for cycle in cycles:
        poses = np.asarray(cycle['raw_target_poses_7d'])
        base = np.asarray(cycle['current_pose_6d'])[0]
        rotation = st.Rotation.from_rotvec(base[3:]).as_matrix()
        local = (rotation.T @ (poses[:, :3] - base[:3]).T).T
        live.append({
            'cycle_index': cycle['cycle_index'],
            'local_step_5_mm': vector_mm(local[5]),
            'local_step_9_mm': vector_mm(local[9]),
            'local_step_15_mm': vector_mm(local[15]),
        })

    device = torch.device(args.device)
    # The checkpoint contains deployment weights; pretrained initialization
    # only needs its local cache and must not contact Hugging Face.
    os.environ['HF_HUB_OFFLINE'] = '1'
    for name in ('http_proxy', 'https_proxy', 'all_proxy',
                 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'):
        os.environ.pop(name, None)
    payload = torch.load(args.checkpoint, map_location='cpu',
                         pickle_module=dill, mmap=True, weights_only=False)
    cfg = payload['cfg']
    policy = hydra.utils.instantiate(cfg.policy)
    weights = 'ema_model' if cfg.training.use_ema else 'model'
    policy.load_state_dict(payload['state_dicts'][weights], strict=True)
    del payload
    policy.num_inference_steps = 16
    policy.to(device).eval()

    store = zarr.ZipStore(args.dataset, mode='r')
    try:
        group = zarr.open(store, mode='r')
        ends = np.asarray(group['meta/episode_ends'][:])
        starts = np.r_[0, ends[:-1]]
        records = []
        for index in indices:
            episode = int(np.searchsorted(ends, index, side='right'))
            if index - 3 < starts[episode] or index + 45 >= ends[episode]:
                raise ValueError(f'frame {index} lacks a full observation/action window')
            observation, target = demo_window(group, index, cfg.task.shape_meta)
            tensor_obs = {
                key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(device)
                for key, value in observation.items()
            }
            generator = torch.Generator(device=device).manual_seed(42)
            with torch.inference_mode():
                prediction = policy.predict_action(
                    tensor_obs, generator=generator)['action_pred'][0].cpu().numpy()
            pred = prediction[:, :3]
            true = target[:, :3]
            records.append({
                'episode': episode + 1,
                'frame_index': index,
                'frame_in_episode': int(index - starts[episode]),
                'recorded_step_5_mm': vector_mm(true[5]),
                'predicted_step_5_mm': vector_mm(pred[5]),
                'recorded_step_9_mm': vector_mm(true[9]),
                'predicted_step_9_mm': vector_mm(pred[9]),
                'recorded_step_15_mm': vector_mm(true[15]),
                'predicted_step_15_mm': vector_mm(pred[15]),
                'direction_cosine_step_9': cosine(true[9], pred[9]),
                'position_rmse_mm': float(np.sqrt(np.mean(
                    np.sum((pred - true) ** 2, axis=1))) * 1000),
            })
            print('demo', episode + 1, 'frame', index,
                  'target9', records[-1]['recorded_step_9_mm'],
                  'pred9', records[-1]['predicted_step_9_mm'], flush=True)
    finally:
        store.close()

    report = {
        'schema': 'vjaw_direction_offline_diagnostic_v1',
        'safety': 'offline only; no camera, serial port, controller, or robot command',
        'checkpoint': str(pathlib.Path(args.checkpoint).resolve()),
        'dataset': str(pathlib.Path(args.dataset).resolve()),
        'trace': str(pathlib.Path(args.trace).resolve()),
        'sampling_seed': 42,
        'denoising_steps': 16,
        'live_raw_predictions_local_to_tcp': live,
        'similar_demo_windows': records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print('report:', output)


if __name__ == '__main__':
    main()
