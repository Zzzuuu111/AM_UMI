#!/usr/bin/env python3
"""Plot a recorded AM2Pro policy run without opening camera or robot devices."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import zarr


def interpolate_position(query_time, readback_time, readback_pos,
                         episode_time, episode_pos):
    if readback_time[0] <= query_time <= readback_time[-1]:
        times, points = readback_time, readback_pos
    elif episode_time[0] <= query_time <= episode_time[-1]:
        times, points = episode_time, episode_pos
    else:
        return np.full(3, np.nan)
    return np.array([np.interp(query_time, times, points[:, axis])
                     for axis in range(3)])


def direction_reversals(points, cycle_indices):
    """Count consecutive Cartesian steps whose directions differ by >90 deg."""
    points = np.asarray(points, dtype=float)
    cycle_indices = np.asarray(cycle_indices, dtype=int)
    steps = np.diff(points, axis=0)
    if len(steps) < 2:
        return {'within_chunk': None, 'across_chunk': None}
    dot = np.sum(steps[:-1] * steps[1:], axis=1)
    within = ((cycle_indices[:-2] == cycle_indices[1:-1]) &
              (cycle_indices[1:-1] == cycle_indices[2:]))
    result = {}
    for label, mask in (('within_chunk', within), ('across_chunk', ~within)):
        result[label] = {
            'reversals': int(np.count_nonzero(dot[mask] < 0)),
            'opportunities': int(np.count_nonzero(mask)),
            'fraction': (float(np.mean(dot[mask] < 0)) if np.any(mask) else None),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--output-suffix', default='',
                        help='Optional suffix for a new plot/report without overwriting prior analysis.')
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    stem = f'{args.episode:03d}'
    artifact_stem = stem + (f'_{args.output_suffix}' if args.output_suffix else '')
    trace_path = run_dir / f'policy_trace_episode_{stem}.jsonl'
    readback_path = run_dir / f'policy_robot_readback_episode_{stem}.npz'
    video_path = run_dir / 'videos' / str(args.episode) / '0.mp4'
    plot_path = run_dir / f'policy_trajectory_episode_{artifact_stem}.png'
    report_path = run_dir / f'policy_diagnostic_episode_{artifact_stem}.json'
    if plot_path.exists() or report_path.exists():
        parser.error('拒绝覆盖已有诊断图或报告')
    if not video_path.is_file():
        parser.error(f'录像不存在: {video_path}')
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()
             if line.strip()]
    if not trace:
        parser.error('策略逐轮日志为空')

    with np.load(readback_path) as readback:
        readback_time = np.asarray(readback['robot_timestamp'], dtype=float)
        readback_pos = np.asarray(readback['ActualTCPPose'][:, :3], dtype=float)
    if len(readback_time) < 2 or np.any(np.diff(readback_time) <= 0):
        parser.error('控制器位姿读数不足或时间戳不递增')

    replay = zarr.open(str(run_dir / 'replay_buffer.zarr'), mode='r')
    ends = np.asarray(replay['meta/episode_ends'][:], dtype=int)
    if args.episode < 0 or args.episode >= len(ends):
        parser.error(f'找不到 episode {args.episode}')
    start = int(ends[args.episode - 1]) if args.episode else 0
    end = int(ends[args.episode])
    episode_time = np.asarray(replay['data/timestamp'][start:end], dtype=float)
    episode_pos = np.asarray(replay['data/robot0_eef_pos'][start:end], dtype=float)
    episode_gripper = np.asarray(
        replay['data/robot0_gripper_width'][start:end], dtype=float).reshape(-1)

    command_times = []
    command_pos = []
    command_gripper = []
    raw_pos = []
    smoothed_pos = []
    predicted_indices = []
    cycle_indices = []
    clip_mm = []
    queued_waypoints_total = 0
    for cycle_index, cycle in enumerate(trace):
        queued_waypoints_total += len(cycle['submitted_timestamps'])
        effective_until = float('inf')
        if cycle.get('queue_policy_horizon'):
            # Future points are replaced when the next horizon is queued.
            effective_until = (trace[cycle_index + 1].get('queue_sent_at')
                               if cycle_index + 1 < len(trace)
                               else cycle['wall_time'])
        for target, target_time, index in zip(
                cycle['submitted_target_poses_7d'],
                cycle['submitted_timestamps'],
                cycle['submitted_prediction_indices']):
            if target_time > effective_until:
                continue
            raw = np.asarray(cycle['raw_target_poses_7d'][index], dtype=float)
            smoothed = np.asarray(
                cycle.get('smoothed_target_poses_7d',
                          cycle['raw_target_poses_7d'])[index], dtype=float)
            limited = np.asarray(target, dtype=float)
            command_times.append(float(target_time))
            command_pos.append(limited[:3])
            command_gripper.append(limited[6])
            raw_pos.append(raw[:3])
            smoothed_pos.append(smoothed[:3])
            predicted_indices.append(int(index))
            cycle_indices.append(cycle_index)
            clip_mm.append(float(np.linalg.norm(raw[:3] - limited[:3]) * 1000))
    command_times = np.asarray(command_times)
    command_pos = np.asarray(command_pos)
    command_gripper = np.asarray(command_gripper)
    raw_pos = np.asarray(raw_pos)
    smoothed_pos = np.asarray(smoothed_pos)
    predicted_indices = np.asarray(predicted_indices)
    clip_mm = np.asarray(clip_mm)
    actual_at_command = np.stack([
        interpolate_position(t, readback_time, readback_pos,
                             episode_time, episode_pos)
        for t in command_times
    ])
    command_error_mm = np.linalg.norm(command_pos - actual_at_command, axis=1) * 1000
    valid_errors = command_error_mm[np.isfinite(command_error_mm)]
    origin = episode_pos[0]
    t0 = episode_time[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, first, second, labels in (
            (axes[0, 0], 0, 1, ('X', 'Y')),
            (axes[0, 1], 0, 2, ('X', 'Z'))):
        actual = (episode_pos - origin) * 1000
        commands = (command_pos - origin) * 1000
        raw = (raw_pos - origin) * 1000
        ax.plot(actual[:, first], actual[:, second], label='actual TCP', lw=2)
        ax.scatter(commands[:, first], commands[:, second], label='submitted target', s=35)
        ax.scatter(raw[:, first], raw[:, second], label='raw policy target', marker='x', s=35)
        ax.set_xlabel(labels[0] + ' relative to start (mm)')
        ax.set_ylabel(labels[1] + ' relative to start (mm)')
        ax.axis('equal')
        ax.grid(True, alpha=.25)
    axes[0, 0].legend(fontsize=8)

    relative_actual = np.linalg.norm(episode_pos - origin, axis=1) * 1000
    relative_target = np.linalg.norm(command_pos - origin, axis=1) * 1000
    axes[0, 2].plot(episode_time - t0, relative_actual, label='actual distance')
    axes[0, 2].scatter(command_times - t0, relative_target, label='target distance')
    axes[0, 2].set_ylabel('distance from start (mm)')
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].plot(command_times - t0, command_error_mm, 'o-')
    axes[1, 0].set_ylabel('target minus actual at target time (mm)')
    axes[1, 1].plot(command_times - t0, clip_mm, 'o-')
    axes[1, 1].set_ylabel('raw target minus limited target (mm)')
    axes[1, 2].plot(episode_time - t0, episode_gripper * 1000,
                    label='actual gripper')
    axes[1, 2].scatter(command_times - t0, command_gripper * 1000,
                       label='submitted gripper')
    axes[1, 2].set_ylabel('gripper width (mm)')
    axes[1, 2].legend(fontsize=8)
    for ax in axes.flat[2:]:
        ax.set_xlabel('time from episode start (s)')
        ax.grid(True, alpha=.25)
    fig.suptitle('AM2Pro policy run: prediction, limiter, and measured motion')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    observation_latency_ms = np.asarray([
        (cycle['wall_time'] - cycle['observation_timestamp']) * 1000
        for cycle in trace])
    inference_ms = np.asarray([cycle['inference_seconds'] * 1000
                               for cycle in trace])
    report = {
        'schema': 'am_umi_policy_live_diagnostic_v1',
        'safety': 'analysis only; no robot or camera opened',
        'video': str(video_path),
        'plot': str(plot_path),
        'trace': str(trace_path),
        'readback': str(readback_path),
        'episode_duration_s': float(episode_time[-1] - episode_time[0]),
        'inference_cycles': len(trace),
        'submitted_waypoints': len(command_times),
        'queued_waypoints_total': queued_waypoints_total,
        'readback_samples': len(readback_time),
        'plotted_actual_source': 'full_episode_replay_buffer_10hz',
        'submitted_prediction_indices': predicted_indices.tolist(),
        'raw_direction_reversals': direction_reversals(raw_pos, cycle_indices),
        'smoothed_direction_reversals': direction_reversals(smoothed_pos, cycle_indices),
        'submitted_direction_reversals': direction_reversals(command_pos, cycle_indices),
        'inference_latency_ms_median': float(np.median(inference_ms)),
        'observation_age_at_queue_ms_median': (float(np.median([
            (cycle['queue_sent_at'] - cycle['observation_timestamp']) * 1000
            for cycle in trace if cycle.get('queue_sent_at') is not None]))
            if any(cycle.get('queue_sent_at') is not None for cycle in trace)
            else None),
        'phase_skipped_waypoints_total': sum(
            len(cycle.get('phase_skipped_indices', [])) for cycle in trace),
        'policy_execution_latency_ms': [
            cycle.get('policy_execution_latency_s', 0.01) * 1000
            for cycle in trace],
        'observation_age_at_trace_ms_median': float(np.median(observation_latency_ms)),
        'raw_to_limited_position_mm_median': float(np.median(clip_mm)),
        'raw_to_limited_position_mm_max': float(np.max(clip_mm)),
        'target_actual_position_mm_median': (float(np.median(valid_errors))
                                             if len(valid_errors) else None),
        'target_actual_position_mm_max': (float(np.max(valid_errors))
                                          if len(valid_errors) else None),
        'target_actual_position_mm_each': [
            float(x) if np.isfinite(x) else None for x in command_error_mm],
        'actual_path_length_mm': float(np.linalg.norm(
            np.diff(episode_pos, axis=0), axis=1).sum() * 1000),
        'actual_start_end_mm': float(np.linalg.norm(episode_pos[-1] - origin) * 1000),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('AM2PRO_POLICY_DIAGNOSTIC_OK')
    print('report:', report_path)
    print('plot:', plot_path)
    print('video:', video_path)
    print('median target-actual position error (mm):',
          report['target_actual_position_mm_median'])


if __name__ == '__main__':
    main()
