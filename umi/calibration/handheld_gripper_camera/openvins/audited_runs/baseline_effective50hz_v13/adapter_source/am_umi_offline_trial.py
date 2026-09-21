#!/usr/bin/env python3
"""Container-only supervisor. Inputs read-only; all generated files in /output."""
import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np


def stop(child, under_gdb=False):
    if child.poll() is None:
        if under_gdb:
            children = pathlib.Path(f'/proc/{child.pid}/task/{child.pid}/children').read_text().split()
            if len(children) != 1:
                raise RuntimeError('无法唯一定位调试器中的测试进程')
            os.kill(int(children[0]), signal.SIGINT)
        else:
            child.send_signal(signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-seconds', type=float, default=0)
    p.add_argument('--speed', type=float, default=0.5)
    p.add_argument('--without-mask', action='store_true', help='diagnostic ablation only')
    p.add_argument('--gripper-envelope', action='store_true', help='mask the fingers and tags visible outside the legacy center mask')
    p.add_argument('--gdb', action='store_true', help='requires isolated debug image')
    args = p.parse_args()
    out = pathlib.Path('/output')
    if any(out.iterdir()):
        raise SystemExit('拒绝覆盖非空测试输出目录')
    source = pathlib.Path('/input-config')
    cfg = out / 'config'
    cfg.mkdir()
    adapter_snapshot = out / 'adapter_source'
    adapter_snapshot.mkdir()
    for name in ('am_umi_offline_trial.py', 'am_umi_replay.py'):
        shutil.copyfile(pathlib.Path('/tools') / name, adapter_snapshot / name)
    provenance = pathlib.Path('/session/preparation_report.json')
    timing_report = pathlib.Path('/session/uvc_payload_timing_report.json')
    if provenance.is_file():
        shutil.copyfile(provenance, out / 'input_preparation_report.json')
    input_timing = json.loads(timing_report.read_text()) if timing_report.is_file() else {}
    estimator = (source / 'estimator_config.yaml').read_text()
    estimator = re.sub(r'^(save_total_state|filepath_est|filepath_std|filepath_gt):.*\n', '', estimator, flags=re.M)
    estimator = re.sub(r'^record_timing_filepath:.*$', 'record_timing_filepath: "/output/timing.txt"', estimator, flags=re.M)
    # No GUI subscribers during this offline run. Avoid the detached image
    # publisher thread (and its shutdown lifetime) entirely.
    estimator += '\nmulti_threading_pubs: false\n'
    estimator = estimator.replace('use_mask: false', 'use_mask: ' + ('false' if args.without_mask else 'true') + '\nmask0: "mask.png"')
    (cfg / 'estimator_config.yaml').write_text(estimator)
    chain = (source / 'kalibr_imucam_chain.yaml').read_text()
    chain = chain.replace('resolution: [1920, 1080]', 'resolution: [960, 540]')
    match = re.search(r'intrinsics: \[([^\]]+)\]', chain)
    intrinsics = [float(x) / 2 for x in match[1].split(',')]
    chain = chain[:match.start()] + 'intrinsics: ' + str(intrinsics) + chain[match.end():]
    if 'timeshift_cam_imu:' not in chain:
        chain += '  timeshift_cam_imu: 0.0\n'
    (cfg / 'kalibr_imucam_chain.yaml').write_text(chain)
    shutil.copyfile(source / 'kalibr_imu_chain.yaml', cfg / 'kalibr_imu_chain.yaml')
    mask = cv2.imread('/mask.png', cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (540, 960):
        raise RuntimeError('遮罩尺寸错误')
    if args.gripper_envelope:
        # Pixel coordinates at 960x540, conservatively covering the rigid
        # fingers/white housings and their tags in this validation session.
        # This is a session-specific diagnostic ROI, not a universal rig mask.
        left = np.array([[100,540],[175,330],[230,305],[400,165],[450,165],[410,540]], np.int32)
        right = np.array([[595,165],[650,165],[785,300],[840,320],[910,540],[640,540]], np.int32)
        cv2.fillPoly(mask, [left, right], 255)
    cv2.imwrite(str(cfg / 'mask.png'), mask)
    # Save visual evidence of the exact mask convention: white = exclude.
    cap = cv2.VideoCapture('/session/raw_video.mp4')
    ok, frame = cap.read()
    cap.release()
    if not ok or frame.shape[:2] != (1080, 1920):
        raise RuntimeError('视频尺寸不匹配内参')
    frame = cv2.resize(frame, (960, 540))
    frame[mask > 127] = frame[mask > 127] // 2 + np.array([0, 0, 127], dtype=np.uint8)
    cv2.imwrite(str(out / 'mask_preview.jpg'), frame)
    cmd = ['/workspace/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf', str(cfg / 'estimator_config.yaml'),
           '--ros-args', '-p', 'save_total_state:=true', '-p', 'filepath_est:=/output/state_estimate.txt',
           '-p', 'filepath_std:=/output/state_std.txt']
    if args.gdb:
        cmd = ['gdb', '--batch', '-ex', 'set pagination off', '-ex', 'handle SIGINT nostop noprint pass',
               '-ex', 'run', '-ex', 'thread apply all bt', '--args'] + cmd
    replay_cmd = [sys.executable, str(adapter_snapshot / 'am_umi_replay.py'), '--session-dir', '/session',
                  '--speed', str(args.speed), '--max-seconds', str(args.max_seconds), '--flush-seconds', '5']
    hashes = {}
    for file in [*cfg.iterdir(), pathlib.Path('/tools/am_umi_replay.py'), pathlib.Path('/tools/am_umi_offline_trial.py'),
                 pathlib.Path('/session/frame_timestamps_uvc_source.csv'), pathlib.Path('/session/imu_data_strict_uvc.json')]:
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (out / 'run_manifest.json').write_text(json.dumps(dict(arguments=vars(args), node_command=cmd,
        replay_command=replay_cmd, sha256=hashes, opencv=cv2.__version__,
        docker_image_id=os.environ.get('AM_UMI_DOCKER_IMAGE_ID')), indent=2)+'\n')
    started = time.monotonic()
    with (out / 'openvins_node.log').open('w') as nf, (out / 'replay.log').open('w') as rf:
        node = subprocess.Popen(cmd, stdout=nf, stderr=subprocess.STDOUT)
        replay = None
        try:
            replay = subprocess.Popen(replay_cmd, stdout=rf, stderr=subprocess.STDOUT)
            deadline = started + 120 + (args.max_seconds or 125) / args.speed * 3
            while replay.poll() is None:
                if node.poll() is not None:
                    raise RuntimeError(f'OpenVINS 提前退出: {node.returncode}')
                if time.monotonic() > deadline:
                    raise RuntimeError('离线回放超时')
                time.sleep(0.5)
            if replay.returncode:
                raise RuntimeError(f'回放失败: {replay.returncode}，查看 replay.log')
        finally:
            if replay is not None:
                stop(replay)
            stop(node, args.gdb)
    state_path = out / 'state_estimate.txt'
    rows = np.loadtxt(state_path, comments='#', ndmin=2) if state_path.is_file() else np.empty((0, 0))
    node_exit = node.returncode
    if args.gdb and 'received signal SIGSEGV' in (out / 'openvins_node.log').read_text():
        node_exit = -11
    result = dict(node_exit=node_exit, replay_exit=replay.returncode, states=len(rows),
                  input_timing_associations_over_50ms=input_timing.get('associations_over_50ms'),
                  input_timing_matching=input_timing.get('frame_event_matching'),
                  elapsed_wall_s=time.monotonic() - started, mask_enabled=not args.without_mask,
                  gripper_envelope=args.gripper_envelope,
                  status='NO_STATE' if not len(rows) else 'REQUIRES_TRAJECTORY_VALIDATION')
    if len(rows):
        pos, vel = rows[:, 5:8], rows[:, 8:11]
        with pathlib.Path('/session/frame_timestamps_uvc_source.csv').open() as file:
            camera_times = np.array([float(r['uvc_source_relative_s']) for r in csv.DictReader(file)])
        if args.max_seconds > 0:
            camera_times = camera_times[camera_times <= args.max_seconds]
        state_times = rows[:, 0] - 1_700_000_000
        post_init = camera_times[camera_times >= state_times[0] - 0.00002]
        index = np.clip(np.searchsorted(state_times, post_init), 0, len(state_times)-1)
        previous = np.maximum(index - 1, 0)
        matches = np.minimum(abs(state_times[index]-post_init), abs(state_times[previous]-post_init)) < 0.00002
        result.update(finite=bool(np.isfinite(rows).all()),
                      expected_post_init_camera_frames=len(post_init),
                      matched_post_init_camera_frames=int(matches.sum()),
                      post_init_coverage=float(matches.mean()) if len(matches) else 0.0,
                      state_duration_s=float(rows[-1, 0] - rows[0, 0]),
                      position_span_m=np.ptp(pos, axis=0).tolist(),
                      max_displacement_m=float(np.linalg.norm(pos - pos[0], axis=1).max()),
                      max_speed_mps=float(np.linalg.norm(vel, axis=1).max()))
        if not result['finite'] or result['max_displacement_m'] > 10 or result['max_speed_mps'] > 10:
            result['status'] = 'DIVERGED_FOR_TABLETOP_SESSION'
    if node_exit != 0:
        result['lifecycle_status'] = 'NODE_EXIT_ERROR'
    else:
        result['lifecycle_status'] = 'CLEAN_EXIT'
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] == 'DIVERGED_FOR_TABLETOP_SESSION':
        return 2
    return 0 if len(rows) and node_exit == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
