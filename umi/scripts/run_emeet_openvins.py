#!/usr/bin/env python3
"""Isolated offline OpenVINS trial; no ROS/Conda installation on the host."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description='使用 Docker 离线测试手持夹爪 OpenVINS，不连接硬件')
    p.add_argument('--session-dir', required=True)
    p.add_argument('--output-dir', required=True, help='必须是尚不存在的新目录')
    p.add_argument('--config-dir', default=str(root / 'calibration/handheld_gripper_camera/openvins/emeet_jy901b_v1'))
    p.add_argument('--image', default='am_umi_openvins_offline:20260905')
    p.add_argument('--speed', type=float, default=0.5, help='回放速度，0.5 为半速；不改变传感器时间戳')
    p.add_argument('--max-seconds', type=float, default=0, help='视频时间上限，0 为全部')
    p.add_argument('--without-mask', action='store_true', help='仅用于比较，不屏蔽夹爪')
    p.add_argument('--gripper-envelope', action='store_true', help='额外屏蔽本验证视频中的左右夹爪与标记，不适用于任意安装位置')
    p.add_argument('--gdb', action='store_true', help='仅调试镜像：保存崩溃调用栈')
    args = p.parse_args()
    session, out, cfg = Path(args.session_dir).resolve(), Path(args.output_dir).resolve(), Path(args.config_dir).resolve()
    if not 0 < args.speed <= 2 or args.max_seconds < 0:
        p.error('speed 应在 (0, 2]，max-seconds 应 >= 0')
    for name in ('raw_video.mp4', 'frame_timestamps_uvc_source.csv', 'imu_data_strict_uvc.json'):
        if not (session / name).is_file():
            p.error(f'缺少输入 {session / name}')
    if out.exists():
        p.error(f'拒绝覆盖已有目录: {out}')
    inspection = subprocess.run(['docker', 'image', 'inspect', args.image], check=True, capture_output=True, text=True)
    image_id = json.loads(inspection.stdout)[0]['Id']
    out.mkdir(parents=True)
    name = f'am_umi_openvins_trial_{os.getpid()}'
    cmd = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--cpus', '2',
           '--memory', '4g', '--memory-swap', '4g', '--pids-limit', '256',
           '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'ROS_LOCALHOST_ONLY=1',
           '-e', f'AM_UMI_DOCKER_IMAGE_ID={image_id}',
           '-e', 'ROS_LOG_DIR=/output/ros_logs', '-e', 'OMP_NUM_THREADS=2',
           '-e', 'OPENBLAS_NUM_THREADS=1']
    for src, dst, mode in [(session, '/session', 'ro'), (cfg, '/input-config', 'ro'),
                           (root / 'third_party/open_vins', '/tools', 'ro'),
                           (root / 'calibration/handheld_gripper_camera/slam/emeet_handheld_gripper_mask_960x540_v1.png', '/mask.png', 'ro'),
                           (out, '/output', 'rw')]:
        cmd += ['-v', f'{src}:{dst}:{mode}']
    cmd += [args.image, '/bin/bash', '-c',
            'source /opt/ros/humble/setup.bash && source /workspace/install/setup.bash && exec python3 /tools/am_umi_offline_trial.py "$@"',
            'trial', '--speed', str(args.speed), '--max-seconds', str(args.max_seconds)]
    if args.without_mask:
        cmd.append('--without-mask')
    if args.gripper_envelope:
        cmd.append('--gripper-envelope')
    if args.gdb:
        cmd.append('--gdb')
    print(f'开始离线测试（CPU 2 核，内存最多 4 GiB，不使用 GPU）\n输出: {out}', flush=True)
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print('正在停止本次测试容器；保留已有日志。', flush=True)
        subprocess.run(['docker', 'stop', '--timeout', '10', name], check=False)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
