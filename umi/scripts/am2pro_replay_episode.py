"""
Replay a recorded demo episode on the AM2Pro arm.

The recorded trajectory lives in the SLAM tag frame, which is NOT aligned
with the robot base frame. Instead of requiring hand-eye alignment, we replay
the RELATIVE motion: each frame's pose delta vs the episode's first frame is
applied on top of the robot's CURRENT pose. This reproduces the exact demo
motion in the robot's own frame.

Usage (umi env):
    python scripts/am2pro_replay_episode.py -rc example/eval_robots_config.yaml \
        --dataset imu_work/demo_batch3_dataset.zarr --episode 0 \
        [--fps 30] [--speed 1.0] [--return]

Keys during replay: Ctrl+C stops immediately.
"""
import json
import os
import sys
import threading
import time

import click
import cv2
import numpy as np
import yaml
import zarr

ROOT = __file__.rsplit('/', 2)[0]
sys.path.insert(0, ROOT)

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs  # noqa: E402
from umi.common.pose_util import pose_to_mat  # noqa: E402
from umi.real_world.am2pro_interpolation_controller import (  # noqa: E402
    AM2ProInterpolationController)

register_codecs()  # required to decode the JpegXl-compressed images in the zarr


@click.command()
@click.option('-rc', '--robot_config', required=True, help='eval_robots_config.yaml')
@click.option('--dataset', required=True, help='zarr dataset (training zip or eval dir)')
@click.option('--episode', type=int, default=0,
              help='single episode index to replay (ignored when --episodes is set)')
@click.option('--episodes', default=None,
              help='comma-separated episodes to replay continuously, e.g. 0,1,2. '
                   'Each later segment is rigidly aligned to the preceding endpoint '
                   'to remove SLAM relocalization jumps.')
@click.option('--fps', type=float, default=30.0, help='replay control rate')
@click.option('--source-fps', type=float, default=30.0, show_default=True,
              help='源 zarr 帧率；当前固定 Tag 手持 demo 为 30。旧 60 FPS 数据需显式设为 60')
@click.option('--speed', type=float, default=1.0, help='time scale (1.0 = real time)')
@click.option('--return', 'do_return', is_flag=True, default=False,
              help='move back to the starting pose after replay')
@click.option('--hold-gripper', is_flag=True, default=False,
              help='首测时保持当前夹爪开度，不执行 demo 的夹爪开合')
@click.option('--record', default=None, help='record the camera view to this mp4 during replay')
@click.option('--camera', default='/dev/video2', help='camera device for --record')
@click.option('--state-log', default=None,
              help='save target/actual robot states to this .npz; defaults to '
                   '<record basename>.npz when --record is used')
@click.option('--start-wrist-flex-deg', type=float, default=None,
              help='before replay, move ONLY wrist_flex to this absolute degree value. '
                   'Allowed safe setup range: -80 to +70. Omit to leave it unchanged.')
@click.option('--start-shoulder-lift-deg', type=click.FloatRange(-90.0, 0.0),
              default=None,
              help='before replay, move ONLY shoulder_lift to this absolute degree '
                   'value. Use a value above the physical -100 degree lower limit '
                   'to reserve downward reach; omit to leave it unchanged.')
@click.option('--start-reference', default=None,
              help='保存的 right_tcp reference.json；与 --move-to-reference 一起使用')
@click.option('--move-to-reference', is_flag=True, default=False,
              help='实际低速移动全部六个关节到 --start-reference 保存的关节基准')
@click.option('--reference-joint-duration', type=click.FloatRange(min=1.0), default=8.0,
              show_default=True, help='回到基准的最短时长（秒）')
@click.option('--reference-max-joint-speed-deg-s', type=click.FloatRange(min=0.1), default=10.0,
              show_default=True, help='回到基准时各关节的最大平均速度')
@click.option('--reference-tolerance-deg', type=click.FloatRange(min=0.1), default=4.0,
              show_default=True, help='基准到位回读容差；超出则中止 replay')
@click.option('--start-joint-duration', type=float, default=3.0,
              help='seconds for each optional joint setup move')
@click.option('--ik-iterations', type=click.IntRange(1, 5), default=1,
              show_default=True,
              help='IK solve steps per 50 Hz control tick. One step prevents wrist '
                   'over-correction during replay; use larger values only after '
                   'a stable short test.')
@click.option('--ik-orientation-weight', type=click.FloatRange(0.05, 1.0),
              default=0.35, show_default=True,
              help='relative orientation priority for replay IK. A lower value '
                   'keeps the wrist continuous near sensitive configurations.')
@click.option('--wrist-flex-min-deg', type=click.FloatRange(0.0, 70.0),
              default=20.0, show_default=True,
              help='keep wrist_flex on its positive branch and at least this many '
                   'degrees from the 0-degree wrist singularity.')
@click.option('--wrist-flex-max-deg', type=click.FloatRange(20.0, 85.0),
              default=70.0, show_default=True,
              help='keep wrist_flex below this value and away from its positive '
                   'joint limit during replay.')
@click.option('--max-tcp-speed', type=click.FloatRange(min=0.001), default=0.06,
              show_default=True,
              help='maximum commanded TCP translation speed in m/s. Filters SLAM '
                   'trajectory spikes; 0.06 is deliberately conservative for a '
                   'first physical replay.')
@click.option('--max-tcp-rot-speed-deg', type=click.FloatRange(min=1.0), default=30.0,
              show_default=True,
              help='maximum commanded TCP angular speed in deg/s. Filters abrupt '
                   'orientation spikes in the reconstructed trajectory.')
@click.option('--yaw-offset-deg', type=float, default=-90.0,
              help='yaw (deg) between the SLAM tag frame and the robot base frame '
                   '(verified default for this robot/desk layout: -90)')
@click.option('--mirror-x', is_flag=True, default=True,
              help='reflect the demo frame x axis before alignment '
                   '(verified default for this robot/desk layout: ON)')
@click.option('--max-frames', type=int, default=None,
              help='stop after N frames (quick direction test; use with 2s previews)')
@click.option('--start-yaw-deg', type=float, default=0.0,
              help='rotate the START pose orientation about z before replay. '
                   'Compensates a deployment camera mounted 180 deg rotated vs the '
                   'handheld demo camera (view-matching then yaws the arm half a turn).')
@click.option('--scale', type=float, default=1.0,
              help='scale the demo motion amplitude (x/y/z translation). The human hand '
                   'works in a ~60cm radius but this arm reaches ~40cm; use 0.4-0.6 so the '
                   'replay fits the workspace. Shape is preserved, orientation is not scaled.')
def main(robot_config, dataset, episode, episodes, fps, source_fps, speed, do_return, hold_gripper, record, camera,
         state_log, start_wrist_flex_deg, start_shoulder_lift_deg,
         start_reference, move_to_reference, reference_joint_duration,
         reference_max_joint_speed_deg_s, reference_tolerance_deg,
         start_joint_duration, yaw_offset_deg, mirror_x, max_frames,
         start_yaw_deg, scale, ik_iterations, ik_orientation_weight,
         wrist_flex_min_deg, wrist_flex_max_deg, max_tcp_speed,
         max_tcp_rot_speed_deg):
    cfg = yaml.safe_load(open(robot_config))
    rc = cfg['robots'][0]
    assert rc['robot_type'] == 'am2pro'

    reference_q = None
    if start_reference is not None:
        with open(os.path.expanduser(start_reference), encoding='utf-8') as file:
            reference_data = json.load(file)
        if reference_data.get('tcp_frame') != 'right_tcp':
            raise click.UsageError('--start-reference 必须是新夹爪 right_tcp 的 reference.json')
        reference_q = np.asarray(
            reference_data.get('robot_state', {}).get('ActualQ', [])[:6], dtype=float)
        if reference_q.shape != (6,) or not np.all(np.isfinite(reference_q)):
            raise click.UsageError('--start-reference 缺少有效 robot_state/ActualQ')
    if move_to_reference and reference_q is None:
        raise click.UsageError('--move-to-reference 必须同时提供 --start-reference')
    # A recorded UMI trajectory is body-relative.  Starting it from an
    # arbitrary pose is unsafe and makes the scene-relative demonstration
    # meaningless, so physical replay is deliberately opt-in only after the
    # saved deployment reference has been reached while torque remains on.
    if not move_to_reference:
        raise click.UsageError(
            '为安全起见，AM2Pro replay 必须使用 --move-to-reference，'
            '并提供对应的 --start-reference；脚本会自动回到基准后连续执行。')

    # load replay buffer (ZipStore file for training data, DirectoryStore for eval)
    if os.path.isdir(dataset):
        rb = ReplayBuffer.create_from_path(dataset, mode='r')
    else:
        store = zarr.ZipStore(dataset, mode='r')
        rb = ReplayBuffer.copy_from_store(store, zarr.MemoryStore())

    episode_ids = _parse_episode_ids(episode, episodes, rb.n_episodes)
    demo_poses, width, source_episode, episode_frame_idx, boundaries = \
        _load_stitched_episodes(rb, episode_ids)
    n = len(demo_poses)
    print(f"selected episodes {episode_ids}: {n} frames "
          f"({(n - 1) / source_fps:.2f}s at {source_fps:.1f} FPS), width range "
          f"[{width.min():.3f}, {width.max():.3f}] m")
    for boundary in boundaries:
        print("  stitched boundary "
              f"{boundary['from_episode']}->{boundary['to_episode']}: raw jump "
              f"{boundary['position_jump_m'] * 1000:.1f} mm, "
              f"{boundary['rotation_jump_deg']:.1f} deg -> aligned continuously")
    if fps <= 0 or source_fps <= 0 or speed <= 0:
        raise click.UsageError('--fps, --source-fps and --speed must be positive')
    if scale <= 0:
        raise click.UsageError('--scale must be positive')
    if start_joint_duration <= 0:
        raise click.UsageError('--start-joint-duration must be positive')
    if (start_wrist_flex_deg is not None and
            not -80.0 <= start_wrist_flex_deg <= 70.0):
        raise click.UsageError(
            '--start-wrist-flex-deg must be within the conservative setup range [-80, 70]')
    if wrist_flex_min_deg > wrist_flex_max_deg:
        raise click.UsageError(
            '--wrist-flex-min-deg cannot exceed --wrist-flex-max-deg')

    # A video without corresponding command/state telemetry is difficult to
    # diagnose: a bad replay may be caused by the source trajectory, IK, or
    # servo tracking. Make the diagnostic log automatic whenever video is
    # requested, while still allowing a caller to select an explicit path.
    if state_log is None and record:
        state_log = os.path.splitext(record)[0] + '.npz'

    with AM2ProInterpolationController(
        robot_usb_port=rc.get('robot_usb_port', '/dev/ttyACM0'),
        frequency=50,
        receive_latency=rc.get('robot_obs_latency', 0.005),
        server_python=rc.get('robot_python'),
        ik_backend=rc.get('ik_backend', 'ros2_dh'),
        urdf_path=rc.get('urdf_path'),
        flip_joints=rc.get('joint_flip', []),
        joint_model_candidate=rc.get('joint_model_candidate'),
        gripper_width_min=rc.get('gripper_width_min', 0.0),
        gripper_width_max=rc.get('gripper_width_max', 0.074),
        gripper_servo_closed=rc.get('gripper_servo_closed', 5.3),
        gripper_servo_open=rc.get('gripper_servo_open', 93.6),
        gripper_width_servo_table=rc.get('gripper_width_servo_table'),
        ik_iterations=ik_iterations,
        ik_orientation_weight=ik_orientation_weight,
        wrist_flex_min_deg=wrist_flex_min_deg,
        wrist_flex_max_deg=wrist_flex_max_deg,
        verbose=True,
    ) as robot:
        # wait for first state
        state = None
        for _ in range(100):
            state = robot.get_state()
            if state is not None:
                break
            time.sleep(0.05)
        if state is None:
            raise SystemExit("no state from robot server")

        if move_to_reference:
            current_q = np.asarray(state['ActualQ'][:6], dtype=float)
            max_delta = float(np.max(np.abs(reference_q - current_q)))
            duration = max(float(reference_joint_duration),
                           max_delta / float(reference_max_joint_speed_deg_s))
            print("REFERENCE PREPOSITION: moving all six joints to saved right_tcp reference "
                  f"over {duration:.1f}s (largest delta {max_delta:.1f} deg; "
                  f"cap {reference_max_joint_speed_deg_s:.1f} deg/s).")
            print("  current:", np.round(current_q, 1))
            print("  target :", np.round(reference_q, 1))
            robot.servoJ(reference_q, duration=duration)
            # Do not declare success partway through the smooth joint move.
            # At e.g. 10 s of a commanded 12 s move the target is still
            # intentionally changing, so a loose tolerance can otherwise
            # cause replay to begin before reaching the saved reference.
            motion_done_at = time.time() + duration
            deadline = motion_done_at + 5.0
            latest = state
            while time.time() < deadline:
                time.sleep(0.1)
                candidate = robot.get_state()
                if candidate is not None:
                    latest = candidate
                    error = float(np.max(np.abs(
                        np.asarray(latest['ActualQ'][:6], dtype=float) - reference_q)))
                    if time.time() >= motion_done_at and error <= reference_tolerance_deg:
                        break
            final_error = float(np.max(np.abs(
                np.asarray(latest['ActualQ'][:6], dtype=float) - reference_q)))
            final_joint_error = np.asarray(latest['ActualQ'][:6], dtype=float) - reference_q
            print("  readback:", np.round(np.asarray(latest['ActualQ'][:6], dtype=float), 1))
            print("  error   :", np.round(final_joint_error, 2), "deg (actual - target)")
            if final_error > reference_tolerance_deg:
                raise SystemExit(
                    f'reference preposition did not converge: max joint error '
                    f'{final_error:.1f} deg > {reference_tolerance_deg:.1f} deg. Replay aborted.')
            state = latest
            print(f"REFERENCE PREPOSITION reached; max readback error {final_error:.2f} deg.")

        def preposition_one_joint(current_state, name, joint_index, target_deg):
            """Move just one joint, confirm readback, and return fresh state."""
            target_q = np.asarray(current_state['ActualQ'][:6], dtype=float).copy()
            target_q[joint_index] = float(target_deg)
            print("PRE-POSITION: keeping five joints at their current readings; "
                  f"moving {name} {current_state['ActualQ'][joint_index]:.1f} -> "
                  f"{target_deg:.1f} deg over {start_joint_duration:.1f}s.")
            robot.servoJ(target_q, duration=start_joint_duration)
            # Wait through the commanded motion, then require a close readback
            # before treating its achieved TCP pose as the replay start.
            deadline = time.time() + start_joint_duration + 5.0
            latest = current_state
            while time.time() < deadline:
                time.sleep(0.1)
                latest = robot.get_state()
                if (latest is not None and
                        abs(float(latest['ActualQ'][joint_index]) - target_deg) <= 3.0):
                    break
            if latest is None:
                raise SystemExit(f'lost robot state during {name} pre-position')
            joint_error = abs(float(latest['ActualQ'][joint_index]) - target_deg)
            if joint_error > 3.0:
                raise SystemExit(
                    f'{name} did not reach its setup target: actual '
                    f'{latest["ActualQ"][joint_index]:.1f} deg, requested '
                    f'{target_deg:.1f} deg. Replay aborted.')
            print(f"PRE-POSITION reached: {name}="
                  f"{latest['ActualQ'][joint_index]:.1f} deg")
            return latest

        if start_shoulder_lift_deg is not None:
            state = preposition_one_joint(
                state, 'shoulder_lift', 1, start_shoulder_lift_deg)
        if start_wrist_flex_deg is not None:
            # Direct joint command is deliberate: a Cartesian pose command
            # cannot guarantee which redundant wrist configuration IK chooses.
            state = preposition_one_joint(
                state, 'wrist_flex', 3, start_wrist_flex_deg)

        start_pose = state['ActualTCPPose'].astype(float)
        T_cur = pose_to_mat(start_pose)
        # start-yaw compensation: rotate the effective start orientation
        # about the world z axis (fixes camera-mount 180-deg flips, etc.)
        if abs(start_yaw_deg) > 1e-6:
            th = np.deg2rad(start_yaw_deg)
            Rz = np.eye(4)
            Rz[:3, :3] = np.array([
                [np.cos(th), -np.sin(th), 0],
                [np.sin(th),  np.cos(th), 0],
                [0, 0, 1]])
            T_cur = Rz @ T_cur
            print(f"start-yaw compensation: {start_yaw_deg} deg")
        # FULLY body-relative replay: T_target = T_cur @ (T_0^-1 @ T_i).
        # Frame-convention invariant -> no tag<->base alignment needed at all.
        # Requires the user to pose the arm like the demo start frame
        # (same location relative to the scene, similar hand orientation).
        T0_inv = np.linalg.inv(demo_poses[0])
        print(f"current TCP pose: {np.round(start_pose, 4)}")
        start_q = state['ActualQ'][:6]
        print(f"current joints  : {np.round(start_q, 1)}  "
              f"(wrist_flex={start_q[3]:.1f}  wrist_yaw={start_q[4]:.1f}  wrist_roll={start_q[5]:.1f})")
        if abs(start_q[3]) < 15:
            print("WARNING: wrist_flex near 0 (yaw/roll axes almost aligned) -> "
                  "the IK may wind up the wrist; start with the wrist visibly bent instead.")
        print("REPLAY MODE: fully body-relative (no alignment). POSE THE ARM like "
              "the demo start: same gripper location relative to the scene, similar "
              "hand orientation (watch demo video frame 0).")
        print(f"replay IK: {ik_iterations} step/tick, orientation weight "
              f"{ik_orientation_weight:.2f}, wrist_flex >= "
              f"{wrist_flex_min_deg:.1f} and <= {wrist_flex_max_deg:.1f} deg")
        print(f"trajectory limiter: {max_tcp_speed:.3f} m/s, "
              f"{max_tcp_rot_speed_deg:.1f} deg/s")
        if hold_gripper:
            print("REPLAY SAFETY: holding current gripper opening; demo gripper commands disabled")

        # Optional camera recording runs in a dedicated thread.  UVC reads can
        # block for tens or hundreds of milliseconds; keeping them out of this
        # control loop prevents recording from turning smooth servo commands
        # into stop-and-go motion.
        recorder = None
        replay_samples = []
        video_frame_timestamps = []
        if record:
            record = os.path.abspath(os.path.expanduser(record))
            os.makedirs(os.path.dirname(record), exist_ok=True)
            recorder = _AsyncVideoRecorder(camera, record, fps=30.0)
            recorder.start()
            print(f"recording camera -> {record}")

        def grab_frame():
            """Return the latest asynchronously captured frame timestamp."""
            return recorder.latest_timestamp() if recorder is not None else np.nan

        def log_replay_state(source_frame_idx, source_episode_id,
                             source_episode_frame_idx, target_pose, target_width,
                             command_time, target_time, camera_time):
            """Capture the actual robot state adjacent to one replay command."""
            observed = robot.get_state()
            actual_pose = np.full(6, np.nan, dtype=np.float64)
            actual_q = np.full(7, np.nan, dtype=np.float64)
            actual_qd = np.full(7, np.nan, dtype=np.float64)
            actual_gripper = np.nan
            robot_time = np.nan
            if observed is not None:
                actual_pose = np.asarray(observed['ActualTCPPose'], dtype=np.float64).copy()
                actual_q = np.asarray(observed['ActualQ'], dtype=np.float64).copy()
                actual_qd = np.asarray(observed['ActualQd'], dtype=np.float64).copy()
                actual_gripper = float(observed.get('gripper_position', np.nan))
                robot_time = float(observed.get('robot_timestamp', np.nan))
            replay_samples.append({
                'source_frame_idx': int(source_frame_idx),
                'source_episode': int(source_episode_id),
                'source_episode_frame_idx': int(source_episode_frame_idx),
                'demo_time': float(source_frame_idx / source_fps),
                'command_time': float(command_time),
                'target_time': float(target_time),
                'camera_time': float(camera_time),
                'target_pose': np.asarray(target_pose, dtype=np.float64).copy(),
                'target_gripper_width': float(target_width),
                'actual_pose': actual_pose,
                'actual_q': actual_q,
                'actual_qd': actual_qd,
                'actual_gripper_width': actual_gripper,
                'robot_time': robot_time,
            })

        print("REPLAYING (Ctrl+C to stop). The arm will follow the demo motion.")

        dt = 1.0 / fps
        command_period = dt / speed
        next_cycle_deadline = time.monotonic() + command_period
        previous_target = None
        limited_steps = 0
        missed_deadlines = 0
        max_deadline_lag = 0.0
        try:
            # Keep the requested replay cadence while sampling the source
            # trajectory at its declared rate. Current hand-held Zarr files
            # are 30 FPS; older imported UMI datasets may be 60 FPS.
            k = max(1, int(round(source_fps / fps)))
            for i in range(0, n, k):
                # fully body-relative: T_target = T_cur @ (T_0^-1 @ T_i)
                M_body = T0_inv @ demo_poses[i]
                if abs(scale - 1.0) > 1e-6:
                    M_body = M_body.copy()
                    M_body[:3, 3] *= float(scale)   # scale translation only
                T_target = T_cur @ M_body
                target = np.concatenate([T_target[:3, 3],
                                         _rotvec(T_target[:3, :3])])
                if previous_target is not None:
                    target, was_limited = _limit_pose_step(
                        previous_target, target,
                        max_translation=max_tcp_speed * dt / speed,
                        max_rotation=np.deg2rad(max_tcp_rot_speed_deg) * dt / speed)
                    limited_steps += int(was_limited)
                previous_target = target.copy()
                w = float(state.get('gripper_position', width[i])) if hold_gripper else float(width[i])
                command_time = time.time()
                target_time = command_time + command_period
                if not hold_gripper:
                    robot.schedule_gripper(w, target_time=target_time)
                robot.servoL(target, duration=command_period)
                camera_time = grab_frame()
                log_replay_state(
                    i, source_episode[i], episode_frame_idx[i], target, w,
                    command_time, target_time, camera_time)
                # periodic wrist diagnostic (every 1s of demo time)
                if i % max(1, int(round(source_fps))) == 0:
                    s = robot.get_state()
                    if s is not None:
                        print(f"  [t={i/60:.1f}s] wrist_roll={s['ActualQ'][5]:7.1f} deg",
                              flush=True)
                remaining = next_cycle_deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    missed_deadlines += 1
                    max_deadline_lag = max(max_deadline_lag, -remaining)
                next_cycle_deadline += command_period
                # Do not burst several stale commands after a rare long OS or
                # serial stall. Rebase the clock and resume at the requested
                # period instead.
                if next_cycle_deadline < time.monotonic():
                    next_cycle_deadline = time.monotonic() + command_period
                if max_frames is not None and i >= max_frames:
                    print(f"Stopped after {max_frames} frames (quick test).")
                    break
            print("Replay finished.")
            if limited_steps:
                print(f"trajectory limiter smoothed {limited_steps} command steps")
            print(f"control timing: {missed_deadlines} missed deadlines, "
                  f"max lag {max_deadline_lag * 1000:.1f} ms")
        except KeyboardInterrupt:
            print("Replay interrupted by user.")

        if do_return:
            print("Returning to starting pose ...")
            robot.servoL(start_pose, duration=2.0)
            time.sleep(2.5)  # background recorder captures the return move
            print("Returned.")

        if recorder is not None:
            video_frame_timestamps = recorder.close()
            print(f"camera recording saved: {record} "
                  f"({len(video_frame_timestamps)} frames)")
        if state_log is not None:
            state_log = os.path.abspath(os.path.expanduser(state_log))
            os.makedirs(os.path.dirname(state_log), exist_ok=True)
            np.savez_compressed(
                state_log,
                source_frame_idx=np.array([x['source_frame_idx'] for x in replay_samples]),
                source_episode=np.array([x['source_episode'] for x in replay_samples]),
                source_episode_frame_idx=np.array(
                    [x['source_episode_frame_idx'] for x in replay_samples]),
                demo_time=np.array([x['demo_time'] for x in replay_samples]),
                command_time=np.array([x['command_time'] for x in replay_samples]),
                target_time=np.array([x['target_time'] for x in replay_samples]),
                camera_time=np.array([x['camera_time'] for x in replay_samples]),
                target_tcp_pose=np.stack([x['target_pose'] for x in replay_samples]),
                target_gripper_width=np.array(
                    [x['target_gripper_width'] for x in replay_samples]),
                actual_tcp_pose=np.stack([x['actual_pose'] for x in replay_samples]),
                actual_joint_pos=np.stack([x['actual_q'] for x in replay_samples]),
                actual_joint_vel=np.stack([x['actual_qd'] for x in replay_samples]),
                actual_gripper_width=np.array(
                    [x['actual_gripper_width'] for x in replay_samples]),
                robot_time=np.array([x['robot_time'] for x in replay_samples]),
                video_frame_time=np.array(video_frame_timestamps),
            )
            print(f"replay state log saved: {state_log} ({len(replay_samples)} samples)")


def _parse_episode_ids(single_episode, episodes, n_episodes):
    """Resolve and validate the requested episode sequence."""
    if episodes is None:
        selected = [int(single_episode)]
    else:
        try:
            selected = [int(x.strip()) for x in episodes.split(',') if x.strip()]
        except ValueError as exc:
            raise click.UsageError(
                '--episodes must be a comma-separated list of integers') from exc
        if not selected:
            raise click.UsageError('--episodes must contain at least one episode')
        if len(set(selected)) != len(selected):
            raise click.UsageError('--episodes must not contain duplicates')
    invalid = [x for x in selected if x < 0 or x >= n_episodes]
    if invalid:
        raise click.UsageError(
            f'dataset has {n_episodes} episodes; invalid selection: {invalid}')
    return selected


def _load_stitched_episodes(replay_buffer, episode_ids):
    """Load episodes and rigidly align each boundary for continuous replay.

    Dataset generation splits a demo whenever SLAM briefly loses tracking.
    Relocalization can place the next segment in a slightly shifted map frame.
    Left-multiplying the complete next segment so its first pose equals the
    preceding endpoint removes that artificial jump while preserving every
    within-segment relative motion.
    """
    from scipy.spatial.transform import Rotation

    pose_parts = []
    width_parts = []
    source_episode_parts = []
    episode_frame_parts = []
    boundaries = []
    previous_adjusted_last = None
    previous_raw_last = None
    previous_episode = None

    for episode_id in episode_ids:
        ep = replay_buffer.get_episode(episode_id)
        pos = ep['robot0_eef_pos']
        rot = ep['robot0_eef_rot_axis_angle']
        widths = ep['robot0_gripper_width'].reshape(-1)
        raw_poses = np.stack([
            pose_to_mat(np.concatenate([pos[i], rot[i]], axis=-1))
            for i in range(len(pos))
        ])
        adjusted_poses = raw_poses

        if previous_adjusted_last is not None:
            position_jump = float(np.linalg.norm(
                raw_poses[0, :3, 3] - previous_raw_last[:3, 3]))
            rotation_jump = float(np.degrees((
                Rotation.from_matrix(raw_poses[0, :3, :3]) *
                Rotation.from_matrix(previous_raw_last[:3, :3]).inv()
            ).magnitude()))
            alignment = previous_adjusted_last @ np.linalg.inv(raw_poses[0])
            adjusted_poses = alignment[None, ...] @ raw_poses
            boundaries.append({
                'from_episode': previous_episode,
                'to_episode': episode_id,
                'position_jump_m': position_jump,
                'rotation_jump_deg': rotation_jump,
            })

        pose_parts.append(adjusted_poses)
        width_parts.append(np.asarray(widths, dtype=np.float64))
        source_episode_parts.append(
            np.full(len(widths), episode_id, dtype=np.int64))
        episode_frame_parts.append(np.arange(len(widths), dtype=np.int64))
        previous_adjusted_last = adjusted_poses[-1]
        previous_raw_last = raw_poses[-1]
        previous_episode = episode_id

    return (
        np.concatenate(pose_parts, axis=0),
        np.concatenate(width_parts, axis=0),
        np.concatenate(source_episode_parts, axis=0),
        np.concatenate(episode_frame_parts, axis=0),
        boundaries,
    )


class _AsyncVideoRecorder:
    """Own a UVC camera and MP4 writer on a background capture thread."""

    def __init__(self, camera, output, fps=30.0):
        self.camera = camera
        self.output = output
        self.fps = float(fps)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_timestamp = np.nan
        self._frame_timestamps = []
        self._error = None
        self._thread = None

        self._cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise SystemExit(f'cannot open camera {camera}')
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (width, height))
        if not self._writer.isOpened():
            self._cap.release()
            raise SystemExit(f'cannot open output {output}')

    def start(self):
        self._thread = threading.Thread(
            target=self._capture_loop, name='replay-camera', daemon=True)
        self._thread.start()

    def _capture_loop(self):
        consecutive_failures = 0
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    self._error = 'camera returned 30 consecutive empty frames'
                    break
                time.sleep(0.01)
                continue
            consecutive_failures = 0
            frame_time = time.time()
            self._writer.write(frame)
            with self._lock:
                self._latest_timestamp = frame_time
                self._frame_timestamps.append(frame_time)

    def latest_timestamp(self):
        with self._lock:
            return float(self._latest_timestamp)

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                # Releasing the camera unblocks a stalled V4L2 read.
                self._cap.release()
                self._thread.join(timeout=2.0)
        self._cap.release()
        self._writer.release()
        if self._error is not None:
            print(f'WARNING: asynchronous camera stopped early: {self._error}')
        with self._lock:
            return list(self._frame_timestamps)


def _rotvec(R):
    from scipy.spatial.transform import Rotation as R2
    return R2.from_matrix(R).as_rotvec()


def _limit_pose_step(previous, target, max_translation, max_rotation):
    """Rate-limit an SE(3) target relative to the preceding command.

    SLAM occasionally produces a one-frame position jump much faster than the
    AM2Pro can safely execute.  Bounding the *commanded* step retains the
    direction of that motion but spreads it over several servo ticks.
    """
    from scipy.spatial.transform import Rotation

    previous = np.asarray(previous, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    result = target.copy()
    limited = False

    delta_pos = target[:3] - previous[:3]
    distance = float(np.linalg.norm(delta_pos))
    if distance > max_translation:
        result[:3] = previous[:3] + delta_pos * (max_translation / distance)
        limited = True

    R_previous = Rotation.from_rotvec(previous[3:])
    R_target = Rotation.from_rotvec(target[3:])
    delta_rot = (R_target * R_previous.inv()).as_rotvec()
    angle = float(np.linalg.norm(delta_rot))
    if angle > max_rotation:
        delta_rot *= max_rotation / angle
        result[3:] = (Rotation.from_rotvec(delta_rot) * R_previous).as_rotvec()
        limited = True

    return result, limited


if __name__ == "__main__":
    main()
