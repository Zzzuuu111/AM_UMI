"""Continuously compare a live AM2Pro or UVC view against a saved reference."""

import argparse
import pathlib
import sys
import tempfile
import time
from multiprocessing.managers import SharedMemoryManager

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.usb_util import filter_v4l_paths, get_sorted_v4l_paths  # noqa: E402
from umi.common.view_reference import (  # noqa: E402
    compare_policy_images,
    load_reference,
    make_alignment_panel,
    preprocess_uvc_bgr,
)
from umi.real_world.bimanual_umi_env import BimanualUmiEnv  # noqa: E402


def save_panel(output_dir, panel_bgr, metrics, elapsed):
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "latest_alignment.png"), panel_bgr)
    with (output_dir / "alignment_history.csv").open("a", encoding="utf-8") as file:
        file.write(f"{elapsed:.3f},{metrics['appearance_score']:.6f},"
                   f"{metrics['mae']:.6f},{metrics['correlation']:.6f}\n")


def compare_robot_state(reference_state, current_state):
    """Compare saved/current AM2Pro state in physically meaningful units."""
    reference_tcp = np.asarray(reference_state["ActualTCPPose"], dtype=np.float64)
    current_tcp = np.asarray(current_state["ActualTCPPose"], dtype=np.float64)
    position_mm = float(np.linalg.norm(current_tcp[:3] - reference_tcp[:3]) * 1000.0)
    # TCP orientation is an axis-angle vector.  Its element-wise difference
    # is not an angular distance near pi, so compose rotations instead.
    rotation_deg = float(np.degrees((
        Rotation.from_rotvec(reference_tcp[3:]).inv()
        * Rotation.from_rotvec(current_tcp[3:])
    ).magnitude()))
    reference_joints = np.asarray(reference_state["ActualQ"], dtype=np.float64)
    current_joints = np.asarray(current_state["ActualQ"], dtype=np.float64)
    joint_error = np.abs(current_joints - reference_joints)
    gripper_mm = abs(
        float(current_state.get("gripper_position", 0.0))
        - float(reference_state.get("gripper_position", 0.0))) * 1000.0
    return {
        "tcp_position_mm": position_mm,
        "tcp_rotation_deg": rotation_deg,
        "joint_max_deg": float(joint_error.max()),
        "joint_rms_deg": float(np.sqrt(np.mean(joint_error ** 2))),
        "gripper_mm": gripper_mm,
    }


def add_robot_alignment_text(panel_bgr, robot_metrics):
    if robot_metrics is None:
        return panel_bgr
    panel_bgr = cv2.copyMakeBorder(
        panel_bgr, 0, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    text = (
        f"ROBOT: TCP position={robot_metrics['tcp_position_mm']:.1f} mm, "
        f"orientation={robot_metrics['tcp_rotation_deg']:.1f} deg, "
        f"max joint={robot_metrics['joint_max_deg']:.1f} deg, "
        f"gripper={robot_metrics['gripper_mm']:.1f} mm"
    )
    cv2.putText(panel_bgr, text, (8, panel_bgr.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 255, 0), 1, cv2.LINE_AA)
    return panel_bgr


def monitor_frames(
        frame_source, reference_rgb, output_dir, seconds, interval, show,
        reference_robot_state=None, robot_state_source=None):
    """Monitor a source returning ``(policy_rgb, raw_bgr_or_none)``."""
    start = time.monotonic()
    next_report = start
    print("columns: elapsed_s, appearance_score, mae, correlation")
    if reference_robot_state is not None:
        print("robot columns: tcp_position_mm, tcp_rotation_deg, joint_max_deg, gripper_mm")
    while time.monotonic() - start < seconds:
        current_rgb, raw_bgr = frame_source()
        metrics = compare_policy_images(reference_rgb, current_rgb)
        panel_bgr = make_alignment_panel(reference_rgb, current_rgb, metrics)
        robot_metrics = None
        if reference_robot_state is not None and robot_state_source is not None:
            robot_metrics = compare_robot_state(
                reference_robot_state, robot_state_source())
            panel_bgr = add_robot_alignment_text(panel_bgr, robot_metrics)
        elapsed = time.monotonic() - start
        save_panel(output_dir, panel_bgr, metrics, elapsed)
        if time.monotonic() >= next_report:
            report = (f"{elapsed:.1f}, {metrics['appearance_score']:.3f}, "
                      f"{metrics['mae']:.3f}, {metrics['correlation']:.3f}")
            if robot_metrics is not None:
                report += (
                    f" | robot: {robot_metrics['tcp_position_mm']:.1f} mm, "
                    f"{robot_metrics['tcp_rotation_deg']:.1f} deg, "
                    f"joint max {robot_metrics['joint_max_deg']:.1f} deg, "
                    f"gripper {robot_metrics['gripper_mm']:.1f} mm")
            print(report, flush=True)
            next_report += 1.0
        if show:
            cv2.imshow("UMI view reference alignment", panel_bgr)
            if raw_bgr is not None:
                raw_display = np.ascontiguousarray(raw_bgr)
                max_width = 960
                if raw_display.shape[1] > max_width:
                    scale = max_width / raw_display.shape[1]
                    raw_display = cv2.resize(
                        raw_display, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
                cv2.putText(raw_display, "RAW CAMERA: use this to manually find the view",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("UMI raw camera view", raw_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("VIEW_REFERENCE_ALIGNMENT_CANCELLED")
                break
        time.sleep(max(0.0, interval))
    if show:
        cv2.destroyAllWindows()


def resolve_device(config, explicit_device):
    if explicit_device:
        return explicit_device
    if config.get("camera_device_path"):
        return config["camera_device_path"]
    paths = filter_v4l_paths(
        get_sorted_v4l_paths(), match_name=config.get("camera_match_name"))
    return paths[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--robot-config", default=str(ROOT_DIR / "example" / "eval_robots_config.yaml"))
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Finite monitor duration; default 60 seconds")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <reference-dir>/live_alignment")
    parser.add_argument("--show", action="store_true",
        help="Show raw-camera and policy-alignment windows; Q/Esc ends early")
    parser.add_argument("--camera-only", action="store_true",
                        help="Use a UVC camera only; no AM2Pro controller is started")
    parser.add_argument("--device", default=None,
                        help="UVC device for --camera-only; defaults to config discovery")
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("--seconds and --interval must be positive")

    reference_dir = pathlib.Path(args.reference_dir).expanduser()
    reference_rgb, metadata = load_reference(reference_dir)
    config = yaml.safe_load(pathlib.Path(args.robot_config).expanduser().read_text())
    output_dir = (pathlib.Path(args.output_dir).expanduser()
                  if args.output_dir else reference_dir / "live_alignment")
    output_dir.mkdir(parents=True, exist_ok=True)
    history = output_dir / "alignment_history.csv"
    if not history.exists():
        history.write_text("elapsed_s,appearance_score,mae,correlation\n", encoding="utf-8")
    print("reference created:", metadata.get("created_at", "unknown"))
    print("panel written repeatedly to:", output_dir / "latest_alignment.png")

    if args.camera_only:
        device = resolve_device(config, args.device)
        capture = cv2.VideoCapture(str(pathlib.Path(device).resolve()), cv2.CAP_V4L2)
        settings = config.get("camera_settings") or {}
        if settings:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.get("width", 1920))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.get("height", 1080))
            capture.set(cv2.CAP_PROP_FPS, settings.get("fps", 30))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open UVC camera: {device}")
        print("camera-only alignment; device:", device)
        try:
            def get_frame():
                ok, frame_bgr = capture.read()
                if not ok:
                    raise RuntimeError("camera frame read failed")
                return (preprocess_uvc_bgr(frame_bgr, config.get("camera_crop")),
                        frame_bgr)
            monitor_frames(get_frame, reference_rgb, output_dir,
                           args.seconds, args.interval, args.show)
        finally:
            capture.release()
        return

    with tempfile.TemporaryDirectory(prefix="am_umi_view_alignment_") as tmp:
        with SharedMemoryManager() as shm_manager:
            env = BimanualUmiEnv(
                output_dir=pathlib.Path(tmp) / "run",
                robots_config=config["robots"],
                grippers_config=config["grippers"],
                frequency=10,
                obs_image_resolution=(224, 224),
                obs_float32=True,
                camera_reorder=[0],
                init_joints=False,
                enable_multi_cam_vis=False,
                enable_camera_raw_buffer=args.show,
                camera_obs_latency=0.17,
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                gripper_obs_horizon=2,
                camera_match_name=config.get("camera_match_name"),
                camera_device_path=config.get("camera_device_path"),
                camera_settings=config.get("camera_settings"),
                camera_crop=config.get("camera_crop"),
                shm_manager=shm_manager,
            )
            try:
                print("Starting camera and AM2Pro holding controller (no trajectory commands) ...")
                env.start()
                time.sleep(1.5)
                def get_frame():
                    observation = env.get_obs()
                    raw_bgr = env.camera.get_vis()["color"][0] if args.show else None
                    return observation["camera0_rgb"][-1], raw_bgr

                reference_robot_state = metadata.get("robot_state")
                if reference_robot_state is None:
                    print("Warning: reference has no robot_state; image alignment only.")
                monitor_frames(
                    get_frame, reference_rgb, output_dir,
                    args.seconds, args.interval, args.show,
                    reference_robot_state=reference_robot_state,
                    robot_state_source=lambda: env.get_robot_state()[0])
                print("VIEW_REFERENCE_ALIGNMENT_COMPLETE")
            finally:
                print("Stopping camera and holding controller ...", flush=True)
                env.stop()


if __name__ == "__main__":
    main()
