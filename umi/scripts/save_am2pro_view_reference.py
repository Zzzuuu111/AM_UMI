"""Save the AM2Pro deployment start view exactly as the policy receives it.

This starts the tested AM2Pro controller in hold mode only.  It does not call
servoL, servoJ, schedule_waypoint, schedule_gripper, or policy inference.

With ``--preview-first``, it first opens a live preview.  Press ``b`` in that
window to save the complete reference (policy image *and* current AM2Pro
state), or ``q``/Esc to leave without writing anything.

In ``--free-drive`` mode the arm stays hand-movable (torque off).  Press ``h``
in the preview window to LOCK the arm at its current pose (torque on,
Goal_Position = current position, so it no longer sags when you let go),
``g`` to release it again, then ``b`` to save.  After a successful save the
arm stays locked; run with ``--release`` to free it later.
"""

import argparse
import datetime as dt
import pathlib
import sys
import tempfile
import time
from multiprocessing.managers import SharedMemoryManager

import cv2
import numpy as np
import yaml


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.common.view_reference import write_reference  # noqa: E402
from umi.common.view_reference import preprocess_uvc_bgr  # noqa: E402
from umi.common.usb_util import filter_v4l_paths, get_sorted_v4l_paths  # noqa: E402
from umi.common.pose_util import mat_to_pose  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402
from umi.real_world.am2pro_joint_mapping import (  # noqa: E402
    encoder_to_model,
    mapping_from_config,
)
from umi.real_world.bimanual_umi_env import BimanualUmiEnv  # noqa: E402


# Deliberately repeated here instead of importing the controller server: this
# mode must never instantiate the controller or issue a motor write.
_MOTOR_SPECS = (
    ("shoulder_pan", 1, "sts3250", "DEGREES"),
    ("shoulder_lift", 2, "sts3095", "DEGREES"),
    ("elbow_flex", 3, "sts3095", "DEGREES"),
    ("wrist_flex", 4, "sts3250", "DEGREES"),
    ("wrist_yaw", 5, "sts3250", "DEGREES"),
    ("wrist_roll", 6, "sts3250", "DEGREES"),
    ("gripper", 7, "sts3250", "RANGE_0_100"),
)
_ARM_MOTOR_NAMES = [item[0] for item in _MOTOR_SPECS[:6]]
_URDF_JOINT_NAMES = [
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
]


def draw_legible_text(image, text, origin, color, scale=0.58):
    """Draw high-contrast preview text over a live camera image."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def draw_policy_crop(raw_bgr, display_bgr, raw_scale, config):
    """Draw the exact source window used for the policy image."""
    crop = config.get("camera_crop")
    if crop is None:
        return
    cx, cy = crop["center"]
    cs = int(crop["size"])
    x0 = max(0, int(cx - cs // 2))
    y0 = max(0, int(cy - cs // 2))
    x1 = min(raw_bgr.shape[1], x0 + cs)
    y1 = min(raw_bgr.shape[0], y0 + cs)
    cv2.rectangle(
        display_bgr,
        (int(x0 * raw_scale), int(y0 * raw_scale)),
        (int(x1 * raw_scale), int(y1 * raw_scale)),
        (0, 255, 0), 2)


def make_tag_detector(tag_ids):
    """Load the shared UMI ArUco dictionary for an operator-only status check."""
    if not tag_ids:
        return None
    config_path = ROOT_DIR / "calibration" / "shared_tags" / "aruco_config.yaml"
    tag_config = yaml.safe_load(config_path.read_text())
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, tag_config["aruco_dict"]["predefined"]))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def annotate_tag_status(raw_bgr, display_bgr, raw_scale, detector, tag_ids):
    """Draw per-Tag fixed-world status without changing the policy image."""
    if detector is None:
        return {}
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY))
    found = set() if ids is None else {int(item) for item in ids.reshape(-1)}
    status = {tag_id: tag_id in found for tag_id in tag_ids}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
            if int(marker_id) in status:
                points = (marker_corners.reshape(-1, 2) * raw_scale).astype(np.int32)
                cv2.polylines(display_bgr, [points], True, (0, 220, 0), 2)
    # A fixed opaque panel keeps every status legible even when the operator
    # resizes the preview window or the task scene is bright/visually busy.
    panel_width = min(440, max(330, display_bgr.shape[1] // 2))
    panel_height = 34 * len(tag_ids) + 16
    # Keep it in the upper-left source image rather than beside the policy
    # panel: this remains visible even when a desktop window is narrow.
    x0 = 12
    y0 = 52
    panel = display_bgr[y0:y0 + panel_height, x0:x0 + panel_width]
    if panel.size:
        overlay = panel.copy()
        overlay[:] = (20, 20, 20)
        cv2.addWeighted(overlay, 0.86, panel, 0.14, 0, panel)
    for index, tag_id in enumerate(tag_ids):
        visible = status[tag_id]
        text = f"WORLD TAG {tag_id}: {'DETECTED' if visible else 'MISSING'}"
        color = (0, 230, 255) if visible else (0, 0, 255)  # yellow / red
        draw_legible_text(display_bgr, text, (x0 + 12, y0 + 30 + 34 * index), color,
                           scale=0.72)
    return status


def positions_to_robot_state(positions, config):
    """Convert raw servo positions into the UMI robot-state dict (model space)."""
    robot = config["robots"][0]
    q_encoder = np.array([float(positions[name]) for name in _ARM_MOTOR_NAMES])
    # References are deliberately stored in URDF model coordinates.  The
    # accepted V-jaw calibration converts raw Feetech readings here; legacy
    # configurations retain their historical joint_flip-only behavior.
    q_model = encoder_to_model(q_encoder, mapping_from_config(robot, ROOT_DIR))
    urdf_path = pathlib.Path(robot.get(
        "urdf_path", ROOT_DIR / "alohamini2pro_right_arm_kinematics.urdf"))
    if not urdf_path.is_absolute():
        urdf_path = (ROOT_DIR / urdf_path).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"机器人 URDF 不存在: {urdf_path}")
    kinematics = create_kinematics_backend(
        robot.get("ik_backend", "ros2_dh"),
        urdf_path=str(urdf_path),
        joint_names=_URDF_JOINT_NAMES,
        target_frame_name="right_tcp",
    )
    gripper_servo = float(positions["gripper"])
    closed = float(robot["gripper_servo_closed"])
    opened = float(robot["gripper_servo_open"])
    ratio = 0.0 if abs(opened - closed) < 1e-9 else (gripper_servo - closed) / (opened - closed)
    gripper_width = float(np.clip(
        float(robot["gripper_width_min"]) + ratio * (
            float(robot["gripper_width_max"]) - float(robot["gripper_width_min"])),
        float(robot["gripper_width_min"]), float(robot["gripper_width_max"])))
    actual_q = np.r_[q_model, gripper_servo]
    return {
        "ActualTCPPose": mat_to_pose(kinematics.forward_kinematics(q_model)),
        "ActualQ": actual_q,
        "ActualQd": np.zeros(7, dtype=np.float64),
        "gripper_position": gripper_width,
    }


def read_free_drive_robot_state(config):
    """Read AM2Pro state without enabling torque or sending any motor write."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    robot = config["robots"][0]
    motors = {
        name: Motor(motor_id, model, getattr(MotorNormMode, norm_mode))
        for name, motor_id, model, norm_mode in _MOTOR_SPECS
    }
    bus = FeetechMotorsBus(port=robot["robot_usb_port"], motors=motors)
    try:
        bus.connect()
        # Read-only operations only.  In particular do NOT call
        # configure_motors(), torque_enabled(), write(), or sync_write().
        bus.calibration = bus.read_calibration()
        positions = bus.sync_read("Present_Position", list(motors))
    finally:
        bus.disconnect()
    return positions_to_robot_state(positions, config)


class ArmHoldController:
    """Explicit torque hold for the free-drive preview.

    The read-only free-drive loop never touches torque.  This controller only
    writes to the servo bus after an explicit H (hold) / G (release) keypress:
    H locks every motor at its current position, G frees the arm again.
    On any hold failure the bus is left with torque OFF so the arm stays free.
    """

    def __init__(self, config):
        from lerobot.motors import Motor, MotorNormMode

        robot = config["robots"][0]
        self.port = robot["robot_usb_port"]
        self.motors = {
            name: Motor(motor_id, model, getattr(MotorNormMode, norm_mode))
            for name, motor_id, model, norm_mode in _MOTOR_SPECS
        }
        self.bus = None
        self.held = False

    def _ensure_bus(self):
        from lerobot.motors.feetech import FeetechMotorsBus

        if self.bus is not None:
            return self.bus
        bus = FeetechMotorsBus(port=self.port, motors=self.motors)
        bus.connect()
        try:
            bus.calibration = bus.read_calibration()
        except BaseException:
            bus.disconnect()
            raise
        self.bus = bus
        return bus

    def hold(self):
        """Enable torque in POSITION mode locked at the current joint positions."""
        from lerobot.motors.feetech import OperatingMode

        bus = self._ensure_bus()
        # Keep torque off while configuring and while setting Goal_Position to
        # the *current* encoder positions, so the moment torque comes back on
        # the servos lock exactly where the operator placed the arm (no jerk
        # toward a stale goal from an earlier session).
        bus.disable_torque(list(bus.motors))
        try:
            positions = bus.sync_read("Present_Position", list(bus.motors))
            bus.sync_write("Goal_Position", {
                name: float(positions[name]) for name in positions})
            bus.configure_motors()
            for name in bus.motors:
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                # Keep the wrist-flex setting consistent with the live
                # controller when the new long/heavy gripper is installed.
                bus.write("P_Coefficient", name, 80 if name == "wrist_flex" else 24)
                bus.write("I_Coefficient", name, 0)
                bus.write("D_Coefficient", name, 32)
                if name == "gripper":
                    bus.write("Max_Torque_Limit", name, 500)
                    bus.write("Protection_Current", name, 250)
                    bus.write("Overload_Torque", name, 25)
        except BaseException:
            self.held = False
            raise
        try:
            bus.enable_torque(list(bus.motors))
        except BaseException:
            # Never leave only some motors enabled: drop torque again so the
            # operator can simply retry.
            try:
                bus.disable_torque(list(bus.motors))
            except BaseException:
                pass
            self.held = False
            raise
        self.held = True
        return positions

    def release(self):
        """Disable torque on every motor so the arm is hand-movable again."""
        self._ensure_bus().disable_torque(list(self.motors))
        self.held = False

    def read_state(self, config):
        """Read positions over the already-open bus (the serial port is exclusive)."""
        positions = self._ensure_bus().sync_read("Present_Position", list(self.motors))
        return positions_to_robot_state(positions, config)

    def close(self, release_first):
        """Disconnect; optionally disable torque first (cancel path frees the arm)."""
        if self.bus is None:
            return
        try:
            if release_first:
                self.bus.disable_torque(list(self.motors))
        finally:
            try:
                self.bus.disconnect()
            finally:
                self.bus = None
                self.held = False


def release_am2pro_torque(config):
    """--release mode: disable torque on every AM2Pro motor and exit."""
    controller = ArmHoldController(config)
    try:
        controller.release()
        print("AM2PRO_ARM_TORQUE_RELEASED", flush=True)
    finally:
        controller.close(release_first=True)


def save_free_drive_reference(args, config, reference_dir):
    """Preview freely movable arm and save camera + read-only pose on B."""
    import av

    device = config.get("camera_device_path")
    if not device:
        device = filter_v4l_paths(
            get_sorted_v4l_paths(), match_name=config.get("camera_match_name"))[0]
    settings = config.get("camera_settings") or {}
    width, height = int(settings.get("width", 1920)), int(settings.get("height", 1080))
    fps = int(settings.get("fps", 30))
    print("FREE_DRIVE_PREVIEW_READY")
    print("Read-only by default.  Torque changes happen only on explicit keys:")
    print("  H = LOCK arm at its current pose (torque on)   G = release (torque off)")
    print("Move the arm by hand. Keep task objects inside the green policy-crop box.")
    print("Click the preview window; H=hold, G=release, B=save image + pose, Q/Esc=cancel.")
    container = av.open(
        str(pathlib.Path(device).resolve()), format="v4l2",
        options={"input_format": "mjpeg", "video_size": f"{width}x{height}", "framerate": str(fps)})
    window_name = "AM2Pro free-drive start view | H: hold | G: release | B: save | Q/Esc: cancel"
    tag_detector = make_tag_detector(args.monitor_tag_ids)
    hold_ctrl = ArmHoldController(config)
    save_done = False
    try:
        for frame in container.decode(video=0):
            raw_bgr = frame.to_ndarray(format="bgr24")
            policy_rgb = preprocess_uvc_bgr(raw_bgr, config.get("camera_crop"))
            policy_bgr = np.ascontiguousarray(policy_rgb[..., ::-1])
            raw_scale = min(1.0, args.display_width / raw_bgr.shape[1])
            raw_display = cv2.resize(raw_bgr, None, fx=raw_scale, fy=raw_scale,
                                     interpolation=cv2.INTER_AREA)
            draw_policy_crop(raw_bgr, raw_display, raw_scale, config)
            tag_status = annotate_tag_status(
                raw_bgr, raw_display, raw_scale, tag_detector, args.monitor_tag_ids)
            policy_height = raw_display.shape[0]
            policy_display = cv2.resize(policy_bgr, (policy_height, policy_height),
                                        interpolation=cv2.INTER_NEAREST)
            display = np.concatenate([raw_display, policy_display], axis=1)
            draw_legible_text(display, "RAW: green box is policy field of view", (12, 28),
                              (255, 255, 255), scale=0.62)
            draw_legible_text(display, "POLICY INPUT", (raw_display.shape[1] + 12, 28),
                              (255, 255, 255), scale=0.54)
            if hold_ctrl.held:
                draw_legible_text(
                    display, "[ARM HELD]   H: hold   G: release   B: save   Q / Esc: cancel",
                    (12, display.shape[0] - 14), (0, 255, 0), scale=0.65)
            else:
                draw_legible_text(display, "H: hold   G: release   B: save   Q / Esc: cancel",
                                  (12, display.shape[0] - 14), (255, 255, 255), scale=0.65)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("AM2PRO_VIEW_REFERENCE_CANCELLED")
                return
            if key in (ord("h"), ord("H")):
                try:
                    hold_ctrl.hold()
                    print("AM2PRO_ARM_HOLDING: torque on, locked at current pose", flush=True)
                except Exception as exc:
                    print(f"AM2PRO_ARM_HOLD_FAILED (torque left off): {exc}", flush=True)
                continue
            if key in (ord("g"), ord("G")):
                try:
                    hold_ctrl.release()
                    print("AM2PRO_ARM_RELEASED: torque off, keep a hand on the arm", flush=True)
                except Exception as exc:
                    print(f"AM2PRO_ARM_RELEASE_FAILED: {exc}", flush=True)
                continue
            if key not in (ord("b"), ord("B")):
                continue
            if args.require_all_tags_visible and not all(
                    tag_status.get(tag_id, False) for tag_id in args.monitor_tag_ids):
                missing_ids = [tag_id for tag_id in args.monitor_tag_ids
                               if not tag_status.get(tag_id, False)]
                print(f"NOT_ALL_MONITORED_WORLD_TAGS_VISIBLE_REFERENCE_NOT_SAVED: missing {missing_ids}",
                      flush=True)
                continue
            if args.require_tag_visible and not any(tag_status.values()):
                print("NO_MONITORED_WORLD_TAG_VISIBLE_REFERENCE_NOT_SAVED", flush=True)
                continue
            if not hold_ctrl.held:
                print("WARNING: arm is not held; its pose may have sagged under gravity. "
                      "Consider H to lock first, then B to save the held pose.", flush=True)
            if hold_ctrl.bus is not None:
                print("Reading AM2Pro pose from the preview bus ...", flush=True)
                robot_state = hold_ctrl.read_state(config)
            else:
                print("Reading AM2Pro pose without enabling torque ...", flush=True)
                robot_state = read_free_drive_robot_state(config)
            metadata = {
                "created_at": dt.datetime.now().astimezone().isoformat(),
                "purpose": "AM2Pro deployment start-view reference",
                "capture_mode": "free_drive_held" if hold_ctrl.held else "free_drive_read_only",
                "arm_held": bool(hold_ctrl.held),
                "tcp_frame": "right_tcp",
                "policy_image": {
                    "key": "camera0_rgb", "resolution": [224, 224],
                    "dtype": "uint8_rgb", "preprocessing": "camera crop + RGB conversion + gripper mask",
                },
                "raw_camera_image": {
                    "filename": "raw_camera.png",
                    "resolution": [int(raw_bgr.shape[1]), int(raw_bgr.shape[0])],
                    "purpose": "operator record of the uncropped camera view; not policy input",
                },
                "camera_config": {
                    "match_name": config.get("camera_match_name"), "device_path": str(device),
                    "settings": settings, "crop": config.get("camera_crop"),
                },
                "robot_state": json_value(robot_state),
                "reference_timestamp": time.time(),
            }
            saved = write_reference(reference_dir, policy_rgb, metadata)
            raw_output = saved / "raw_camera.png"
            if not cv2.imwrite(str(raw_output), raw_bgr):
                raise RuntimeError(f"failed to write raw camera image: {raw_output}")
            print("AM2PRO_VIEW_REFERENCE_SAVED")
            print("reference_dir:", saved)
            print("policy_image:", saved / "policy_input_rgb.png")
            print("raw_camera_image:", raw_output)
            save_done = True
            if hold_ctrl.held:
                print("ARM_STAYS_HOLDING_AT_SAVED_POSE", flush=True)
                print("Release later with: python scripts/save_am2pro_view_reference.py "
                      f"--release --robot-config {args.robot_config}", flush=True)
            else:
                print("FREE_DRIVE_NO_MOTOR_COMMAND_SENT", flush=True)
            return
    finally:
        container.close()
        cv2.destroyAllWindows()
        hold_ctrl.close(release_first=not save_done)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", default=None,
                        help="New directory for policy_input_rgb.png and reference.json")
    parser.add_argument("--robot-config", default=str(ROOT_DIR / "example" / "eval_robots_config.yaml"))
    parser.add_argument(
        "--preview-first", action="store_true",
        help="Preview the policy image first; press B to save, Q/Esc to cancel.")
    parser.add_argument("--display-width", type=int, default=960,
                        help="Raw-preview width in pixels when --preview-first is used.")
    parser.add_argument("--free-drive", action="store_true",
                        help="Camera preview while the arm remains hand-movable; H locks torque at "
                             "the current pose, G releases, B reads and saves the pose.")
    parser.add_argument("--release", action="store_true",
                        help="Disable AM2Pro torque and exit; frees an arm that was left holding.")
    parser.add_argument("--monitor-tag-id", type=int, default=13,
                        help="单个固定世界 Tag；未给 --monitor-tag-ids 时使用。负数关闭。")
    parser.add_argument("--monitor-tag-ids", default=None,
                        help="多个固定世界 Tag，例如 13,14,15；任一可见即可满足 --require-tag-visible")
    parser.add_argument("--require-tag-visible", action="store_true",
                        help="With --free-drive, B saves only when --monitor-tag-id is visible.")
    parser.add_argument("--require-all-tags-visible", action="store_true",
                        help="With --free-drive, B saves only when every --monitor-tag-ids Tag is visible.")
    args = parser.parse_args()

    if args.monitor_tag_ids is None:
        args.monitor_tag_ids = [] if args.monitor_tag_id < 0 else [args.monitor_tag_id]
    else:
        try:
            args.monitor_tag_ids = [int(item.strip()) for item in args.monitor_tag_ids.split(",")
                                    if item.strip()]
        except ValueError:
            parser.error("--monitor-tag-ids 必须为逗号分隔的整数，例如 13,14,15")
        if not args.monitor_tag_ids or len(set(args.monitor_tag_ids)) != len(args.monitor_tag_ids) \
                or any(tag_id < 0 for tag_id in args.monitor_tag_ids):
            parser.error("--monitor-tag-ids 需要至少一个互不相同的非负编号")

    if args.require_tag_visible and args.require_all_tags_visible:
        parser.error("--require-tag-visible 与 --require-all-tags-visible 不能同时使用")

    config_path = pathlib.Path(args.robot_config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())

    if args.release:
        release_am2pro_torque(config)
        return

    if not args.reference_dir:
        parser.error("--reference-dir is required (unless --release is given)")
    reference_dir = pathlib.Path(args.reference_dir).expanduser()
    if reference_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing reference: {reference_dir}")

    if args.free_drive:
        save_free_drive_reference(args, config, reference_dir)
        return

    with tempfile.TemporaryDirectory(prefix="am_umi_view_reference_") as tmp:
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
                camera_obs_latency=0.17,
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                gripper_obs_horizon=2,
                camera_match_name=config.get("camera_match_name"),
                camera_device_path=config.get("camera_device_path"),
                camera_settings=config.get("camera_settings"),
                camera_crop=config.get("camera_crop"),
                # A raw frame is needed only for this operator preview.  The
                # policy still receives the separately transformed 224 image.
                enable_camera_raw_buffer=args.preview_first,
                shm_manager=shm_manager,
            )
            try:
                print("Starting camera and holding AM2Pro at its current pose ...", flush=True)
                env.start()
                time.sleep(1.5)
                if args.preview_first:
                    print("Left: raw full camera view. Right: exact policy input.")
                    print("Click the preview window, then press B to save; Q/Esc cancels.")
                    window_name = "AM2Pro start-view | B: save | Q/Esc: cancel"
                    while True:
                        observation = env.get_obs()
                        image_rgb = observation["camera0_rgb"][-1]
                        policy_bgr = np.ascontiguousarray(
                            np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)[..., ::-1])
                        raw_bgr = env.camera.get_vis()["color"][0]
                        raw_h, raw_w = raw_bgr.shape[:2]
                        raw_scale = min(1.0, args.display_width / raw_w)
                        raw_display = cv2.resize(
                            raw_bgr, None, fx=raw_scale, fy=raw_scale,
                            interpolation=cv2.INTER_AREA)
                        draw_policy_crop(raw_bgr, raw_display, raw_scale, config)
                        policy_height = raw_display.shape[0]
                        policy_display = cv2.resize(
                            policy_bgr, (policy_height, policy_height),
                            interpolation=cv2.INTER_NEAREST)
                        display = np.concatenate([raw_display, policy_display], axis=1)
                        raw_label = "RAW CAMERA (use this to find the view)"
                        policy_label = "POLICY INPUT (this is what B saves)"
                        cv2.putText(display, raw_label, (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(display, policy_label,
                                    (raw_display.shape[1] + 12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(display, "B: save reference   Q / Esc: cancel", (12, display.shape[0] - 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                                    cv2.LINE_AA)
                        cv2.imshow(window_name, display)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("b"), ord("B")):
                            break
                        if key in (ord("q"), ord("Q"), 27):
                            print("AM2PRO_VIEW_REFERENCE_CANCELLED")
                            return
                    cv2.destroyWindow(window_name)
                else:
                    observation = env.get_obs()

                robot_state = env.get_robot_state()[0]
                metadata = {
                    "created_at": dt.datetime.now().astimezone().isoformat(),
                    "purpose": "AM2Pro deployment start-view reference",
                    "policy_image": {
                        "key": "camera0_rgb",
                        "resolution": [224, 224],
                        "dtype": "float32_rgb_0_to_1",
                        "preprocessing": "BimanualUmiEnv camera crop + RGB conversion + gripper mask",
                    },
                    "camera_config": {
                        "match_name": config.get("camera_match_name"),
                        "device_path": config.get("camera_device_path"),
                        "settings": config.get("camera_settings"),
                        "crop": config.get("camera_crop"),
                    },
                    "robot_state": json_value(robot_state),
                    "reference_timestamp": float(observation["timestamp"][-1]),
                }
                saved = write_reference(
                    reference_dir, observation["camera0_rgb"][-1], metadata)
                print("AM2PRO_VIEW_REFERENCE_SAVED")
                print("reference_dir:", saved)
                print("policy_image:", saved / "policy_input_rgb.png")
                print("This script sent no trajectory action to the robot.")
            finally:
                cv2.destroyAllWindows()
                print("Stopping camera and holding controller ...", flush=True)
                env.stop()


if __name__ == "__main__":
    main()
