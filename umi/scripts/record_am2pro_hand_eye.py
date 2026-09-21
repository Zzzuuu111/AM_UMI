"""Collect static AM2Pro wrist-camera hand-eye calibration samples safely.

By default this tool is read-only: it never enables torque and never writes a
motor register.  With the explicit ``--manual-torque-controls`` option it
also exposes H/G controls for a supervised free-drive workflow: G releases
all AM2Pro motors, and H locks all of them at their current encoder positions.
It always saves an RGB frame plus calibrated joint positions when the operator
types ``c``.

Run in the AM_UMI environment.  After collecting at least eight diverse,
static poses with Tag 13 visible, feed the resulting pickle to
``scripts/calibrate_robot_world_hand_eye.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
import cv2
import numpy as np
import yaml

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from umi.common.cv_util import (
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import mat_to_pose
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend


MOTOR_SPECS = (
    ("shoulder_pan", 1, "sts3250", MotorNormMode.DEGREES),
    ("shoulder_lift", 2, "sts3095", MotorNormMode.DEGREES),
    ("elbow_flex", 3, "sts3095", MotorNormMode.DEGREES),
    ("wrist_flex", 4, "sts3250", MotorNormMode.DEGREES),
    ("wrist_yaw", 5, "sts3250", MotorNormMode.DEGREES),
    ("wrist_roll", 6, "sts3250", MotorNormMode.DEGREES),
    ("gripper", 7, "sts3250", MotorNormMode.RANGE_0_100),
)
ARM_NAMES = [item[0] for item in MOTOR_SPECS[:6]]
ALL_MOTOR_NAMES = [item[0] for item in MOTOR_SPECS]
URDF_JOINT_NAMES = (
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
)


class HandEyeWebPreview:
    """Small local browser preview for systems where OpenCV GUI cannot open."""

    def __init__(self, port: int, allowed_commands=None):
        self._lock = threading.Lock()
        self._jpeg = None
        self._status = "正在连接相机…"
        self._commands = queue.Queue()
        self._allowed_commands = set(allowed_commands or {"c", "p", "d", "q"})
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def do_GET(self):
                if self.path.startswith("/frame.jpg"):
                    with owner._lock:
                        data = owner._jpeg
                    if data is None:
                        self.send_response(204)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Cache-Control", "no-store")
                        self.send_header("Content-Length", str(len(data)))
                        self.end_headers()
                        self.wfile.write(data)
                        return
                    self.end_headers()
                    return
                if self.path.startswith("/status.json"):
                    with owner._lock:
                        status = owner._status
                    data = json.dumps({"status": status}, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if self.path.startswith("/command/"):
                    command = self.path.rsplit("/", 1)[-1].lower()
                    if command in owner._allowed_commands:
                        owner._commands.put(command)
                        self.send_response(204)
                    else:
                        self.send_response(400)
                    self.end_headers()
                    return
                page = '''<!doctype html><meta charset="utf-8"><title>AM2Pro 手眼标定</title>
<style>body{background:#111;color:#eee;font:18px sans-serif;text-align:center}img{max-width:96vw;max-height:76vh}button{font-size:22px;margin:6px;padding:10px 18px}</style>
<h2>AM2Pro 腕部相机手眼标定</h2><div id="status">正在连接…</div><img id="v" src="/frame.jpg"><p id="controls"></p>
<p>只在 Tag 13 显示“已检测到”、机械臂完全静止时采样。</p>
<script>const v=document.getElementById('v'),s=document.getElementById('status'),controls=document.getElementById('controls'),allowed=%s;setInterval(()=>v.src='/frame.jpg?t='+Date.now(),120);setInterval(()=>fetch('/status.json?t='+Date.now()).then(r=>r.json()).then(x=>s.textContent=x.status).catch(()=>{}),180);function cmd(x){fetch('/command/'+encodeURIComponent(x))}for(const k of allowed){const b=document.createElement('button');b.textContent=k.toUpperCase();b.onclick=()=>cmd(k);controls.appendChild(b)}document.addEventListener('keydown',e=>{let k=e.key.toLowerCase();if(allowed.includes(k)){e.preventDefault();cmd(k)}})</script>''' % json.dumps(sorted(owner._allowed_commands))
                page = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def publish(self, image_bgr: np.ndarray):
        image_bgr = cv2.resize(image_bgr, (960, 540), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if ok:
            with self._lock:
                self._jpeg = encoded.tobytes()

    def set_status(self, text: str):
        with self._lock:
            self._status = text

    def get_command(self):
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


def next_rgb_frame(frame_iter):
    """Consume a few buffered frames and return a current RGB image."""
    frame = None
    for _ in range(4):
        frame = next(frame_iter)
    return frame.to_ndarray(format="rgb24")


def main():
    parser = argparse.ArgumentParser(
        description="Read-only AM2Pro wrist-camera hand-eye sample collector.")
    parser.add_argument("--output", required=True,
                        help="New .pkl output file; never overwritten.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument(
        "--camera-device",
        default="/dev/v4l/by-id/usb-EMEET_WXSJ_GC02_1080P_SN0001-video-index0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument(
        "--intrinsics",
        default="calibration/robot_wrist_camera/intrinsics/"
                "emeet_wrist_1920x1080_30fps_fisheye_v1.json")
    parser.add_argument("--aruco-yaml", default="calibration/shared_tags/aruco_config.yaml")
    parser.add_argument("--ik-backend", choices=("ros2_dh", "placo"), default="ros2_dh")
    parser.add_argument("--web-preview", action="store_true",
                        help="show live preview and C/P/D/Q controls in a browser")
    parser.add_argument("--web-preview-port", type=int, default=8766)
    parser.add_argument("--capture-settle-frames", type=int, default=15,
                        help="frames to discard after C before sampling (default: 15, about 0.5 s)")
    parser.add_argument("--manual-torque-controls", action="store_true",
                        help=("启用显式 H=锁定当前位置、G=释放全部电机的人工摆臂模式。"
                              "默认仍不会写入电机；此选项仅用于全程人工监督的 TCP 采样。"))
    args = parser.parse_args()

    out_path = (ROOT_DIR / args.output).resolve()
    if out_path.exists():
        parser.error(f"refusing to overwrite existing output: {out_path}")
    if args.capture_settle_frames < 1:
        parser.error("--capture-settle-frames must be positive")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    intr_path = (ROOT_DIR / args.intrinsics).resolve()
    aruco_path = (ROOT_DIR / args.aruco_yaml).resolve()
    if not intr_path.is_file() or not aruco_path.is_file():
        parser.error("intrinsics or ArUco configuration file does not exist")

    aruco_cfg = parse_aruco_config(yaml.safe_load(aruco_path.read_text()))
    raw_intr = parse_fisheye_intrinsics(json.loads(intr_path.read_text()))
    intr = convert_fisheye_intrinsics_resolution(
        raw_intr, target_resolution=(args.width, args.height))

    motors = {name: Motor(mid, model, norm) for name, mid, model, norm in MOTOR_SPECS}
    bus = FeetechMotorsBus(port=args.port, motors=motors)
    kin = create_kinematics_backend(
        args.ik_backend,
        urdf_path=str(ROOT_DIR / "alohamini2pro_right_arm_kinematics.urdf"),
        joint_names=URDF_JOINT_NAMES,
        target_frame_name="right_Fixed_Jaw",
    )

    resolved_camera = os.path.realpath(args.camera_device)
    camera = None
    web_preview = None
    try:
        # connect/read_calibration/sync_read are read-only operations.  In
        # particular, do not call configure_motors, torque_enabled or write.
        bus.connect()
        bus.calibration = bus.read_calibration()
        camera = av.open(resolved_camera, format="v4l2", options={
            "input_format": "mjpeg",
            "video_size": f"{args.width}x{args.height}",
            "framerate": str(args.fps),
        })
        frames = camera.decode(video=0)
        _ = next_rgb_frame(frames)

        manual_held = False

        def hold_at_current_pose():
            """Safely lock every servo at its encoder position after explicit H."""
            from lerobot.motors.feetech import OperatingMode

            nonlocal manual_held
            bus.disable_torque(ALL_MOTOR_NAMES)
            positions = bus.sync_read("Present_Position", ALL_MOTOR_NAMES)
            # Set goals while torque is off.  Re-enabling torque therefore
            # cannot pull a manually placed arm toward an old command.
            bus.sync_write("Goal_Position", {
                name: float(positions[name]) for name in ALL_MOTOR_NAMES})
            for name in ALL_MOTOR_NAMES:
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            try:
                bus.enable_torque(ALL_MOTOR_NAMES)
            except BaseException:
                bus.disable_torque(ALL_MOTOR_NAMES)
                manual_held = False
                raise
            manual_held = True
            return positions

        def release_for_manual_move():
            """Free every servo only after explicit G."""
            nonlocal manual_held
            bus.disable_torque(ALL_MOTOR_NAMES)
            manual_held = False

        allowed_commands = {"c", "p", "d", "q"}
        if args.manual_torque_controls:
            allowed_commands.update({"h", "g"})

        if args.web_preview:
            web_preview = HandEyeWebPreview(args.web_preview_port, allowed_commands=allowed_commands)
            print("WEB_PREVIEW_READY:", web_preview.url, flush=True)
            print("请在浏览器打开该地址。页面可预览相机，使用 C/P/D/Q 按钮。", flush=True)

        samples = []

        def read_pose():
            # Save the gripper encoder together with every image.  Existing
            # hand-eye consumers ignore this extra field; TCP/grasp-center
            # calibration can use it to associate a fitted point with J7.
            pos = bus.sync_read("Present_Position", ALL_MOTOR_NAMES)
            q_deg = np.array([float(pos[name]) for name in ARM_NAMES], dtype=float)
            gripper_position = float(pos["gripper"])
            q_model = q_deg.copy()
            q_model[5] *= -1.0  # physical wrist_roll is reversed vs URDF
            return q_deg, gripper_position, mat_to_pose(kin.forward_kinematics(q_model))

        def settled_capture():
            """Flush recent camera frames and reject a sample if joints moved."""
            q_before, _, _ = read_pose()
            last_frame = None
            for _ in range(args.capture_settle_frames):
                last_frame = next(frames)
            image = last_frame.to_ndarray(format="rgb24")
            q_after, gripper_after, tcp_after = read_pose()
            return (image, q_after, gripper_after, tcp_after,
                    float(np.max(np.abs(q_after - q_before))))
        print("AM2PRO_HAND_EYE_COLLECTOR_READY", flush=True)
        if args.manual_torque_controls:
            print("人工摆臂模式：G=释放全部电机；H=锁在当前位置；C 只允许在 H 后采样。", flush=True)
            print("先托住机械臂再按 G；退出时维持最后一次 H/G 的扭矩状态。", flush=True)
        else:
            print("安全说明：本程序不会开启扭矩、不会发送关节或夹爪动作。", flush=True)
        print(f"camera: {args.camera_device} -> {resolved_camera}", flush=True)
        print(f"Tag {args.tag_id} 必须清晰可见。每个姿态静止后输入 c 再回车采样。", flush=True)
        print(f"每次 C 后自动丢弃 {args.capture_settle_frames} 帧；关节漂移超过 0.5° 会拒绝保存。", flush=True)
        controls_text = "c=采样，p=读取当前姿态，d=删除最后一张，q=保存并退出"
        if args.manual_torque_controls:
            controls_text += "，h=锁定当前位置，g=释放全部电机"
        print("命令：" + controls_text + "。", flush=True)

        while not args.web_preview:
            try:
                command = input("hand_eye> ").strip().lower()
            except EOFError:
                command = "q"
            if command == "q":
                break
            if args.manual_torque_controls and command == "g":
                release_for_manual_move()
                print("AM2PRO_MANUAL_MODE_FREE: torque off; 请托住机械臂后手动摆动。", flush=True)
                continue
            if args.manual_torque_controls and command == "h":
                positions = hold_at_current_pose()
                print("AM2PRO_MANUAL_MODE_HELD: torque on at", {
                    name: round(float(positions[name]), 2) for name in ALL_MOTOR_NAMES}, flush=True)
                continue
            if command == "d":
                if samples:
                    samples.pop()
                    print(f"删除成功；当前有效样本数：{len(samples)}", flush=True)
                    pickle.dump(samples, out_path.open("wb"))
                else:
                    print("还没有样本。", flush=True)
                continue
            if command not in ("c", "p"):
                print("请输入有效控制键。", flush=True)
                continue

            if command == "c" and args.manual_torque_controls and not manual_held:
                print("请先按 H 锁定当前位置，再按 C 采样。", flush=True)
                continue

            if command == "p":
                q_deg, gripper_position, tcp_pose = read_pose()
            else:
                img, q_deg, gripper_position, tcp_pose, joint_drift_deg = settled_capture()
                if joint_drift_deg > 0.5:
                    print(f"关节在采样期间变化 {joint_drift_deg:.3f}°；本次拒绝保存，请静止后重试。", flush=True)
                    continue
            print("joint_deg:", np.round(q_deg, 3).tolist(), flush=True)
            print(f"gripper_position_range_0_100: {gripper_position:.3f}", flush=True)
            print("fixed_jaw_pose_base:", np.round(tcp_pose, 6).tolist(), flush=True)
            if command == "p":
                continue

            tags = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_cfg["aruco_dict"],
                marker_size_map=aruco_cfg["marker_size_map"],
                fisheye_intr_dict=intr,
            )
            if args.tag_id not in tags:
                print(f"未检测到 Tag {args.tag_id}；本次没有保存。请调整姿态/光线后重试。", flush=True)
                continue
            samples.append({
                "img": img,
                "tcp_pose": tcp_pose,
                "joint_deg": q_deg,
                "gripper_position_range_0_100": gripper_position,
                "capture_monotonic_s": time.monotonic(),
                "tag_id": args.tag_id,
            })
            pickle.dump(samples, out_path.open("wb"))
            print(f"HAND_EYE_SAMPLE_SAVED: {len(samples)}; file: {out_path}", flush=True)

        while args.web_preview:
            img = next_rgb_frame(frames)
            tags = detect_localize_aruco_tags(
                img=img, aruco_dict=aruco_cfg["aruco_dict"],
                marker_size_map=aruco_cfg["marker_size_map"], fisheye_intr_dict=intr)
            tag_ok = args.tag_id in tags
            display = img[..., ::-1].copy()
            torque_text = ("｜扭矩：锁定" if manual_held else "｜扭矩：释放") if args.manual_torque_controls else ""
            status = (f"Tag {args.tag_id}：{'已检测到；可在静止时采集' if tag_ok else '未检测到；请调整机械臂'}"
                      f"｜已保存 {len(samples)} 张{torque_text}")
            color = (0, 220, 0) if tag_ok else (0, 0, 255)
            cv2.putText(display, status, (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2, cv2.LINE_AA)
            web_preview.publish(display)
            web_preview.set_status(status)
            command = web_preview.get_command()
            if command is None:
                continue
            if command == "q":
                break
            if args.manual_torque_controls and command == "g":
                release_for_manual_move()
                print("AM2PRO_MANUAL_MODE_FREE: torque off; 请托住机械臂后手动摆动。", flush=True)
                continue
            if args.manual_torque_controls and command == "h":
                positions = hold_at_current_pose()
                print("AM2PRO_MANUAL_MODE_HELD: torque on at", {
                    name: round(float(positions[name]), 2) for name in ALL_MOTOR_NAMES}, flush=True)
                continue
            if command == "d":
                if samples:
                    samples.pop()
                    pickle.dump(samples, out_path.open("wb"))
                    print(f"删除成功；当前有效样本数：{len(samples)}", flush=True)
                continue
            if command == "p":
                q_deg, gripper_position, tcp_pose = read_pose()
            else:
                if args.manual_torque_controls and not manual_held:
                    print("请先按 H 锁定当前位置，再按 C 采样。", flush=True)
                    continue
                img, q_deg, gripper_position, tcp_pose, joint_drift_deg = settled_capture()
                if joint_drift_deg > 0.5:
                    print(f"关节在采样期间变化 {joint_drift_deg:.3f}°；本次拒绝保存，请静止后重试。", flush=True)
                    continue
            print("joint_deg:", np.round(q_deg, 3).tolist(), flush=True)
            print(f"gripper_position_range_0_100: {gripper_position:.3f}", flush=True)
            print("fixed_jaw_pose_base:", np.round(tcp_pose, 6).tolist(), flush=True)
            if command == "p":
                continue
            tags = detect_localize_aruco_tags(
                img=img, aruco_dict=aruco_cfg["aruco_dict"],
                marker_size_map=aruco_cfg["marker_size_map"], fisheye_intr_dict=intr)
            if args.tag_id not in tags:
                print(f"未检测到 Tag {args.tag_id}；本次没有保存。", flush=True)
                continue
            samples.append({
                "img": img, "tcp_pose": tcp_pose, "joint_deg": q_deg,
                "gripper_position_range_0_100": gripper_position,
                "capture_monotonic_s": time.monotonic(), "tag_id": args.tag_id,
            })
            pickle.dump(samples, out_path.open("wb"))
            print(f"HAND_EYE_SAMPLE_SAVED: {len(samples)}; file: {out_path}", flush=True)

        pickle.dump(samples, out_path.open("wb"))
        print("AM2PRO_HAND_EYE_COLLECTION_SAVED", flush=True)
        print("samples:", len(samples), flush=True)
        print("output:", out_path, flush=True)
        if len(samples) < 8:
            print("提示：建议至少采集 8 个不同末端位置和朝向的样本。", flush=True)
    finally:
        if camera is not None:
            camera.close()
        if web_preview is not None:
            web_preview.stop()
        bus.disconnect()


if __name__ == "__main__":
    main()
