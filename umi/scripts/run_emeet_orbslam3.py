"""Run the local EMEET/JY901B ORB-SLAM3 image on one prepared session.

Run export_handheld_imu_to_orbslam3.py first.  This program is offline: it
only reads a recorded session directory and writes a trajectory CSV there.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="运行 EMEET + JY901B ORB-SLAM3 离线建图")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument(
        "--settings",
        default=str(ROOT_DIR / "calibration" / "handheld_gripper_camera" / "slam" /
                    "emeet_jy901b_orbslam3_v1.yaml"),
    )
    parser.add_argument(
        "--image", default="am_umi_orb_slam3_emeet_tagonly_v6:latest",
        help=("当前 EMEET 修复镜像；包含鱼眼 Tag 初始化、IMU 读取边界与 "
              "Tag 可见时禁止退回不稳定普通两视图初始化的修复。"),
    )
    parser.add_argument("--trajectory-name", default="camera_trajectory.csv")
    parser.add_argument("--imu-json", default="imu_data.json",
                        help="session-dir 内的已导出 IMU JSON 文件名")
    parser.add_argument("--save-map", default=None,
                        help="optional atlas filename stored inside session-dir, e.g. map_atlas.osa")
    parser.add_argument("--max-lost-frames", type=int, default=120)
    parser.add_argument("--disable-imu-init", action="store_true",
                        help="diagnostic fallback: keep Tag-anchored visual scale and skip VI initialization")
    parser.add_argument(
        "--mask",
        default=str(ROOT_DIR / "calibration" / "handheld_gripper_camera" / "slam" /
                    "emeet_handheld_gripper_mask_960x540_v1.png"),
        help="灰度遮罩；白色区域会被 SLAM 忽略（默认遮掉相机刚性连接的手持夹爪）")
    parser.add_argument("--no-mask", action="store_true",
                        help="仅用于诊断：不遮掉手持夹爪区域")
    args = parser.parse_args()

    session_dir = pathlib.Path(args.session_dir).expanduser().resolve()
    settings = pathlib.Path(args.settings).expanduser().resolve()
    trajectory = session_dir / args.trajectory_name
    imu_json_name = pathlib.Path(args.imu_json)
    if imu_json_name.name != args.imu_json:
        raise SystemExit("--imu-json 必须是 session-dir 内的单个文件名")
    imu_json = session_dir / imu_json_name
    required = [session_dir / "raw_video.mp4", imu_json, settings]
    absent = [str(path) for path in required if not path.is_file()]
    if absent:
        raise SystemExit("缺少必需文件：\n  " + "\n  ".join(absent))
    if trajectory.exists():
        raise SystemExit(f"为保护已有轨迹，拒绝覆盖：{trajectory}")
    map_path = None
    if args.save_map is not None:
        map_name = pathlib.Path(args.save_map)
        if map_name.name != args.save_map or map_name.suffix != ".osa":
            raise SystemExit("--save-map 必须是 session-dir 内的单个 .osa 文件名")
        map_path = session_dir / map_name
        if map_path.exists():
            raise SystemExit(f"为保护已有地图，拒绝覆盖：{map_path}")
    if args.max_lost_frames <= 0:
        parser.error("--max-lost-frames 必须为正数")
    mask = None if args.no_mask else pathlib.Path(args.mask).expanduser().resolve()
    if mask is not None and not mask.is_file():
        raise SystemExit(f"mask 文件不存在：{mask}")

    docker_options = ["docker", "run", "--rm"]
    if args.disable_imu_init:
        docker_options.extend(["--env", "ORB_SLAM3_DISABLE_IMU_INIT=1"])
    docker_options.extend([
        "--volume", f"{session_dir}:/data",
        "--volume", f"{settings}:/settings/emeet_jy901b_orbslam3.yaml:ro",
    ])
    if mask is not None:
        docker_options.extend(["--volume", f"{mask}:/settings/slam_mask.png:ro"])

    command = docker_options + [
        args.image,
        "/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam",
        "--vocabulary", "/ORB_SLAM3/Vocabulary/ORBvoc.txt",
        "--setting", "/settings/emeet_jy901b_orbslam3.yaml",
        "--input_video", "/data/raw_video.mp4",
        "--input_imu_json", f"/data/{args.imu_json}",
        "--output_trajectory_csv", f"/data/{args.trajectory_name}",
        "--max_lost_frames", str(args.max_lost_frames),
    ]
    if mask is not None:
        command.extend(["--mask_img", "/settings/slam_mask.png"])
    if map_path is not None:
        command.extend(["--save_map", f"/data/{args.save_map}"])

    print("EMEET_ORB_SLAM3_STARTED")
    print("session:", session_dir)
    print("image:", args.image)
    if args.disable_imu_init:
        print("mode: Tag-anchored visual baseline (IMU initialization disabled)")
    if mask is not None:
        print("mask:", mask)
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(f"ORB-SLAM3 失败，退出码：{result.returncode}")
    if not trajectory.is_file() or trajectory.stat().st_size == 0:
        raise SystemExit("ORB-SLAM3 已退出但没有生成轨迹 CSV")
    print("EMEET_ORB_SLAM3_OK")
    print("trajectory:", trajectory)
    if map_path is not None:
        if not map_path.is_file() or map_path.stat().st_size == 0:
            raise SystemExit("ORB-SLAM3 已退出但没有生成地图 atlas")
        print("map:", map_path)


if __name__ == "__main__":
    main()
