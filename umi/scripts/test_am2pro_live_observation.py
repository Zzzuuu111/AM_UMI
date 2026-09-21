"""Read one synchronized AM2Pro wrist-camera and robot observation safely."""

import pathlib
import sys
import tempfile
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import yaml


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from umi.real_world.bimanual_umi_env import BimanualUmiEnv  # noqa: E402


CONFIG_PATH = ROOT_DIR / "example" / "eval_robots_config.yaml"


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    # This starts the same holding controller validated in the preceding test.
    # It never calls servoL, servoJ, schedule_waypoint, or schedule_gripper.
    with tempfile.TemporaryDirectory(prefix="am_umi_observation_") as tmp:
        output_dir = pathlib.Path(tmp) / "run"
        with SharedMemoryManager() as shm_manager:
            print("[1/4] Creating camera + AM2Pro observation environment...", flush=True)
            env = BimanualUmiEnv(
                output_dir=output_dir,
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
                shm_manager=shm_manager,
            )
            for camera in env.camera.cameras.values():
                camera.verbose = True
            try:
                print("[2/4] Starting wrist camera and holding controller...", flush=True)
                env.start()
                print("[3/4] Components ready; filling observation buffers...", flush=True)
                # Fill both the camera and robot state buffers before aligning.
                time.sleep(1.5)
                print(
                    "camera_buffer_frames:", env.camera.min_buffer_count,
                    flush=True,
                )
                print("[4/4] Reading one synchronized observation...", flush=True)
                obs = env.get_obs()
                nonfinite = [
                    key for key, value in obs.items()
                    if isinstance(value, np.ndarray) and not np.isfinite(value).all()
                ]
                if nonfinite:
                    raise RuntimeError(f"non-finite observation keys: {nonfinite}")
                print("AM2PRO_LIVE_OBSERVATION_OK")
                for key, value in obs.items():
                    if isinstance(value, np.ndarray):
                        print(f"{key}: shape={value.shape} dtype={value.dtype}")
            finally:
                print("Stopping camera and controller...", flush=True)
                env.stop()


if __name__ == "__main__":
    main()
