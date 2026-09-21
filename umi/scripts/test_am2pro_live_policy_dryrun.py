"""Generate one live AM2Pro policy action chunk without executing it."""

import argparse
import pathlib
import sys
import tempfile
import time
from multiprocessing.managers import SharedMemoryManager

import dill
import hydra
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from umi.real_world.bimanual_umi_env import BimanualUmiEnv  # noqa: E402
from umi.real_world.real_inference_util import (  # noqa: E402
    get_real_umi_action,
    get_real_umi_obs_dict,
)


CHECKPOINT_PATH = pathlib.Path(
    "/home/zzzjh/universal_manipulation_interface/data/outputs/2026.08.27/"
    "14.10.09_train_diffusion_unet_image_umi/checkpoints/"
    "epoch=0060-test_mean_score=-0.019.ckpt"
)
CONFIG_PATH = ROOT_DIR / "example" / "eval_robots_config.yaml"
GPU_MEMORY_FRACTION = 0.45


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-inference-steps", type=int, default=16)
    args = parser.parse_args()
    if args.num_inference_steps <= 0:
        parser.error("--num-inference-steps must be positive")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    # Read only the metadata before forking camera/controller processes. The
    # full CPU tensors and CUDA context are intentionally created afterwards.
    metadata = torch.load(
        CHECKPOINT_PATH, map_location="meta", mmap=True,
        pickle_module=dill, weights_only=False,
    )
    cfg = metadata["cfg"]
    del metadata

    tx_robot1_robot0 = np.asarray(config["tx_left_right"])
    with tempfile.TemporaryDirectory(prefix="am_umi_policy_dryrun_") as tmp:
        output_dir = pathlib.Path(tmp) / "run"
        with SharedMemoryManager() as shm_manager:
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
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                camera_match_name=config.get("camera_match_name"),
                camera_device_path=config.get("camera_device_path"),
                camera_settings=config.get("camera_settings"),
                camera_crop=config.get("camera_crop"),
                shm_manager=shm_manager,
            )
            try:
                print("Starting live observation sources (hold only)...", flush=True)
                env.start()
                time.sleep(1.5)
                obs = env.get_obs()
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is unavailable")
                payload = torch.load(
                    CHECKPOINT_PATH, map_location="cpu", mmap=True,
                    pickle_module=dill, weights_only=False,
                )
                state_name = "ema_model" if cfg.training.use_ema else "model"
                policy = hydra.utils.instantiate(cfg.policy)
                policy.load_state_dict(
                    payload["state_dicts"][state_name], strict=True
                )
                del payload
                torch.cuda.set_per_process_memory_fraction(
                    GPU_MEMORY_FRACTION, device=0
                )
                policy.num_inference_steps = args.num_inference_steps
                policy.eval().cuda()
                episode_start_pose = [
                    np.concatenate([
                        obs[f"robot{robot_id}_eef_pos"],
                        obs[f"robot{robot_id}_eef_rot_axis_angle"],
                    ], axis=-1)[-1]
                    for robot_id in range(len(config["robots"]))
                ]
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs,
                    shape_meta=cfg.task.shape_meta,
                    obs_pose_repr=cfg.task.pose_repr.obs_pose_repr,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose,
                )
                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda value: torch.from_numpy(value).unsqueeze(0).cuda(),
                )
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                with torch.inference_mode():
                    policy.reset()
                    result = policy.predict_action(obs_dict)
                torch.cuda.synchronize()
                raw_action = result["action_pred"][0].cpu().numpy()
                action = get_real_umi_action(
                    raw_action, obs, cfg.task.pose_repr.action_pose_repr
                )
                if not np.isfinite(action).all():
                    raise RuntimeError("policy returned non-finite live action")
                if action.shape[-1] != 7 * len(config["robots"]):
                    raise RuntimeError(f"unexpected action shape: {action.shape}")
                current_pose = np.concatenate([
                    obs["robot0_eef_pos"][-1],
                    obs["robot0_eef_rot_axis_angle"][-1],
                ])
                current_gripper = float(obs["robot0_gripper_width"][-1, 0])
                first_action = action[0, :7]
                delta_pos_m = first_action[:3] - current_pose[:3]
                delta_rot_rad = (
                    Rotation.from_rotvec(first_action[3:6])
                    * Rotation.from_rotvec(current_pose[3:6]).inv()
                ).magnitude()
                trajectory_delta_pos_norm_m = np.linalg.norm(
                    action[:, :3] - current_pose[:3], axis=1)
                trajectory_gripper_width_m = action[:, 6]
                print("AM2PRO_LIVE_POLICY_DRYRUN_OK")
                print("observation_keys:", sorted(obs_dict_np.keys()))
                print("action_shape:", action.shape)
                print("num_inference_steps:", args.num_inference_steps)
                print(f"inference_seconds: {time.monotonic() - start:.3f}")
                print(f"peak_MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
                print("first_action_delta_pos_m:", np.round(delta_pos_m, 5).tolist())
                print(f"first_action_delta_pos_norm_m: {np.linalg.norm(delta_pos_m):.5f}")
                print(f"first_action_delta_rot_rad: {delta_rot_rad:.5f}")
                print(f"current_gripper_width_m: {current_gripper:.5f}")
                print(f"first_action_gripper_width_m: {first_action[6]:.5f}")
                print("trajectory_delta_pos_norm_m:",
                      np.round(trajectory_delta_pos_norm_m, 5).tolist())
                print("trajectory_gripper_width_m:",
                      np.round(trajectory_gripper_width_m, 5).tolist())
                print("No predicted action was sent to the robot.")
            finally:
                print("Stopping camera and holding controller...", flush=True)
                env.stop()


if __name__ == "__main__":
    main()
