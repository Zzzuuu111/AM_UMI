"""Run one GPU inference from the archived UMI checkpoint and copied Zarr."""

import argparse
import contextlib
import os
import pathlib
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import default_collate


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

# Registers the saved config's ${eval:...} resolver.
import diffusion_policy.workspace.train_diffusion_unet_image_workspace  # noqa: F401,E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


DEFAULT_CHECKPOINT = pathlib.Path(
    "/home/zzzjh/universal_manipulation_interface/data/outputs/2026.08.27/"
    "14.10.09_train_diffusion_unet_image_umi/checkpoints/"
    "epoch=0060-test_mean_score=-0.019.ckpt"
)
TEST_DATASET = "test_data/replay_case_0099/replay_case_0099_dataset.zarr"
GPU_MEMORY_FRACTION = 0.45


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=pathlib.Path,
                        default=DEFAULT_CHECKPOINT)
    parser.add_argument("--precision", choices=("fp32", "fp16"),
                        default="fp32")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the AM_UMI environment")

    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
    payload = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    cfg = payload["cfg"]
    # The archived config has an already-resolved nested path, whereas current
    # configs interpolate this value from task.dataset_path.  Replace both so
    # either format always uses the copied AM_UMI replay data.
    OmegaConf.update(cfg, "task.dataset_path", TEST_DATASET, merge=False)
    OmegaConf.update(cfg, "task.dataset.dataset_path", TEST_DATASET,
                     merge=False)

    policy = instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"]["model"], strict=True)
    del payload

    dataset = instantiate(cfg.task.dataset)
    obs = dict_apply(
        default_collate([dataset[0]])["obs"],
        lambda value: value.cuda(non_blocking=True),
    )
    policy = policy.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if args.precision == "fp16" else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast:
        result = policy.predict_action(obs)
    torch.cuda.synchronize()

    action = result.get("action_pred", result.get("action"))
    if action is None:
        raise RuntimeError(f"model returned no action; keys={list(result)}")
    if not torch.isfinite(action).all():
        status = {
            key: bool(torch.isfinite(value).all())
            for key, value in result.items() if torch.is_tensor(value)
        }
        raise RuntimeError(f"model returned non-finite action; finite={status}")
    free, _ = torch.cuda.mem_get_info()
    print("CHECKPOINT_GPU_INFERENCE_OK")
    print("action_shape:", tuple(action.shape))
    print("action_dtype:", action.dtype)
    print("precision:", args.precision)
    print(f"peak_MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")
    print(f"reserved_MiB: {torch.cuda.max_memory_reserved() / 1024**2:.1f}")
    print(f"free_MiB: {free / 1024**2:.1f}")
    print(f"cap_fraction: {GPU_MEMORY_FRACTION}")


if __name__ == "__main__":
    main()
