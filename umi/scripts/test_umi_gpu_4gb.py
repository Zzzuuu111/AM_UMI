"""Run one UMI FP16 training update under the AM_UMI 4 GB GPU ceiling.

This is a hardware/environment smoke test, not a full training launcher: it
does not start W&B, instantiate a real robot runner, or save checkpoints.
"""

import os
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import default_collate

# Importing the workspace registers its ${eval:...} OmegaConf resolver and the
# low-memory optimizer constructor used by the experiment profile.
import diffusion_policy.workspace.train_diffusion_unet_image_workspace  # noqa: F401,E402
from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the AM_UMI environment")

    initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT_DIR / "diffusion_policy" / "config"),
    )
    cfg = compose(
        config_name="train_diffusion_unet_image_workspace",
        overrides=[
            "task=umi",
            "task.dataset_path=test_data/replay_case_0099/"
            "replay_case_0099_dataset.zarr",
            "+experiment=am_umi_gpu_4gb",
        ],
    )

    torch.cuda.set_per_process_memory_fraction(
        float(cfg.training.gpu_memory_fraction), device=0
    )
    dataset = instantiate(cfg.task.dataset)
    policy = instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy = policy.cuda()

    batch = dict_apply(
        default_collate([dataset[0]]),
        lambda value: value.cuda(non_blocking=True),
    )
    optimizer = instantiate(cfg.optimizer, params=policy.parameters())
    scaler = torch.amp.GradScaler("cuda")

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = policy.compute_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()

    free, _ = torch.cuda.mem_get_info()
    print(
        "GPU_4GB_CEILING_TRAIN_STEP_OK "
        f"loss={loss.item():.6f} "
        f"peak_MiB={torch.cuda.max_memory_allocated() / 1024**2:.1f} "
        f"reserved_MiB={torch.cuda.max_memory_reserved() / 1024**2:.1f} "
        f"free_MiB={free / 1024**2:.1f} "
        f"cap_fraction={cfg.training.gpu_memory_fraction}"
    )


if __name__ == "__main__":
    main()
