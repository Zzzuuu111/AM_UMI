"""Restore the model weights from the archived UMI checkpoint on CPU only."""

import argparse
import pathlib
import sys

import torch
from hydra.utils import instantiate


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Registers the OmegaConf ${eval:...} resolver used in the saved config.
import diffusion_policy.workspace.train_diffusion_unet_image_workspace  # noqa: F401,E402


DEFAULT_CHECKPOINT = pathlib.Path(
    "/home/zzzjh/universal_manipulation_interface/data/outputs/2026.08.27/"
    "14.10.09_train_diffusion_unet_image_umi/checkpoints/"
    "epoch=0060-test_mean_score=-0.019.ckpt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=pathlib.Path,
                        default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    # mmap prevents eager tensor copies while the checkpoint is read.  Only the
    # policy state is restored; the archived AdamW state is intentionally left
    # untouched for this compatibility check.
    payload = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    policy = instantiate(payload["cfg"].policy)
    policy.load_state_dict(payload["state_dicts"]["model"], strict=True)

    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    print("CHECKPOINT_MODEL_RESTORE_OK")
    print("checkpoint:", args.checkpoint)
    print("parameters:", parameter_count)
    print("device:", next(policy.parameters()).device)


if __name__ == "__main__":
    main()
