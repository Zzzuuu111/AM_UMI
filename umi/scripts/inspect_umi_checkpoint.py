"""Inspect the saved structure of a UMI checkpoint without loading tensors."""

import argparse
from pathlib import Path

import torch


DEFAULT_CHECKPOINT = Path(
    "/home/zzzjh/universal_manipulation_interface/data/outputs/2026.08.27/"
    "14.10.09_train_diffusion_unet_image_umi/checkpoints/"
    "epoch=0060-test_mean_score=-0.019.ckpt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", type=Path,
                        default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="meta", mmap=True, weights_only=False
    )
    print("checkpoint:", args.checkpoint)
    print("top-level:", list(checkpoint.keys()))
    for name, value in checkpoint.items():
        keys = list(value.keys()) if hasattr(value, "keys") else None
        print(f"{name}: {type(value).__name__}", keys if keys is not None else "")


if __name__ == "__main__":
    main()
