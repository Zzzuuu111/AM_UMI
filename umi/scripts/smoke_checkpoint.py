"""
Smoke-test a trained checkpoint WITHOUT hardware: load the policy on CPU,
feed a synthetic observation batch, run one diffusion rollout, and report
the action output (shape / finite / value range).

Usage (umi_torch27 env):
    python scripts/smoke_checkpoint.py \
        --run_dir data/outputs/2026.08.27/14.10.09_train_diffusion_unet_image_umi \
        [--ckpt epoch=0010-test_mean_score=-0.025.ckpt]   # default: latest.ckpt

Safe to run while training (CPU only, no robot, no camera).
"""
import argparse
import pathlib
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT_DIR)

# allow ${eval:} resolvers in configs
OmegaConf.register_new_resolver("eval", eval, replace=True)
# hydra's built-in `now` resolver is not available outside hydra.main; stub it.
OmegaConf.register_new_resolver("now", lambda pattern: "", replace=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="Hydra training run directory")
    parser.add_argument("--ckpt", default="latest.ckpt", help="Checkpoint filename in run_dir/checkpoints")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / args.ckpt
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    OmegaConf.resolve(cfg)

    print(f"[1/4] building policy from {run_dir.name} ...")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.eval()

    print(f"[2/4] loading {ckpt_path.name} ...")
    # weights_only=False: our own checkpoints embed omegaconf DictConfig
    # objects, which torch 2.6+ refuses under the default weights_only=True.
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dicts = payload.get("state_dicts", payload)
    # prefer EMA weights (training uses EMA for evaluation)
    if isinstance(state_dicts, dict) and "ema" in state_dicts:
        sd = state_dicts["ema"]
        print("      using EMA weights")
    elif isinstance(state_dicts, dict) and "model" in state_dicts:
        sd = state_dicts["model"]
        print("      using raw model weights (no EMA found)")
    else:
        sd = state_dicts
        print("      raw state dict")
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    if missing:
        print(f"      missing keys: {len(missing)} e.g. {missing[:3]}")
    if unexpected:
        print(f"      unexpected keys: {len(unexpected)} e.g. {unexpected[:3]}")

    print("[3/4] building synthetic obs batch (B=2) ...")
    B = 2
    shape_meta = cfg.task.shape_meta
    obs_dict = {}
    for key, attr in shape_meta.obs.items():
        horizon = attr.horizon
        shape = attr.shape
        if attr.type == "rgb":
            obs_dict[key] = torch.rand(B, horizon, *shape, dtype=torch.float32)
        else:
            obs_dict[key] = torch.zeros(B, horizon, *shape, dtype=torch.float32)

    print("[4/4] running one diffusion rollout on CPU (may take 1-3 min) ...")
    with torch.no_grad():
        result = policy.predict_action(obs_dict)
    action = result["action_pred"].cpu().numpy()

    print("\n========== SMOKE RESULT ==========")
    print(f"action shape : {action.shape}  (expect (B=2, n_action_steps=8, 10))")
    print(f"all finite   : {np.isfinite(action).all()}")
    print(f"pos range    : {action[..., :3].min():.4f} ~ {action[..., :3].max():.4f}")
    print(f"rot range    : {action[..., 3:9].min():.4f} ~ {action[..., 3:9].max():.4f}")
    print(f"width range  : {action[..., 9].min():.4f} ~ {action[..., 9].max():.4f}")
    print("===================================")
    ok = np.isfinite(action).all()
    print("SMOKE OK: checkpoint loads and produces finite actions" if ok else "SMOKE FAILED: non-finite output")


if __name__ == "__main__":
    main()
