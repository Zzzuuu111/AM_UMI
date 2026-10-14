#!/usr/bin/env bash
# Short fine-tune on synthetic gripper labels; isolated from the original run.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--smoke" ) ]]; then
    echo "用法: bash scripts/start_vjaw_gripper_width_ablation.sh [--smoke]" >&2
    exit 2
fi

collection="data/handheld_demos_vjaw/v13_gripper_plus5_close0_20261010"
dataset="$collection/vjaw_train60_gripper_plus5_close0.zarr.zip"
init_checkpoint="data/outputs/vjaw_original60_finetune_20261008_181742/checkpoints/latest.ckpt"
stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_gripper_plus5_close0_5epoch_${stamp}"
if [[ "${1:-}" == "--smoke" ]]; then
    run="vjaw_gripper_plus5_close0_smoke_${stamp}"
fi
out="data/outputs/$run"
log="$collection/$run.log"
if [[ ! -f "$dataset" || ! -f "$init_checkpoint" ]]; then
    echo "测试数据或初始化模型不存在" >&2
    exit 2
fi
if [[ -e "$out" || -e "$log" ]]; then
    echo "拒绝覆盖已有输出: $out 或 $log" >&2
    exit 2
fi

export UMI_NORMALIZER_WORKERS=0
export HF_HUB_OFFLINE=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
args=(
    --config-name train_diffusion_transformer_umi_workspace
    task=umi
    "task.dataset_path=$dataset"
    task.dataset.seed=45
    +policy.obs_encoder.pretrained_image_norm=true
    dataloader.batch_size=1
    dataloader.num_workers=0
    dataloader.persistent_workers=false
    val_dataloader.batch_size=1
    val_dataloader.num_workers=0
    val_dataloader.persistent_workers=false
    +training.mixed_precision=fp16
    +training.save_resume_state=true
    "+training.init_checkpoint=$init_checkpoint"
    +training.init_weights=ema_model
    training.gradient_accumulate_every=64
    training.num_epochs=5
    training.checkpoint_every=1
    training.lr_warmup_steps=50
    optimizer.lr=3e-5
    optimizer.obs_encoder_lr=3e-6
    training.rollout_every=1000000
    logging.mode=offline
    "exp_name=$run"
    "hydra.run.dir=$out"
)
if [[ "${1:-}" == "--smoke" ]]; then
    args+=(
        training.num_epochs=1
        training.max_train_steps=1
        training.gradient_accumulate_every=1
        training.lr_warmup_steps=0
        training.sample_every=1000000
    )
    python -u train.py "${args[@]}" 2>&1 | tee "$log"
    echo "SMOKE_OK: $out"
else
    nohup setsid python -u train.py "${args[@]}" >"$log" 2>&1 &
    pid=$!
    echo "$pid" >"$collection/latest_gripper_ablation.pid"
    sleep 5
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "训练启动后退出，日志: $log" >&2
        tail -40 "$log" >&2
        exit 1
    fi
    echo "TRAIN_STARTED PID=$pid"
    echo "log: $log"
    echo "output: $out"
fi
