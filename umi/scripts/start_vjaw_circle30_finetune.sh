#!/usr/bin/env bash
# Fine-tune the original 60-demo EMA policy using only 30 new circle demos.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi
if [[ $# -gt 0 && "$1" != "--smoke" && "$1" != "--resume" ]] || \
   [[ "${1:-}" == "--smoke" && $# -ne 1 ]] || \
   [[ "${1:-}" == "--resume" && $# -ne 2 ]] || \
   [[ $# -gt 2 ]]; then
    echo "用法: bash scripts/start_vjaw_circle30_finetune.sh [--smoke | --resume RUN_DIR]" >&2
    exit 2
fi

collection="data/handheld_demos_vjaw/v14_circle30_20261013"
dataset="$collection/v14_circle_030.zarr.zip"
init_checkpoint="data/outputs/vjaw_original60_finetune_20261008_181742/checkpoints/latest.ckpt"
if [[ ! -f "$dataset" || ! -f "$init_checkpoint" ]]; then
    echo "缺少新 30 条训练集或原 60 条 EMA checkpoint" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_circle30_finetune_${stamp}"
if [[ "${1:-}" == "--smoke" ]]; then
    run="vjaw_circle30_finetune_smoke_${stamp}"
fi
out="data/outputs/$run"
log="$collection/$run.log"
pidfile="$collection/latest_circle30_finetune.pid"
if [[ "${1:-}" == "--resume" ]]; then
    out="${2%/}"
    run="$(basename "$out")"
    log="$collection/${run}_resume_${stamp}.log"
    if [[ ! -f "$out/checkpoints/latest.ckpt" ]]; then
        echo "完整续训 checkpoint 不存在: $out/checkpoints/latest.ckpt" >&2
        exit 2
    fi
else
    if [[ -e "$out" || -e "$log" ]]; then
        echo "拒绝覆盖已有输出: $out 或 $log" >&2
        exit 2
    fi
fi
if [[ -f "$pidfile" ]]; then
    old_pid=$(cat "$pidfile")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "训练进程 $old_pid 仍在运行，拒绝并发启动" >&2
        exit 2
    fi
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
    training.num_epochs=70
    training.checkpoint_every=5
    training.lr_warmup_steps=100
    optimizer.lr=1e-5
    optimizer.obs_encoder_lr=1e-6
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
        training.checkpoint_every=1
    )
    echo "SMOKE_START: $out"
    python -u train.py "${args[@]}" 2>&1 | tee "$log"
    echo "SMOKE_OK: $out"
    exit 0
fi
if [[ "${1:-}" == "--resume" ]]; then
    args+=(training.resume=true +training.require_exact_resume=true)
fi
nohup setsid python -u train.py "${args[@]}" >"$log" 2>&1 &
pid=$!
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
    echo "训练启动后已退出，日志: $log" >&2
    tail -40 "$log" >&2
    exit 1
fi
echo "$pid" >"$pidfile"
ln -sfn "$(basename "$log")" "$collection/latest_circle30_finetune.log"
echo "TRAIN_STARTED PID=$pid"
echo "log: $log"
echo "output: $out"
