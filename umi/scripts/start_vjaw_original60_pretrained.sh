#!/usr/bin/env bash
# Train the upstream UMI ViT-Base/action-Transformer architecture on V13-60.
# Use --smoke for one optimizer step or --resume RUN_DIR after interruption.
# Add --no-time-limit to a new or resumed run to skip the 12-hour wall-time cap.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

mode="${1:-train}"
if [[ "$mode" == "--no-time-limit" ]]; then
    mode="train"
fi
if [[ "$mode" != "train" && "$mode" != "--smoke" && "$mode" != "--resume" ]]; then
    echo "用法: bash scripts/start_vjaw_original60_pretrained.sh [--no-time-limit | --smoke | --resume RUN_DIR [--no-time-limit]]" >&2
    exit 2
fi
if [[ "$mode" == "--resume" && ( $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != "--no-time-limit" ) ) ]]; then
    echo "--resume 需要原训练输出目录，可选 --no-time-limit" >&2
    exit 2
fi
if [[ "$mode" == "train" && $# -ne 0 && ( $# -ne 1 || "$1" != "--no-time-limit" ) || "$mode" == "--smoke" && $# -ne 1 ]]; then
    echo "启动参数不正确" >&2
    exit 2
fi

collection="data/handheld_demos_vjaw/v13_train60_20261008"
dataset="$collection/v13_train_060.zarr.zip"
if [[ ! -f "$dataset" ]]; then
    echo "训练集不存在: $dataset" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_original60_pretrained_${stamp}"
if [[ "$mode" == "--smoke" ]]; then
    run="vjaw_original60_pretrained_smoke_${stamp}"
fi
out="data/outputs/$run"
log="$collection/$run.log"
if [[ "$mode" == "--resume" ]]; then
    out="${2%/}"
    run="$(basename "$out")"
    log="$collection/${run}_resume_${stamp}.log"
    if [[ ! -f "$out/checkpoints/latest.ckpt" ]]; then
        echo "续训 checkpoint 不存在: $out/checkpoints/latest.ckpt" >&2
        exit 2
    fi
    pidfile="$collection/latest_original60_pretrained.pid"
    if [[ -f "$pidfile" ]]; then
        old_pid=$(cat "$pidfile")
        if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "训练进程 $old_pid 仍在运行，拒绝并发写同一目录" >&2
            exit 2
        fi
    fi
elif [[ -e "$out" || -e "$log" ]]; then
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
    +policy.obs_encoder.pretrained_image_norm=true
    dataloader.batch_size=1
    dataloader.num_workers=0
    dataloader.persistent_workers=false
    val_dataloader.batch_size=1
    val_dataloader.num_workers=0
    val_dataloader.persistent_workers=false
    +training.mixed_precision=fp16
    +training.save_resume_state=true
    training.gradient_accumulate_every=64
    training.rollout_every=1000000
    logging.mode=offline
    "exp_name=$run"
    "hydra.run.dir=$out"
)

if [[ "$mode" == "--smoke" ]]; then
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
else
    if [[ "$mode" == "--resume" ]]; then
        args+=(training.resume=true +training.require_exact_resume=true)
    fi
    if [[ "${1:-}" == "--no-time-limit" || "${3:-}" == "--no-time-limit" ]]; then
        nohup setsid python -u train.py "${args[@]}" >"$log" 2>&1 &
    else
        nohup timeout --signal=TERM --kill-after=120s 12h \
            python -u train.py "${args[@]}" >"$log" 2>&1 &
    fi
    pid=$!
    if [[ "${1:-}" == "--no-time-limit" || "${3:-}" == "--no-time-limit" ]]; then
        sleep 5
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "无时限训练启动后已退出，日志: $log" >&2
            tail -40 "$log" >&2
            exit 1
        fi
    fi
    echo "$pid" >"$collection/latest_original60_pretrained.pid"
    ln -sfn "$(basename "$log")" "$collection/latest_original60_pretrained.log"
    echo "TRAIN_STARTED PID=$pid"
    echo "log: $log"
    echo "output: $out"
fi
