#!/usr/bin/env bash
# Train the 60-demo synthetic gripper-label dataset with random model weights.
# No wall-clock timeout. Use --smoke for one optimizer step or --resume RUN_DIR.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

mode="${1:-train}"
if [[ "$mode" != "train" && "$mode" != "--smoke" && "$mode" != "--resume" ]]; then
    echo "用法: bash scripts/start_vjaw_gripper_width_scratch.sh [--smoke | --resume RUN_DIR]" >&2
    exit 2
fi
if [[ "$mode" == "--resume" && $# -ne 2 || "$mode" == "--smoke" && $# -ne 1 || "$mode" == "train" && $# -ne 0 ]]; then
    echo "启动参数不正确" >&2
    exit 2
fi

collection="data/handheld_demos_vjaw/v13_gripper_plus5_close0_20261010"
dataset="$collection/vjaw_train60_gripper_plus5_close0.zarr.zip"
if [[ ! -f "$dataset" ]]; then
    echo "训练集不存在: $dataset" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_gripper_plus5_close0_scratch_${stamp}"
if [[ "$mode" == "--smoke" ]]; then
    run="vjaw_gripper_plus5_close0_scratch_smoke_${stamp}"
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
    pidfile="$collection/latest_gripper_scratch.pid"
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
    task.dataset.seed=45
    policy.obs_encoder.pretrained=false
    dataloader.batch_size=1
    dataloader.num_workers=0
    dataloader.persistent_workers=false
    val_dataloader.batch_size=1
    val_dataloader.num_workers=0
    val_dataloader.persistent_workers=false
    +training.mixed_precision=fp16
    +training.save_resume_state=true
    training.gradient_accumulate_every=64
    training.num_epochs=200
    training.checkpoint_every=1
    training.rollout_every=1000000
    checkpoint.topk.k=3
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
    )
    python -u train.py "${args[@]}" 2>&1 | tee "$log"
    echo "SMOKE_OK: $out"
else
    if [[ "$mode" == "--resume" ]]; then
        args+=(training.resume=true +training.require_exact_resume=true)
    fi
    nohup setsid python -u train.py "${args[@]}" >"$log" 2>&1 &
    pid=$!
    echo "$pid" >"$collection/latest_gripper_scratch.pid"
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
