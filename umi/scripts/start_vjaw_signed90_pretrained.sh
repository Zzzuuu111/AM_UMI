#!/usr/bin/env bash
# Train a new policy on the 90-demo signed-gripper-label candidate dataset.
# The vision encoder uses its configured pretrained weights; policy weights
# start fresh. No checkpoint trained on the old gripper labels is imported.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

mode="${1:-train}"
if [[ "$mode" == "--no-time-limit" ]]; then mode="train"; fi
if [[ "$mode" != "train" && "$mode" != "--smoke" && "$mode" != "--resume" ]]; then
    echo "用法: bash scripts/start_vjaw_signed90_pretrained.sh [--smoke | --no-time-limit | --resume RUN_DIR [--no-time-limit]]" >&2
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

collection="data/handheld_demos_vjaw/v14_train90_circle_20261013"
dataset="$collection/v14_train_090_signed_gripper_zeroed_candidate.zarr.zip"
quality="$dataset.quality_report.json"
if [[ ! -f "$dataset" || ! -f "$quality" ]]; then
    echo "修正数据集或质量报告不存在" >&2
    exit 2
fi
if ! python -c 'import json,sys; q=json.load(open(sys.argv[1])); sys.exit(0 if q.get("result")=="PASS" and q.get("episodes")==90 else 1)' "$quality"; then
    echo "修正数据集未通过 90 条质量检查" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_signed90_pretrained_${stamp}"
if [[ "$mode" == "--smoke" ]]; then run="vjaw_signed90_pretrained_smoke_${stamp}"; fi
out="data/outputs/$run"
log="$collection/$run.log"
if [[ "$mode" == "--resume" ]]; then
    out="${2%/}"
    run="$(basename "$out")"
    log="$collection/${run}_resume_${stamp}.log"
    if [[ ! -f "$out/checkpoints/latest.ckpt" ]]; then
        echo "完整续训 checkpoint 不存在: $out/checkpoints/latest.ckpt" >&2
        exit 2
    fi
elif [[ -e "$out" || -e "$log" ]]; then
    echo "拒绝覆盖已有输出: $out 或 $log" >&2
    exit 2
fi

pidfile="$collection/latest_signed90_pretrained.pid"
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
    training.gradient_accumulate_every=64
    training.num_epochs=200
    training.checkpoint_every=10
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
    exit 0
fi

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
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
    echo "训练启动后已退出，日志: $log" >&2
    tail -40 "$log" >&2
    exit 1
fi
echo "$pid" >"$pidfile"
ln -sfn "$(basename "$log")" "$collection/latest_signed90_pretrained.log"
echo "TRAIN_STARTED PID=$pid"
echo "log: $log"
echo "output: $out"
