#!/usr/bin/env bash
# Start the screened 30-episode V-jaw Transformer training run in background.
# Usage (after conda activate AM_UMI): bash scripts/start_vjaw_transformer_6h.sh
set -euo pipefail

cd "$(dirname "$0")/.."

collection="data/handheld_demos_vjaw/v12_train30_20260921"
dataset="$collection/v12_train_030.zarr.zip"
if [ ! -f "$dataset" ]; then
    echo "训练集不存在: $dataset" >&2
    exit 2
fi
if [ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]; then
    echo "请先执行 conda activate AM_UMI；当前环境: ${CONDA_DEFAULT_ENV:-未激活}" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_v12_030_transformer_6h_$stamp"
log="$collection/$run.log"
out="data/outputs/$run"
latest_log="$collection/latest_transformer_6h.log"
latest_pid="$collection/latest_transformer_6h.pid"

export UMI_NORMALIZER_WORKERS=0
nohup timeout --signal=TERM --kill-after=120s 6h \
    python -u train.py \
    --config-name train_diffusion_transformer_umi_workspace \
    task=umi \
    task.dataset_path="$dataset" \
    +experiment=am_umi_transformer_gpu_6gb \
    training.num_epochs=80 \
    training.lr_warmup_steps=1000 \
    training.checkpoint_every=5 \
    checkpoint.topk.k=3 \
    training.rollout_every=1000000 \
    training.sample_every=20 \
    logging.mode=offline \
    exp_name="$run" \
    hydra.run.dir="$out" \
    >"$log" 2>&1 &

pid=$!
ln -sfn "$(basename "$log")" "$latest_log"
printf '%s\n' "$pid" > "$latest_pid"
echo "VJAW_TRAINING_STARTED"
echo "PID: $pid"
echo "log: $log"
echo "tail -f $latest_log"
