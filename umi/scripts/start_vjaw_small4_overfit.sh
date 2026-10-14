#!/usr/bin/env bash
# Start a small, offline-only-data Transformer overfit diagnostic.
# The run intentionally trains on all 4 demos to test whether the policy can
# overfit a tiny dataset. Generalization is not measured by this diagnostic.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI；当前环境: ${CONDA_DEFAULT_ENV:-未激活}" >&2
    exit 2
fi

dataset="data/handheld_demos_vjaw/v12_train30_20260921/small_overfit_4demo_20260923/vjaw_small_004.zarr.zip"
if [[ ! -f "$dataset" ]]; then
    echo "小数据集文件不存在: $dataset" >&2
    exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_transformer_small4_overfit_$stamp"
out="data/outputs/$run"
log="data/handheld_demos_vjaw/v12_train30_20260921/small_overfit_4demo_20260923/$run.log"

if [[ -e "$out" || -e "$log" ]]; then
    echo "拒绝覆盖已有实验输出: $out 或 $log" >&2
    exit 2
fi

export UMI_NORMALIZER_WORKERS=0
nohup python -u train.py \
    --config-name train_diffusion_transformer_umi_workspace \
    task=umi \
    task.dataset_path="$dataset" \
    task.dataset.val_ratio=0.0 \
    +experiment=am_umi_transformer_gpu_6gb \
    training.num_epochs=80 \
    training.lr_warmup_steps=100 \
    training.checkpoint_every=5 \
    checkpoint.topk.k=3 \
    training.rollout_every=1000000 \
    training.sample_every=10 \
    logging.mode=offline \
    exp_name="$run" \
    hydra.run.dir="$out" \
    >"$log" 2>&1 &

pid=$!
echo "VJAW_SMALL_OVERFIT_STARTED"
echo "PID: $pid"
echo "dataset: $dataset"
echo "log: $log"
echo "model/checkpoints: $out/checkpoints/"
