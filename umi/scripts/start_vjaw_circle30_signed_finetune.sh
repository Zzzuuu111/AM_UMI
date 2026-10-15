#!/usr/bin/env bash
# Fine-tune epoch-60 EMA weights on the exact corrected circle30 subset.
set -euo pipefail
cd "$(dirname "$0")/.."
usage() {
    echo "用法: bash scripts/start_vjaw_circle30_signed_finetune.sh [--epochs 30 | --smoke | --resume RUN_DIR] [--no-time-limit]"
    echo "默认 30 epochs，无墙钟超时；每 5 轮和最后一轮保存，支持完整续训。"
}
mode=train
epochs=30
epochs_set=false
resume_dir=''
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { usage >&2; exit 2; }
            epochs="$2"; epochs_set=true; shift 2 ;;
        --smoke)
            [[ "$mode" == train ]] || { usage >&2; exit 2; }
            mode=smoke; shift ;;
        --resume)
            [[ "$mode" == train && -n "${2:-}" ]] || { usage >&2; exit 2; }
            mode=resume; resume_dir="$2"; shift 2 ;;
        --no-time-limit) shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "$mode" != train && "$epochs_set" == true ]]; then
    echo "--epochs 仅用于新微调；续训沿用原运行的完整配置。" >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != AM_UMI ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi
collection='data/handheld_demos_vjaw/v14_circle30_signed_from90_20261015'
dataset="$collection/v14_circle_030_signed_zeroed_from90.zarr.zip"
init_checkpoint='data/pretrained/vjaw_signed90_epoch60/epoch=0060-train_loss=0.019.ckpt'
for required in "$dataset" "$dataset.manifest.json" "$dataset.quality_report.json"; do
    [[ -s "$required" ]] || { echo "缺少文件: $required；先运行 python scripts/fetch_vjaw_circle30_finetune_assets.py" >&2; exit 2; }
done
val_indices=$(python - "$dataset" <<'PY'
import json,sys
p=sys.argv[1]
m=json.load(open(p+'.manifest.json')); q=json.load(open(p+'.quality_report.json'))
assert m['source_sha256']=='c747a021c573b1a00dbd2c71f3e0147cf269317726bad7cdc331f9df92454d4f'
assert m['all_numeric_arrays_equal'] and m['all_decoded_image_frames_equal']
assert m['episodes']==30 and m['frames']==16208
assert m['val_episode_indices']==[5,20]
assert q['result']=='PASS' and q['episodes']==30
print(json.dumps(m['val_episode_indices'],separators=(',',':')))
PY
)
stamp=$(date +%Y%m%d_%H%M%S)
run="vjaw_circle30_signed_finetune_${stamp}"
[[ "$mode" != smoke ]] || run="vjaw_circle30_signed_finetune_smoke_${stamp}"
out="data/outputs/$run"
if [[ "$mode" == resume ]]; then
    out=$(realpath -e "$resume_dir")
    run=$(basename "$out")
    [[ -s "$out/.hydra/config.yaml" && -s "$out/checkpoints/latest.ckpt" ]] || {
        echo "续训需要原运行的 .hydra/config.yaml 和 checkpoints/latest.ckpt" >&2; exit 2; }
    log="$collection/${run}_resume_${stamp}.log"
    args=(--config-path "$out/.hydra" --config-name config
          training.resume=true ++training.require_exact_resume=true "hydra.run.dir=$out")
else
    [[ -s "$init_checkpoint" ]] || { echo "缺少 epoch 60 checkpoint，请先下载训练资源" >&2; exit 2; }
    log="$collection/$run.log"
    [[ ! -e "$out" ]] || { echo "拒绝覆盖输出目录: $out" >&2; exit 2; }
    args=(--config-name train_diffusion_transformer_umi_workspace task=umi
        "task.dataset_path=$dataset" task.dataset.seed=45
        "+task.dataset.val_episode_indices=$val_indices"
        +policy.obs_encoder.pretrained_image_norm=true
        +policy.obs_encoder.load_pretrained_weights=false
        dataloader.batch_size=1 dataloader.num_workers=0 dataloader.persistent_workers=false
        val_dataloader.batch_size=1 val_dataloader.num_workers=0 val_dataloader.persistent_workers=false
        +training.mixed_precision=fp16 +training.save_resume_state=true
        "+training.init_checkpoint='$init_checkpoint'" +training.init_weights=ema_model
        training.gradient_accumulate_every=64 "training.num_epochs=$epochs"
        training.checkpoint_every=5 training.lr_warmup_steps=100
        optimizer.lr=1e-5 optimizer.obs_encoder_lr=1e-6
        training.rollout_every=1000000 logging.mode=offline
        "exp_name=$run" "hydra.run.dir=$out")
fi
[[ ! -e "$log" ]] || { echo "拒绝覆盖日志: $log" >&2; exit 2; }
pidfile="$collection/latest_circle30_signed_finetune.pid"
if [[ -f "$pidfile" ]]; then
    old_pid=$(cat "$pidfile")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "微调进程 $old_pid 仍在运行，拒绝重复启动" >&2; exit 2
    fi
fi
export UMI_NORMALIZER_WORKERS=0 HF_HUB_OFFLINE=1 WANDB_MODE=offline
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
if [[ "$mode" == smoke ]]; then
    args+=(training.num_epochs=1 training.max_train_steps=1 training.max_val_steps=1
           training.gradient_accumulate_every=1 training.lr_warmup_steps=0
           training.sample_every=1000000 training.checkpoint_every=1)
    echo "SMOKE_START: $out"
    python -u train.py "${args[@]}" 2>&1 | tee "$log"
    echo "SMOKE_OK: $out"
    exit 0
fi
nohup setsid python -u train.py "${args[@]}" >"$log" 2>&1 &
pid=$!
sleep 5
if ! kill -0 "$pid" 2>/dev/null; then
    tail -40 "$log" >&2
    exit 1
fi
echo "$pid" >"$pidfile"
ln -sfn "$(basename "$log")" "$collection/latest_circle30_signed_finetune.log"
echo "TRAIN_STARTED PID=$pid"
echo "log: $log"
echo "output: $out"
