#!/usr/bin/env bash
# Supervised trial of a numbered checkpoint from the corrected 90-demo run.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
    echo "用法: bash scripts/start_vjaw_signed90_supervised_trial.sh [--epoch 60] [--smoke | --duration 秒数]"
    echo "默认: epoch 60，最多推理 120 秒；--smoke 为 5 秒。"
}
epoch=60
duration=120
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epoch)
            if [[ ! "${2:-}" =~ ^[0-9]{1,3}$ ]]; then
                usage >&2
                exit 2
            fi
            epoch=$((10#$2))
            if (( epoch > 199 )); then
                echo "本轮共 200 epochs，编号为 0..199" >&2
                exit 2
            fi
            shift 2
            ;;
        --smoke) duration=5; shift ;;
        --full) duration=120; shift ;;
        --duration)
            if [[ ! "${2:-}" =~ ^[1-9][0-9]*$ ]]; then
                usage >&2
                exit 2
            fi
            duration="$2"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi
printf -v epoch_tag '%04d' "$epoch"
checkpoint_dir="data/outputs/vjaw_signed90_pretrained_20261014_101005/checkpoints"
shopt -s nullglob
checkpoints=("$checkpoint_dir"/epoch="${epoch_tag}"-train_loss=*.ckpt)
if [[ ${#checkpoints[@]} -ne 1 ]]; then
    echo "需要且只能找到一个 epoch ${epoch} 的具名 checkpoint；当前找到 ${#checkpoints[@]} 个。" >&2
    echo "请确认该 epoch 已训练结束并保存完成：$checkpoint_dir" >&2
    exit 2
fi
checkpoint="${checkpoints[0]}"
robot_config="example/eval_robots_config_vjaw_ros2.yaml"
start_reference="calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json"
tcp_transform="calibration/robot_wrist_camera/tcp/vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json"
for required in "$checkpoint" "$robot_config" "$start_reference" "$tcp_transform"; do
    if [[ ! -s "$required" ]]; then
        echo "文件不存在或为空: $required" >&2
        exit 2
    fi
done
output="data/outputs/vjaw_signed90_epoch${epoch_tag}_live_${duration}s_$(date +%Y%m%d_%H%M%S)"
log="${output}.log"
if [[ -e "$output" || -e "$log" ]]; then
    echo "拒绝覆盖已有输出: $output 或 $log" >&2
    exit 2
fi

export HF_HUB_OFFLINE=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
echo "模型: $checkpoint"
echo "数据版本: 修正夹爪标签后的 90 条示教"
echo "本次输出目录: $output"
echo "本次最多推理 ${duration} 秒；启动后低速回到 v6 基准并对齐基准夹爪宽度，读数检查通过后自动开始。"
echo "正常停止或超时后，停稳并低速回到 v6 基准；异常或 Ctrl+C 时原地保持。"
echo "运行中可在此终端输入 s 或 x 后回车停止。"
python -u eval_real.py \
    -i "$checkpoint" \
    -o "$output" \
    -rc "$robot_config" \
    --no_spacemouse \
    --terminal-controls \
    --start-reference "$start_reference" \
    --move-to-reference \
    --hold-on-stop \
    --single-run-return-to-reference \
    --auto-confirm-reference \
    --save-policy-trace \
    --policy-right-tcp-to-demo-tcp "$tcp_transform" \
    --allow-candidate-policy-tcp-transform \
    --max_duration "$duration" \
    --max_pos_speed 0.03 \
    --max_rot_speed 0.15 \
    --max_gripper_speed 0.02 \
    --queue-policy-horizon \
    --policy-execution-latency 0.15 \
    --steps_per_inference 6 \
    --frequency 10 \
    --policy-sampling-seed 42 \
    --smooth-action-horizon 2>&1 | tee "$log"
