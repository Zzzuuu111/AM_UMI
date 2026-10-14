#!/usr/bin/env bash
# Supervised trial of the original 60-demo policy fine-tuned on 30 circle demos.
set -euo pipefail

cd "$(dirname "$0")/.."
usage() {
    echo "用法: bash scripts/start_vjaw_circle30_supervised_trial.sh [--smoke | --full | --duration 秒数]" >&2
}
if [[ $# -gt 2 ]]; then
    usage
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

checkpoint="data/outputs/vjaw_circle30_finetune_20261013_163842/checkpoints/latest.ckpt"
robot_config="example/eval_robots_config_vjaw_ros2.yaml"
start_reference="calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json"
tcp_transform="calibration/robot_wrist_camera/tcp/vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json"
duration=120
run_name="vjaw_circle30_live_full"
if [[ "${1:-}" == "--smoke" ]]; then
    if [[ $# -ne 1 ]]; then
        usage
        exit 2
    fi
    duration=5
    run_name="vjaw_circle30_live_smoke"
elif [[ "${1:-}" == "--full" ]]; then
    if [[ $# -ne 1 ]]; then
        usage
        exit 2
    fi
elif [[ "${1:-}" == "--duration" ]]; then
    if [[ $# -ne 2 || ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        usage
        exit 2
    fi
    duration="$2"
    run_name="vjaw_circle30_live_${duration}s"
elif [[ $# -ne 0 ]]; then
    usage
    exit 2
fi
output="data/outputs/${run_name}_$(date +%Y%m%d_%H%M%S)"
log="${output}.log"
if [[ ! -f "$checkpoint" || ! -f "$robot_config" || ! -f "$start_reference" || ! -f "$tcp_transform" ]]; then
    echo "模型、机械臂配置、v6 起始基准或 TCP 坐标转换不存在" >&2
    exit 2
fi
if [[ -e "$output" || -e "$log" ]]; then
    echo "拒绝覆盖已有输出: $output 或 $log" >&2
    exit 2
fi

export HF_HUB_OFFLINE=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
echo "模型: $checkpoint"
echo "本次输出目录: $output"
echo "本次最多推理 ${duration} 秒；启动后低速回到 v6 基准，将夹爪闭合到 0 mm 并检查读数，然后自动开始。"
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
