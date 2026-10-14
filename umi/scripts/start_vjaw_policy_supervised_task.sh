#!/usr/bin/env bash
# Supervised AM2Pro task trial with video, a time cap, and a slow reference return.
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--smoke" ) ]]; then
    echo "用法: bash scripts/start_vjaw_policy_supervised_task.sh [--smoke]" >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "AM_UMI" ]]; then
    echo "请先执行 conda activate AM_UMI" >&2
    exit 2
fi

checkpoint="data/outputs/vjaw_original60_finetune_20261008_181742/checkpoints/latest.ckpt"
robot_config="example/eval_robots_config_vjaw_ros2.yaml"
start_reference="calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint/reference.json"
tcp_transform="calibration/robot_wrist_camera/tcp/vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json"
duration=120
run_name="vjaw_policy_live_task"
run_mode_args=(--single-run-return-to-reference --auto-confirm-reference)
tcp_transform_args=(--allow-candidate-policy-tcp-transform)
if [[ "${1:-}" == "--smoke" ]]; then
    duration=5
    run_name="vjaw_policy_stream_smoke"
    run_mode_args=()
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
echo "本次输出目录: $output"
echo "启动后低速回到 v6 起始基准，并检查关节及 TCP 读数。"
echo "把未来预测点排成连续轨迹；下一轮推理替换未执行部分。"
echo "按观测时间戳跳过过时点，预留 0.15 秒执行余量；同向运动时跳过落在当前位置后方的点。"
echo "若新预测要求反向，最近三轮新观测中至少两轮同意才允许低速返回。"
echo "推理使用 checkpoint 的去噪步数、固定采样种子和三点轨迹平滑。"
echo "使用与 fixed-tip replay 相同的候选 TCP 转换；爪尖点位可能仍有毫米级偏差。"
if [[ "${1:-}" == "--smoke" ]]; then
    echo "本次最多 ${duration} 秒。"
    echo "在此终端输入 c 后回车开始；输入 s 后回车停止；输入 x 后回车退出。"
    echo "停止后保持当前位姿和夹爪宽度，不自动返回起点。"
else
    echo "本次最多推理 ${duration} 秒；启动后无需输入 y 或 c，回基准检查通过便自动开始推理。"
    echo "看到任务完成时在此终端输入 s 或 x 后回车；超时也会停稳并低速回到 v6 基准，随后退出程序。"
    echo "Ctrl+C 或异常时停在当前位置，不自动返回。"
fi
python -u eval_real.py \
    -i "$checkpoint" \
    -o "$output" \
    -rc "$robot_config" \
    --no_spacemouse \
    --terminal-controls \
    --start-reference "$start_reference" \
    --move-to-reference \
    --hold-on-stop \
    "${run_mode_args[@]}" \
    --save-policy-trace \
    --policy-right-tcp-to-demo-tcp "$tcp_transform" \
    "${tcp_transform_args[@]}" \
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
