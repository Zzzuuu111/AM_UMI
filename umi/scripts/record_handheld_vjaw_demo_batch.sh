#!/usr/bin/env bash
# Interactively record a numbered batch of visual-only V-jaw demonstrations.
#
# Every saved session is automatically processed and screened offline before
# the next requested sample number is used.  It never opens a robot controller.
# Physical empty-workspace replay remains a deliberate, supervised later gate.
#
# Usage:
#   bash scripts/record_handheld_vjaw_demo_batch.sh \
#     --count 30 --first-demo-number 31 --first-sample-index 1 \
#     --reference-dir calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint

set -euo pipefail

count=30
first_demo_number=31
first_sample_index=1
prefix="v12_train"
reference_dir="calibration/robot_wrist_camera/view_references/follow_umi_vjaw_start_v6_safe_midpoint"
auto_screen=1
collection_dir=""

usage() {
    echo "用法: bash scripts/record_handheld_vjaw_demo_batch.sh [选项]"
    echo "  --count N                 要录制的候选条数，默认 30"
    echo "  --first-demo-number N     第一条的 demo 编号，默认 31"
    echo "  --first-sample-index N    第一条的登记编号，默认 1"
    echo "  --prefix NAME             会话名中 demo 编号后的名称，默认 v12_train"
    echo "  --reference-dir DIR       V 型夹爪基准目录"
    echo "  --collection-dir NAME     在 data/handheld_demos_vjaw/ 下新建/使用的集合子目录"
    echo "  --no-auto-screen          只录制，不自动执行视觉/IK 离线筛选"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --count) count="$2"; shift 2 ;;
        --first-demo-number) first_demo_number="$2"; shift 2 ;;
        --first-sample-index) first_sample_index="$2"; shift 2 ;;
        --prefix) prefix="$2"; shift 2 ;;
        --reference-dir) reference_dir="$2"; shift 2 ;;
        --collection-dir) collection_dir="$2"; shift 2 ;;
        --no-auto-screen) auto_screen=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数: $1"; usage; exit 2 ;;
    esac
done

if ! [[ "$count" =~ ^[1-9][0-9]*$ ]] || ! [[ "$first_demo_number" =~ ^[0-9]+$ ]] || \
   ! [[ "$first_sample_index" =~ ^[0-9]+$ ]]; then
    echo "--count、--first-demo-number、--first-sample-index 必须是正整数"
    exit 2
fi

cd "$(dirname "$0")/.."
if [ ! -f "$reference_dir/reference.json" ]; then
    echo "找不到 V 型夹爪 reference.json: $reference_dir/reference.json"
    exit 2
fi
if [[ "$collection_dir" == /* || "$collection_dir" == *".."* ]]; then
    echo "--collection-dir 必须是 data/handheld_demos_vjaw 下的相对单层目录名"
    exit 2
fi
session_root="data/handheld_demos_vjaw"
if [ -n "$collection_dir" ]; then
    if [[ "$collection_dir" == */* ]]; then
        echo "--collection-dir 当前只接受单层目录名，例如 v13_train15"
        exit 2
    fi
    session_root="$session_root/$collection_dir"
fi

echo "VJAW_BATCH_RECORDING_READY"
echo "候选条数: $count；demo 编号: $first_demo_number..$((first_demo_number + count - 1))"
echo "集合目录: $session_root"
if [ "$auto_screen" -eq 1 ]; then
    echo "每条：R 开始、S 保存；随后自动转换 + 固定 TCP 离线 IK 筛选。失败会保留诊断并重录同一编号。"
else
    echo "每条：浏览器按 R 开始、S 保存。按 X 丢弃并自动重录同一编号。"
fi
echo "要提前停止整个批次，请在此终端按 Ctrl+C。"

ledger="$session_root/auto_screen_ledger.jsonl"
accepted_list="$session_root/accepted_sessions.txt"

screen_saved_session() {
    local session_dir="$1"
    local session_relative="$2"
    local processed_zarr="$session_dir/processed.zarr"
    local plan_json="$session_dir/fixed_tip_constrained_lookahead_plan.json"
    local process_log="$session_dir/auto_process.log"
    local ik_log="$session_dir/auto_offline_ik.log"
    local screen_log="$session_dir/auto_screen.log"

    if ! bash scripts/process_handheld_vjaw_demo.sh \
        "$session_relative" "$reference_dir" "$processed_zarr" >"$process_log" 2>&1; then
        echo "AUTO_SCREEN_REJECT: 转换/视觉门失败；详见 $process_log"
        return 1
    fi
    if ! python -u scripts/retarget_vjaw_demo_taskspace.py \
        --dataset "$processed_zarr" \
        --start-reference "$reference_dir/reference.json" \
        --robot-config example/eval_robots_config_vjaw_ros2.yaml \
        --out "$plan_json" \
        --right-tcp-to-source-tcp calibration/robot_wrist_camera/tcp/vjaw_right_tcp_to_fixed_jaw_inner_front_tip_v1_candidate.json \
        --allow-provisional-tcp-frame-transform \
        --orientation-mode constrained_lookahead \
        --lookahead-beam-width 8 \
        --lookahead-joint-branches 2 \
        --constrained-orientation-alpha-step 0.05 \
        --constrained-max-joint-step-deg 3.0 \
        --position-tolerance-mm 10 \
        --orientation-tolerance-deg 60 \
        --orientation-weight 0.35 \
        --joint-margin-deg 5 \
        --ik-steps 30 \
        --random-restarts 0 >"$ik_log" 2>&1; then
        echo "AUTO_SCREEN_REJECT: 离线 IK 运行异常；详见 $ik_log"
        return 1
    fi
    if ! python -u scripts/check_vjaw_training_episode.py \
        --session-dir "$session_dir" --plan "$plan_json" >"$screen_log" 2>&1; then
        cat "$screen_log"
        return 1
    fi
    return 0
}

append_ledger() {
    local sample_name="$1"
    local session_dir="$2"
    local attempt="$3"
    local verdict="$4"
    python - "$ledger" "$sample_name" "$session_dir" "$attempt" "$verdict" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ledger, sample, session, attempt, verdict = sys.argv[1:]
record = {
    "time": datetime.now(timezone.utc).isoformat(),
    "sample": sample,
    "session_dir": session,
    "attempt": int(attempt),
    "offline_auto_screen": verdict,
}
with Path(ledger).open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

for ((offset = 0; offset < count; offset++)); do
    demo_number=$((first_demo_number + offset))
    sample_index=$((first_sample_index + offset))
    demo_label=$(printf '%03d' "$demo_number")
    sample_label=$(printf '%03d' "$sample_index")
    demo_name="vjaw_demo_${demo_label}_${prefix}_${sample_label}"
    demo_relative="$demo_name"
    if [ -n "$collection_dir" ]; then
        demo_relative="$collection_dir/$demo_name"
    fi
    output_dir="$session_root/$demo_name"
    if [ -e "$output_dir" ]; then
        echo "拒绝覆盖已有会话: $output_dir"
        echo "请改 --first-demo-number/--first-sample-index，或先人工检查该目录。"
        exit 2
    fi

    echo
    echo "========== 目标合格样本 $sample_label/$count: $demo_name =========="
    attempt=1
    # A prior interrupted batch may already have preserved one or more
    # rejected attempts.  Continue with the next suffix instead of trying to
    # overwrite attempt_01 again on the next rejection.
    while [ -e "${output_dir}__rejected_attempt_$(printf '%02d' "$attempt")" ]; do
        attempt=$((attempt + 1))
    done
    if [ "$attempt" -gt 1 ]; then
        echo "检测到此前拒绝归档；本次将从 attempt $attempt 继续。"
    fi
    while true; do
        bash scripts/record_handheld_vjaw_demo_web.sh "$demo_relative" "$reference_dir"
        if [ -f "$output_dir/raw_video.mp4" ]; then
            echo "BATCH_SESSION_SAVED: $demo_name"
            if [ "$auto_screen" -eq 0 ]; then
                break
            fi
            echo "AUTO_SCREEN_STARTED: $demo_name（只读离线；不连接机器人）"
            if screen_saved_session "$output_dir" "$demo_relative"; then
                append_ledger "$demo_name" "$output_dir" "$attempt" "pass"
                printf '%s\n' "$output_dir/processed.zarr" >> "$accepted_list"
                echo "AUTO_SCREEN_PASS: $demo_name；已登记为第 $sample_label 条离线合格候选。"
                echo "仍需在 30 条收集后按计划进行人工监督的空载 replay 抽检。"
                break
            fi
            append_ledger "$demo_name" "$output_dir" "$attempt" "reject"
            rejected_dir="${output_dir}__rejected_attempt_$(printf '%02d' "$attempt")"
            if [ -e "$rejected_dir" ]; then
                echo "拒绝目录已存在，停止以避免覆盖: $rejected_dir"
                exit 2
            fi
            mv "$output_dir" "$rejected_dir"
            echo "AUTO_SCREEN_REJECT: 已保留在 $rejected_dir"
            echo "请重录同一编号 $demo_name（下一次保存后会再次自动筛选）。"
            attempt=$((attempt + 1))
            continue
        fi
        echo "BATCH_SESSION_DISCARDED: $demo_name；现在重录同一编号。"
    done
done

echo "VJAW_BATCH_RECORDING_FINISHED"
if [ "$auto_screen" -eq 1 ]; then
    echo "自动通过 zarr 列表: $accepted_list"
    echo "离线筛选日志: $ledger"
    echo "这些是视觉+离线 IK 通过的训练候选；仍须人工监督空载 replay，不能自动替代该实物安全门。"
fi
