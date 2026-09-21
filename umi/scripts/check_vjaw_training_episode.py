#!/usr/bin/env python3
"""Decide whether one recorded V-jaw episode passes the offline intake gate.

This intentionally has no robot dependency: it checks the reports produced by
the fixed-table-Tag conversion and constrained-lookahead planner.  A passing
result is *not* a collision certificate or an unattended physical-replay
approval.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"缺少报告: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线检查一条 V 型夹爪录制是否可进入训练候选集；绝不连接机器人。")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--plan", required=True,
                        help="retarget_vjaw_demo_taskspace.py 生成的计划报告")
    parser.add_argument("--output", default=None, help="输出 JSON；默认写入 session 目录")
    parser.add_argument("--min-visible-ratio", type=float, default=0.95)
    parser.add_argument("--min-final-valid-ratio", type=float, default=1.0)
    parser.add_argument("--max-remaining-lost-frames", type=int, default=0)
    parser.add_argument("--min-joint-margin-deg", type=float, default=5.0)
    args = parser.parse_args()

    if not 0 < args.min_visible_ratio <= 1 or not 0 < args.min_final_valid_ratio <= 1:
        parser.error("可见率阈值必须在 (0, 1] 内")
    if args.max_remaining_lost_frames < 0 or args.min_joint_margin_deg < 0:
        parser.error("丢失帧和关节余量阈值不能为负")

    session = Path(args.session_dir).expanduser().resolve()
    plan_path = Path(args.plan).expanduser().resolve()
    output = (Path(args.output).expanduser().resolve() if args.output else
              session / "v12_auto_screen.json")
    reasons: list[str] = []
    details: dict = {}
    try:
        visual = read_json(session / "camera_trajectory_multi_tag_robust.fixed_tag_report.json")
        refined = read_json(session / "camera_trajectory_multi_tag_refined.refinement_report.json")
        plan = read_json(plan_path)
        visible_ratio = float(visual.get("visible_ratio", 0.0))
        final_valid_ratio = float(refined.get("final_valid_ratio", 0.0))
        remaining_lost = int(refined.get("remaining_lost_frames", -1))
        plan_status = plan.get("status")
        margin = float(plan.get("minimum_joint_margin_deg", float("-inf")))
        branch_ranges = plan.get("branch_discontinuity_frame_ranges", [])
        details = {
            "visible_ratio": visible_ratio,
            "final_valid_ratio": final_valid_ratio,
            "remaining_lost_frames": remaining_lost,
            "plan_status": plan_status,
            "minimum_joint_margin_deg": margin,
            "branch_discontinuity_frame_ranges": branch_ranges,
            "position_error_mm": plan.get("position_error_mm"),
            "orientation_deviation_deg": plan.get("orientation_deviation_deg"),
        }
        if visible_ratio < args.min_visible_ratio or not visual.get("meets_min_visible_ratio", False):
            reasons.append("fixed_tag_visibility")
        if final_valid_ratio < args.min_final_valid_ratio:
            reasons.append("refined_trajectory_coverage")
        if remaining_lost > args.max_remaining_lost_frames:
            reasons.append("remaining_lost_frames")
        if plan_status != "candidate":
            reasons.append("offline_ik_plan")
        if margin < args.min_joint_margin_deg:
            reasons.append("joint_margin")
        if branch_ranges:
            reasons.append("joint_branch_discontinuity")
    except Exception as exc:
        reasons.append("missing_or_invalid_report")
        details["error"] = str(exc)

    result = {
        "schema": "am_umi_vjaw_training_auto_screen_v1",
        "safety": ("offline visual/conversion/IK gate only; it does not run a robot "
                   "or approve unattended physical replay"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "plan": str(plan_path),
        "accepted": not reasons,
        "reasons": reasons,
        "thresholds": {
            "min_visible_ratio": args.min_visible_ratio,
            "min_final_valid_ratio": args.min_final_valid_ratio,
            "max_remaining_lost_frames": args.max_remaining_lost_frames,
            "min_joint_margin_deg": args.min_joint_margin_deg,
        },
        "details": details,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("VJAW_TRAINING_AUTO_SCREEN_" + ("PASS" if result["accepted"] else "REJECT"))
    print("report:", output)
    if reasons:
        print("reasons:", ", ".join(reasons))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
