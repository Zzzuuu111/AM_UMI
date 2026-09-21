#!/usr/bin/env python3
"""Validate a width-dependent V-jaw grasp-center candidate on held-out data.

This is deliberately offline-only.  Two accepted cylinder-axis fits define a
linear *candidate* grasp_center(width); an independently collected third fit
is compared against that interpolation and is never used to create it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_candidate(path: Path, label: str) -> tuple[np.ndarray, float, dict]:
    if not path.is_file():
        raise ValueError(f"{label} 文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "candidate":
        raise ValueError(f"{label} 不是通过的轴线候选: status={data.get('status')}")
    point = np.asarray(data.get("grasp_center_translation_parent_m"), dtype=float)
    width = data.get("j7", {}).get("inlier_width_m_median")
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{label} 缺少有效 grasp_center_translation_parent_m")
    if width is None or not np.isfinite(float(width)):
        raise ValueError(f"{label} 缺少有效 J7 实际开口中位数")
    return point, float(width), data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="离线验证 grasp_center(width) 线性插值；不会连接机器人。")
    parser.add_argument("--endpoint-a", required=True, help="已通过的小/大开口候选 JSON")
    parser.add_argument("--endpoint-b", required=True, help="另一已通过开口候选 JSON")
    parser.add_argument("--held-out", required=True, help="中圆柱独立拟合候选 JSON")
    parser.add_argument("--output", required=True, help="新的验证报告 JSON；拒绝覆盖")
    parser.add_argument("--max-error-mm", type=float, default=5.0,
                        help="预测中心与留出实测中心的最大允许三维差异（默认 5）")
    args = parser.parse_args()
    if args.max_error_mm <= 0:
        parser.error("--max-error-mm 必须为正")

    endpoint_a_path = resolve(args.endpoint_a)
    endpoint_b_path = resolve(args.endpoint_b)
    held_out_path = resolve(args.held_out)
    output_path = resolve(args.output)
    if output_path.exists():
        parser.error(f"拒绝覆盖已有结果: {output_path}")

    try:
        point_a, width_a, data_a = load_candidate(endpoint_a_path, "endpoint-a")
        point_b, width_b, data_b = load_candidate(endpoint_b_path, "endpoint-b")
        point_test, width_test, data_test = load_candidate(held_out_path, "held-out")
    except ValueError as exc:
        parser.error(str(exc))
    if data_a.get("parent_frame") != data_b.get("parent_frame") or \
            data_a.get("parent_frame") != data_test.get("parent_frame"):
        parser.error("三个候选不在同一 parent_frame，不能比较")
    if abs(width_b - width_a) < 1e-6:
        parser.error("两个端点开口几乎相同，无法形成插值")

    # Order only for clear reporting; the interpolation formula itself is
    # symmetric.  The held-out opening must be genuinely inside the interval.
    if width_a > width_b:
        point_a, point_b = point_b, point_a
        width_a, width_b = width_b, width_a
        endpoint_a_path, endpoint_b_path = endpoint_b_path, endpoint_a_path
    alpha = (width_test - width_a) / (width_b - width_a)
    inside_interval = 0.0 <= alpha <= 1.0
    predicted = point_a + alpha * (point_b - point_a)
    delta = point_test - predicted
    error_mm = float(np.linalg.norm(delta) * 1000.0)
    accepted = inside_interval and error_mm <= args.max_error_mm

    report = {
        "schema": "am_umi_vjaw_grasp_center_width_holdout_validation_v1",
        "status": "candidate" if accepted else "rejected",
        "definition": ("Held-out candidate is compared with the linear interpolation of two "
                       "accepted width-dependent grasp-center candidates. No robot was connected."),
        "parent_frame": data_a.get("parent_frame"),
        "endpoint_a": {"path": str(endpoint_a_path), "width_m": width_a,
                       "point_parent_m": point_a.tolist()},
        "endpoint_b": {"path": str(endpoint_b_path), "width_m": width_b,
                       "point_parent_m": point_b.tolist()},
        "held_out": {"path": str(held_out_path), "width_m": width_test,
                     "point_parent_m": point_test.tolist()},
        "interpolation_alpha": float(alpha),
        "predicted_point_parent_m": predicted.tolist(),
        "held_out_minus_prediction_m": delta.tolist(),
        "error_mm": error_mm,
        "max_error_mm": float(args.max_error_mm),
        "inside_endpoint_width_interval": inside_interval,
        "limitations": [
            "This validates only one held-out width; it does not promote right_tcp or modify URDF.",
            "If accepted, use the resulting width-dependent center only as a replay candidate and verify physically at low speed.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("AM2PRO_GRASP_CENTER_WIDTH_HOLDOUT_" + ("CANDIDATE" if accepted else "REJECTED"))
    print("endpoint widths mm: {:.2f} .. {:.2f}; held-out: {:.2f}; alpha={:.3f}".format(
        width_a * 1000.0, width_b * 1000.0, width_test * 1000.0, alpha))
    print("prediction error mm: {:.3f} (limit {:.3f})".format(error_mm, args.max_error_mm))
    print("saved:", output_path)


if __name__ == "__main__":
    main()
