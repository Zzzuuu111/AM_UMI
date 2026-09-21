#!/usr/bin/env python3
"""Offline fit of encoder-to-URDF joint candidates for the V-jaw robot.

This is a diagnostic calibration tool.  Given static images of a fixed world
Tag and their *actual* servo readbacks, it keeps the official URDF camera
mount fixed and searches for a per-joint sign plus constant encoder offset:

    q_urdf_deg = sign * q_encoder_deg + offset_deg

The result is only a candidate report.  It never talks to a serial port and
does not alter a robot configuration or a URDF.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umi.common.cv_util import (  # noqa: E402
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    parse_aruco_config,
    parse_fisheye_intrinsics,
)
from umi.common.pose_util import mat_to_pose, pose_to_mat  # noqa: E402
from umi.real_world.am2pro_kinematics_backends import create_kinematics_backend  # noqa: E402


JOINT_NAMES = (
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_yaw_joint", "right_wrist_roll",
)
ENCODER_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_yaw", "wrist_roll",
)


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def origin_matrix(node: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    if node is None:
        return matrix
    matrix[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    matrix[:3, :3] = Rotation.from_euler(
        "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return matrix


def root_transform(urdf_path: Path, link: str) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    parents: dict[str, tuple[str, np.ndarray]] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            parents[child.attrib["link"]] = (parent.attrib["link"], origin_matrix(joint.find("origin")))
    chain: list[np.ndarray] = []
    current = link
    while current in parents:
        current, transform = parents[current]
        chain.append(transform)
    if not chain:
        raise ValueError(f"URDF 中找不到 link: {link}")
    result = np.eye(4, dtype=np.float64)
    for transform in reversed(chain):
        result = result @ transform
    return result


def tag_observations(samples, intrinsics_path: Path, aruco_path: Path, tag_id: int):
    aruco = parse_aruco_config(yaml.safe_load(aruco_path.read_text(encoding="utf-8")))
    raw_intrinsics = parse_fisheye_intrinsics(json.loads(intrinsics_path.read_text(encoding="utf-8")))
    observed, q_values, ids = [], [], []
    for index, sample in enumerate(samples):
        image = sample.get("img")
        q = np.asarray(sample.get("joint_deg"), dtype=np.float64)
        if image is None or q.shape != (6,) or not np.isfinite(q).all():
            continue
        intrinsics = convert_fisheye_intrinsics_resolution(
            raw_intrinsics, target_resolution=(image.shape[1], image.shape[0]))
        tags = detect_localize_aruco_tags(
            image, aruco["aruco_dict"], aruco["marker_size_map"], intrinsics)
        if tag_id not in tags:
            continue
        tag = tags[tag_id]
        camera_from_tag = np.eye(4, dtype=np.float64)
        camera_from_tag[:3, :3] = cv2.Rodrigues(tag["rvec"])[0]
        camera_from_tag[:3, 3] = tag["tvec"]
        observed.append(camera_from_tag)
        q_values.append(q)
        ids.append(index + 1)
    if len(observed) < 10:
        raise ValueError(f"有效 Tag {tag_id} 样本只有 {len(observed)}，至少需要 10 张")
    return np.asarray(observed), np.asarray(q_values), ids


def pose_residual(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    delta = np.linalg.inv(observed) @ predicted
    return np.r_[delta[:3, 3] / 0.010,
                 Rotation.from_matrix(delta[:3, :3]).as_rotvec() / np.deg2rad(3.0)]


def summary_errors(residuals: np.ndarray) -> dict:
    position_mm = np.linalg.norm(residuals[:, :3], axis=1) * 10.0
    rotation_deg = np.linalg.norm(residuals[:, 3:], axis=1) * 3.0
    return {
        "translation_mm": {
            "mean": float(position_mm.mean()), "median": float(np.median(position_mm)),
            "p95": float(np.percentile(position_mm, 95)), "max": float(position_mm.max()),
        },
        "rotation_deg": {
            "mean": float(rotation_deg.mean()), "median": float(np.median(rotation_deg)),
            "p95": float(np.percentile(rotation_deg, 95)), "max": float(rotation_deg.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="V 型夹爪 keyboard hand-eye .pkl")
    parser.add_argument("--robot-config", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--aruco-yaml", required=True)
    parser.add_argument("--tag-id", type=int, default=13)
    parser.add_argument("--offset-bound-deg", type=float, default=45.0)
    parser.add_argument("--max-nfev", type=int, default=160)
    parser.add_argument("--refine-sign-candidates", type=int, default=8,
                        help="先快速枚举 64 种符号后，精修得分最好的几个（默认 8）")
    parser.add_argument("--out", required=True, help="新的候选报告 JSON；不得覆盖")
    args = parser.parse_args()
    if args.offset_bound_deg <= 0 or args.max_nfev <= 0 or args.refine_sign_candidates <= 0:
        parser.error("offset-bound-deg、max-nfev 与 refine-sign-candidates 必须为正")

    output = resolve(args.out)
    if output.exists():
        parser.error(f"拒绝覆盖已有报告：{output}")
    config = yaml.safe_load(resolve(args.robot_config).read_text(encoding="utf-8"))
    robot = config["robots"][0]
    urdf_path = Path(robot["urdf_path"]).expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"URDF 不存在: {urdf_path}")
    backend_name = robot.get("ik_backend", "placo")
    if backend_name != "placo":
        parser.error("此诊断需要 placo 直接读取官方 right_tcp；robot-config 的 ik_backend 必须为 placo")

    samples = pickle.load(resolve(args.input).open("rb"))
    observed, q_encoder, sample_ids = tag_observations(
        samples, resolve(args.intrinsics), resolve(args.aruco_yaml), args.tag_id)
    span = np.ptp(q_encoder, axis=0)
    weak = [ENCODER_NAMES[i] for i, value in enumerate(span) if value < 12.0]
    if weak:
        parser.error("关节覆盖不足 12°: " + ", ".join(weak))

    # The camera and right_tcp share the fixed jaw, hence their relative URDF
    # transform contains no unknown joint angle.
    tcp_from_root = root_transform(urdf_path, "right_tcp")
    camera_from_root = root_transform(urdf_path, "right_camera")
    tcp_from_camera = np.linalg.inv(tcp_from_root) @ camera_from_root
    backend = create_kinematics_backend("placo", str(urdf_path), JOINT_NAMES, "right_tcp")

    def base_from_camera(q_model: np.ndarray) -> np.ndarray:
        return backend.forward_kinematics(q_model) @ tcp_from_camera

    quick_candidates = []
    print("VJAW_JOINT_MODEL_FIT_STARTED")
    print("valid_samples:", len(observed), "joint_span_deg:", np.round(span, 2).tolist())
    # First score all sign conventions without optimizing.  This inexpensive
    # pass avoids 64 expensive finite-difference nonlinear fits.
    for signs_tuple in itertools.product((-1.0, 1.0), repeat=6):
        signs = np.asarray(signs_tuple, dtype=np.float64)
        initial_base_from_world = base_from_camera(signs * q_encoder[0]) @ observed[0]
        quick_residuals = np.asarray([
            pose_residual(measurement,
                          np.linalg.inv(base_from_camera(signs * encoder)) @ initial_base_from_world)
            for encoder, measurement in zip(q_encoder, observed)])
        quick_errors = summary_errors(quick_residuals)
        quick_score = quick_errors["translation_mm"]["mean"] + 10.0 * quick_errors["rotation_deg"]["mean"]
        quick_candidates.append({"quick_score": float(quick_score), "sign": signs.astype(int).tolist()})
    quick_candidates.sort(key=lambda item: item["quick_score"])

    candidates = []
    # Only the most plausible sign patterns get bounded zero-offset fitting.
    for quick in quick_candidates[:min(args.refine_sign_candidates, len(quick_candidates))]:
        signs = np.asarray(quick["sign"], dtype=np.float64)
        initial_base_from_world = base_from_camera(signs * q_encoder[0]) @ observed[0]
        initial = np.r_[mat_to_pose(initial_base_from_world), np.zeros(6)]
        lower = np.r_[[-3.0, -3.0, -3.0], [-np.pi, -np.pi, -np.pi],
                      np.full(6, -args.offset_bound_deg)]
        upper = np.r_[[3.0, 3.0, 3.0], [np.pi, np.pi, np.pi],
                      np.full(6, args.offset_bound_deg)]

        def residual(vector: np.ndarray) -> np.ndarray:
            base_from_world = pose_to_mat(vector[:6])
            offsets = vector[6:]
            values = []
            for encoder, measurement in zip(q_encoder, observed):
                predicted = np.linalg.inv(base_from_camera(signs * encoder + offsets)) @ base_from_world
                values.extend(pose_residual(measurement, predicted))
            return np.asarray(values)

        fit = least_squares(residual, initial, bounds=(lower, upper), loss="huber",
                            f_scale=1.0, max_nfev=args.max_nfev)
        residual_matrix = residual(fit.x).reshape(-1, 6)
        errors = summary_errors(residual_matrix)
        score = errors["translation_mm"]["mean"] + 10.0 * errors["rotation_deg"]["mean"]
        candidates.append({
            "score": float(score), "sign": signs.astype(int).tolist(),
            "quick_score": quick["quick_score"],
            "offset_deg": fit.x[6:].tolist(), "base_from_world": pose_to_mat(fit.x[:6]).tolist(),
            "errors": errors, "optimizer_cost": float(fit.cost), "optimizer_success": bool(fit.success),
        })
    candidates.sort(key=lambda item: item["score"])
    best = candidates[0]
    accepted = (best["errors"]["translation_mm"]["mean"] <= 10.0 and
                best["errors"]["translation_mm"]["p95"] <= 20.0 and
                best["errors"]["rotation_deg"]["mean"] <= 2.0 and
                best["errors"]["rotation_deg"]["p95"] <= 4.0)
    report = {
        "schema": "am_umi_vjaw_encoder_to_urdf_candidate_v1",
        "status": "candidate" if accepted else "not_accepted",
        "safety": "offline diagnostic only; no robot command, serial write, URDF change, or config change occurred",
        "input": str(resolve(args.input)), "urdf": str(urdf_path), "tag_id": args.tag_id,
        "valid_sample_indices_1based": sample_ids,
        "joint_encoder_span_deg": {name: float(span[i]) for i, name in enumerate(ENCODER_NAMES)},
        "model": "q_urdf_deg = sign * q_encoder_deg + offset_deg",
        "best_candidate": best,
        "top_candidates": candidates[:8],
        "quick_sign_ranking": quick_candidates[:16],
        "acceptance": {
            "criteria": "mean/p95 translation <= 10/20 mm and mean/p95 rotation <= 2/4 deg",
            "recommendation": "YES" if accepted else "NO",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("VJAW_JOINT_MODEL_FIT_OK")
    print("best_sign:", best["sign"])
    print("best_offset_deg:", np.array2string(np.asarray(best["offset_deg"]), precision=3))
    print("translation_mm:", best["errors"]["translation_mm"])
    print("rotation_deg:", best["errors"]["rotation_deg"])
    print("recommendation:", report["acceptance"]["recommendation"])
    print("saved:", output)


if __name__ == "__main__":
    main()
