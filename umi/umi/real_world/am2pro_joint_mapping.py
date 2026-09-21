"""Explicit encoder <-> URDF joint-coordinate mappings for AM2Pro.

The mapping is deliberately applied only at the motor boundary.  IK, FK,
stored references and offline replay use URDF model coordinates throughout.
No function in this module opens a serial port or writes hardware state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MOTOR_ARM_NAMES = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_yaw", "wrist_roll",
)


def _validate(sign, offset, source: str) -> dict:
    sign = np.asarray(sign, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if sign.shape != (6,) or offset.shape != (6,):
        raise ValueError(f"{source}: joint sign/offset 必须各有 6 项")
    if not np.all(np.isin(sign, (-1.0, 1.0))) or not np.isfinite(offset).all():
        raise ValueError(f"{source}: joint sign 必须为 ±1，offset 必须有限")
    return {"sign": sign, "offset_deg": offset, "source": source}


def load_candidate(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "am_umi_vjaw_encoder_to_urdf_candidate_v1":
        raise ValueError(f"未知 joint model candidate 格式: {path}")
    if data.get("acceptance", {}).get("recommendation") != "YES":
        raise ValueError(f"joint model candidate 尚未通过验证: {path}")
    best = data.get("best_candidate", {})
    return _validate(best.get("sign"), best.get("offset_deg"), str(path))


def mapping_from_config(robot: dict, root: Path) -> dict:
    """Read an accepted candidate, or retain legacy flip-only behavior."""
    candidate = robot.get("joint_model_candidate")
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (root / path).resolve()
        return load_candidate(path)
    sign = np.ones(6, dtype=np.float64)
    for index, name in enumerate(MOTOR_ARM_NAMES):
        if name in robot.get("joint_flip", []):
            sign[index] = -1.0
    return _validate(sign, np.zeros(6), "legacy joint_flip")


def encoder_to_model(encoder_deg, mapping: dict) -> np.ndarray:
    encoder = np.asarray(encoder_deg, dtype=np.float64)
    if encoder.shape[-1] != 6:
        raise ValueError("encoder joint vector 最后一维必须为 6")
    return encoder * mapping["sign"] + mapping["offset_deg"]


def model_to_encoder(model_deg, mapping: dict) -> np.ndarray:
    model = np.asarray(model_deg, dtype=np.float64)
    if model.shape[-1] != 6:
        raise ValueError("model joint vector 最后一维必须为 6")
    # sign is ±1, so its inverse is itself.
    return (model - mapping["offset_deg"]) * mapping["sign"]


def encoder_interval_to_model(interval_deg, index: int, mapping: dict) -> tuple[float, float]:
    """Map a raw encoder interval into its sorted URDF-model interval."""
    low, high = map(float, interval_deg)
    if low > high:
        raise ValueError("encoder interval 下界大于上界")
    values = encoder_to_model(np.eye(6)[index] * low, mapping)[index], \
        encoder_to_model(np.eye(6)[index] * high, mapping)[index]
    return float(min(values)), float(max(values))


def model_safe_limits_from_config(robot: dict, mapping: dict) -> dict[str, tuple[float, float]]:
    """Return explicit model-coordinate safety bounds from a robot config.

    ``encoder_safe_limits_deg`` documents a physical, calibrated motor range;
    it is transformed through the accepted mapping.  A
    ``model_safe_limits_deg`` entry may deliberately supersede an inaccurate
    CAD/URDF limit, but is accepted only when it is contained in the physical
    encoder range declared in the same config.  This prevents a convenience
    config edit from silently widening past the motor's own EEPROM range.
    """
    encoder_bounds: dict[str, tuple[float, float]] = {}
    for name, interval in robot.get("encoder_safe_limits_deg", {}).items():
        if name not in MOTOR_ARM_NAMES:
            raise ValueError(f"encoder_safe_limits_deg 包含未知关节: {name}")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"encoder_safe_limits_deg[{name}] 必须是 [lower, upper]")
        encoder_bounds[name] = encoder_interval_to_model(
            interval, MOTOR_ARM_NAMES.index(name), mapping)

    result = dict(encoder_bounds)
    for name, interval in robot.get("model_safe_limits_deg", {}).items():
        if name not in MOTOR_ARM_NAMES:
            raise ValueError(f"model_safe_limits_deg 包含未知关节: {name}")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError(f"model_safe_limits_deg[{name}] 必须是 [lower, upper]")
        low, high = map(float, interval)
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"model_safe_limits_deg[{name}] 范围无效")
        physical = encoder_bounds.get(name)
        if physical is not None and (low < physical[0] - 1e-6 or high > physical[1] + 1e-6):
            raise ValueError(
                f"model_safe_limits_deg[{name}] 超出 encoder_safe_limits_deg 换算范围 "
                f"[{physical[0]:.3f}, {physical[1]:.3f}]")
        result[name] = (low, high)
    return result
