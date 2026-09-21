"""Utilities for comparing a live policy image with a saved start-view reference.

The reference is deliberately the *post-processed* RGB image used by the
policy, rather than a convenient raw camera screenshot.  That makes the
comparison useful when checking camera crop, zoom, and gripper masking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from umi.common.cv_util import draw_predefined_mask


REFERENCE_IMAGE_NAME = "policy_input_rgb.png"
REFERENCE_METADATA_NAME = "reference.json"


def to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HWC RGB policy image (uint8 or float in [0, 1]) to uint8."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image with 3 channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).round().astype(np.uint8)
    else:
        image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image)


def preprocess_uvc_bgr(
        frame_bgr: np.ndarray,
        camera_crop: dict[str, Any] | None,
        output_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Match the AM2Pro UVC crop/mask path in ``BimanualUmiEnv``.

    This supports a camera-only alignment monitor for a hand-held UVC camera.
    For a different camera model, pass its calibrated crop in the config; a
    raw image comparison across unmatched lenses is only a coarse guide.
    """
    image = np.asarray(frame_bgr)
    if camera_crop is not None:
        cx, cy = camera_crop["center"]
        crop_size = int(camera_crop["size"])
        height, width = image.shape[:2]
        x0 = int(max(0, cx - crop_size // 2))
        x1 = x0 + crop_size
        if x1 > width:
            x1 = width
            x0 = max(0, width - crop_size)
        y0 = int(max(0, cy - crop_size // 2))
        y1 = y0 + crop_size
        if y1 > height:
            y1 = height
            y0 = max(0, height - crop_size)
        image = cv2.resize(
            image[y0:y1, x0:x1], output_size, interpolation=cv2.INTER_AREA)
    else:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    image = np.ascontiguousarray(image[..., ::-1])  # BGR -> RGB
    return draw_predefined_mask(
        image, color=(0, 0, 0), mirror=False, gripper=True,
        finger=False, use_aa=True)


def write_reference(
        reference_dir: str | Path,
        image_rgb: np.ndarray,
        metadata: dict[str, Any],
) -> Path:
    """Write a new reference directory without overwriting an existing one."""
    reference_dir = Path(reference_dir).expanduser()
    if reference_dir.exists():
        raise FileExistsError(
            f"reference directory already exists: {reference_dir}. "
            "Choose a new name to preserve the previous calibration.")
    reference_dir.mkdir(parents=True)
    image_rgb = to_uint8_rgb(image_rgb)
    if not cv2.imwrite(str(reference_dir / REFERENCE_IMAGE_NAME), image_rgb[..., ::-1]):
        raise RuntimeError("failed to write reference image")
    with (reference_dir / REFERENCE_METADATA_NAME).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return reference_dir


def load_reference(reference_dir: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    reference_dir = Path(reference_dir).expanduser()
    image_bgr = cv2.imread(str(reference_dir / REFERENCE_IMAGE_NAME), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(reference_dir / REFERENCE_IMAGE_NAME)
    with (reference_dir / REFERENCE_METADATA_NAME).open(encoding="utf-8") as file:
        metadata = json.load(file)
    return np.ascontiguousarray(image_bgr[..., ::-1]), metadata


def compare_policy_images(reference_rgb: np.ndarray, current_rgb: np.ndarray) -> dict[str, float]:
    """Return simple, interpretable image-alignment diagnostics.

    These are not a camera-pose solver.  Use them to guide visual alignment of
    fixed tags/object scale while inspecting the panel produced below.
    """
    reference = to_uint8_rgb(reference_rgb)
    current = to_uint8_rgb(current_rgb)
    if reference.shape != current.shape:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]))
    mae = float(np.mean(np.abs(reference.astype(np.float32) - current.astype(np.float32))) / 255.0)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cur_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_std, cur_std = float(ref_gray.std()), float(cur_gray.std())
    if ref_std < 1e-6 or cur_std < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(ref_gray.ravel(), cur_gray.ravel())[0, 1])
    return {
        "mae": mae,
        "correlation": correlation,
        "appearance_score": float(max(0.0, 1.0 - mae)),
    }


def make_alignment_panel(
        reference_rgb: np.ndarray,
        current_rgb: np.ndarray,
        metrics: dict[str, float],
        scale: int = 2,
) -> np.ndarray:
    """Create a BGR panel: reference | live | overlay | amplified difference."""
    reference = to_uint8_rgb(reference_rgb)
    current = to_uint8_rgb(current_rgb)
    if reference.shape != current.shape:
        current = cv2.resize(current, (reference.shape[1], reference.shape[0]))
    overlay = cv2.addWeighted(reference, 0.5, current, 0.5, 0)
    difference = cv2.absdiff(reference, current)
    difference = np.clip(difference.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
    panel_rgb = np.concatenate([reference, current, overlay, difference], axis=1)
    if scale > 1:
        panel_rgb = cv2.resize(
            panel_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    panel_bgr = np.ascontiguousarray(panel_rgb[..., ::-1])
    labels = ["reference", "live", "overlay", "difference x3"]
    cell_width = panel_bgr.shape[1] // 4
    for index, label in enumerate(labels):
        cv2.putText(panel_bgr, label, (index * cell_width + 8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    text = (f"appearance={metrics['appearance_score']:.3f}  "
            f"MAE={metrics['mae']:.3f}  corr={metrics['correlation']:.3f}")
    cv2.putText(panel_bgr, text, (8, panel_bgr.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return panel_bgr

