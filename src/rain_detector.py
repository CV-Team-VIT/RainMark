"""
Main rain streak detection algorithm translated from detect_rainstreaks.m.
"""
from __future__ import annotations

from typing import Any, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

from src.contrast_analyzer import (
    function_contrast_at_5_percent_gt,
    function_contrast_at_5_percent_rain,
)
from src.mask_cleanup import adaptive_rain_mask_cleanup
from utils.image_utils import get_image_brightness, rgb_to_gray


def _as_rgb_float(image: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"{name} must be an RGB image")
    return image[:, :, :3].astype(np.float64, copy=False)


def _resize_to_match(
    array: np.ndarray,
    target_shape: tuple[int, int],
    interpolation: int,
) -> np.ndarray:
    if array.shape == target_shape:
        return array
    return cv2.resize(array, (target_shape[1], target_shape[0]), interpolation=interpolation)


def _create_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = np.clip(image, 0, 255).astype(np.uint8).copy()
    mask = mask.astype(bool)
    overlay[mask, 0] = 57
    overlay[mask, 1] = 255
    overlay[mask, 2] = 20
    return overlay


def detect_rainstreaks(
    GT: np.ndarray,
    Rain: np.ndarray,
    S: int = 7,
    visibility_percent: float = 5.0,
    bright_thresh: int = 150,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    Detect and quantify rain streaks using the MATLAB RainMark algorithm.
    """
    GT = _as_rgb_float(GT, "GT")
    Rain = _as_rgb_float(Rain, "Rain")

    if GT.shape[:2] != Rain.shape[:2]:
        raise ValueError("GT and Rain images must have matching height and width")

    GT_gray = rgb_to_gray(GT).astype(np.float64)
    Rain_gray = rgb_to_gray(Rain).astype(np.float64)

    (
        mask_rain,
        contrast_rain,
        dominant_angle_primary,
        dominant_angle_weighted,
        roi,
        angles,
    ) = function_contrast_at_5_percent_rain(
        Rain_gray,
        S=S,
        percentage=visibility_percent,
        verbose=show_plots,
    )

    gt_smooth = ndimage.gaussian_filter(GT_gray, sigma=1.5)
    mask_gt, contrast_gt = function_contrast_at_5_percent_gt(
        gt_smooth,
        S=S,
        percentage=visibility_percent,
        angles=angles,
        verbose=show_plots,
    )

    mask_gt = _resize_to_match(
        mask_gt.astype(np.uint8),
        mask_rain.shape,
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    epsilon = 1e-6
    wI = np.sum(mask_rain)
    wJ = np.sum(mask_gt)
    e1 = max(0.0, float(wI - wJ)) / (max(float(wI), float(wJ)) + epsilon)

    bright_Rain = get_image_brightness(Rain).astype(np.float64)
    bright_GT = get_image_brightness(GT).astype(np.float64)
    bright_GT = _resize_to_match(
        bright_GT,
        bright_Rain.shape,
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float64)

    kernel = np.ones((3, 3), dtype=np.float64)
    new_saturated = signal.convolve2d(
        ((bright_Rain >= bright_thresh) & (bright_GT < bright_thresh)).astype(np.float64),
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    ) > 0
    ns1 = float(np.sum(new_saturated) / bright_GT.size)

    neighborhood_size = 7
    tau = 2.0
    epsv = 1e-6
    muJ = ndimage.uniform_filter(bright_GT, size=neighborhood_size, mode="nearest")
    brightness_diff = bright_Rain - bright_GT
    delta_b = brightness_diff / (np.maximum(muJ, tau) + epsv)
    delta = float(np.percentile(delta_b, 80))

    rain_streak_mask = (
        (mask_rain == 1)
        & (mask_gt == 0)
        & ((bright_Rain > bright_GT + delta) | (bright_Rain < bright_GT - delta))
    )

    if show_plots:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rain_streak_mask, cmap="gray")
        plt.title("Initial Rain Streak Mask")
        plt.axis("off")

    mask_clean = adaptive_rain_mask_cleanup(rain_streak_mask)
    streak_area = float(np.sum(mask_clean) / mask_clean.size)
    percentage_streak_area = 100.0 * streak_area

    if show_plots:
        plt.subplot(1, 3, 2)
        plt.imshow(mask_clean, cmap="gray")
        plt.title("Cleaned Rain Streak Mask")
        plt.axis("off")

    overlay = _create_overlay(Rain, mask_clean)

    if show_plots:
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Rain Streaks Highlighted (Neon)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    return {
        "e1": e1,
        "ns1": ns1,
        "streak_area": streak_area,
        "percentage_streak_area": percentage_streak_area,
        "overlay": overlay,
        "rain_mask": mask_clean,
        "edge_map_gt": mask_gt,
        "edge_map_rain": mask_rain,
        "brightness_diff": brightness_diff,
        "delta_threshold": delta,
        "mask_clean": mask_clean,
        "mask_gt": mask_gt,
        "mask_rain": mask_rain,
        "rain_streak_mask": rain_streak_mask,
        "contrast_map_gt": contrast_gt,
        "contrast_map_rain": contrast_rain,
        "dominant_angle_primary": dominant_angle_primary,
        "dominant_angle_weighted": dominant_angle_weighted,
        "roi": roi,
        "angles": angles,
    }


def calculate_rain_severity(results: Dict[str, Any]) -> float:
    """
    Calculate rain severity score using the formula from the original MATLAB code.
    """
    return (
        17.138146 * results["e1"]
        + 0.132285 * results["ns1"]
        + 0.887244 * results["streak_area"]
    )
