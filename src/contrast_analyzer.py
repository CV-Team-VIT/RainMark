"""
Orientation-aware Weber contrast functions translated from the MATLAB RainMark
reference files:
- functionContrastAt5PerCentRain.m
- functionContrastAt5PerCentGT.m
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage

from src.morphology import create_line_strel
from src.roi import find_adaptive_roi_check


def _validate_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale")
    return image.astype(np.float64, copy=False)


def _sample_angles_from_roi(roi: np.ndarray, num_angles: int = 8) -> np.ndarray:
    roi = np.asarray(roi, dtype=float)

    if roi.size == 0:
        roi = np.array([[0.0, 180.0]])
    elif roi.ndim == 1:
        if roi.size == 1:
            roi = np.array([[roi[0] - 5.0, roi[0] + 5.0]])
        else:
            roi = roi.reshape(1, -1)

    if roi.shape[1] != 2:
        raise ValueError("ROI must be an Nx2 array of [start, end] degrees")

    roi[:, 0] = np.maximum(0.0, roi[:, 0])
    roi[:, 1] = np.minimum(180.0, roi[:, 1])
    widths = np.maximum(0.0, roi[:, 1] - roi[:, 0])
    total_width = float(np.sum(widths))

    if total_width <= 0:
        roi = np.array([[0.0, 180.0]])
        widths = np.array([180.0])
        total_width = 180.0

    angles = []
    remaining = num_angles
    for idx, interval in enumerate(roi):
        if remaining <= 0:
            break

        if idx == len(roi) - 1:
            n_samples = remaining
        else:
            n_samples = max(1, int(round(num_angles * (widths[idx] / total_width))))
            n_samples = min(n_samples, remaining)

        angles.extend(np.linspace(interval[0], interval[1], n_samples))
        remaining = num_angles - len(angles)

    if len(angles) < num_angles:
        angles.extend(np.linspace(roi[0, 0], roi[-1, 1], num_angles - len(angles)))

    return np.asarray(angles[:num_angles], dtype=float).reshape(-1)


def _direction_label(dominant_angle: float) -> str:
    if 0.0 <= dominant_angle <= 15.0 or 165.0 <= dominant_angle <= 180.0:
        return "Near-vertical"
    if 105.0 < dominant_angle <= 165.0:
        return "Right-slanted"
    if 15.0 < dominant_angle <= 75.0:
        return "Left-slanted"
    return "Near-horizontal"


def _oriented_min_max(image: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    line_lengths = [3]
    pad_size = int(np.ceil(max(line_lengths) / 2))

    image_pad = np.pad(image.astype(np.float64), pad_size, mode="symmetric")
    image_min = np.full_like(image_pad, np.inf, dtype=np.float64)
    image_max = np.full_like(image_pad, -np.inf, dtype=np.float64)

    for length in line_lengths:
        for angle in angles:
            footprint = create_line_strel(length, angle)
            eroded = ndimage.grey_erosion(image_pad, footprint=footprint)
            dilated = ndimage.grey_dilation(image_pad, footprint=footprint)
            image_min = np.minimum(image_min, eroded)
            image_max = np.maximum(image_max, dilated)

    height, width = image.shape
    return (
        image_min[pad_size : pad_size + height, pad_size : pad_size + width],
        image_max[pad_size : pad_size + height, pad_size : pad_size + width],
    )


def _contrast_from_angles(
    image: np.ndarray,
    S: int,
    percentage: float,
    angles: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    image = _validate_grayscale(image)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    if angles.size == 0:
        raise ValueError("angles must contain at least one orientation")

    height, width = image.shape
    percentage = percentage / 2.0

    image_min, image_max = _oriented_min_max(image, angles)

    image_pad = np.pad(image, S, mode="symmetric")
    min_pad = np.pad(image_min, S, mode="symmetric")
    max_pad = np.pad(image_max, S, mode="symmetric")

    mask = np.zeros((height + 2 * S, width + 2 * S), dtype=bool)
    contrast_map = np.zeros((height + 2 * S, width + 2 * S), dtype=np.float64)

    step = max(1, round(S / 2))
    for ii in range(0, height, step):
        for jj in range(0, width, step):
            row = S + ii
            col = S + jj
            window = image_pad[row : row + S, col : col + S].astype(np.float64)
            window_min = min_pad[row : row + S, col : col + S].astype(np.float64)
            window_max = max_pad[row : row + S, col : col + S].astype(np.float64)

            is_min = max(1, int(round(float(np.min(window_min)))))
            is_max = min(255, int(round(float(np.max(window_max)))))
            if is_max < is_min:
                continue

            f_cube = np.zeros((S, S, is_max - is_min + 1), dtype=bool)
            contrasts = np.zeros(is_max + 1, dtype=np.float64)

            for threshold in range(is_min, is_max + 1):
                values = []
                for nn in range(1, S):
                    for mm in range(1, S):
                        if window_min[nn, mm] <= threshold < window_max[nn, mm]:
                            c1 = min(
                                abs(threshold - window[nn, mm])
                                / max(threshold, window[nn, mm]),
                                abs(threshold - window[nn, mm - 1])
                                / max(threshold, window[nn, mm - 1]),
                            )
                            values.append(c1)
                            f_cube[nn, mm, threshold - is_min] = True
                            f_cube[nn, mm - 1, threshold - is_min] = True

                if values:
                    contrasts[threshold] = float(np.sum(values) / len(values))

            s0 = max(int(np.argmax(contrasts)), is_min)
            metric = 256.0 * contrasts[s0]
            if metric <= 256.0 * (percentage / 100.0):
                continue

            cube_index = s0 - is_min
            if cube_index < 0 or cube_index >= f_cube.shape[2]:
                continue

            window_mask = f_cube[:, :, cube_index]
            target = mask[row : row + S, col : col + S]
            target |= window_mask

            crr = np.zeros((S, S), dtype=np.float64)
            crr[target] = 2.0 * metric / 256.0
            contrast_map[row : row + S, col : col + S] = crr

    return mask[S : S + height, S : S + width], contrast_map[S : S + height, S : S + width]


def function_contrast_at_5_percent_rain(
    image: np.ndarray,
    S: int = 7,
    percentage: float = 5.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Compute the rainy image contrast mask and sampled morphology angles.
    """
    image = _validate_grayscale(image)

    gradient_x = ndimage.sobel(image, axis=1, mode="reflect")
    gradient_y = ndimage.sobel(image, axis=0, mode="reflect")
    gradient_mag = np.hypot(gradient_x, gradient_y)
    gradient_dir = np.degrees(np.arctan2(gradient_y, gradient_x))
    gradient_dir_180 = np.mod(gradient_dir, 180.0)

    max_mag = float(np.max(gradient_mag)) if gradient_mag.size else 0.0
    valid_angles = gradient_dir_180[gradient_mag > 0.2 * max_mag] if max_mag > 0 else np.array([])

    hist_edges = np.arange(0.0, 185.0, 5.0)
    counts, hist_edges = np.histogram(valid_angles, bins=hist_edges)
    bin_centers = hist_edges[:-1] + np.diff(hist_edges) / 2.0

    (
        dominant_angle_primary,
        dominant_angle_weighted,
        roi,
        _is_near_zero_or_180,
        _peak_angles,
    ) = find_adaptive_roi_check(bin_centers, counts, tolerance=5.0)

    angles = _sample_angles_from_roi(roi, num_angles=8)
    mask, contrast_map = _contrast_from_angles(image, S, percentage, angles)

    if verbose:
        edge_angle = np.mod(dominant_angle_primary + 90.0, 180.0)
        print(f"Dominant Gradient Angle theta = {dominant_angle_primary:.2f} deg")
        print(f"Edge Orientation phi = {edge_angle:.2f} deg")
        print(f"Detected Rain Streak Type = {_direction_label(dominant_angle_primary)}")
        print(f"Angles sampled for RainMark: {np.array2string(angles, precision=3)}")

    return (
        mask,
        contrast_map,
        dominant_angle_primary,
        dominant_angle_weighted,
        roi,
        angles,
    )


def function_contrast_at_5_percent_gt(
    image: np.ndarray,
    S: int,
    percentage: float,
    angles: np.ndarray,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the GT contrast mask with the rainy image's sampled angles.
    """
    if verbose:
        angles_array = np.asarray(angles, dtype=float).reshape(-1)
        print(f"Using {angles_array.size} orientation angles transferred from rainy image:")
        print(np.array2string(angles_array, precision=3))

    return _contrast_from_angles(image, S, percentage, angles)


def function_contrast_at_5_percent(
    image: np.ndarray,
    S: int = 7,
    percentage: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper for older callers.
    """
    mask, contrast_map, *_ = function_contrast_at_5_percent_rain(
        image,
        S=S,
        percentage=percentage,
        verbose=False,
    )
    return mask, contrast_map
