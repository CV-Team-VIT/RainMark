"""
Adaptive orientation ROI detection translated from findAdaptiveROI_Check.m.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def find_adaptive_roi_check(
    bin_centers: np.ndarray,
    counts: np.ndarray,
    tolerance: float = 5.0,
) -> Tuple[float, float, np.ndarray, bool, np.ndarray]:
    """
    Detect dominant orientation regions with circular 0/180 degree handling.
    """
    bin_centers = np.asarray(bin_centers, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()

    if bin_centers.size == 0 or counts.size == 0:
        roi = np.array([[0.0, 180.0]])
        return 0.0, 0.0, roi, True, np.array([0.0])

    extend_bins = min(3, counts.size)
    counts_ext = np.concatenate([counts[-extend_bins:], counts, counts[:extend_bins]])
    centers_ext = np.concatenate(
        [
            bin_centers[-extend_bins:] - 180.0,
            bin_centers,
            bin_centers[:extend_bins] + 180.0,
        ]
    )

    smooth_counts = gaussian_filter1d(counts_ext, sigma=1.0, mode="nearest")
    max_smooth = float(np.max(smooth_counts)) if smooth_counts.size else 0.0

    if max_smooth > 0:
        peaks, properties = find_peaks(
            smooth_counts,
            prominence=0.05 * max_smooth,
            distance=1,
        )
    else:
        peaks = np.array([], dtype=int)
        properties = {"prominences": np.array([])}

    if peaks.size == 0:
        idx_max = int(np.argmax(counts))
        pks = np.array([counts[idx_max]], dtype=float)
        locs = np.array([bin_centers[idx_max]], dtype=float)
    else:
        pks = smooth_counts[peaks].astype(float)
        locs = centers_ext[peaks].astype(float)

    locs = np.mod(locs, 180.0)

    wrap_mask = (locs < tolerance) | (locs > 180.0 - tolerance)
    if np.count_nonzero(wrap_mask) >= 2:
        wrap_count = np.sum(pks[wrap_mask])
        wrap_angle = np.mean(locs[wrap_mask])
        pks = np.concatenate([pks[~wrap_mask], np.array([wrap_count])])
        locs = np.concatenate([locs[~wrap_mask], np.array([wrap_angle])])

    if pks.size == 0 or np.max(pks) <= 0:
        pks = np.array([1.0])
        locs = np.array([0.0])

    strong_mask = pks >= 0.2 * np.max(pks)
    pks = pks[strong_mask]
    locs = locs[strong_mask]
    peak_angles = locs.copy()

    roi_list = []
    for ang, cnt in zip(locs, pks):
        idx = int(np.argmin(np.abs(bin_centers - ang)))
        start = max(0, idx - 2)
        stop = min(counts.size, idx + 3)
        local_mean = float(np.mean(counts[start:stop])) if stop > start else 0.0
        flatness_ratio = cnt / max(1.0, local_mean)

        if flatness_ratio < 1.3:
            roi_width = 25.0
        elif flatness_ratio > 2.0:
            roi_width = 10.0
        else:
            roi_width = 15.0

        if (ang < tolerance) or (ang > 180.0 - tolerance):
            roi_list.append([0.0, roi_width])
            roi_list.append([180.0 - roi_width, 180.0])
        else:
            roi_list.append([max(0.0, ang - roi_width), min(180.0, ang + roi_width)])

    roi = np.array(roi_list, dtype=float) if roi_list else np.array([[0.0, 0.0]])
    roi = roi[np.argsort(roi[:, 0])]

    merged = []
    current = roi[0].copy()
    for interval in roi[1:]:
        if interval[0] <= current[1]:
            current[1] = max(current[1], interval[1])
        else:
            merged.append(current)
            current = interval.copy()
    merged.append(current)
    roi = np.array(merged, dtype=float)

    idx_max = int(np.argmax(pks))
    dominant_angle_primary = float(locs[idx_max])
    dominant_angle_weighted = float(np.sum((pks / np.sum(pks)) * locs))
    is_near_zero_or_180 = (
        dominant_angle_primary < tolerance
        or dominant_angle_primary > 180.0 - tolerance
    )

    return (
        dominant_angle_primary,
        dominant_angle_weighted,
        roi,
        is_near_zero_or_180,
        peak_angles,
    )
