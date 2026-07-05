"""
Small morphology helpers used by the MATLAB RainMark port.
"""
from __future__ import annotations

import numpy as np


def create_line_strel(length: int, angle: float) -> np.ndarray:
    """
    Create a binary line footprint similar to MATLAB's strel('line', ...).
    """
    if length < 1:
        raise ValueError("length must be positive")

    angle_rad = np.deg2rad(angle)
    half_len = (length - 1) / 2.0
    offsets = np.arange(-half_len, half_len + 1)

    x = np.round(np.cos(angle_rad) * offsets).astype(int)
    y = np.round(np.sin(angle_rad) * offsets).astype(int)

    strel = np.zeros((y.max() - y.min() + 1, x.max() - x.min() + 1), dtype=bool)
    strel[y - y.min(), x - x.min()] = True
    return strel
