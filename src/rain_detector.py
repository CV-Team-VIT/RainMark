"""
Main rain streak detection algorithm - Python equivalent of detect_rainstreaks.m
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, Any, Optional

import os
import sys

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.contrast_analyzer import function_contrast_at_5_percent
from src.mask_cleanup import adaptive_rain_mask_cleanup
from utils.image_utils import rgb_to_gray, get_image_brightness


def detect_rainstreaks(GT: np.ndarray, Rain: np.ndarray, S: int = 7, 
                      visibility_percent: float = 5.0, bright_thresh: int = 150,
                      show_plots: bool = False) -> Dict[str, Any]:
    """
    Detect and quantify rain streaks in color images using contrast and saturation analysis.
    
    This is the Python equivalent of detect_rainstreaks.m
    
    Args:
        GT: Ground truth RGB image (numpy array)
        Rain: Rainy RGB image (numpy array)  
        S: Subwindow size for local contrast computation (default: 7)
        visibility_percent: Visibility threshold percentage (default: 5.0)
        bright_thresh: Threshold for detecting saturated (bright) pixels (default: 150)
        show_plots: Whether to display intermediate results (default: False)
        
    Returns:
        Dictionary with results:
        - e1: Edge amplification ratio
        - ns1: Percentage of newly saturated pixels
        - streak_area: Rain streak area ratio
        - percentage_streak_area: Percentage of image area covered by rain streaks
        - overlay: RGB image with streaks highlighted in neon green
        - rain_mask: Binary mask of detected rain streaks
    """
    
    # Ensure inputs are in proper format
    GT = GT.astype(np.float64)
    Rain = Rain.astype(np.float64)
    
    # 1. Compute gradient magnitude (convert RGB to grayscale for gradient computation)
    GT_gray = rgb_to_gray(GT)
    Rain_gray = rgb_to_gray(Rain)
    
    # 2. Contrast maps using grayscale images
    edges_GT, _ = function_contrast_at_5_percent(GT_gray, S, visibility_percent)
    edges_Rain, _ = function_contrast_at_5_percent(Rain_gray, S, visibility_percent)
    
    # 3. Edge amplification metric
    epsilon = 1e-6
    wI = np.sum(edges_Rain)  # W_I
    wJ = np.sum(edges_GT)    # W_J
    e1 = max(0, (wI - wJ)) / (max(wI, wJ) + epsilon)
    
    # 4. RESPO: Rain-streak Pixel Occupancy for newly saturated pixel detection
    bright_Rain = get_image_brightness(Rain)  # B_I, brightness of rainy image
    bright_GT = get_image_brightness(GT)      # B_J, brightness of GT image
    
    # 3x3 convolution kernel (equivalent to MATLAB's ones(3))
    kernel = np.ones((3, 3))
    
    # Detect newly saturated pixels
    newly_saturated_condition = (bright_Rain >= bright_thresh) & (bright_GT < bright_thresh)
    new_saturated = ndimage.convolve(newly_saturated_condition.astype(float), kernel, mode='constant') > 0
    ns1 = np.sum(new_saturated) / bright_GT.size
    
    # 5. Adaptive brightness delta
    # Candidate edge region: edges in Rain but not in GT
    Nmap = (edges_Rain == 1) & (edges_GT == 0)
    
    # Absolute brightness difference
    diff = bright_Rain - bright_GT
    
    # Percentile only over candidate region
    if np.any(Nmap):
        delta = np.percentile(np.abs(diff[Nmap]), 50)  # 50th percentile (median)
    else:
        delta = 0
    
    # 6. Rain streak coverage
    rain_streak_mask = Nmap & (np.abs(diff) > delta)
    
    if show_plots:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rain_streak_mask, cmap='gray')
        plt.title('Initial Rain Streak Mask')
        plt.axis('off')
    
    # Clean up the mask
    mask_clean = adaptive_rain_mask_cleanup(rain_streak_mask)
    streak_area = np.sum(mask_clean) / mask_clean.size
    percentage_streak_area = 100 * streak_area
    
    if show_plots:
        plt.subplot(1, 3, 2)
        plt.imshow(mask_clean, cmap='gray')
        plt.title('Cleaned Rain Streak Mask')
        plt.axis('off')
    
    # 7. Visualization (overlay neon green on rain streaks)
    overlay = Rain.copy().astype(np.uint8)
    
    # Neon green RGB values
    neon_r, neon_g, neon_b = 57, 255, 20
    
    # Apply neon green to masked pixels
    overlay[mask_clean, 0] = neon_r  # Red channel
    overlay[mask_clean, 1] = neon_g  # Green channel  
    overlay[mask_clean, 2] = neon_b  # Blue channel
    
    if show_plots:
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title('Rain Streaks Highlighted (Neon)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # 8. Output structure
    results = {
        'e1': e1,
        'ns1': ns1,
        'streak_area': streak_area,
        'percentage_streak_area': percentage_streak_area,
        'overlay': overlay,
        'rain_mask': mask_clean,
        'edge_map_gt': edges_GT,
        'edge_map_rain': edges_Rain,
        'brightness_diff': diff,
        'delta_threshold': delta
    }
    
    return results


def calculate_rain_severity(results: Dict[str, Any]) -> float:
    """
    Calculate rain severity score using the formula from the original MATLAB code.
    
    Args:
        results: Results dictionary from detect_rainstreaks
        
    Returns:
        Rain severity score
    """
    # Formula from MATLAB: Rain_Severity = 17.138146 * e1 + 0.132285 * ns1 + 0.887244 * streak_area
    rain_severity = (17.138146 * results['e1'] + 
                    0.132285 * results['ns1'] + 
                    0.887244 * results['streak_area'])
    
    return rain_severity
