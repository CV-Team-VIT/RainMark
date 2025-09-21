"""
Adaptive mask cleanup functions - Python equivalent of adaptiveRainMaskCleanup.m
"""
import numpy as np
from skimage import measure, morphology
from skimage.measure import regionprops
from typing import Dict, Any


def adaptive_rain_mask_cleanup(mask: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Clean up rain mask by removing small noise blobs using adaptive thresholding.
    
    This is the Python equivalent of adaptiveRainMaskCleanup.m
    
    Args:
        mask: Binary mask to clean up
        verbose: Whether to print debug information
        
    Returns:
        Cleaned binary mask
    """
    # Ensure binary logical
    mask = mask > 0
    
    # Count components before cleanup
    labeled_before = measure.label(mask)
    num_components_before = labeled_before.max()
    
    if verbose:
        print(f'Components before cleanup: {num_components_before}')
    
    # Get blob statistics
    props = regionprops(labeled_before)
    
    if len(props) == 0:
        if verbose:
            print('No blobs found.')
        return mask
    
    # Extract areas
    areas = [prop.area for prop in props]
    
    # Adaptive cutoff calculation
    median_area = np.median(areas)
    cutoff = 1.5 * median_area
    cutoff = max(10, round(cutoff))  # Safety lower bound
    
    # Remove small blobs using area opening
    mask_clean = morphology.remove_small_objects(mask, min_size=cutoff)
    
    # Count components after cleanup
    if verbose:
        labeled_after = measure.label(mask_clean)
        num_components_after = labeled_after.max()
        print(f'Components after cleanup: {num_components_after}')
    
    return mask_clean


def connected_components_analysis(mask: np.ndarray) -> Dict[str, Any]:
    """
    Perform connected components analysis on binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Dictionary with component statistics
    """
    labeled = measure.label(mask)
    props = regionprops(labeled)
    
    if len(props) == 0:
        return {
            'num_components': 0,
            'areas': [],
            'centroids': [],
            'bounding_boxes': []
        }
    
    areas = [prop.area for prop in props]
    centroids = [prop.centroid for prop in props]
    bounding_boxes = [prop.bbox for prop in props]
    
    return {
        'num_components': len(props),
        'areas': areas,
        'centroids': centroids,
        'bounding_boxes': bounding_boxes,
        'total_area': sum(areas),
        'mean_area': np.mean(areas),
        'median_area': np.median(areas)
    }
