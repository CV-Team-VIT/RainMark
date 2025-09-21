"""
Additional utility functions for the rain detection system
"""
import numpy as np
from typing import List, Dict, Any
import json
import os


def batch_evaluate_images(image_pairs: List[tuple], output_file: str = None) -> List[Dict[str, Any]]:
    """
    Evaluate multiple image pairs and return results.
    
    Args:
        image_pairs: List of (gt_path, rain_path) tuples
        output_file: Optional file to save results
        
    Returns:
        List of result dictionaries
    """
    from ..src.rain_detector import detect_rainstreaks, calculate_rain_severity
    from .image_utils import load_image
    
    results = []
    
    for i, (gt_path, rain_path) in enumerate(image_pairs):
        try:
            gt_img = load_image(gt_path)
            rain_img = load_image(rain_path)
            
            detection_results = detect_rainstreaks(gt_img, rain_img)
            rain_severity = calculate_rain_severity(detection_results)
            
            result = {
                'pair_id': i,
                'gt_path': gt_path,
                'rain_path': rain_path,
                'edge_amplification': detection_results['e1'],
                'newly_saturated_pixels': detection_results['ns1'],
                'streak_area': detection_results['streak_area'],
                'percentage_streak_area': detection_results['percentage_streak_area'],
                'rain_severity': rain_severity
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics from evaluation results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with statistics
    """
    if not results:
        return {}
    
    metrics = ['edge_amplification', 'newly_saturated_pixels', 'streak_area', 
               'percentage_streak_area', 'rain_severity']
    
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in results if metric in r]
        if values:
            stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return stats


def create_evaluation_report(results: List[Dict[str, Any]], output_dir: str):
    """
    Create a comprehensive evaluation report.
    
    Args:
        results: List of result dictionaries
        output_dir: Directory to save the report
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    stats = compute_statistics(results)
    
    # Save statistics
    with open(os.path.join(output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Create plots
    if results:
        metrics = ['edge_amplification', 'newly_saturated_pixels', 'streak_area', 
                  'percentage_streak_area', 'rain_severity']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in results if metric in r]
            if values and i < len(axes):
                axes[i].hist(values, bins=20, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"📊 Evaluation report saved to {output_dir}")


def validate_image_paths(gt_dir: str, rain_dir: str, num_images: int) -> List[tuple]:
    """
    Validate and return existing image pairs.
    
    Args:
        gt_dir: Ground truth images directory
        rain_dir: Rainy images directory  
        num_images: Number of images to check
        
    Returns:
        List of valid (gt_path, rain_path) tuples
    """
    valid_pairs = []
    
    for i in range(1, num_images + 1):
        idx = f"{i:03d}"
        gt_path = os.path.join(gt_dir, f'norain-{idx}.png')
        rain_path = os.path.join(rain_dir, f'rain-{idx}.png')
        
        if os.path.exists(gt_path) and os.path.exists(rain_path):
            valid_pairs.append((gt_path, rain_path))
    
    return valid_pairs
