"""
Main processing script - Python equivalent of Main.m
"""
import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from typing import Optional
import sys

# Add the project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.image_utils import load_image, save_image


def process_dataset(gt_dir: str, rain_dir: str, save_dir: str, 
                   num_gt_images: int = 100, rain_variants: int = 1,
                   save_as_mat: bool = True) -> None:
    """
    Process dataset by converting images to .mat files for faster loading.
    
    This is the Python equivalent of Main.m preprocessing step.
    
    Args:
        gt_dir: Directory containing ground truth images
        rain_dir: Directory containing rainy images  
        save_dir: Directory to save processed .mat files
        num_gt_images: Number of ground truth images to process
        rain_variants: Number of rain variants per GT image
        save_as_mat: Whether to save as .mat files or keep as arrays
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print('🔄 Processing RGB images...')
    
    processed_count = 0
    
    for i in tqdm(range(1, num_gt_images + 1), desc="Processing images"):
        # Format index with leading zeros (3 digits: 001, 002, ...)
        idx = f"{i:03d}"
        
        # Ground truth image path
        gt_path = os.path.join(gt_dir, f'norain-{idx}.png')
        
        if os.path.isfile(gt_path):
            try:
                GT = load_image(gt_path)  # Keep in RGB
                
                if save_as_mat:
                    # Save as .mat file
                    sio.savemat(os.path.join(save_dir, f'GT_{i}.mat'), {'GT': GT})
                else:
                    # Save as numpy array
                    np.save(os.path.join(save_dir, f'GT_{i}.npy'), GT)
                    
            except Exception as e:
                print(f'⚠️ Error processing GT image {gt_path}: {e}')
                continue
        else:
            print(f'⚠️ Missing GT image: {gt_path}')
            continue
        
        # Rainy image path  
        rain_path = os.path.join(rain_dir, f'rain-{idx}.png')
        
        if os.path.isfile(rain_path):
            try:
                Rain = load_image(rain_path)  # Keep in RGB
                
                if save_as_mat:
                    # Save as .mat file
                    sio.savemat(os.path.join(save_dir, f'Rain_{i}.mat'), {'Rain': Rain})
                else:
                    # Save as numpy array
                    np.save(os.path.join(save_dir, f'Rain_{i}.npy'), Rain)
                    
                processed_count += 1
                
            except Exception as e:
                print(f'⚠️ Error processing Rain image {rain_path}: {e}')
                continue
        else:
            print(f'⚠️ Missing Rainy image: {rain_path}')
        
        # Optional: display progress every 20 images
        if i % 20 == 0:
            print(f'✅ Processed {i} / {num_gt_images} images.')
    
    file_type = ".mat files" if save_as_mat else "numpy arrays"
    print(f'🎉 All GT and Rainy images saved as RGB {file_type}.')
    print(f'📊 Successfully processed {processed_count} image pairs.')


def load_processed_image(file_path: str, image_key: str = None) -> np.ndarray:
    """
    Load a processed image from .mat or .npy file.
    
    Args:
        file_path: Path to the processed file
        image_key: Key for .mat files (e.g., 'GT', 'Rain')
        
    Returns:
        Loaded image as numpy array
    """
    if file_path.endswith('.mat'):
        if image_key is None:
            raise ValueError("image_key must be specified for .mat files")
        mat_data = sio.loadmat(file_path)
        return mat_data[image_key]
    elif file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        raise ValueError("Unsupported file format. Use .mat or .npy files.")


def batch_process_rain_detection(gt_dir: str, rain_dir: str, output_dir: str,
                                num_images: int = 100, S: int = 7,
                                visibility_percent: float = 5.0,
                                bright_thresh: int = 150) -> None:
    """
    Process multiple image pairs for rain detection.
    
    Args:
        gt_dir: Directory containing ground truth images or .mat files
        rain_dir: Directory containing rainy images or .mat files  
        output_dir: Directory to save results
        num_images: Number of image pairs to process
        S: Subwindow size for contrast computation
        visibility_percent: Visibility threshold percentage
        bright_thresh: Brightness threshold for saturation detection
    """
    from src.rain_detector import detect_rainstreaks, calculate_rain_severity
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for i in tqdm(range(1, num_images + 1), desc="Detecting rain streaks"):
        try:
            # Try to load from .mat files first, then from image files
            idx = f"{i:03d}"
            
            # Try .mat files
            gt_mat_path = os.path.join(gt_dir, f'GT_{i}.mat')
            rain_mat_path = os.path.join(rain_dir, f'Rain_{i}.mat')
            
            if os.path.exists(gt_mat_path) and os.path.exists(rain_mat_path):
                GT = load_processed_image(gt_mat_path, 'GT')
                Rain = load_processed_image(rain_mat_path, 'Rain')
            else:
                # Try image files
                gt_img_path = os.path.join(gt_dir, f'norain-{idx}.png')
                rain_img_path = os.path.join(rain_dir, f'rain-{idx}.png')
                
                if os.path.exists(gt_img_path) and os.path.exists(rain_img_path):
                    GT = load_image(gt_img_path)
                    Rain = load_image(rain_img_path)
                else:
                    print(f'⚠️ Missing image pair for index {i}')
                    continue
            
            # Detect rain streaks
            results = detect_rainstreaks(GT, Rain, S, visibility_percent, bright_thresh)
            
            # Calculate rain severity
            rain_severity = calculate_rain_severity(results)
            
            # Save overlay image
            overlay_path = os.path.join(output_dir, f'overlay_{i:03d}.png')
            save_image(results['overlay'], overlay_path)
            
            # Save results
            result_summary = {
                'image_id': i,
                'edge_amplification': results['e1'],
                'newly_saturated_pixels': results['ns1'],
                'streak_area': results['streak_area'],
                'percentage_streak_area': results['percentage_streak_area'],
                'rain_severity': rain_severity
            }
            
            results_summary.append(result_summary)
            
        except Exception as e:
            print(f'⚠️ Error processing image pair {i}: {e}')
            continue
    
    # Save summary results
    import json
    summary_path = os.path.join(output_dir, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f'🎉 Processed {len(results_summary)} image pairs.')
    print(f'📊 Results saved to {output_dir}')


if __name__ == "__main__":
    # Example usage
    gt_dir = 'Rain100H/'
    rain_dir = 'Rain100H/rainy/'
    save_dir = 'Rain100H/mat/'
    
    # Process dataset
    process_dataset(gt_dir, rain_dir, save_dir, num_gt_images=100)
