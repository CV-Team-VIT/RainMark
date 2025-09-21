"""
Image utility functions for rain streak detection
"""
import numpy as np
import cv2
from PIL import Image
import os
from typing import Union, Tuple


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy array of the image in RGB format
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Use PIL to maintain RGB format (OpenCV uses BGR)
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def rgb_to_gray(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale using standard weights.
    
    Args:
        rgb_image: RGB image as numpy array
        
    Returns:
        Grayscale image as numpy array
    """
    if len(rgb_image.shape) == 3:
        # Standard RGB to grayscale conversion weights
        weights = np.array([0.299, 0.587, 0.114])
        return np.dot(rgb_image, weights)
    return rgb_image


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to file.
    
    Args:
        image: Image array
        output_path: Output file path
    """
    # Ensure image is in proper format
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(image)
    img.save(output_path)


def pad_image(image: np.ndarray, pad_size: Union[int, Tuple[int, int]], 
              mode: str = 'symmetric') -> np.ndarray:
    """
    Pad an image with specified padding mode.
    
    Args:
        image: Input image
        pad_size: Padding size (int for uniform, tuple for (vertical, horizontal))
        mode: Padding mode ('symmetric', 'constant', 'reflect', etc.)
        
    Returns:
        Padded image
    """
    if isinstance(pad_size, int):
        pad_width = ((pad_size, pad_size), (pad_size, pad_size))
    else:
        pad_width = ((pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]))
    
    if len(image.shape) == 3:
        pad_width = pad_width + ((0, 0),)
    
    return np.pad(image, pad_width, mode=mode)


def morphological_operations(image: np.ndarray, operation: str, 
                           kernel_size: int = 3) -> np.ndarray:
    """
    Perform morphological operations on binary image.
    
    Args:
        image: Binary image
        operation: 'erosion', 'dilation', 'opening', 'closing'
        kernel_size: Size of the structural element
        
    Returns:
        Processed binary image
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'erosion':
        return cv2.erode(image.astype(np.uint8), kernel, iterations=1)
    elif operation == 'dilation':
        return cv2.dilate(image.astype(np.uint8), kernel, iterations=1)
    elif operation == 'opening':
        return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def get_image_brightness(image: np.ndarray) -> np.ndarray:
    """
    Get brightness of image by taking maximum across RGB channels.
    
    Args:
        image: RGB image
        
    Returns:
        Brightness map
    """
    if len(image.shape) == 3:
        return np.max(image, axis=2)
    return image
