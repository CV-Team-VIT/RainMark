"""
Contrast analysis functions - Python equivalent of functionContrastAt5PerCent.m
"""
import numpy as np
from scipy import ndimage
from skimage import morphology
from tqdm import tqdm
from typing import Tuple


def function_contrast_at_5_percent(image: np.ndarray, S: int = 7, 
                                 percentage: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify visibly contrasting edges in a grayscale image based on local contrast
    estimation using Weber contrast (Human-perceived contrast metric).
    
    This is the Python equivalent of functionContrastAt5PerCent.m
    
    Args:
        image: Grayscale input image
        S: Local patch size to assess contrast (default: 7)
        percentage: Minimum required contrast visibility threshold (default: 5.0%)
        
    Returns:
        Tuple of (Mask, Contrast_map):
        - Mask: Binary mask where pixels above contrast threshold are marked True
        - Contrast_map: Contrast strength map for visible pixels
    """
    
    # Ensure image is grayscale and double precision
    if len(image.shape) > 2:
        raise ValueError("Input image must be grayscale")
    
    image = image.astype(np.float64)
    nl, nc = image.shape
    
    # 4-neighborhood structural elements for horizontal and vertical directions
    # Horizontal structure: vertical line
    seh = np.array([[0, 1, 0],
                    [0, 1, 0], 
                    [0, 0, 0]], dtype=np.uint8)
    
    # Vertical structure: horizontal line  
    seg = np.array([[0, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0]], dtype=np.uint8)
    
    # Pad image for morphological operations
    image_pad = np.pad(image, ((3, 3), (3, 3)), mode='symmetric')
    
    # Compute min/max using morphological operations
    # Erosion gives minimum, dilation gives maximum
    Igmin11 = morphology.erosion(image_pad, seg)
    Ihmin11 = morphology.erosion(image_pad, seh) 
    Igmax11 = morphology.dilation(image_pad, seg)
    Ihmax11 = morphology.dilation(image_pad, seh)
    
    # Extract the original size
    Igmin1 = Igmin11[3:3+nl, 3:3+nc]
    Ihmin1 = Ihmin11[3:3+nl, 3:3+nc]
    Igmax1 = Igmax11[3:3+nl, 3:3+nc]
    Ihmax1 = Ihmax11[3:3+nl, 3:3+nc]
    
    # Pad image and min/max matrices with window size S
    image_pad = np.pad(image, ((S, S), (S, S)), mode='symmetric')
    Igmin1_pad = np.pad(Igmin1, ((S, S), (S, S)), mode='symmetric')
    Ihmin1_pad = np.pad(Ihmin1, ((S, S), (S, S)), mode='symmetric')
    Igmax1_pad = np.pad(Igmax1, ((S, S), (S, S)), mode='symmetric')
    Ihmax1_pad = np.pad(Ihmax1, ((S, S), (S, S)), mode='symmetric')
    
    # Initialize output arrays
    Mask = np.zeros((nl + 2*S, nc + 2*S), dtype=bool)
    Crr = np.zeros((nl + 2*S, nc + 2*S), dtype=np.float64)
    
    # Adjust percentage for Weber contrast calculation
    percentage = percentage / 2
    
    # Process subwindows with step size S/2
    step_size = max(1, round(S/2))
    
    # Create progress bar
    total_windows = len(range(0, nl, step_size)) * len(range(0, nc, step_size))
    
    with tqdm(total=total_windows, desc="Processing contrast windows") as pbar:
        for ii in range(0, nl, step_size):
            for jj in range(0, nc, step_size):
                # Extract subwindow
                Is = image_pad[S+ii:2*S+ii, S+jj:2*S+jj].astype(np.float64)
                
                Isgmin = Igmin1_pad[S+ii:2*S+ii, S+jj:2*S+jj]
                Ishmin = Ihmin1_pad[S+ii:2*S+ii, S+jj:2*S+jj]
                Isgmax = Igmax1_pad[S+ii:2*S+ii, S+jj:2*S+jj]
                Ishmax = Ihmax1_pad[S+ii:2*S+ii, S+jj:2*S+jj]
                
                # Get intensity range
                Ismin = max(1, min(255, int(np.round(np.min(Is)))))
                Ismax = max(1, min(255, int(np.round(np.max(Is)))))
                
                if Ismin >= Ismax:
                    pbar.update(1)
                    continue
                
                # Initialize arrays for this subwindow
                Fcube = np.zeros((S, S, Ismax-Ismin+1), dtype=bool)
                C = np.zeros(256, dtype=np.float64)
                
                # Process each threshold level
                for s in range(Ismin, Ismax + 1):
                    Fg = 0  # Count of vertically separated pixels
                    Fh = 0  # Count of horizontally separated pixels
                    Cgxx1 = []  # Vertical contrasts
                    Chxx1 = []  # Horizontal contrasts
                    
                    # Process inner pixels (avoid boundaries)
                    for nn in range(1, S):
                        for mm in range(1, S):
                            # Vertical contrast
                            if (Isgmin[nn, mm] <= s) and (Isgmax[nn, mm] > s):
                                # Weber contrast calculation
                                contrast = min(
                                    abs(s - Is[nn, mm]) / max(s, Is[nn, mm]),
                                    abs(s - Is[nn, mm-1]) / max(s, Is[nn, mm-1])
                                )
                                Cgxx1.append(contrast)
                                Fg += 1
                                
                                # Mark pixels in cube
                                Fcube[nn, mm, s-Ismin] = True
                                Fcube[nn, mm-1, s-Ismin] = True
                            
                            # Horizontal contrast  
                            if (Ishmin[nn, mm] <= s) and (Ishmax[nn, mm] > s):
                                # Weber contrast calculation
                                contrast = min(
                                    abs(s - Is[nn, mm]) / max(s, Is[nn, mm]),
                                    abs(s - Is[nn-1, mm]) / max(s, Is[nn-1, mm])
                                )
                                Chxx1.append(contrast)
                                Fh += 1
                                
                                # Mark pixels in cube
                                Fcube[nn, mm, s-Ismin] = True
                                Fcube[nn-1, mm, s-Ismin] = True
                    
                    # Calculate average contrast for this threshold
                    if (Fg + Fh) > 0:
                        C[s] = (sum(Cgxx1) + sum(Chxx1)) / (Fg + Fh)
                
                # Find threshold that maximizes contrast
                s0 = max(Ismin, np.argmax(C))
                M = 256 * C[s0]
                
                # Check if contrast exceeds threshold
                if M > (256 * (percentage / 100)):
                    # Update mask using logical OR (for overlapping windows)
                    if s0-Ismin < Fcube.shape[2]:
                        window_mask = Fcube[:, :, s0-Ismin]
                        Mask[S+ii:2*S+ii, S+jj:2*S+jj] |= window_mask
                        
                        # Update contrast map
                        Crr1 = np.zeros((S, S))
                        Crr1[window_mask] = 2 * M / 256
                        Crr[S+ii:2*S+ii, S+jj:2*S+jj] = np.maximum(
                            Crr[S+ii:2*S+ii, S+jj:2*S+jj], Crr1
                        )
                
                pbar.update(1)
    
    # Extract final results (remove padding)
    final_mask = Mask[S:S+nl, S:S+nc]
    final_crr = Crr[S:S+nl, S:S+nc]
    
    return final_mask, final_crr
