"""Utility functions for frequency domain operations."""
#-Utility functions for frequency domain operations.
import numpy as np
from typing import Tuple


def fft2d(img: np.ndarray) -> np.ndarray:
    """Apply 2D Fast Fourier Transform to an image.
    
    Args:
        img: Input image in numpy format [H, W, C]
        
    Returns:
        Frequency domain representation of the image
    """
    return np.fft.fft2(img, axes=(0, 1))


def ifft2d(freq: np.ndarray) -> np.ndarray:
    """Apply inverse 2D Fast Fourier Transform.
    
    Args:
        freq: Frequency domain representation
        
    Returns:
        Spatial domain image [H, W, C]
    """
    img = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
    img = np.clip(img, 0, 255)
    img = img.astype('uint8')
    return img


def square_avg(freq: np.ndarray, index: int, max_num: int) -> float:
    """Calculate average magnitude of frequency components in a square pattern.
    
    This creates a square-shaped selection by taking:
    1. A horizontal line at the top (index) from index to max_num-index
    2. A horizontal line at the bottom (max_num-1-index) from index to max_num-index
    3. A vertical line on the left from index+1 to max_num-1-index
    4. A vertical line on the right from index+1 to max_num-1-index
    
    Args:
        freq: Magnitude of frequency components
        index: Index of the square (distance from the edge)
        max_num: Maximum size of the frequency grid
        
    Returns:
        Average magnitude in the square
    """
    # Top horizontal line
    rank1 = np.sum(freq[index, index:max_num-index, :])
    
    # Bottom horizontal line
    rank2 = np.sum(freq[max_num-1-index, index:max_num-index, :])
    
    # Left vertical line (excluding corners already counted)
    col1 = np.sum(freq[index+1:max_num-1-index, index, :])
    
    # Right vertical line (excluding corners already counted)
    col2 = np.sum(freq[index+1:max_num-1-index, max_num-1-index, :])
    
    # Calculate total number of elements in the square
    num = 4 * (max_num - 2 * index) - 4
    
    # Return average magnitude
    return (rank1 + rank2 + col1 + col2) / float(max(1, num))


def square_zero(freq: np.ndarray, index: int, max_num: int) -> np.ndarray:
    """Zero out frequency components in a square pattern.
    
    Args:
        freq: Frequency domain representation
        index: Index of the square (distance from the edge)
        max_num: Maximum size of the frequency grid
        
    Returns:
        Modified frequency domain with zeroed components
    """
    freq_modified = freq.copy()
    
    # Top horizontal line
    freq_modified[index, index:max_num-index, :] = 0
    
    # Bottom horizontal line
    freq_modified[max_num-1-index, index:max_num-index, :] = 0
    
    # Left vertical line
    freq_modified[index:max_num-index, index, :] = 0
    
    # Right vertical line
    freq_modified[index:max_num-index, max_num-1-index, :] = 0
    
    return freq_modified


def square_recover(freq_modified: np.ndarray, freq_ori: np.ndarray, 
                  index: int, max_num: int) -> np.ndarray:
    """Recover frequency components in a square pattern from original.
    
    Args:
        freq_modified: Modified frequency domain
        freq_ori: Original frequency domain
        index: Index of the square to recover
        max_num: Maximum size of the frequency grid
        
    Returns:
        Frequency domain with recovered components
    """
    result = freq_modified.copy()
    
    # Top horizontal line
    result[index, index:max_num-index, :] = freq_ori[index, index:max_num-index, :]
    
    # Bottom horizontal line
    result[max_num-1-index, index:max_num-index, :] = freq_ori[max_num-1-index, index:max_num-index, :]
    
    # Left vertical line
    result[index:max_num-index, index, :] = freq_ori[index:max_num-index, index, :]
    
    # Right vertical line
    result[index:max_num-index, max_num-1-index, :] = freq_ori[index:max_num-index, max_num-1-index, :]
    
    return result


def low_pass_filter(freq: np.ndarray, cutoff: int) -> np.ndarray:
    """Apply a low-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        cutoff: Frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.zeros((h, w))
    mask[distance <= cutoff] = 1
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def high_pass_filter(freq: np.ndarray, cutoff: int) -> np.ndarray:
    """Apply a high-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        cutoff: Frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.ones((h, w))
    mask[distance <= cutoff] = 0
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def band_pass_filter(freq: np.ndarray, low_cutoff: int, high_cutoff: int) -> np.ndarray:
    """Apply a band-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        low_cutoff: Lower frequency cutoff radius
        high_cutoff: Higher frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.zeros((h, w))
    mask[(distance >= low_cutoff) & (distance <= high_cutoff)] = 1
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def visualize_spectrum(freq: np.ndarray) -> np.ndarray:
    """Create a visualization of the frequency spectrum.
    
    Args:
        freq: Frequency domain representation
        
    Returns:
        Visualization image [H, W, C]
    """
    # Get magnitude spectrum
    mag_spectrum = np.abs(freq)
    
    # Apply log transform for better visualization
    mag_spectrum = np.log(1 + mag_spectrum)
    
    # Normalize to 0-255
    for i in range(mag_spectrum.shape[2]):
        mag_min = mag_spectrum[:, :, i].min()
        mag_max = mag_spectrum[:, :, i].max()
        mag_spectrum[:, :, i] = 255 * (mag_spectrum[:, :, i] - mag_min) / (mag_max - mag_min)
    
    # Shift to center
    mag_spectrum = np.fft.fftshift(mag_spectrum, axes=(0, 1))
    
    return mag_spectrum.astype(np.uint8)

import numpy as np
from typing import Tuple


def fft2d(img: np.ndarray) -> np.ndarray:
    """Apply 2D Fast Fourier Transform to an image.
    
    Args:
        img: Input image in numpy format [H, W, C]
        
    Returns:
        Frequency domain representation of the image
    """
    return np.fft.fft2(img, axes=(0, 1))


def ifft2d(freq: np.ndarray) -> np.ndarray:
    """Apply inverse 2D Fast Fourier Transform.
    
    Args:
        freq: Frequency domain representation
        
    Returns:
        Spatial domain image [H, W, C]
    """
    img = np.abs(np.fft.ifft2(freq, axes=(0, 1)))
    img = np.clip(img, 0, 255)
    img = img.astype('uint8')
    return img


def square_avg(freq: np.ndarray, index: int, max_num: int) -> float:
    """Calculate average magnitude of frequency components in a square pattern.
    
    This creates a square-shaped selection by taking:
    1. A horizontal line at the top (index) from index to max_num-index
    2. A horizontal line at the bottom (max_num-1-index) from index to max_num-index
    3. A vertical line on the left from index+1 to max_num-1-index
    4. A vertical line on the right from index+1 to max_num-1-index
    
    Args:
        freq: Magnitude of frequency components
        index: Index of the square (distance from the edge)
        max_num: Maximum size of the frequency grid
        
    Returns:
        Average magnitude in the square
    """
    # Top horizontal line
    rank1 = np.sum(freq[index, index:max_num-index, :])
    
    # Bottom horizontal line
    rank2 = np.sum(freq[max_num-1-index, index:max_num-index, :])
    
    # Left vertical line (excluding corners already counted)
    col1 = np.sum(freq[index+1:max_num-1-index, index, :])
    
    # Right vertical line (excluding corners already counted)
    col2 = np.sum(freq[index+1:max_num-1-index, max_num-1-index, :])
    
    # Calculate total number of elements in the square
    num = 4 * (max_num - 2 * index) - 4
    
    # Return average magnitude
    return (rank1 + rank2 + col1 + col2) / float(max(1, num))


def square_zero(freq: np.ndarray, index: int, max_num: int) -> np.ndarray:
    """Zero out frequency components in a square pattern.
    
    Args:
        freq: Frequency domain representation
        index: Index of the square (distance from the edge)
        max_num: Maximum size of the frequency grid
        
    Returns:
        Modified frequency domain with zeroed components
    """
    freq_modified = freq.copy()
    
    # Top horizontal line
    freq_modified[index, index:max_num-index, :] = 0
    
    # Bottom horizontal line
    freq_modified[max_num-1-index, index:max_num-index, :] = 0
    
    # Left vertical line
    freq_modified[index:max_num-index, index, :] = 0
    
    # Right vertical line
    freq_modified[index:max_num-index, max_num-1-index, :] = 0
    
    return freq_modified


def square_recover(freq_modified: np.ndarray, freq_ori: np.ndarray, 
                  index: int, max_num: int) -> np.ndarray:
    """Recover frequency components in a square pattern from original.
    
    Args:
        freq_modified: Modified frequency domain
        freq_ori: Original frequency domain
        index: Index of the square to recover
        max_num: Maximum size of the frequency grid
        
    Returns:
        Frequency domain with recovered components
    """
    result = freq_modified.copy()
    
    # Top horizontal line
    result[index, index:max_num-index, :] = freq_ori[index, index:max_num-index, :]
    
    # Bottom horizontal line
    result[max_num-1-index, index:max_num-index, :] = freq_ori[max_num-1-index, index:max_num-index, :]
    
    # Left vertical line
    result[index:max_num-index, index, :] = freq_ori[index:max_num-index, index, :]
    
    # Right vertical line
    result[index:max_num-index, max_num-1-index, :] = freq_ori[index:max_num-index, max_num-1-index, :]
    
    return result


def low_pass_filter(freq: np.ndarray, cutoff: int) -> np.ndarray:
    """Apply a low-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        cutoff: Frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.zeros((h, w))
    mask[distance <= cutoff] = 1
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def high_pass_filter(freq: np.ndarray, cutoff: int) -> np.ndarray:
    """Apply a high-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        cutoff: Frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.ones((h, w))
    mask[distance <= cutoff] = 0
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def band_pass_filter(freq: np.ndarray, low_cutoff: int, high_cutoff: int) -> np.ndarray:
    """Apply a band-pass filter in the frequency domain.
    
    Args:
        freq: Frequency domain representation
        low_cutoff: Lower frequency cutoff radius
        high_cutoff: Higher frequency cutoff radius
        
    Returns:
        Filtered frequency domain
    """
    h, w, c = freq.shape
    center_h, center_w = h // 2, w // 2
    
    # Create a distance matrix from center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # Create mask and apply fftshift
    mask = np.zeros((h, w))
    mask[(distance >= low_cutoff) & (distance <= high_cutoff)] = 1
    mask = np.fft.fftshift(mask)
    
    # Apply to all channels
    result = freq.copy()
    for i in range(c):
        result[:, :, i] = freq[:, :, i] * mask[:, :, np.newaxis]
        
    return result


def visualize_spectrum(freq: np.ndarray) -> np.ndarray:
    """Create a visualization of the frequency spectrum.
    
    Args:
        freq: Frequency domain representation
        
    Returns:
        Visualization image [H, W, C]
    """
    # Get magnitude spectrum
    mag_spectrum = np.abs(freq)
    
    # Apply log transform for better visualization
    mag_spectrum = np.log(1 + mag_spectrum)
    
    # Normalize to 0-255
    for i in range(mag_spectrum.shape[2]):
        mag_min = mag_spectrum[:, :, i].min()
        mag_max = mag_spectrum[:, :, i].max()
        mag_spectrum[:, :, i] = 255 * (mag_spectrum[:, :, i] - mag_min) / (mag_max - mag_min)
    
    # Shift to center
    mag_spectrum = np.fft.fftshift(mag_spectrum, axes=(0, 1))
    
    return mag_spectrum.astype(np.uint8)