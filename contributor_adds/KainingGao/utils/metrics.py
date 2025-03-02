"""Metrics for evaluating adversarial attacks and defenses."""
#-Metrics for evaluating adversarial attacks and defenses.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable


def attack_success_rate(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    original_labels: torch.Tensor,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> float:
    """Calculate the success rate of an adversarial attack.
    
    Args:
        model: Target model
        original_images: Original clean images
        adversarial_images: Adversarial images
        original_labels: Ground truth labels
        targeted: Whether the attack is targeted
        target_labels: Target labels for targeted attack
        
    Returns:
        Attack success rate (0.0 to 1.0)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to model device if needed
    original_labels = original_labels.to(device)
    if target_labels is not None:
        target_labels = target_labels.to(device)
    
    # Predict on adversarial images
    with torch.no_grad():
        outputs = model(adversarial_images.to(device))
        _, predicted_labels = torch.max(outputs, dim=1)
    
    # Calculate success rate
    if targeted:
        # For targeted attacks, success means classifying as the target class
        success = (predicted_labels == target_labels).float().mean().item()
    else:
        # For untargeted attacks, success means misclassifying
        success = (predicted_labels != original_labels).float().mean().item()
    
    return success


def perturbation_norm(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    norm_type: str = 'L2'
) -> torch.Tensor:
    """Calculate the perturbation norm between original and adversarial images.
    
    Args:
        original_images: Original clean images
        adversarial_images: Adversarial images
        norm_type: Type of norm to calculate ('L0', 'L1', 'L2', 'Linf')
        
    Returns:
        Tensor of perturbation norms for each image
    """
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    # Calculate perturbation
    perturbation = adversarial_images - original_images
    
    # Calculate norm based on type
    if norm_type.lower() == 'l0':
        # L0 norm counts the number of non-zero elements
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = (perturbation_flat != 0).float().sum(dim=1)
        
    elif norm_type.lower() == 'l1':
        # L1 norm is the sum of absolute values
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=1, dim=1)
        
    elif norm_type.lower() == 'l2':
        # L2 norm is the Euclidean distance
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=2, dim=1)
        
    elif norm_type.lower() == 'linf':
        # Linf norm is the maximum absolute value
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=float('inf'), dim=1)
        
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
        
    return norm


def structural_similarity(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor
) -> torch.Tensor:
    """Calculate the structural similarity index (SSIM) between original and adversarial images.
    
    Args:
        original_images: Original clean images [B, C, H, W]
        adversarial_images: Adversarial images [B, C, H, W]
        
    Returns:
        Tensor of SSIM values for each image
    """
    # Constants for SSIM calculation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    batch_size, channels, height, width = original_images.shape
    ssim_values = torch.zeros(batch_size, device=device)
    
    # Calculate SSIM for each image in the batch
    for i in range(batch_size):
        # Extract current images
        img1 = original_images[i]
        img2 = adversarial_images[i]
        
        # Calculate means
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
        
        # Calculate variances and covariance
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Average over spatial dimensions and channels
        ssim_values[i] = ssim_map.mean()
        
    return ssim_values


def query_efficiency(
    success_rate: float,
    avg_queries: float,
    baseline_queries: Optional[float] = None
) -> Dict[str, float]:
    """Calculate query efficiency metrics for an attack.
    
    Args:
        success_rate: Attack success rate (0.0 to 1.0)
        avg_queries: Average number of queries used by the attack
        baseline_queries: Average number of queries for a baseline attack
        
    Returns:
        Dictionary of query efficiency metrics
    """
    metrics = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,
        'queries_per_success': avg_queries / max(success_rate, 1e-10)
    }
    
    if baseline_queries is not None:
        metrics['query_reduction'] = 1.0 - (avg_queries / baseline_queries)
        metrics['relative_efficiency'] = baseline_queries / max(avg_queries, 1e-10)
        
    return metrics


def frequency_domain_analysis(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Analyze perturbation in frequency domain.
    
    Args:
        original_images: Original clean images [B, C, H, W]
        adversarial_images: Adversarial images [B, C, H, W]
        
    Returns:
        Dictionary of frequency domain metrics
    """
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    batch_size, channels, height, width = original_images.shape
    metrics = {}
    
    # Calculate perturbation
    perturbation = adversarial_images - original_images
    
    # Convert to numpy for FFT
    perturbation_np = perturbation.detach().cpu().numpy()
    
    # Average over all images in the batch for visualization
    avg_perturbation = np.mean(np.abs(perturbation_np), axis=0)
    
    # Compute 2D FFT for each channel
    freq_domain = np.zeros((channels, height, width), dtype=complex)
    for c in range(channels):
        freq_domain[c] = np.fft.fft2(avg_perturbation[c])
        
    # Shift zero frequency to center
    freq_domain_shifted = np.zeros_like(freq_domain)
    for c in range(channels):
        freq_domain_shifted[c] = np.fft.fftshift(freq_domain[c])
        
    # Calculate magnitude spectrum (log scale for visualization)
    magnitude_spectrum = np.log(1 + np.abs(freq_domain_shifted))
    
    # Normalize for visualization
    magnitude_max = np.max(magnitude_spectrum)
    if magnitude_max > 0:
        magnitude_spectrum = magnitude_spectrum / magnitude_max
        
    # Calculate frequency statistics
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Calculate energy distribution at different frequency ranges
    low_freq_mask = distance_from_center <= min(height, width) * 0.25
    mid_freq_mask = (distance_from_center > min(height, width) * 0.25) & \
                     (distance_from_center <= min(height, width) * 0.5)
    high_freq_mask = distance_from_center > min(height, width) * 0.5
    
    # Calculate energy in each frequency band
    total_energy = np.sum(np.abs(freq_domain_shifted) ** 2)
    low_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * low_freq_mask[:, :, np.newaxis])
    mid_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * mid_freq_mask[:, :, np.newaxis])
    high_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * high_freq_mask[:, :, np.newaxis])
    
    # Normalize energies
    if total_energy > 0:
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
    else:
        low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0.0
        
    # Store metrics
    metrics['magnitude_spectrum'] = torch.tensor(magnitude_spectrum)
    metrics['low_freq_ratio'] = torch.tensor(low_freq_ratio)
    metrics['mid_freq_ratio'] = torch.tensor(mid_freq_ratio)
    metrics['high_freq_ratio'] = torch.tensor(high_freq_ratio)
    
    return metrics


def defense_effectiveness(
    model: nn.Module,
    adversarial_images: torch.Tensor,
    original_labels: torch.Tensor,
    defended_images: torch.Tensor,
    original_images: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate the effectiveness of a defense.
    
    Args:
        model: Target model
        adversarial_images: Adversarial images before defense
        original_labels: Ground truth labels
        defended_images: Adversarial images after defense
        original_images: Original clean images (optional)
        
    Returns:
        Dictionary of defense effectiveness metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to model device if needed
    original_labels = original_labels.to(device)
    adversarial_images = adversarial_images.to(device)
    defended_images = defended_images.to(device)
    
    metrics = {}
    
    # Predict on adversarial and defended images
    with torch.no_grad():
        adv_outputs = model(adversarial_images)
        def_outputs = model(defended_images)
        
        _, adv_predictions = torch.max(adv_outputs, dim=1)
        _, def_predictions = torch.max(def_outputs, dim=1)
        
    # Calculate accuracy before and after defense
    adv_accuracy = (adv_predictions == original_labels).float().mean().item()
    def_accuracy = (def_predictions == original_labels).float().mean().item()
    
    # Calculate defense improvement
    metrics['adv_accuracy'] = adv_accuracy
    metrics['def_accuracy'] = def_accuracy
    metrics['defense_improvement'] = def_accuracy - adv_accuracy
    
    # Calculate image quality metrics if original images are provided
    if original_images is not None:
        original_images = original_images.to(device)
        
        # Calculate L2 distance between original and defended images
        batch_size = original_images.shape[0]
        original_flat = original_images.view(batch_size, -1)
        defended_flat = defended_images.view(batch_size, -1)
        
        l2_distance = torch.norm(original_flat - defended_flat, p=2, dim=1).mean().item()
        metrics['defense_distortion'] = l2_distance
        
        # Calculate SSIM between original and defended images
        ssim_values = structural_similarity(original_images, defended_images)
        metrics['defense_ssim'] = ssim_values.mean().item()
        
    return metrics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable


def attack_success_rate(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    original_labels: torch.Tensor,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> float:
    """Calculate the success rate of an adversarial attack.
    
    Args:
        model: Target model
        original_images: Original clean images
        adversarial_images: Adversarial images
        original_labels: Ground truth labels
        targeted: Whether the attack is targeted
        target_labels: Target labels for targeted attack
        
    Returns:
        Attack success rate (0.0 to 1.0)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to model device if needed
    original_labels = original_labels.to(device)
    if target_labels is not None:
        target_labels = target_labels.to(device)
    
    # Predict on adversarial images
    with torch.no_grad():
        outputs = model(adversarial_images.to(device))
        _, predicted_labels = torch.max(outputs, dim=1)
    
    # Calculate success rate
    if targeted:
        # For targeted attacks, success means classifying as the target class
        success = (predicted_labels == target_labels).float().mean().item()
    else:
        # For untargeted attacks, success means misclassifying
        success = (predicted_labels != original_labels).float().mean().item()
    
    return success


def perturbation_norm(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    norm_type: str = 'L2'
) -> torch.Tensor:
    """Calculate the perturbation norm between original and adversarial images.
    
    Args:
        original_images: Original clean images
        adversarial_images: Adversarial images
        norm_type: Type of norm to calculate ('L0', 'L1', 'L2', 'Linf')
        
    Returns:
        Tensor of perturbation norms for each image
    """
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    # Calculate perturbation
    perturbation = adversarial_images - original_images
    
    # Calculate norm based on type
    if norm_type.lower() == 'l0':
        # L0 norm counts the number of non-zero elements
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = (perturbation_flat != 0).float().sum(dim=1)
        
    elif norm_type.lower() == 'l1':
        # L1 norm is the sum of absolute values
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=1, dim=1)
        
    elif norm_type.lower() == 'l2':
        # L2 norm is the Euclidean distance
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=2, dim=1)
        
    elif norm_type.lower() == 'linf':
        # Linf norm is the maximum absolute value
        batch_size = perturbation.shape[0]
        perturbation_flat = perturbation.view(batch_size, -1)
        norm = torch.norm(perturbation_flat, p=float('inf'), dim=1)
        
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
        
    return norm


def structural_similarity(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor
) -> torch.Tensor:
    """Calculate the structural similarity index (SSIM) between original and adversarial images.
    
    Args:
        original_images: Original clean images [B, C, H, W]
        adversarial_images: Adversarial images [B, C, H, W]
        
    Returns:
        Tensor of SSIM values for each image
    """
    # Constants for SSIM calculation
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    batch_size, channels, height, width = original_images.shape
    ssim_values = torch.zeros(batch_size, device=device)
    
    # Calculate SSIM for each image in the batch
    for i in range(batch_size):
        # Extract current images
        img1 = original_images[i]
        img2 = adversarial_images[i]
        
        # Calculate means
        mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
        
        # Calculate variances and covariance
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1_mu2
        
        # Calculate SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Average over spatial dimensions and channels
        ssim_values[i] = ssim_map.mean()
        
    return ssim_values


def query_efficiency(
    success_rate: float,
    avg_queries: float,
    baseline_queries: Optional[float] = None
) -> Dict[str, float]:
    """Calculate query efficiency metrics for an attack.
    
    Args:
        success_rate: Attack success rate (0.0 to 1.0)
        avg_queries: Average number of queries used by the attack
        baseline_queries: Average number of queries for a baseline attack
        
    Returns:
        Dictionary of query efficiency metrics
    """
    metrics = {
        'success_rate': success_rate,
        'avg_queries': avg_queries,
        'queries_per_success': avg_queries / max(success_rate, 1e-10)
    }
    
    if baseline_queries is not None:
        metrics['query_reduction'] = 1.0 - (avg_queries / baseline_queries)
        metrics['relative_efficiency'] = baseline_queries / max(avg_queries, 1e-10)
        
    return metrics


def frequency_domain_analysis(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Analyze perturbation in frequency domain.
    
    Args:
        original_images: Original clean images [B, C, H, W]
        adversarial_images: Adversarial images [B, C, H, W]
        
    Returns:
        Dictionary of frequency domain metrics
    """
    # Ensure inputs are on the same device
    device = original_images.device
    adversarial_images = adversarial_images.to(device)
    
    batch_size, channels, height, width = original_images.shape
    metrics = {}
    
    # Calculate perturbation
    perturbation = adversarial_images - original_images
    
    # Convert to numpy for FFT
    perturbation_np = perturbation.detach().cpu().numpy()
    
    # Average over all images in the batch for visualization
    avg_perturbation = np.mean(np.abs(perturbation_np), axis=0)
    
    # Compute 2D FFT for each channel
    freq_domain = np.zeros((channels, height, width), dtype=complex)
    for c in range(channels):
        freq_domain[c] = np.fft.fft2(avg_perturbation[c])
        
    # Shift zero frequency to center
    freq_domain_shifted = np.zeros_like(freq_domain)
    for c in range(channels):
        freq_domain_shifted[c] = np.fft.fftshift(freq_domain[c])
        
    # Calculate magnitude spectrum (log scale for visualization)
    magnitude_spectrum = np.log(1 + np.abs(freq_domain_shifted))
    
    # Normalize for visualization
    magnitude_max = np.max(magnitude_spectrum)
    if magnitude_max > 0:
        magnitude_spectrum = magnitude_spectrum / magnitude_max
        
    # Calculate frequency statistics
    center_y, center_x = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Calculate energy distribution at different frequency ranges
    low_freq_mask = distance_from_center <= min(height, width) * 0.25
    mid_freq_mask = (distance_from_center > min(height, width) * 0.25) & \
                     (distance_from_center <= min(height, width) * 0.5)
    high_freq_mask = distance_from_center > min(height, width) * 0.5
    
    # Calculate energy in each frequency band
    total_energy = np.sum(np.abs(freq_domain_shifted) ** 2)
    low_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * low_freq_mask[:, :, np.newaxis])
    mid_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * mid_freq_mask[:, :, np.newaxis])
    high_freq_energy = np.sum(np.abs(freq_domain_shifted) ** 2 * high_freq_mask[:, :, np.newaxis])
    
    # Normalize energies
    if total_energy > 0:
        low_freq_ratio = low_freq_energy / total_energy
        mid_freq_ratio = mid_freq_energy / total_energy
        high_freq_ratio = high_freq_energy / total_energy
    else:
        low_freq_ratio = mid_freq_ratio = high_freq_ratio = 0.0
        
    # Store metrics
    metrics['magnitude_spectrum'] = torch.tensor(magnitude_spectrum)
    metrics['low_freq_ratio'] = torch.tensor(low_freq_ratio)
    metrics['mid_freq_ratio'] = torch.tensor(mid_freq_ratio)
    metrics['high_freq_ratio'] = torch.tensor(high_freq_ratio)
    
    return metrics


def defense_effectiveness(
    model: nn.Module,
    adversarial_images: torch.Tensor,
    original_labels: torch.Tensor,
    defended_images: torch.Tensor,
    original_images: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Evaluate the effectiveness of a defense.
    
    Args:
        model: Target model
        adversarial_images: Adversarial images before defense
        original_labels: Ground truth labels
        defended_images: Adversarial images after defense
        original_images: Original clean images (optional)
        
    Returns:
        Dictionary of defense effectiveness metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Move inputs to model device if needed
    original_labels = original_labels.to(device)
    adversarial_images = adversarial_images.to(device)
    defended_images = defended_images.to(device)
    
    metrics = {}
    
    # Predict on adversarial and defended images
    with torch.no_grad():
        adv_outputs = model(adversarial_images)
        def_outputs = model(defended_images)
        
        _, adv_predictions = torch.max(adv_outputs, dim=1)
        _, def_predictions = torch.max(def_outputs, dim=1)
        
    # Calculate accuracy before and after defense
    adv_accuracy = (adv_predictions == original_labels).float().mean().item()
    def_accuracy = (def_predictions == original_labels).float().mean().item()
    
    # Calculate defense improvement
    metrics['adv_accuracy'] = adv_accuracy
    metrics['def_accuracy'] = def_accuracy
    metrics['defense_improvement'] = def_accuracy - adv_accuracy
    
    # Calculate image quality metrics if original images are provided
    if original_images is not None:
        original_images = original_images.to(device)
        
        # Calculate L2 distance between original and defended images
        batch_size = original_images.shape[0]
        original_flat = original_images.view(batch_size, -1)
        defended_flat = defended_images.view(batch_size, -1)
        
        l2_distance = torch.norm(original_flat - defended_flat, p=2, dim=1).mean().item()
        metrics['defense_distortion'] = l2_distance
        
        # Calculate SSIM between original and defended images
        ssim_values = structural_similarity(original_images, defended_images)
        metrics['defense_ssim'] = ssim_values.mean().item()
        
    return metrics