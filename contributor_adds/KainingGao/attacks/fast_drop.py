"""Implementation of the FastDrop attack with improvements."""
#-Implementation of the FastDrop attack with improvements
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from .base_attack import BaseAttack
from ..utils.frequency_utils import (
    fft2d, 
    ifft2d, 
    square_avg, 
    square_zero, 
    square_recover
)


class FastDrop(BaseAttack):
    """FastDrop attack implementation with improvements.
    
    This attack operates in the frequency domain, strategically dropping
    frequency components to create adversarial examples with minimal queries.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        norm_type: str = 'l2',
        norm_bound: float = 5.0,
        max_queries: int = 100,
        square_max_num: int = 32,  # Size of frequency grid
        targeted: bool = False,
        freq_recovery_rounds: int = 2,
        progressive_scan: bool = True,
        verbose: bool = False
    ):
        """Initialize FastDrop attack.
        
        Args:
            model: Target model to attack
            device: Device to run the attack on
            norm_type: Type of norm to constrain perturbation ('l2' or 'linf')
            norm_bound: Maximum perturbation size
            max_queries: Maximum number of model queries allowed
            square_max_num: Size of frequency grid (32 for CIFAR, 224 for ImageNet)
            targeted: Whether the attack is targeted
            freq_recovery_rounds: Number of rounds for recovery optimization
            progressive_scan: Whether to use progressive frequency scanning
            verbose: Whether to print progress information
        """
        super().__init__(model, device, norm_type, norm_bound, targeted, verbose)
        
        self.max_queries = max_queries
        self.square_max_num = square_max_num
        self.freq_recovery_rounds = freq_recovery_rounds
        self.progressive_scan = progressive_scan
        
        # Determine mean and std for normalization based on dataset
        if square_max_num == 32:  # CIFAR-10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        else:  # ImageNet
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image for model input.
        
        Args:
            image: Image tensor
            
        Returns:
            Normalized image tensor
        """
        # Create normalized copy of the image
        img = image.clone()
        
        # Apply normalization
        for i in range(3):
            img[:, i, :, :] = (img[:, i, :, :] - self.mean[i]) / self.std[i]
            
        return img
    
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate adversarial examples using FastDrop.
        
        Args:
            images: Clean images to perturb [B, C, H, W]
            labels: True labels of the images
            target_labels: Target labels for targeted attack
            
        Returns:
            Tuple containing adversarial examples and attack metadata
        """
        batch_size = images.shape[0]
        adv_images = images.clone()
        metadata = {
            'success': [False] * batch_size,
            'queries': [0] * batch_size,
            'perturbation_norm': [0.0] * batch_size
        }
        
        for i in range(batch_size):
            # Process single image
            orig_img = images[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
            orig_img = orig_img.astype(np.uint8)
            
            # Get original prediction
            img_tensor = self.preprocess_image(images[i:i+1].to(self.device))
            with torch.no_grad():
                output = self.model(img_tensor)
                _, orig_label = torch.max(output, dim=1)
            
            self.query_count += 1
            metadata['queries'][i] += 1
            
            # Skip if original prediction is incorrect or matches target
            if self.targeted:
                if orig_label.item() == target_labels[i].item():
                    continue
                target_label = target_labels[i].item()
            else:
                if orig_label.item() != labels[i].item():
                    continue
                target_label = None
                
            # Perform FFT on the image
            freq = fft2d(orig_img)
            freq_ori = freq.copy()
            freq_ori_m = np.abs(freq_ori)
            
            # Stage 1: Analyze frequency components
            freq_abs = np.abs(freq)
            num_block = int(self.square_max_num / 2)
            block_sum = np.zeros(num_block)
            
            for j in range(num_block):
                block_sum[j] = square_avg(freq_abs, j, self.square_max_num)
                
            # Sort frequency components by importance
            block_sum_ind = np.argsort(block_sum)
            block_sum_ind_flag = np.zeros(num_block)
            
            # Stage 2: Find minimal set of frequencies to drop
            if self.progressive_scan:
                # Custom scan ranges based on dataset
                if self.square_max_num == 32:  # CIFAR-10
                    ranges = [1, 2] + list(range(3, num_block + 1))
                else:  # ImageNet
                    ranges = [60, 80, 90] + list(range(91, num_block + 1))
                    
                freq_modified = freq.copy()
                freq_modified_m = np.abs(freq_modified)
                freq_modified_p = np.angle(freq_modified)
                
                success = False
                mag_start = 0
                
                for mag in ranges:
                    # Drop frequencies in batches
                    for j in range(mag_start, mag):
                        ind = block_sum_ind[j]
                        freq_modified_m = square_zero(freq_modified_m, ind, self.square_max_num)
                        
                    # Reconstruct complex frequency representation
                    freq_modified = freq_modified_m * np.exp(1j * freq_modified_p)
                    
                    # Convert back to image
                    img_adv = ifft2d(freq_modified)
                    
                    # Check if attack succeeded
                    img_adv_tensor = torch.tensor(
                        img_adv.transpose(2, 0, 1) / 255.0, 
                        dtype=torch.float32
                    ).unsqueeze(0)
                    img_adv_tensor = self.preprocess_image(img_adv_tensor.to(self.device))
                    
                    self.query_count += 1
                    metadata['queries'][i] += 1
                    
                    with torch.no_grad():
                        output = self.model(img_adv_tensor)
                        _, adv_label = torch.max(output, dim=1)
                    
                    # Check for success based on attack type
                    if self.targeted:
                        attack_success = (adv_label.item() == target_label)
                    else:
                        attack_success = (adv_label.item() != orig_label.item())
                        
                    if attack_success:
                        # Calculate perturbation norm
                        l2_norm = torch.norm(
                            self.preprocess_image(images[i:i+1].to(self.device)) - 
                            img_adv_tensor,
                            p=2
                        ).item()
                        
                        # If within bounds, we're done
                        if l2_norm < self.norm_bound:
                            adv_images[i] = torch.tensor(
                                img_adv.transpose(2, 0, 1) / 255.0, 
                                dtype=torch.float32
                            )
                            metadata['success'][i] = True
                            metadata['perturbation_norm'][i] = l2_norm
                            success = True
                            break
                    
                    mag_start = mag
                    
                    # Check if we've exceeded query budget
                    if metadata['queries'][i] >= self.max_queries:
                        break
                        
                if success or metadata['queries'][i] >= self.max_queries:
                    continue
            
                # Stage 3: Optimize by recovering unnecessary frequency components
                max_i = mag_start - 1
                block_sum_ind_flag[:max_i+1] = 1
                freq_m = freq_modified_m.copy()
                freq_p = np.angle(freq_ori)  # Use original phase
                
                img_temp = img_adv.copy()
                
                # Stage 4: Iterate through recovery rounds
                for _ in range(self.freq_recovery_rounds):
                    for j in range(max_i, -1, -1):
                        if block_sum_ind_flag[j] == 1:
                            ind = block_sum_ind[j]
                            
                            # Try restoring this frequency component
                            freq_m_test = freq_m.copy()
                            freq_m_test = square_recover(freq_m_test, freq_ori_m, ind, self.square_max_num)
                            freq_test = freq_m_test * np.exp(1j * freq_p)
                            
                            # Convert back to image
                            img_adv_test = ifft2d(freq_test)
                            
                            # Test if still adversarial
                            img_adv_tensor = torch.tensor(
                                img_adv_test.transpose(2, 0, 1) / 255.0, 
                                dtype=torch.float32
                            ).unsqueeze(0)
                            img_adv_tensor = self.preprocess_image(img_adv_tensor.to(self.device))
                            
                            self.query_count += 1
                            metadata['queries'][i] += 1
                            
                            with torch.no_grad():
                                output = self.model(img_adv_tensor)
                                _, adv_label = torch.max(output, dim=1)
                                
                            if self.targeted:
                                still_adversarial = (adv_label.item() == target_label)
                            else:
                                still_adversarial = (adv_label.item() != orig_label.item())
                                
                            if still_adversarial:
                                # Keep the restored frequency
                                freq_m = freq_m_test
                                img_temp = img_adv_test.copy()
                                block_sum_ind_flag[j] = 0
                            else:
                                # Revert to zeroed frequency
                                pass
                                
                            # Check if we've exceeded query budget
                            if metadata['queries'][i] >= self.max_queries:
                                break
                                
                    if metadata['queries'][i] >= self.max_queries:
                        break
                
                # Final adversarial example
                l2_norm = torch.norm(
                    self.preprocess_image(images[i:i+1].to(self.device)) - 
                    self.preprocess_image(torch.tensor(
                        img_temp.transpose(2, 0, 1) / 255.0, 
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)),
                    p=2
                ).item()
                
                adv_images[i] = torch.tensor(
                    img_temp.transpose(2, 0, 1) / 255.0, 
                    dtype=torch.float32
                )
                metadata['success'][i] = True
                metadata['perturbation_norm'][i] = l2_norm
        
        # Calculate overall attack statistics
        metadata['overall_success_rate'] = sum(metadata['success']) / batch_size
        metadata['avg_queries'] = sum(metadata['queries']) / batch_size
        metadata['avg_perturbation'] = sum(metadata['perturbation_norm']) / max(1, sum(metadata['success']))
        metadata['total_queries'] = self.query_count
        
        return adv_images, metadata

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from .base_attack import BaseAttack
from ..utils.frequency_utils import (
    fft2d, 
    ifft2d, 
    square_avg, 
    square_zero, 
    square_recover
)


class FastDrop(BaseAttack):
    """FastDrop attack implementation with improvements.
    
    This attack operates in the frequency domain, strategically dropping
    frequency components to create adversarial examples with minimal queries.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        norm_type: str = 'l2',
        norm_bound: float = 5.0,
        max_queries: int = 100,
        square_max_num: int = 32,  # Size of frequency grid
        targeted: bool = False,
        freq_recovery_rounds: int = 2,
        progressive_scan: bool = True,
        verbose: bool = False
    ):
        """Initialize FastDrop attack.
        
        Args:
            model: Target model to attack
            device: Device to run the attack on
            norm_type: Type of norm to constrain perturbation ('l2' or 'linf')
            norm_bound: Maximum perturbation size
            max_queries: Maximum number of model queries allowed
            square_max_num: Size of frequency grid (32 for CIFAR, 224 for ImageNet)
            targeted: Whether the attack is targeted
            freq_recovery_rounds: Number of rounds for recovery optimization
            progressive_scan: Whether to use progressive frequency scanning
            verbose: Whether to print progress information
        """
        super().__init__(model, device, norm_type, norm_bound, targeted, verbose)
        
        self.max_queries = max_queries
        self.square_max_num = square_max_num
        self.freq_recovery_rounds = freq_recovery_rounds
        self.progressive_scan = progressive_scan
        
        # Determine mean and std for normalization based on dataset
        if square_max_num == 32:  # CIFAR-10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2023, 0.1994, 0.2010)
        else:  # ImageNet
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image for model input.
        
        Args:
            image: Image tensor
            
        Returns:
            Normalized image tensor
        """
        # Create normalized copy of the image
        img = image.clone()
        
        # Apply normalization
        for i in range(3):
            img[:, i, :, :] = (img[:, i, :, :] - self.mean[i]) / self.std[i]
            
        return img
    
    def attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate adversarial examples using FastDrop.
        
        Args:
            images: Clean images to perturb [B, C, H, W]
            labels: True labels of the images
            target_labels: Target labels for targeted attack
            
        Returns:
            Tuple containing adversarial examples and attack metadata
        """
        batch_size = images.shape[0]
        adv_images = images.clone()
        metadata = {
            'success': [False] * batch_size,
            'queries': [0] * batch_size,
            'perturbation_norm': [0.0] * batch_size
        }
        
        for i in range(batch_size):
            # Process single image
            orig_img = images[i].detach().cpu().numpy().transpose(1, 2, 0) * 255
            orig_img = orig_img.astype(np.uint8)
            
            # Get original prediction
            img_tensor = self.preprocess_image(images[i:i+1].to(self.device))
            with torch.no_grad():
                output = self.model(img_tensor)
                _, orig_label = torch.max(output, dim=1)
            
            self.query_count += 1
            metadata['queries'][i] += 1
            
            # Skip if original prediction is incorrect or matches target
            if self.targeted:
                if orig_label.item() == target_labels[i].item():
                    continue
                target_label = target_labels[i].item()
            else:
                if orig_label.item() != labels[i].item():
                    continue
                target_label = None
                
            # Perform FFT on the image
            freq = fft2d(orig_img)
            freq_ori = freq.copy()
            freq_ori_m = np.abs(freq_ori)
            
            # Stage 1: Analyze frequency components
            freq_abs = np.abs(freq)
            num_block = int(self.square_max_num / 2)
            block_sum = np.zeros(num_block)
            
            for j in range(num_block):
                block_sum[j] = square_avg(freq_abs, j, self.square_max_num)
                
            # Sort frequency components by importance
            block_sum_ind = np.argsort(block_sum)
            block_sum_ind_flag = np.zeros(num_block)
            
            # Stage 2: Find minimal set of frequencies to drop
            if self.progressive_scan:
                # Custom scan ranges based on dataset
                if self.square_max_num == 32:  # CIFAR-10
                    ranges = [1, 2] + list(range(3, num_block + 1))
                else:  # ImageNet
                    ranges = [60, 80, 90] + list(range(91, num_block + 1))
                    
                freq_modified = freq.copy()
                freq_modified_m = np.abs(freq_modified)
                freq_modified_p = np.angle(freq_modified)
                
                success = False
                mag_start = 0
                
                for mag in ranges:
                    # Drop frequencies in batches
                    for j in range(mag_start, mag):
                        ind = block_sum_ind[j]
                        freq_modified_m = square_zero(freq_modified_m, ind, self.square_max_num)
                        
                    # Reconstruct complex frequency representation
                    freq_modified = freq_modified_m * np.exp(1j * freq_modified_p)
                    
                    # Convert back to image
                    img_adv = ifft2d(freq_modified)
                    
                    # Check if attack succeeded
                    img_adv_tensor = torch.tensor(
                        img_adv.transpose(2, 0, 1) / 255.0, 
                        dtype=torch.float32
                    ).unsqueeze(0)
                    img_adv_tensor = self.preprocess_image(img_adv_tensor.to(self.device))
                    
                    self.query_count += 1
                    metadata['queries'][i] += 1
                    
                    with torch.no_grad():
                        output = self.model(img_adv_tensor)
                        _, adv_label = torch.max(output, dim=1)
                    
                    # Check for success based on attack type
                    if self.targeted:
                        attack_success = (adv_label.item() == target_label)
                    else:
                        attack_success = (adv_label.item() != orig_label.item())
                        
                    if attack_success:
                        # Calculate perturbation norm
                        l2_norm = torch.norm(
                            self.preprocess_image(images[i:i+1].to(self.device)) - 
                            img_adv_tensor,
                            p=2
                        ).item()
                        
                        # If within bounds, we're done
                        if l2_norm < self.norm_bound:
                            adv_images[i] = torch.tensor(
                                img_adv.transpose(2, 0, 1) / 255.0, 
                                dtype=torch.float32
                            )
                            metadata['success'][i] = True
                            metadata['perturbation_norm'][i] = l2_norm
                            success = True
                            break
                    
                    mag_start = mag
                    
                    # Check if we've exceeded query budget
                    if metadata['queries'][i] >= self.max_queries:
                        break
                        
                if success or metadata['queries'][i] >= self.max_queries:
                    continue
            
                # Stage 3: Optimize by recovering unnecessary frequency components
                max_i = mag_start - 1
                block_sum_ind_flag[:max_i+1] = 1
                freq_m = freq_modified_m.copy()
                freq_p = np.angle(freq_ori)  # Use original phase
                
                img_temp = img_adv.copy()
                
                # Stage 4: Iterate through recovery rounds
                for _ in range(self.freq_recovery_rounds):
                    for j in range(max_i, -1, -1):
                        if block_sum_ind_flag[j] == 1:
                            ind = block_sum_ind[j]
                            
                            # Try restoring this frequency component
                            freq_m_test = freq_m.copy()
                            freq_m_test = square_recover(freq_m_test, freq_ori_m, ind, self.square_max_num)
                            freq_test = freq_m_test * np.exp(1j * freq_p)
                            
                            # Convert back to image
                            img_adv_test = ifft2d(freq_test)
                            
                            # Test if still adversarial
                            img_adv_tensor = torch.tensor(
                                img_adv_test.transpose(2, 0, 1) / 255.0, 
                                dtype=torch.float32
                            ).unsqueeze(0)
                            img_adv_tensor = self.preprocess_image(img_adv_tensor.to(self.device))
                            
                            self.query_count += 1
                            metadata['queries'][i] += 1
                            
                            with torch.no_grad():
                                output = self.model(img_adv_tensor)
                                _, adv_label = torch.max(output, dim=1)
                                
                            if self.targeted:
                                still_adversarial = (adv_label.item() == target_label)
                            else:
                                still_adversarial = (adv_label.item() != orig_label.item())
                                
                            if still_adversarial:
                                # Keep the restored frequency
                                freq_m = freq_m_test
                                img_temp = img_adv_test.copy()
                                block_sum_ind_flag[j] = 0
                            else:
                                # Revert to zeroed frequency
                                pass
                                
                            # Check if we've exceeded query budget
                            if metadata['queries'][i] >= self.max_queries:
                                break
                                
                    if metadata['queries'][i] >= self.max_queries:
                        break
                
                # Final adversarial example
                l2_norm = torch.norm(
                    self.preprocess_image(images[i:i+1].to(self.device)) - 
                    self.preprocess_image(torch.tensor(
                        img_temp.transpose(2, 0, 1) / 255.0, 
                        dtype=torch.float32
                    ).unsqueeze(0).to(self.device)),
                    p=2
                ).item()
                
                adv_images[i] = torch.tensor(
                    img_temp.transpose(2, 0, 1) / 255.0, 
                    dtype=torch.float32
                )
                metadata['success'][i] = True
                metadata['perturbation_norm'][i] = l2_norm
        
        # Calculate overall attack statistics
        metadata['overall_success_rate'] = sum(metadata['success']) / batch_size
        metadata['avg_queries'] = sum(metadata['queries']) / batch_size
        metadata['avg_perturbation'] = sum(metadata['perturbation_norm']) / max(1, sum(metadata['success']))
        metadata['total_queries'] = self.query_count
        
        return adv_images, metadata