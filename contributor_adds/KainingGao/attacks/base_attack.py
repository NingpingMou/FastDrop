"""Base class for all adversarial attacks."""
#-Base class for all adversarial attacks
import abc
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


class BaseAttack(abc.ABC):
    """Abstract base class for all attack methods.
    
    This provides a common interface for implementing adversarial attacks
    and capturing metrics about the attack performance.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        device: torch.device = None,
        norm_type: str = 'l2',
        norm_bound: float = 5.0,
        targeted: bool = False,
        verbose: bool = False
    ):
        """Initialize attack.
        
        Args:
            model: Target model to attack
            device: Device to run the attack on
            norm_type: Type of norm to constrain perturbation ('l2' or 'linf')
            norm_bound: Maximum perturbation size
            targeted: Whether the attack is targeted (aims for specific class)
            verbose: Whether to print progress information
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_type = norm_type
        self.norm_bound = norm_bound
        self.targeted = targeted
        self.verbose = verbose
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Attack statistics
        self.query_count = 0
        self.attack_time = 0.0
        self.success_rate = 0.0
        self.avg_perturbation_size = 0.0
        
    def reset_stats(self):
        """Reset attack statistics."""
        self.query_count = 0
        self.attack_time = 0.0
        
    def compute_perturbation_norm(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> float:
        """Calculate the perturbation norm between original and perturbed images.
        
        Args:
            original: Original image
            perturbed: Perturbed image
            
        Returns:
            Norm of the perturbation
        """
        if self.norm_type == 'l2':
            return torch.norm(original - perturbed, p=2).item()
        elif self.norm_type == 'linf':
            return torch.norm(original - perturbed, p=float('inf')).item()
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
    
    @abc.abstractmethod
    def attack(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate adversarial examples.
        
        Args:
            images: Clean images to perturb
            labels: True labels of the images
            target_labels: Target labels for targeted attack
            
        Returns:
            Tuple containing adversarial examples and attack metadata
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Call attack method and record statistics."""
        start_time = time.time()
        self.reset_stats()
        result = self.attack(*args, **kwargs)
        self.attack_time = time.time() - start_time
        return result

import abc
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


class BaseAttack(abc.ABC):
    """Abstract base class for all attack methods.
    
    This provides a common interface for implementing adversarial attacks
    and capturing metrics about the attack performance.
    """
    
    def __init__(
        self, 
        model: nn.Module,
        device: torch.device = None,
        norm_type: str = 'l2',
        norm_bound: float = 5.0,
        targeted: bool = False,
        verbose: bool = False
    ):
        """Initialize attack.
        
        Args:
            model: Target model to attack
            device: Device to run the attack on
            norm_type: Type of norm to constrain perturbation ('l2' or 'linf')
            norm_bound: Maximum perturbation size
            targeted: Whether the attack is targeted (aims for specific class)
            verbose: Whether to print progress information
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.norm_type = norm_type
        self.norm_bound = norm_bound
        self.targeted = targeted
        self.verbose = verbose
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Attack statistics
        self.query_count = 0
        self.attack_time = 0.0
        self.success_rate = 0.0
        self.avg_perturbation_size = 0.0
        
    def reset_stats(self):
        """Reset attack statistics."""
        self.query_count = 0
        self.attack_time = 0.0
        
    def compute_perturbation_norm(
        self, 
        original: torch.Tensor, 
        perturbed: torch.Tensor
    ) -> float:
        """Calculate the perturbation norm between original and perturbed images.
        
        Args:
            original: Original image
            perturbed: Perturbed image
            
        Returns:
            Norm of the perturbation
        """
        if self.norm_type == 'l2':
            return torch.norm(original - perturbed, p=2).item()
        elif self.norm_type == 'linf':
            return torch.norm(original - perturbed, p=float('inf')).item()
        else:
            raise ValueError(f"Unsupported norm type: {self.norm_type}")
    
    @abc.abstractmethod
    def attack(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate adversarial examples.
        
        Args:
            images: Clean images to perturb
            labels: True labels of the images
            target_labels: Target labels for targeted attack
            
        Returns:
            Tuple containing adversarial examples and attack metadata
        """
        pass
    
    def __call__(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Call attack method and record statistics."""
        start_time = time.time()
        self.reset_stats()
        result = self.attack(*args, **kwargs)
        self.attack_time = time.time() - start_time
        return result