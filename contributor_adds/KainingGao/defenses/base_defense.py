"""Base class for all adversarial defenses."""
#-Base class for all adversarial defenses.
import abc
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseDefense(abc.ABC):
    """Abstract base class for all defense methods.
    
    This provides a common interface for implementing adversarial defenses
    and capturing metrics about defense performance.
    """
    
    def __init__(
        self,
        device: torch.device = None
    ):
        """Initialize defense.
        
        Args:
            device: Device to run the defense on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.defense_time = 0.0
        
    @abc.abstractmethod
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply defense to input images.
        
        Args:
            images: Input images to defend
            
        Returns:
            Defended images
        """
        pass
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Call defend method and record time."""
        start_time = time.time()
        result = self.defend(images)
        self.defense_time = time.time() - start_time
        return result


class ModelWrapper(nn.Module):
    """Wrapper to apply defense preprocessing to model.
    
    This wrapper applies a defense method before passing the input to the model,
    effectively creating a defended model.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        defense: BaseDefense
    ):
        """Initialize wrapped model.
        
        Args:
            model: Base model to wrap
            defense: Defense method to apply
        """
        super().__init__()
        self.model = model
        self.defense = defense
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply defense and then model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        defended_x = self.defense(x)
        return self.model(defended_x)


class EnsembleDefense(BaseDefense):
    """Ensemble of multiple defense methods.
    
    This defense applies multiple defense methods and outputs the average
    or a random selection of the defended images.
    """
    
    def __init__(
        self,
        defenses: List[BaseDefense],
        mode: str = "average",
        device: torch.device = None
    ):
        """Initialize ensemble defense.
        
        Args:
            defenses: List of defense methods to apply
            mode: Combination mode ("average" or "random")
            device: Device to run the defense on
        """
        super().__init__(device)
        self.defenses = defenses
        self.mode = mode
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ensemble defense to input images.
        
        Args:
            images: Input images to defend
            
        Returns:
            Defended images
        """
        if self.mode == "average":
            # Apply all defenses and average the results
            outputs = [defense(images) for defense in self.defenses]
            return torch.stack(outputs).mean(dim=0)
        elif self.mode == "random":
            # Randomly select one defense to apply
            defense_idx = torch.randint(0, len(self.defenses), (1,)).item()
            return self.defenses[defense_idx](images)
        else:
            raise ValueError(f"Unsupported ensemble mode: {self.mode}")


class NullDefense(BaseDefense):
    """Null defense that doesn't modify the input.
    
    This is useful as a baseline for comparison.
    """
    
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Return input images unchanged.
        
        Args:
            images: Input images
            
        Returns:
            Input images unchanged
        """
        return images

import abc
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


class BaseDefense(abc.ABC):
    """Abstract base class for all defense methods.
    
    This provides a common interface for implementing adversarial defenses
    and capturing metrics about defense performance.
    """
    
    def __init__(
        self,
        device: torch.device = None
    ):
        """Initialize defense.
        
        Args:
            device: Device to run the defense on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.defense_time = 0.0
        
    @abc.abstractmethod
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply defense to input images.
        
        Args:
            images: Input images to defend
            
        Returns:
            Defended images
        """
        pass
    
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Call defend method and record time."""
        start_time = time.time()
        result = self.defend(images)
        self.defense_time = time.time() - start_time
        return result


class ModelWrapper(nn.Module):
    """Wrapper to apply defense preprocessing to model.
    
    This wrapper applies a defense method before passing the input to the model,
    effectively creating a defended model.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        defense: BaseDefense
    ):
        """Initialize wrapped model.
        
        Args:
            model: Base model to wrap
            defense: Defense method to apply
        """
        super().__init__()
        self.model = model
        self.defense = defense
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply defense and then model.
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        defended_x = self.defense(x)
        return self.model(defended_x)


class EnsembleDefense(BaseDefense):
    """Ensemble of multiple defense methods.
    
    This defense applies multiple defense methods and outputs the average
    or a random selection of the defended images.
    """
    
    def __init__(
        self,
        defenses: List[BaseDefense],
        mode: str = "average",
        device: torch.device = None
    ):
        """Initialize ensemble defense.
        
        Args:
            defenses: List of defense methods to apply
            mode: Combination mode ("average" or "random")
            device: Device to run the defense on
        """
        super().__init__(device)
        self.defenses = defenses
        self.mode = mode
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply ensemble defense to input images.
        
        Args:
            images: Input images to defend
            
        Returns:
            Defended images
        """
        if self.mode == "average":
            # Apply all defenses and average the results
            outputs = [defense(images) for defense in self.defenses]
            return torch.stack(outputs).mean(dim=0)
        elif self.mode == "random":
            # Randomly select one defense to apply
            defense_idx = torch.randint(0, len(self.defenses), (1,)).item()
            return self.defenses[defense_idx](images)
        else:
            raise ValueError(f"Unsupported ensemble mode: {self.mode}")


class NullDefense(BaseDefense):
    """Null defense that doesn't modify the input.
    
    This is useful as a baseline for comparison.
    """
    
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Return input images unchanged.
        
        Args:
            images: Input images
            
        Returns:
            Input images unchanged
        """
        return images