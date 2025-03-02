"""Utilities for loading various model architectures."""
#-Utilities for loading various model architectures.
import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Union

from .resnet import ResNet18


def load_model(
    model_name: str,
    dataset: str,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """Load a model architecture.
    
    Args:
        model_name: Name of the model architecture
        dataset: Name of the dataset the model is trained on
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to custom checkpoint file
        
    Returns:
        Loaded model
    """
    # Convert names to lowercase
    model_name = model_name.lower()
    dataset = dataset.lower()
    
    # CIFAR-10 models
    if dataset == "cifar10":
        num_classes = 10
        
        if model_name == "resnet18":
            model = ResNet18()
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "net" in state_dict:
                    state_dict = state_dict["net"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            # Otherwise use a pre-trained checkpoint
            elif pretrained:
                # Check if default checkpoint exists
                default_path = "ResNet10-CIFAR10.pth"
                if os.path.exists(default_path):
                    state_dict = torch.load(default_path, map_location="cpu")
                    
                    # Handle different checkpoint formats
                    if isinstance(state_dict, dict) and "net" in state_dict:
                        state_dict = state_dict["net"]
                    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                        
                    model.load_state_dict(state_dict)
                else:
                    print(f"Warning: Pretrained checkpoint {default_path} not found.")
                    print("Using randomly initialized weights.")
            
            return model
            
        # Add more CIFAR-10 models as needed
        
    # ImageNet models
    elif dataset == "imagenet":
        if model_name == "resnet50":
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load custom checkpoint
                model = models.resnet50(pretrained=False)
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            else:
                # Use torchvision's pretrained model
                model = models.resnet50(pretrained=pretrained)
                
            return model
            
        elif model_name == "resnet18":
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load custom checkpoint
                model = models.resnet18(pretrained=False)
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            else:
                # Use torchvision's pretrained model
                model = models.resnet18(pretrained=pretrained)
                
            return model
            
        # Add more ImageNet models as needed
        
    # Handle unsupported model/dataset combinations
    raise ValueError(f"Unsupported model {model_name} for dataset {dataset}")


def get_model_info(model: nn.Module) -> Dict:
    """Get information about a model.
    
    Args:
        model: Model to get information about
        
    Returns:
        Dictionary containing model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model name
    model_name = model.__class__.__name__
    
    return {
        "name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }


def create_ensemble(models: list, voting: str = "soft") -> nn.Module:
    """Create an ensemble of models.
    
    Args:
        models: List of models to ensemble
        voting: Voting method ("soft" or "hard")
        
    Returns:
        Ensemble model
    """
    class ModelEnsemble(nn.Module):
        def __init__(self, models, voting):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.voting = voting
            
        def forward(self, x):
            if self.voting == "soft":
                # Average the probabilities
                outputs = [nn.functional.softmax(model(x), dim=1) for model in self.models]
                return torch.stack(outputs).mean(dim=0)
            else:  # Hard voting
                # Count the votes for each class
                outputs = [model(x).argmax(dim=1) for model in self.models]
                votes = torch.stack(outputs, dim=0)
                return torch.zeros(x.size(0), self.models[0].fc.out_features, device=x.device).scatter_add(
                    1, votes.T.unsqueeze(1).expand(-1, 1, -1), 1
                )
                
    return ModelEnsemble(models, voting)

import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Union

from .resnet import ResNet18


def load_model(
    model_name: str,
    dataset: str,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None
) -> nn.Module:
    """Load a model architecture.
    
    Args:
        model_name: Name of the model architecture
        dataset: Name of the dataset the model is trained on
        pretrained: Whether to load pretrained weights
        checkpoint_path: Path to custom checkpoint file
        
    Returns:
        Loaded model
    """
    # Convert names to lowercase
    model_name = model_name.lower()
    dataset = dataset.lower()
    
    # CIFAR-10 models
    if dataset == "cifar10":
        num_classes = 10
        
        if model_name == "resnet18":
            model = ResNet18()
            
            # Load checkpoint if provided
            if checkpoint_path and os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "net" in state_dict:
                    state_dict = state_dict["net"]
                elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            # Otherwise use a pre-trained checkpoint
            elif pretrained:
                # Check if default checkpoint exists
                default_path = "ResNet10-CIFAR10.pth"
                if os.path.exists(default_path):
                    state_dict = torch.load(default_path, map_location="cpu")
                    
                    # Handle different checkpoint formats
                    if isinstance(state_dict, dict) and "net" in state_dict:
                        state_dict = state_dict["net"]
                    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                        
                    model.load_state_dict(state_dict)
                else:
                    print(f"Warning: Pretrained checkpoint {default_path} not found.")
                    print("Using randomly initialized weights.")
            
            return model
            
        # Add more CIFAR-10 models as needed
        
    # ImageNet models
    elif dataset == "imagenet":
        if model_name == "resnet50":
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load custom checkpoint
                model = models.resnet50(pretrained=False)
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            else:
                # Use torchvision's pretrained model
                model = models.resnet50(pretrained=pretrained)
                
            return model
            
        elif model_name == "resnet18":
            if checkpoint_path and os.path.exists(checkpoint_path):
                # Load custom checkpoint
                model = models.resnet18(pretrained=False)
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                
                # Handle different checkpoint formats
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                    
                model.load_state_dict(state_dict)
            else:
                # Use torchvision's pretrained model
                model = models.resnet18(pretrained=pretrained)
                
            return model
            
        # Add more ImageNet models as needed
        
    # Handle unsupported model/dataset combinations
    raise ValueError(f"Unsupported model {model_name} for dataset {dataset}")


def get_model_info(model: nn.Module) -> Dict:
    """Get information about a model.
    
    Args:
        model: Model to get information about
        
    Returns:
        Dictionary containing model information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get model name
    model_name = model.__class__.__name__
    
    return {
        "name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
    }


def create_ensemble(models: list, voting: str = "soft") -> nn.Module:
    """Create an ensemble of models.
    
    Args:
        models: List of models to ensemble
        voting: Voting method ("soft" or "hard")
        
    Returns:
        Ensemble model
    """
    class ModelEnsemble(nn.Module):
        def __init__(self, models, voting):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.voting = voting
            
        def forward(self, x):
            if self.voting == "soft":
                # Average the probabilities
                outputs = [nn.functional.softmax(model(x), dim=1) for model in self.models]
                return torch.stack(outputs).mean(dim=0)
            else:  # Hard voting
                # Count the votes for each class
                outputs = [model(x).argmax(dim=1) for model in self.models]
                votes = torch.stack(outputs, dim=0)
                return torch.zeros(x.size(0), self.models[0].fc.out_features, device=x.device).scatter_add(
                    1, votes.T.unsqueeze(1).expand(-1, 1, -1), 1
                )
                
    return ModelEnsemble(models, voting)