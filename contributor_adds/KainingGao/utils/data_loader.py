"""Data loading utilities for various datasets."""
#-Data loading utilities for various datasets.
import os
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_dataset(
    dataset_name: str,
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False,
    download: bool = True
) -> torch.utils.data.DataLoader:
    """Load a dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads test set)
        download: Whether to download the dataset if not found
        
    Returns:
        DataLoader for the specified dataset
    """
    if dataset_name.lower() == "cifar10":
        return load_cifar10(data_dir, batch_size, num_workers, train, download)
    elif dataset_name.lower() == "imagenet":
        return load_imagenet(data_dir, batch_size, num_workers, train)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_cifar10(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False,
    download: bool = True
) -> torch.utils.data.DataLoader:
    """Load the CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load CIFAR-10
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads test set)
        download: Whether to download the dataset if not found
        
    Returns:
        DataLoader for CIFAR-10
    """
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Create dataset
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def load_imagenet(
    data_dir: str = "data/imagenet",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False
) -> torch.utils.data.DataLoader:
    """Load the ImageNet dataset.
    
    Args:
        data_dir: Directory containing ImageNet
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads validation set)
        
    Returns:
        DataLoader for ImageNet
    """
    # ImageNet cannot be automatically downloaded
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"ImageNet directory {data_dir} not found. "
            "Please download ImageNet manually and provide the correct path."
        )
    
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        imagenet_dir = os.path.join(data_dir, 'train')
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        imagenet_dir = os.path.join(data_dir, 'val')
    
    # Create dataset
    dataset = datasets.ImageFolder(
        root=imagenet_dir,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def load_custom_dataset(
    data_dir: str,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """Load a custom image dataset from a directory.
    
    Args:
        data_dir: Directory containing the dataset
        transform: Transformations to apply to the images
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the custom dataset
    """
    # Create dataset
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_normalization_params(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Get normalization parameters for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (mean, std) normalization parameters
    """
    if dataset_name.lower() == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset_name.lower() == "imagenet":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing dataset information
    """
    if dataset_name.lower() == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "resolution": (32, 32),
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "class_names": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        }
    elif dataset_name.lower() == "imagenet":
        return {
            "name": "ImageNet",
            "num_classes": 1000,
            "resolution": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            # Class names omitted for brevity
            "class_names": list(range(1000))
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

import os
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_dataset(
    dataset_name: str,
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False,
    download: bool = True
) -> torch.utils.data.DataLoader:
    """Load a dataset.
    
    Args:
        dataset_name: Name of the dataset to load
        data_dir: Directory to store/load the dataset
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads test set)
        download: Whether to download the dataset if not found
        
    Returns:
        DataLoader for the specified dataset
    """
    if dataset_name.lower() == "cifar10":
        return load_cifar10(data_dir, batch_size, num_workers, train, download)
    elif dataset_name.lower() == "imagenet":
        return load_imagenet(data_dir, batch_size, num_workers, train)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_cifar10(
    data_dir: str = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False,
    download: bool = True
) -> torch.utils.data.DataLoader:
    """Load the CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load CIFAR-10
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads test set)
        download: Whether to download the dataset if not found
        
    Returns:
        DataLoader for CIFAR-10
    """
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Create dataset
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=download,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def load_imagenet(
    data_dir: str = "data/imagenet",
    batch_size: int = 32,
    num_workers: int = 4,
    train: bool = False
) -> torch.utils.data.DataLoader:
    """Load the ImageNet dataset.
    
    Args:
        data_dir: Directory containing ImageNet
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        train: Whether to load the training set (otherwise loads validation set)
        
    Returns:
        DataLoader for ImageNet
    """
    # ImageNet cannot be automatically downloaded
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"ImageNet directory {data_dir} not found. "
            "Please download ImageNet manually and provide the correct path."
        )
    
    # Define transformations
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        imagenet_dir = os.path.join(data_dir, 'train')
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        imagenet_dir = os.path.join(data_dir, 'val')
    
    # Create dataset
    dataset = datasets.ImageFolder(
        root=imagenet_dir,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def load_custom_dataset(
    data_dir: str,
    transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False
) -> torch.utils.data.DataLoader:
    """Load a custom image dataset from a directory.
    
    Args:
        data_dir: Directory containing the dataset
        transform: Transformations to apply to the images
        batch_size: Batch size for the data loader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the custom dataset
    """
    # Create dataset
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_normalization_params(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Get normalization parameters for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (mean, std) normalization parameters
    """
    if dataset_name.lower() == "cifar10":
        return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif dataset_name.lower() == "imagenet":
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary containing dataset information
    """
    if dataset_name.lower() == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "resolution": (32, 32),
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "class_names": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        }
    elif dataset_name.lower() == "imagenet":
        return {
            "name": "ImageNet",
            "num_classes": 1000,
            "resolution": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            # Class names omitted for brevity
            "class_names": list(range(1000))
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")