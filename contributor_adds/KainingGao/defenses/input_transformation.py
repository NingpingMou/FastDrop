"""Input transformation defenses against adversarial attacks."""
#-Input transformation defenses against adversarial attacks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Union, Callable

from .base_defense import BaseDefense


class JPEG_Compression(BaseDefense):
    """Defense based on JPEG compression.
    
    JPEG compression can remove subtle perturbations added by adversarial attacks
    while preserving most of the semantic content.
    """
    
    def __init__(
        self,
        quality: int = 75,
        random_quality: bool = False,
        quality_range: Tuple[int, int] = (70, 90),
        device: torch.device = None
    ):
        """Initialize JPEG compression defense.
        
        Args:
            quality: JPEG quality level (0-100)
            random_quality: Whether to use random quality
            quality_range: Range of random quality values
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.quality = quality
        self.random_quality = random_quality
        self.quality_range = quality_range
        
        # Import PIL only when needed to avoid dependency issues
        try:
            from PIL import Image
            import io
            self.Image = Image
            self.io = io
        except ImportError:
            raise ImportError("JPEG_Compression defense requires PIL to be installed.")
            
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply JPEG compression to input images.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Compressed images
        """
        batch_size = images.shape[0]
        defended_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get current image and convert to numpy
            img = images[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Scale to [0, 255]
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            
            # Convert to PIL image
            pil_img = self.Image.fromarray(img)
            
            # Determine quality
            if self.random_quality:
                quality = np.random.randint(self.quality_range[0], self.quality_range[1])
            else:
                quality = self.quality
                
            # Apply JPEG compression
            buffer = self.io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_img = self.Image.open(buffer)
            
            # Convert back to numpy
            compressed_img = np.array(compressed_img).astype(np.float32) / 255.0
            
            # Convert back to tensor
            tensor_img = torch.tensor(compressed_img, device=images.device).permute(2, 0, 1)
            defended_batch.append(tensor_img)
            
        # Stack back into a batch
        return torch.stack(defended_batch)


class BitDepthReduction(BaseDefense):
    """Defense based on reducing bit depth of images.
    
    Reducing bit depth quantizes pixel values, which can remove
    fine-grained adversarial perturbations.
    """
    
    def __init__(
        self,
        bits: int = 5,
        random_bits: bool = False,
        bits_range: Tuple[int, int] = (3, 6),
        device: torch.device = None
    ):
        """Initialize bit depth reduction defense.
        
        Args:
            bits: Number of bits to keep (1-8)
            random_bits: Whether to use random bit depth
            bits_range: Range of random bit depths
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.bits = bits
        self.random_bits = random_bits
        self.bits_range = bits_range
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply bit depth reduction to input images.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Images with reduced bit depth
        """
        # Determine bit depth
        if self.random_bits:
            bits = np.random.randint(self.bits_range[0], self.bits_range[1] + 1)
        else:
            bits = self.bits
            
        # Calculate number of bins
        bins = 2 ** bits
        
        # Scale to [0, bins-1]
        scaled = images * (bins - 1)
        
        # Quantize and scale back
        quantized = torch.round(scaled) / (bins - 1)
        
        return quantized


class SpatialSmoothing(BaseDefense):
    """Defense based on spatial smoothing.
    
    Spatial smoothing methods like Gaussian blur can reduce the impact of
    adversarial perturbations by averaging neighboring pixels.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1.0,
        random_sigma: bool = False,
        sigma_range: Tuple[float, float] = (0.5, 2.0),
        device: torch.device = None
    ):
        """Initialize spatial smoothing defense.
        
        Args:
            kernel_size: Size of the Gaussian kernel
            sigma: Gaussian kernel standard deviation
            random_sigma: Whether to use random sigma
            sigma_range: Range of random sigma values
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.sigma_range = sigma_range
        
    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.
        
        Args:
            size: Kernel size
            sigma: Gaussian standard deviation
            
        Returns:
            2D Gaussian kernel
        """
        # Create 1D Gaussian kernel
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32, device=self.device)
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        
        return kernel_2d
    
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Smoothed images
        """
        # Determine sigma
        if self.random_sigma:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        else:
            sigma = self.sigma
            
        # Generate Gaussian kernel
        kernel = self.gaussian_kernel(self.kernel_size, sigma)
        
        # Reshape kernel for depthwise convolution
        batch_size, channels = images.shape[:2]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        # Apply padding
        padding = self.kernel_size // 2
        padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution separately for each channel to avoid cross-channel mixing
        smoothed = F.conv2d(
            padded.view(batch_size * channels, 1, *padded.shape[2:]),
            kernel,
            padding=0,
            groups=channels
        )
        
        # Reshape back to [B, C, H, W]
        smoothed = smoothed.view(batch_size, channels, *images.shape[2:])
        
        return smoothed


class MedianFilter(BaseDefense):
    """Defense based on median filtering.
    
    Median filtering replaces each pixel with the median value of its neighborhood,
    which can effectively remove salt-and-pepper noise and sparse perturbations.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        random_size: bool = False,
        size_range: Tuple[int, int] = (3, 5),
        device: torch.device = None
    ):
        """Initialize median filter defense.
        
        Args:
            kernel_size: Size of the median filter kernel (odd number)
            random_size: Whether to use random kernel size
            size_range: Range of random kernel sizes (odd numbers)
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.random_size = random_size
        self.size_range = (
            size_range[0] if size_range[0] % 2 == 1 else size_range[0] + 1,
            size_range[1] if size_range[1] % 2 == 1 else size_range[1] + 1
        )
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply median filtering to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Median filtered images
        """
        # Determine kernel size
        if self.random_size:
            sizes = list(range(self.size_range[0], self.size_range[1] + 1, 2))
            kernel_size = np.random.choice(sizes)
        else:
            kernel_size = self.kernel_size
            
        # Convert to CPU for median filtering (faster)
        cpu_images = images.detach().cpu()
        batch_size, channels = cpu_images.shape[:2]
        filtered_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            filtered_channels = []
            
            # Process each channel separately
            for c in range(channels):
                channel = cpu_images[i, c]
                
                # Apply median filtering
                # Use PyTorch's implementation if available, otherwise use numpy
                try:
                    filtered = transforms.functional.medianblur(
                        channel.unsqueeze(0).unsqueeze(0),
                        kernel_size=kernel_size
                    ).squeeze()
                except (AttributeError, NotImplementedError):
                    # Fallback to numpy implementation
                    import scipy.ndimage as ndimage
                    filtered = ndimage.median_filter(
                        channel.numpy(),
                        size=kernel_size
                    )
                    filtered = torch.tensor(filtered)
                    
                filtered_channels.append(filtered)
                
            # Stack channels back together
            filtered_img = torch.stack(filtered_channels)
            filtered_batch.append(filtered_img)
            
        # Stack back into a batch and move to device
        return torch.stack(filtered_batch).to(images.device)


class RandomResizePadding(BaseDefense):
    """Defense based on random resizing and padding.
    
    Randomly resizes the image and adds padding, which disrupts
    the spatial structure of adversarial perturbations.
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        padding_range: Tuple[int, int] = (0, 10),
        device: torch.device = None
    ):
        """Initialize random resize and padding defense.
        
        Args:
            scale_range: Range of random scaling factors
            padding_range: Range of random padding amounts
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.scale_range = scale_range
        self.padding_range = padding_range
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random resizing and padding to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Transformed images
        """
        batch_size, channels, height, width = images.shape
        defended_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            img = images[i]
            
            # Random scaling factor
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            
            # New dimensions
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Resize image
            resized = F.interpolate(
                img.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Random padding amounts
            pad_top = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_bottom = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_left = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_right = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            
            # Apply padding
            padded = F.pad(
                resized,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0
            )
            
            # Resize back to original dimensions
            transformed = F.interpolate(
                padded.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            defended_batch.append(transformed)
            
        # Stack back into a batch
        return torch.stack(defended_batch)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Union, Callable

from .base_defense import BaseDefense


class JPEG_Compression(BaseDefense):
    """Defense based on JPEG compression.
    
    JPEG compression can remove subtle perturbations added by adversarial attacks
    while preserving most of the semantic content.
    """
    
    def __init__(
        self,
        quality: int = 75,
        random_quality: bool = False,
        quality_range: Tuple[int, int] = (70, 90),
        device: torch.device = None
    ):
        """Initialize JPEG compression defense.
        
        Args:
            quality: JPEG quality level (0-100)
            random_quality: Whether to use random quality
            quality_range: Range of random quality values
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.quality = quality
        self.random_quality = random_quality
        self.quality_range = quality_range
        
        # Import PIL only when needed to avoid dependency issues
        try:
            from PIL import Image
            import io
            self.Image = Image
            self.io = io
        except ImportError:
            raise ImportError("JPEG_Compression defense requires PIL to be installed.")
            
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply JPEG compression to input images.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Compressed images
        """
        batch_size = images.shape[0]
        defended_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            # Get current image and convert to numpy
            img = images[i].detach().cpu().permute(1, 2, 0).numpy()
            
            # Scale to [0, 255]
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            
            # Convert to PIL image
            pil_img = self.Image.fromarray(img)
            
            # Determine quality
            if self.random_quality:
                quality = np.random.randint(self.quality_range[0], self.quality_range[1])
            else:
                quality = self.quality
                
            # Apply JPEG compression
            buffer = self.io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_img = self.Image.open(buffer)
            
            # Convert back to numpy
            compressed_img = np.array(compressed_img).astype(np.float32) / 255.0
            
            # Convert back to tensor
            tensor_img = torch.tensor(compressed_img, device=images.device).permute(2, 0, 1)
            defended_batch.append(tensor_img)
            
        # Stack back into a batch
        return torch.stack(defended_batch)


class BitDepthReduction(BaseDefense):
    """Defense based on reducing bit depth of images.
    
    Reducing bit depth quantizes pixel values, which can remove
    fine-grained adversarial perturbations.
    """
    
    def __init__(
        self,
        bits: int = 5,
        random_bits: bool = False,
        bits_range: Tuple[int, int] = (3, 6),
        device: torch.device = None
    ):
        """Initialize bit depth reduction defense.
        
        Args:
            bits: Number of bits to keep (1-8)
            random_bits: Whether to use random bit depth
            bits_range: Range of random bit depths
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.bits = bits
        self.random_bits = random_bits
        self.bits_range = bits_range
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply bit depth reduction to input images.
        
        Args:
            images: Input images [B, C, H, W] in range [0, 1]
            
        Returns:
            Images with reduced bit depth
        """
        # Determine bit depth
        if self.random_bits:
            bits = np.random.randint(self.bits_range[0], self.bits_range[1] + 1)
        else:
            bits = self.bits
            
        # Calculate number of bins
        bins = 2 ** bits
        
        # Scale to [0, bins-1]
        scaled = images * (bins - 1)
        
        # Quantize and scale back
        quantized = torch.round(scaled) / (bins - 1)
        
        return quantized


class SpatialSmoothing(BaseDefense):
    """Defense based on spatial smoothing.
    
    Spatial smoothing methods like Gaussian blur can reduce the impact of
    adversarial perturbations by averaging neighboring pixels.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        sigma: float = 1.0,
        random_sigma: bool = False,
        sigma_range: Tuple[float, float] = (0.5, 2.0),
        device: torch.device = None
    ):
        """Initialize spatial smoothing defense.
        
        Args:
            kernel_size: Size of the Gaussian kernel
            sigma: Gaussian kernel standard deviation
            random_sigma: Whether to use random sigma
            sigma_range: Range of random sigma values
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.random_sigma = random_sigma
        self.sigma_range = sigma_range
        
    def gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel.
        
        Args:
            size: Kernel size
            sigma: Gaussian standard deviation
            
        Returns:
            2D Gaussian kernel
        """
        # Create 1D Gaussian kernel
        x = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32, device=self.device)
        kernel_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D Gaussian kernel
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        
        return kernel_2d
    
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply spatial smoothing to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Smoothed images
        """
        # Determine sigma
        if self.random_sigma:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        else:
            sigma = self.sigma
            
        # Generate Gaussian kernel
        kernel = self.gaussian_kernel(self.kernel_size, sigma)
        
        # Reshape kernel for depthwise convolution
        batch_size, channels = images.shape[:2]
        kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        # Apply padding
        padding = self.kernel_size // 2
        padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution separately for each channel to avoid cross-channel mixing
        smoothed = F.conv2d(
            padded.view(batch_size * channels, 1, *padded.shape[2:]),
            kernel,
            padding=0,
            groups=channels
        )
        
        # Reshape back to [B, C, H, W]
        smoothed = smoothed.view(batch_size, channels, *images.shape[2:])
        
        return smoothed


class MedianFilter(BaseDefense):
    """Defense based on median filtering.
    
    Median filtering replaces each pixel with the median value of its neighborhood,
    which can effectively remove salt-and-pepper noise and sparse perturbations.
    """
    
    def __init__(
        self,
        kernel_size: int = 3,
        random_size: bool = False,
        size_range: Tuple[int, int] = (3, 5),
        device: torch.device = None
    ):
        """Initialize median filter defense.
        
        Args:
            kernel_size: Size of the median filter kernel (odd number)
            random_size: Whether to use random kernel size
            size_range: Range of random kernel sizes (odd numbers)
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.random_size = random_size
        self.size_range = (
            size_range[0] if size_range[0] % 2 == 1 else size_range[0] + 1,
            size_range[1] if size_range[1] % 2 == 1 else size_range[1] + 1
        )
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply median filtering to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Median filtered images
        """
        # Determine kernel size
        if self.random_size:
            sizes = list(range(self.size_range[0], self.size_range[1] + 1, 2))
            kernel_size = np.random.choice(sizes)
        else:
            kernel_size = self.kernel_size
            
        # Convert to CPU for median filtering (faster)
        cpu_images = images.detach().cpu()
        batch_size, channels = cpu_images.shape[:2]
        filtered_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            filtered_channels = []
            
            # Process each channel separately
            for c in range(channels):
                channel = cpu_images[i, c]
                
                # Apply median filtering
                # Use PyTorch's implementation if available, otherwise use numpy
                try:
                    filtered = transforms.functional.medianblur(
                        channel.unsqueeze(0).unsqueeze(0),
                        kernel_size=kernel_size
                    ).squeeze()
                except (AttributeError, NotImplementedError):
                    # Fallback to numpy implementation
                    import scipy.ndimage as ndimage
                    filtered = ndimage.median_filter(
                        channel.numpy(),
                        size=kernel_size
                    )
                    filtered = torch.tensor(filtered)
                    
                filtered_channels.append(filtered)
                
            # Stack channels back together
            filtered_img = torch.stack(filtered_channels)
            filtered_batch.append(filtered_img)
            
        # Stack back into a batch and move to device
        return torch.stack(filtered_batch).to(images.device)


class RandomResizePadding(BaseDefense):
    """Defense based on random resizing and padding.
    
    Randomly resizes the image and adds padding, which disrupts
    the spatial structure of adversarial perturbations.
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        padding_range: Tuple[int, int] = (0, 10),
        device: torch.device = None
    ):
        """Initialize random resize and padding defense.
        
        Args:
            scale_range: Range of random scaling factors
            padding_range: Range of random padding amounts
            device: Device to run the defense on
        """
        super().__init__(device)
        
        self.scale_range = scale_range
        self.padding_range = padding_range
        
    def defend(self, images: torch.Tensor) -> torch.Tensor:
        """Apply random resizing and padding to input images.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Transformed images
        """
        batch_size, channels, height, width = images.shape
        defended_batch = []
        
        # Process each image in the batch
        for i in range(batch_size):
            img = images[i]
            
            # Random scaling factor
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            
            # New dimensions
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Resize image
            resized = F.interpolate(
                img.unsqueeze(0),
                size=(new_height, new_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            # Random padding amounts
            pad_top = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_bottom = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_left = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            pad_right = np.random.randint(self.padding_range[0], self.padding_range[1] + 1)
            
            # Apply padding
            padded = F.pad(
                resized,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='constant',
                value=0
            )
            
            # Resize back to original dimensions
            transformed = F.interpolate(
                padded.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            defended_batch.append(transformed)
            
        # Stack back into a batch
        return torch.stack(defended_batch)