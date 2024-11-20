"""Real-world attack examples against commercial API services.
#-Real-world attack examples against commercial API services.
This script demonstrates how adversarial attacks like FastDrop can be used
to test the robustness of real-world computer vision APIs.

Note: To use this script, you need to obtain API keys for commercial vision services
and customize the API wrapper functions accordingly.
"""

import os
import argparse
import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

from ..attacks.fast_drop import FastDrop
from ..utils.frequency_utils import fft2d, ifft2d
from ..models.model_loader import load_model


class APIWrapper:
    """Base wrapper for vision API services."""
    
    def __init__(self, api_key: str):
        """Initialize API wrapper.
        
        Args:
            api_key: API key for the service
        """
        self.api_key = api_key
        
    def predict(self, image: np.ndarray) -> Dict:
        """Get predictions from the API.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Dictionary containing prediction results
        """
        raise NotImplementedError("Subclasses must implement predict method")
        
    def preprocess_image(self, image: np.ndarray) -> bytes:
        """Preprocess image for API submission.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Image bytes ready for API submission
        """
        img = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


class DummyAPIWrapper(APIWrapper):
    """Dummy API wrapper that uses a local model instead of an actual API.
    
    This is useful for testing without actual API access.
    """
    
    def __init__(self, model_name: str = "resnet18"):
        """Initialize dummy API wrapper.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__("dummy_key")
        
        # Load ImageNet class names
        with open("imagenet_classes.txt", "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        # Load a local model
        self.model = load_model(model_name, "imagenet")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image: np.ndarray) -> Dict:
        """Get predictions from the local model.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Dictionary containing prediction results
        """
        # Convert to PIL image
        img = Image.fromarray(image.astype(np.uint8))
        
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        # Format results
        results = {
            "predictions": [
                {
                    "class_id": idx.item(),
                    "class_name": self.class_names[idx.item()],
                    "confidence": prob.item()
                }
                for idx, prob in zip(top5_indices[0], top5_probs[0])
            ]
        }
        
        return results


class FastDropAPIAttack:
    """Attack commercial API services using FastDrop."""
    
    def __init__(
        self,
        api_wrapper: APIWrapper,
        square_max_num: int = 224,
        norm_bound: float = 20.0,
        max_queries: int = 50,
        save_dir: str = "api_attacks"
    ):
        """Initialize FastDrop API attack.
        
        Args:
            api_wrapper: Wrapper for the API service
            square_max_num: Size of frequency grid (224 for ImageNet-sized images)
            norm_bound: Maximum perturbation norm
            max_queries: Maximum number of API queries
            save_dir: Directory to save results
        """
        self.api = api_wrapper
        self.square_max_num = square_max_num
        self.norm_bound = norm_bound
        self.max_queries = max_queries
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def attack_image(
        self,
        image: np.ndarray,
        target_label: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Attack a single image.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            target_label: Target label for targeted attack (if None, performs untargeted attack)
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (adversarial image, attack metadata)
        """
        # Get original prediction
        if verbose:
            print("Getting original prediction...")
            
        orig_results = self.api.predict(image)
        orig_label = orig_results["predictions"][0]["class_name"]
        orig_confidence = orig_results["predictions"][0]["confidence"]
        
        if verbose:
            print(f"Original prediction: {orig_label} ({orig_confidence:.3f})")
            
        # Initialize attack parameters
        query_count = 1  # Count the initial prediction
        adv_image = image.copy()
        success = False
        
        # Compute FFT of the image
        freq = fft2d(image)
        freq_mag = np.abs(freq)
        freq_phase = np.angle(freq)
        
        # Calculate average magnitude for each square
        num_block = self.square_max_num // 2
        block_sum = np.zeros(num_block)
        
        for i in range(num_block):
            # Calculate sum along the square pattern
            rank1 = np.sum(freq_mag[i, i:self.square_max_num-i, :])
            rank2 = np.sum(freq_mag[self.square_max_num-1-i, i:self.square_max_num-i, :])
            col1 = np.sum(freq_mag[i+1:self.square_max_num-1-i, i, :])
            col2 = np.sum(freq_mag[i+1:self.square_max_num-1-i, self.square_max_num-1-i, :])
            
            # Calculate total elements in the square
            num_elements = 4 * (self.square_max_num - 2 * i) - 4
            
            # Calculate average
            block_sum[i] = (rank1 + rank2 + col1 + col2) / max(1, num_elements)
            
        # Sort blocks by importance
        block_indices = np.argsort(block_sum)
        block_mask = np.ones(num_block, dtype=bool)
        
        # Stage 1: Find minimal frequency components to remove
        if verbose:
            print("Stage 1: Finding minimal frequency components to remove...")
            
        # Start with larger steps for efficiency
        steps = [
            list(range(5, 101, 5)),     # 5, 10, 15, ..., 100
            list(range(101, 201, 10))   # 110, 120, ..., 200
        ]
        steps = [item for sublist in steps for item in sublist]  # Flatten
        
        current_idx = 0
        for step in steps:
            if current_idx >= step:
                continue
                
            # Zero out frequency components up to the current step
            freq_modified = freq.copy()
            freq_mag_modified = np.abs(freq_modified)
            
            for i in range(current_idx, min(step, num_block)):
                idx = block_indices[i]
                
                # Zero out the square pattern
                freq_mag_modified[idx, idx:self.square_max_num-idx, :] = 0
                freq_mag_modified[self.square_max_num-1-idx, idx:self.square_max_num-idx, :] = 0
                freq_mag_modified[idx:self.square_max_num-idx, idx, :] = 0
                freq_mag_modified[idx:self.square_max_num-idx, self.square_max_num-1-idx, :] = 0
                
            # Reconstruct frequency representation with modified magnitude
            freq_modified = freq_mag_modified * np.exp(1j * freq_phase)
            
            # Convert back to image
            candidate = ifft2d(freq_modified)
            
            # Check if perturbation is within bounds
            perturbation = candidate - image
            l2_norm = np.sqrt(np.sum(perturbation ** 2))
            
            if l2_norm > self.norm_bound:
                if verbose:
                    print(f"  Step {step}: L2 norm {l2_norm:.2f} exceeds bound, skipping")
                continue
                
            # Get prediction on candidate image
            candidate_results = self.api.predict(candidate)
            query_count += 1
            
            candidate_label = candidate_results["predictions"][0]["class_name"]
            candidate_confidence = candidate_results["predictions"][0]["confidence"]
            
            if verbose:
                print(f"  Step {step}: Prediction {candidate_label} ({candidate_confidence:.3f}), L2: {l2_norm:.2f}")
                
            # Check for success
            if target_label is None:
                # Untargeted attack: any misclassification is success
                attack_success = candidate_label != orig_label
            else:
                # Targeted attack: must match target label
                attack_success = candidate_label == target_label
                
            if attack_success:
                adv_image = candidate
                success = True
                current_idx = step
                block_mask[block_indices[:step]] = False
                
                if verbose:
                    print(f"  Success! Found adversarial example at step {step}")
                
                # Don't break here - continue to find minimal perturbation
            else:
                current_idx = step
                
            # Check if we've reached query limit
            if query_count >= self.max_queries:
                if verbose:
                    print(f"  Reached query limit ({self.max_queries})")
                break
                
        # Stage 2: Refine by restoring unnecessary components
        if success and query_count < self.max_queries:
            if verbose:
                print("Stage 2: Refining by restoring unnecessary components...")
                
            # Track which blocks we've zeroed out
            zeroed_blocks = np.where(~block_mask)[0]
            
            for i in reversed(zeroed_blocks):
                idx = block_indices[i]
                
                # Try restoring this block
                freq_refined = freq_modified.copy()
                freq_mag_refined = np.abs(freq_refined)
                
                # Restore the square pattern from original magnitude
                freq_mag_refined[idx, idx:self.square_max_num-idx, :] = freq_mag[idx, idx:self.square_max_num-idx, :]
                freq_mag_refined[self.square_max_num-1-idx, idx:self.square_max_num-idx, :] = freq_mag[self.square_max_num-1-idx, idx:self.square_max_num-idx, :]
                freq_mag_refined[idx:self.square_max_num-idx, idx, :] = freq_mag[idx:self.square_max_num-idx, idx, :]
                freq_mag_refined[idx:self.square_max_num-idx, self.square_max_num-1-idx, :] = freq_mag[idx:self.square_max_num-idx, self.square_max_num-1-idx, :]
                
                # Reconstruct with refined magnitude
                freq_refined = freq_mag_refined * np.exp(1j * freq_phase)
                
                # Convert back to image
                candidate = ifft2d(freq_refined)
                
                # Get prediction on refined image
                candidate_results = self.api.predict(candidate)
                query_count += 1
                
                candidate_label = candidate_results["predictions"][0]["class_name"]
                candidate_confidence = candidate_results["predictions"][0]["confidence"]
                
                # Calculate norm
                perturbation = candidate - image
                l2_norm = np.sqrt(np.sum(perturbation ** 2))
                
                if verbose:
                    print(f"  Restoring block {idx}: Prediction {candidate_label} ({candidate_confidence:.3f}), L2: {l2_norm:.2f}")
                    
                # Check if still adversarial
                if target_label is None:
                    still_adversarial = candidate_label != orig_label
                else:
                    still_adversarial = candidate_label == target_label
                    
                if still_adversarial:
                    # Keep the restoration
                    freq_modified = freq_refined
                    adv_image = candidate
                    block_mask[block_indices[i]] = True
                    
                    if verbose:
                        print(f"  Successfully restored block {idx}")
                        
                # Check if we've reached query limit
                if query_count >= self.max_queries:
                    if verbose:
                        print(f"  Reached query limit ({self.max_queries})")
                    break
        
        # Calculate final metrics
        perturbation = adv_image - image
        l2_norm = np.sqrt(np.sum(perturbation ** 2))
        linf_norm = np.max(np.abs(perturbation))
        
        # Final prediction
        final_results = self.api.predict(adv_image)
        final_label = final_results["predictions"][0]["class_name"]
        final_confidence = final_results["predictions"][0]["confidence"]
        
        # Compile metadata
        metadata = {
            "success": success,
            "queries": query_count,
            "l2_norm": l2_norm,
            "linf_norm": linf_norm,
            "original_label": orig_label,
            "original_confidence": orig_confidence,
            "adversarial_label": final_label,
            "adversarial_confidence": final_confidence,
            "target_label": target_label,
            "zeroed_blocks": int(np.sum(~block_mask)),
            "total_blocks": num_block
        }
        
        if verbose:
            print("\nAttack summary:")
            print(f"  Success: {success}")
            print(f"  Queries: {query_count}")
            print(f"  L2 norm: {l2_norm:.2f}")
            print(f"  L∞ norm: {linf_norm:.2f}")
            print(f"  Original: {orig_label} ({orig_confidence:.3f})")
            print(f"  Adversarial: {final_label} ({final_confidence:.3f})")
            print(f"  Zeroed blocks: {int(np.sum(~block_mask))}/{num_block}")
        
        return adv_image, metadata
    
    def attack_directory(
        self,
        image_dir: str,
        results_file: str = "api_attack_results.json",
        num_images: int = 10
    ):
        """Attack all images in a directory.
        
        Args:
            image_dir: Directory containing images
            results_file: File to save results
            num_images: Maximum number of images to attack
        """
        # Get list of image files
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        # Limit number of images
        image_files = image_files[:num_images]
        
        results = {}
        
        for i, filename in enumerate(tqdm(image_files, desc="Attacking images")):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
            
            # Load image
            image_path = os.path.join(image_dir, filename)
            image = np.array(Image.open(image_path).convert("RGB"))
            
            # Resize if needed
            if image.shape[:2] != (224, 224):
                # Preserve aspect ratio
                height, width = image.shape[:2]
                if height > width:
                    new_height = int(height * 224 / width)
                    new_width = 224
                else:
                    new_height = 224
                    new_width = int(width * 224 / height)
                    
                image = np.array(Image.fromarray(image).resize((new_width, new_height)))
                
                # Center crop
                height, width = image.shape[:2]
                start_h = (height - 224) // 2
                start_w = (width - 224) // 2
                image = image[start_h:start_h+224, start_w:start_w+224]
            
            # Attack image
            adv_image, metadata = self.attack_image(image, verbose=True)
            
            # Save adversarial image
            adv_filename = f"adv_{filename}"
            adv_path = os.path.join(self.save_dir, adv_filename)
            Image.fromarray(adv_image.astype(np.uint8)).save(adv_path)
            
            # Save side-by-side comparison
            comparison = np.hstack([image, adv_image])
            comp_filename = f"comp_{filename}"
            comp_path = os.path.join(self.save_dir, comp_filename)
            Image.fromarray(comparison.astype(np.uint8)).save(comp_path)
            
            # Store results
            results[filename] = {
                "metadata": metadata,
                "adversarial_image": adv_filename,
                "comparison_image": comp_filename
            }
            
        # Save results
        results_path = os.path.join(self.save_dir, results_file)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        # Calculate overall statistics
        success_rate = sum(1 for r in results.values() if r["metadata"]["success"]) / len(results)
        avg_queries = sum(r["metadata"]["queries"] for r in results.values()) / len(results)
        avg_l2 = sum(r["metadata"]["l2_norm"] for r in results.values() if r["metadata"]["success"]) / max(1, sum(1 for r in results.values() if r["metadata"]["success"]))
        
        print("\nOverall statistics:")
        print(f"  Success rate: {success_rate:.2f}")
        print(f"  Average queries: {avg_queries:.2f}")
        print(f"  Average L2 norm: {avg_l2:.2f}")
        print(f"  Results saved to {results_path}")


def main():
    """Run API attack from command line."""
    parser = argparse.ArgumentParser(description="Attack commercial API services")
    parser.add_argument("--api", type=str, default="dummy", choices=["dummy", "custom"], 
                        help="API service to attack")
    parser.add_argument("--api-key", type=str, default=None, 
                        help="API key for the service")
    parser.add_argument("--image-dir", type=str, default="data/imagenet", 
                        help="Directory containing images to attack")
    parser.add_argument("--save-dir", type=str, default="api_attacks", 
                        help="Directory to save results")
    parser.add_argument("--max-queries", type=int, default=30, 
                        help="Maximum number of API queries per image")
    parser.add_argument("--num-images", type=int, default=10, 
                        help="Number of images to attack")
    parser.add_argument("--norm-bound", type=float, default=20.0, 
                        help="Maximum L2 norm of perturbation")
    
    args = parser.parse_args()
    
    # Create API wrapper
    if args.api == "dummy":
        api_wrapper = DummyAPIWrapper()
    elif args.api == "custom":
        if args.api_key is None:
            raise ValueError("API key must be provided for custom API")
        # Implement your custom API wrapper here
        raise NotImplementedError("Custom API wrapper not implemented yet")
    else:
        raise ValueError(f"Unsupported API service: {args.api}")
        
    # Create attack
    attack = FastDropAPIAttack(
        api_wrapper=api_wrapper,
        square_max_num=224,
        norm_bound=args.norm_bound,
        max_queries=args.max_queries,
        save_dir=args.save_dir
    )
    
    # Run attack
    attack.attack_directory(
        image_dir=args.image_dir,
        num_images=args.num_images
    )
    

if __name__ == "__main__":
    main()

This script demonstrates how adversarial attacks like FastDrop can be used
to test the robustness of real-world computer vision APIs.

Note: To use this script, you need to obtain API keys for commercial vision services
and customize the API wrapper functions accordingly.
"""

import os
import argparse
import time
import json
import requests
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from tqdm import tqdm

from ..attacks.fast_drop import FastDrop
from ..utils.frequency_utils import fft2d, ifft2d
from ..models.model_loader import load_model


class APIWrapper:
    """Base wrapper for vision API services."""
    
    def __init__(self, api_key: str):
        """Initialize API wrapper.
        
        Args:
            api_key: API key for the service
        """
        self.api_key = api_key
        
    def predict(self, image: np.ndarray) -> Dict:
        """Get predictions from the API.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Dictionary containing prediction results
        """
        raise NotImplementedError("Subclasses must implement predict method")
        
    def preprocess_image(self, image: np.ndarray) -> bytes:
        """Preprocess image for API submission.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Image bytes ready for API submission
        """
        img = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()


class DummyAPIWrapper(APIWrapper):
    """Dummy API wrapper that uses a local model instead of an actual API.
    
    This is useful for testing without actual API access.
    """
    
    def __init__(self, model_name: str = "resnet18"):
        """Initialize dummy API wrapper.
        
        Args:
            model_name: Name of the model to use
        """
        super().__init__("dummy_key")
        
        # Load ImageNet class names
        with open("imagenet_classes.txt", "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]
            
        # Load a local model
        self.model = load_model(model_name, "imagenet")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image: np.ndarray) -> Dict:
        """Get predictions from the local model.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            
        Returns:
            Dictionary containing prediction results
        """
        # Convert to PIL image
        img = Image.fromarray(image.astype(np.uint8))
        
        # Preprocess
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        
        # Format results
        results = {
            "predictions": [
                {
                    "class_id": idx.item(),
                    "class_name": self.class_names[idx.item()],
                    "confidence": prob.item()
                }
                for idx, prob in zip(top5_indices[0], top5_probs[0])
            ]
        }
        
        return results


class FastDropAPIAttack:
    """Attack commercial API services using FastDrop."""
    
    def __init__(
        self,
        api_wrapper: APIWrapper,
        square_max_num: int = 224,
        norm_bound: float = 20.0,
        max_queries: int = 50,
        save_dir: str = "api_attacks"
    ):
        """Initialize FastDrop API attack.
        
        Args:
            api_wrapper: Wrapper for the API service
            square_max_num: Size of frequency grid (224 for ImageNet-sized images)
            norm_bound: Maximum perturbation norm
            max_queries: Maximum number of API queries
            save_dir: Directory to save results
        """
        self.api = api_wrapper
        self.square_max_num = square_max_num
        self.norm_bound = norm_bound
        self.max_queries = max_queries
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
    def attack_image(
        self,
        image: np.ndarray,
        target_label: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Attack a single image.
        
        Args:
            image: Input image [H, W, C] in range [0, 255]
            target_label: Target label for targeted attack (if None, performs untargeted attack)
            verbose: Whether to print progress information
            
        Returns:
            Tuple of (adversarial image, attack metadata)
        """
        # Get original prediction
        if verbose:
            print("Getting original prediction...")
            
        orig_results = self.api.predict(image)
        orig_label = orig_results["predictions"][0]["class_name"]
        orig_confidence = orig_results["predictions"][0]["confidence"]
        
        if verbose:
            print(f"Original prediction: {orig_label} ({orig_confidence:.3f})")
            
        # Initialize attack parameters
        query_count = 1  # Count the initial prediction
        adv_image = image.copy()
        success = False
        
        # Compute FFT of the image
        freq = fft2d(image)
        freq_mag = np.abs(freq)
        freq_phase = np.angle(freq)
        
        # Calculate average magnitude for each square
        num_block = self.square_max_num // 2
        block_sum = np.zeros(num_block)
        
        for i in range(num_block):
            # Calculate sum along the square pattern
            rank1 = np.sum(freq_mag[i, i:self.square_max_num-i, :])
            rank2 = np.sum(freq_mag[self.square_max_num-1-i, i:self.square_max_num-i, :])
            col1 = np.sum(freq_mag[i+1:self.square_max_num-1-i, i, :])
            col2 = np.sum(freq_mag[i+1:self.square_max_num-1-i, self.square_max_num-1-i, :])
            
            # Calculate total elements in the square
            num_elements = 4 * (self.square_max_num - 2 * i) - 4
            
            # Calculate average
            block_sum[i] = (rank1 + rank2 + col1 + col2) / max(1, num_elements)
            
        # Sort blocks by importance
        block_indices = np.argsort(block_sum)
        block_mask = np.ones(num_block, dtype=bool)
        
        # Stage 1: Find minimal frequency components to remove
        if verbose:
            print("Stage 1: Finding minimal frequency components to remove...")
            
        # Start with larger steps for efficiency
        steps = [
            list(range(5, 101, 5)),     # 5, 10, 15, ..., 100
            list(range(101, 201, 10))   # 110, 120, ..., 200
        ]
        steps = [item for sublist in steps for item in sublist]  # Flatten
        
        current_idx = 0
        for step in steps:
            if current_idx >= step:
                continue
                
            # Zero out frequency components up to the current step
            freq_modified = freq.copy()
            freq_mag_modified = np.abs(freq_modified)
            
            for i in range(current_idx, min(step, num_block)):
                idx = block_indices[i]
                
                # Zero out the square pattern
                freq_mag_modified[idx, idx:self.square_max_num-idx, :] = 0
                freq_mag_modified[self.square_max_num-1-idx, idx:self.square_max_num-idx, :] = 0
                freq_mag_modified[idx:self.square_max_num-idx, idx, :] = 0
                freq_mag_modified[idx:self.square_max_num-idx, self.square_max_num-1-idx, :] = 0
                
            # Reconstruct frequency representation with modified magnitude
            freq_modified = freq_mag_modified * np.exp(1j * freq_phase)
            
            # Convert back to image
            candidate = ifft2d(freq_modified)
            
            # Check if perturbation is within bounds
            perturbation = candidate - image
            l2_norm = np.sqrt(np.sum(perturbation ** 2))
            
            if l2_norm > self.norm_bound:
                if verbose:
                    print(f"  Step {step}: L2 norm {l2_norm:.2f} exceeds bound, skipping")
                continue
                
            # Get prediction on candidate image
            candidate_results = self.api.predict(candidate)
            query_count += 1
            
            candidate_label = candidate_results["predictions"][0]["class_name"]
            candidate_confidence = candidate_results["predictions"][0]["confidence"]
            
            if verbose:
                print(f"  Step {step}: Prediction {candidate_label} ({candidate_confidence:.3f}), L2: {l2_norm:.2f}")
                
            # Check for success
            if target_label is None:
                # Untargeted attack: any misclassification is success
                attack_success = candidate_label != orig_label
            else:
                # Targeted attack: must match target label
                attack_success = candidate_label == target_label
                
            if attack_success:
                adv_image = candidate
                success = True
                current_idx = step
                block_mask[block_indices[:step]] = False
                
                if verbose:
                    print(f"  Success! Found adversarial example at step {step}")
                
                # Don't break here - continue to find minimal perturbation
            else:
                current_idx = step
                
            # Check if we've reached query limit
            if query_count >= self.max_queries:
                if verbose:
                    print(f"  Reached query limit ({self.max_queries})")
                break
                
        # Stage 2: Refine by restoring unnecessary components
        if success and query_count < self.max_queries:
            if verbose:
                print("Stage 2: Refining by restoring unnecessary components...")
                
            # Track which blocks we've zeroed out
            zeroed_blocks = np.where(~block_mask)[0]
            
            for i in reversed(zeroed_blocks):
                idx = block_indices[i]
                
                # Try restoring this block
                freq_refined = freq_modified.copy()
                freq_mag_refined = np.abs(freq_refined)
                
                # Restore the square pattern from original magnitude
                freq_mag_refined[idx, idx:self.square_max_num-idx, :] = freq_mag[idx, idx:self.square_max_num-idx, :]
                freq_mag_refined[self.square_max_num-1-idx, idx:self.square_max_num-idx, :] = freq_mag[self.square_max_num-1-idx, idx:self.square_max_num-idx, :]
                freq_mag_refined[idx:self.square_max_num-idx, idx, :] = freq_mag[idx:self.square_max_num-idx, idx, :]
                freq_mag_refined[idx:self.square_max_num-idx, self.square_max_num-1-idx, :] = freq_mag[idx:self.square_max_num-idx, self.square_max_num-1-idx, :]
                
                # Reconstruct with refined magnitude
                freq_refined = freq_mag_refined * np.exp(1j * freq_phase)
                
                # Convert back to image
                candidate = ifft2d(freq_refined)
                
                # Get prediction on refined image
                candidate_results = self.api.predict(candidate)
                query_count += 1
                
                candidate_label = candidate_results["predictions"][0]["class_name"]
                candidate_confidence = candidate_results["predictions"][0]["confidence"]
                
                # Calculate norm
                perturbation = candidate - image
                l2_norm = np.sqrt(np.sum(perturbation ** 2))
                
                if verbose:
                    print(f"  Restoring block {idx}: Prediction {candidate_label} ({candidate_confidence:.3f}), L2: {l2_norm:.2f}")
                    
                # Check if still adversarial
                if target_label is None:
                    still_adversarial = candidate_label != orig_label
                else:
                    still_adversarial = candidate_label == target_label
                    
                if still_adversarial:
                    # Keep the restoration
                    freq_modified = freq_refined
                    adv_image = candidate
                    block_mask[block_indices[i]] = True
                    
                    if verbose:
                        print(f"  Successfully restored block {idx}")
                        
                # Check if we've reached query limit
                if query_count >= self.max_queries:
                    if verbose:
                        print(f"  Reached query limit ({self.max_queries})")
                    break
        
        # Calculate final metrics
        perturbation = adv_image - image
        l2_norm = np.sqrt(np.sum(perturbation ** 2))
        linf_norm = np.max(np.abs(perturbation))
        
        # Final prediction
        final_results = self.api.predict(adv_image)
        final_label = final_results["predictions"][0]["class_name"]
        final_confidence = final_results["predictions"][0]["confidence"]
        
        # Compile metadata
        metadata = {
            "success": success,
            "queries": query_count,
            "l2_norm": l2_norm,
            "linf_norm": linf_norm,
            "original_label": orig_label,
            "original_confidence": orig_confidence,
            "adversarial_label": final_label,
            "adversarial_confidence": final_confidence,
            "target_label": target_label,
            "zeroed_blocks": int(np.sum(~block_mask)),
            "total_blocks": num_block
        }
        
        if verbose:
            print("\nAttack summary:")
            print(f"  Success: {success}")
            print(f"  Queries: {query_count}")
            print(f"  L2 norm: {l2_norm:.2f}")
            print(f"  L∞ norm: {linf_norm:.2f}")
            print(f"  Original: {orig_label} ({orig_confidence:.3f})")
            print(f"  Adversarial: {final_label} ({final_confidence:.3f})")
            print(f"  Zeroed blocks: {int(np.sum(~block_mask))}/{num_block}")
        
        return adv_image, metadata
    
    def attack_directory(
        self,
        image_dir: str,
        results_file: str = "api_attack_results.json",
        num_images: int = 10
    ):
        """Attack all images in a directory.
        
        Args:
            image_dir: Directory containing images
            results_file: File to save results
            num_images: Maximum number of images to attack
        """
        # Get list of image files
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        # Limit number of images
        image_files = image_files[:num_images]
        
        results = {}
        
        for i, filename in enumerate(tqdm(image_files, desc="Attacking images")):
            print(f"\nProcessing image {i+1}/{len(image_files)}: {filename}")
            
            # Load image
            image_path = os.path.join(image_dir, filename)
            image = np.array(Image.open(image_path).convert("RGB"))
            
            # Resize if needed
            if image.shape[:2] != (224, 224):
                # Preserve aspect ratio
                height, width = image.shape[:2]
                if height > width:
                    new_height = int(height * 224 / width)
                    new_width = 224
                else:
                    new_height = 224
                    new_width = int(width * 224 / height)
                    
                image = np.array(Image.fromarray(image).resize((new_width, new_height)))
                
                # Center crop
                height, width = image.shape[:2]
                start_h = (height - 224) // 2
                start_w = (width - 224) // 2
                image = image[start_h:start_h+224, start_w:start_w+224]
            
            # Attack image
            adv_image, metadata = self.attack_image(image, verbose=True)
            
            # Save adversarial image
            adv_filename = f"adv_{filename}"
            adv_path = os.path.join(self.save_dir, adv_filename)
            Image.fromarray(adv_image.astype(np.uint8)).save(adv_path)
            
            # Save side-by-side comparison
            comparison = np.hstack([image, adv_image])
            comp_filename = f"comp_{filename}"
            comp_path = os.path.join(self.save_dir, comp_filename)
            Image.fromarray(comparison.astype(np.uint8)).save(comp_path)
            
            # Store results
            results[filename] = {
                "metadata": metadata,
                "adversarial_image": adv_filename,
                "comparison_image": comp_filename
            }
            
        # Save results
        results_path = os.path.join(self.save_dir, results_file)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
            
        # Calculate overall statistics
        success_rate = sum(1 for r in results.values() if r["metadata"]["success"]) / len(results)
        avg_queries = sum(r["metadata"]["queries"] for r in results.values()) / len(results)
        avg_l2 = sum(r["metadata"]["l2_norm"] for r in results.values() if r["metadata"]["success"]) / max(1, sum(1 for r in results.values() if r["metadata"]["success"]))
        
        print("\nOverall statistics:")
        print(f"  Success rate: {success_rate:.2f}")
        print(f"  Average queries: {avg_queries:.2f}")
        print(f"  Average L2 norm: {avg_l2:.2f}")
        print(f"  Results saved to {results_path}")


def main():
    """Run API attack from command line."""
    parser = argparse.ArgumentParser(description="Attack commercial API services")
    parser.add_argument("--api", type=str, default="dummy", choices=["dummy", "custom"], 
                        help="API service to attack")
    parser.add_argument("--api-key", type=str, default=None, 
                        help="API key for the service")
    parser.add_argument("--image-dir", type=str, default="data/imagenet", 
                        help="Directory containing images to attack")
    parser.add_argument("--save-dir", type=str, default="api_attacks", 
                        help="Directory to save results")
    parser.add_argument("--max-queries", type=int, default=30, 
                        help="Maximum number of API queries per image")
    parser.add_argument("--num-images", type=int, default=10, 
                        help="Number of images to attack")
    parser.add_argument("--norm-bound", type=float, default=20.0, 
                        help="Maximum L2 norm of perturbation")
    
    args = parser.parse_args()
    
    # Create API wrapper
    if args.api == "dummy":
        api_wrapper = DummyAPIWrapper()
    elif args.api == "custom":
        if args.api_key is None:
            raise ValueError("API key must be provided for custom API")
        # Implement your custom API wrapper here
        raise NotImplementedError("Custom API wrapper not implemented yet")
    else:
        raise ValueError(f"Unsupported API service: {args.api}")
        
    # Create attack
    attack = FastDropAPIAttack(
        api_wrapper=api_wrapper,
        square_max_num=224,
        norm_bound=args.norm_bound,
        max_queries=args.max_queries,
        save_dir=args.save_dir
    )
    
    # Run attack
    attack.attack_directory(
        image_dir=args.image_dir,
        num_images=args.num_images
    )
    

if __name__ == "__main__":
    main()