"""Visualization tools for comparing different adversarial attacks."""
#-Visualization tools for comparing different adversarial attacks.
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image

from ..attacks.base_attack import BaseAttack
from ..utils.frequency_utils import fft2d, visualize_spectrum
from ..utils.metrics import perturbation_norm, structural_similarity


class AttackComparison:
    """Visualization tool for comparing different adversarial attacks.
    
    This class provides visualization methods to compare the effects
    of different adversarial attacks on the same input images.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attacks: List[BaseAttack],
        class_names: List[str],
        device: torch.device = None,
        save_dir: str = "visualizations"
    ):
        """Initialize attack comparison visualizer.
        
        Args:
            model: Target model to visualize attacks against
            attacks: List of attack methods to compare
            class_names: List of class names for the model
            device: Device to run the visualizer on
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.attacks = attacks
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def compare_single_image(
        self,
        image: torch.Tensor,
        label: int,
        filename: Optional[str] = None,
        show: bool = True,
        return_images: bool = False
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compare different attacks on a single image.
        
        Args:
            image: Input image tensor [C, H, W] or [1, C, H, W]
            label: Ground truth label
            filename: Filename to save visualization (if None, doesn't save)
            show: Whether to display the visualization
            return_images: Whether to return the adversarial images
            
        Returns:
            Dictionary of adversarial images if return_images=True, otherwise None
        """
        # Ensure image is 4D
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Move to device
        image = image.to(self.device)
        label_tensor = torch.tensor([label], device=self.device)
        
        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.max(output, dim=1)
            confidence = probs[0, pred].item()
            
        orig_pred = pred.item()
        orig_confidence = confidence
        
        # Generate adversarial examples for each attack
        adv_images = {}
        attack_results = {}
        
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            
            # Generate adversarial example
            adv_image, metadata = attack(image, label_tensor)
            
            # Get prediction on adversarial example
            with torch.no_grad():
                output = self.model(adv_image)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                confidence = probs[0, pred].item()
                
            # Calculate perturbation metrics
            l2_norm = perturbation_norm(image, adv_image, 'L2')[0].item()
            linf_norm = perturbation_norm(image, adv_image, 'Linf')[0].item()
            ssim = structural_similarity(image, adv_image)[0].item()
            
            # Store results
            adv_images[attack_name] = adv_image
            attack_results[attack_name] = {
                'prediction': pred.item(),
                'confidence': confidence,
                'l2_norm': l2_norm,
                'linf_norm': linf_norm,
                'ssim': ssim,
                'success': pred.item() != label,
                'queries': metadata.get('avg_queries', 0)
            }
            
        # Visualize comparison
        self._visualize_comparison(
            image, adv_images, attack_results, 
            label, orig_pred, orig_confidence,
            filename, show
        )
        
        if return_images:
            # Convert to numpy for easier handling
            np_images = {
                name: img[0].detach().cpu().numpy() 
                for name, img in adv_images.items()
            }
            np_images['original'] = image[0].detach().cpu().numpy()
            return np_images
            
        return None
    
    def _visualize_comparison(
        self,
        original: torch.Tensor,
        adversarial: Dict[str, torch.Tensor],
        results: Dict[str, Dict],
        true_label: int,
        orig_pred: int,
        orig_confidence: float,
        filename: Optional[str] = None,
        show: bool = True
    ):
        """Create visualization of attack comparison.
        
        Args:
            original: Original image tensor [1, C, H, W]
            adversarial: Dictionary of adversarial images
            results: Dictionary of attack results
            true_label: Ground truth label
            orig_pred: Original prediction
            orig_confidence: Original confidence
            filename: Filename to save visualization
            show: Whether to display the visualization
        """
        num_attacks = len(adversarial)
        fig, axes = plt.subplots(3, num_attacks + 1, figsize=(4 * (num_attacks + 1), 10))
        
        # Remove axes for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
                
        # Original image
        orig_np = original[0].detach().cpu().permute(1, 2, 0).numpy()
        
        # Normalize to [0, 1] if needed
        if orig_np.max() > 1.0:
            orig_np = orig_np / 255.0
            
        # Handle grayscale images
        if orig_np.shape[2] == 1:
            orig_np = np.repeat(orig_np, 3, axis=2)
            
        axes[0, 0].imshow(np.clip(orig_np, 0, 1))
        axes[0, 0].set_title(f"Original\n{self.class_names[true_label]}\n"
                             f"Pred: {self.class_names[orig_pred]}\n"
                             f"Conf: {orig_confidence:.3f}")
        
        # Original frequency spectrum
        orig_freq = fft2d(np.clip(orig_np * 255, 0, 255).astype(np.uint8))
        axes[1, 0].imshow(visualize_spectrum(orig_freq))
        axes[1, 0].set_title("Original Spectrum")
        
        # Original perturbation (empty)
        axes[2, 0].imshow(np.zeros_like(orig_np))
        axes[2, 0].set_title("No Perturbation")
        
        # For each attack
        for i, (attack_name, adv_image) in enumerate(adversarial.items(), 1):
            # Get attack results
            result = results[attack_name]
            pred = result['prediction']
            conf = result['confidence']
            l2 = result['l2_norm']
            linf = result['linf_norm']
            ssim = result['ssim']
            queries = result.get('queries', 0)
            
            # Convert to numpy
            adv_np = adv_image[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Normalize to [0, 1] if needed
            if adv_np.max() > 1.0:
                adv_np = adv_np / 255.0
                
            # Handle grayscale images
            if adv_np.shape[2] == 1:
                adv_np = np.repeat(adv_np, 3, axis=2)
                
            # Adversarial image
            axes[0, i].imshow(np.clip(adv_np, 0, 1))
            axes[0, i].set_title(
                f"{attack_name}\n"
                f"Pred: {self.class_names[pred]}\n"
                f"Conf: {conf:.3f}\n"
                f"L2: {l2:.2f}, L∞: {linf:.2f}\n"
                f"SSIM: {ssim:.3f}, Q: {queries:.1f}"
            )
            
            # Color border based on attack success
            for spine in axes[0, i].spines.values():
                spine.set_visible(True)
                spine.set_color('red' if pred != true_label else 'blue')
                spine.set_linewidth(3)
                
            # Adversarial frequency spectrum
            adv_freq = fft2d(np.clip(adv_np * 255, 0, 255).astype(np.uint8))
            axes[1, i].imshow(visualize_spectrum(adv_freq))
            axes[1, i].set_title(f"{attack_name} Spectrum")
            
            # Perturbation visualization
            perturbation = adv_np - orig_np
            
            # Normalize perturbation for visibility
            abs_pert = np.abs(perturbation)
            max_pert = max(abs_pert.max(), 1e-8)
            norm_pert = abs_pert / max_pert
            
            # Use a heatmap colormap for perturbation
            axes[2, i].imshow(norm_pert, cmap='hot')
            axes[2, i].set_title(
                f"Perturbation\n"
                f"Max: {max_pert:.4f}"
            )
            
        # Add overall title
        plt.suptitle(f"Comparison of Adversarial Attacks", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if filename provided
        if filename:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_multiple_images(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        base_filename: str = "comparison",
        show: bool = False
    ):
        """Compare attacks on multiple images.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Ground truth labels [B]
            base_filename: Base filename for saved visualizations
            show: Whether to display each visualization
        """
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            filename = f"{base_filename}_{i}.png"
            self.compare_single_image(
                images[i:i+1],
                labels[i].item(),
                filename=filename,
                show=show
            )
    
    def generate_attack_report(
        self,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int = 10,
        save_path: str = "attack_report.html"
    ):
        """Generate an HTML report comparing attacks on multiple samples.
        
        Args:
            test_loader: DataLoader for test set
            num_samples: Number of samples to include in report
            save_path: Path to save HTML report
        """
        # Get samples from test loader
        images = []
        labels = []
        
        for data, target in test_loader:
            images.append(data)
            labels.append(target)
            
            if len(images) * data.shape[0] >= num_samples:
                break
                
        # Concatenate and limit to num_samples
        images = torch.cat(images)[:num_samples]
        labels = torch.cat(labels)[:num_samples]
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Adversarial Attack Comparison Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { text-align: center; }",
            "        .sample { margin-bottom: 40px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }",
            "        .sample-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }",
            "        img { max-width: 100%; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        .success { color: red; font-weight: bold; }",
            "        .failure { color: blue; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Adversarial Attack Comparison Report</h1>"
        ]
        
        # Process each sample
        for i in range(num_samples):
            image = images[i:i+1]
            label = labels[i].item()
            
            # Save comparison visualization
            filename = f"comparison_{i}.png"
            adv_images = self.compare_single_image(
                image, label, filename=filename, show=False, return_images=True
            )
            
            # Generate adversarial examples for each attack
            attack_results = {}
            
            for attack in self.attacks:
                attack_name = attack.__class__.__name__
                
                # Generate adversarial example
                image = image.to(self.device)
                label_tensor = torch.tensor([label], device=self.device)
                adv_image, metadata = attack(image, label_tensor)
                
                # Get prediction on adversarial example
                with torch.no_grad():
                    output = self.model(adv_image)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    _, pred = torch.max(output, dim=1)
                    confidence = probs[0, pred].item()
                    
                # Calculate perturbation metrics
                l2_norm = perturbation_norm(image, adv_image, 'L2')[0].item()
                linf_norm = perturbation_norm(image, adv_image, 'Linf')[0].item()
                ssim = structural_similarity(image, adv_image)[0].item()
                
                # Store results
                attack_results[attack_name] = {
                    'prediction': pred.item(),
                    'confidence': confidence,
                    'l2_norm': l2_norm,
                    'linf_norm': linf_norm,
                    'ssim': ssim,
                    'success': pred.item() != label,
                    'queries': metadata.get('avg_queries', 0)
                }
            
            # Add sample to HTML report
            html_content.append(f"    <div class='sample'>")
            html_content.append(f"        <div class='sample-title'>Sample {i+1} - True Class: {self.class_names[label]}</div>")
            html_content.append(f"        <img src='{filename}' alt='Attack comparison for sample {i+1}'>")
            
            # Add table with metrics
            html_content.append(f"        <h3>Attack Metrics</h3>")
            html_content.append(f"        <table>")
            html_content.append(f"            <tr>")
            html_content.append(f"                <th>Attack</th>")
            html_content.append(f"                <th>Prediction</th>")
            html_content.append(f"                <th>Confidence</th>")
            html_content.append(f"                <th>L2 Norm</th>")
            html_content.append(f"                <th>L∞ Norm</th>")
            html_content.append(f"                <th>SSIM</th>")
            html_content.append(f"                <th>Queries</th>")
            html_content.append(f"                <th>Success</th>")
            html_content.append(f"            </tr>")
            
            # Original prediction
            with torch.no_grad():
                output = self.model(image)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                confidence = probs[0, pred].item()
                
            html_content.append(f"            <tr>")
            html_content.append(f"                <td>Original</td>")
            html_content.append(f"                <td>{self.class_names[pred.item()]}</td>")
            html_content.append(f"                <td>{confidence:.3f}</td>")
            html_content.append(f"                <td>0.0</td>")
            html_content.append(f"                <td>0.0</td>")
            html_content.append(f"                <td>1.0</td>")
            html_content.append(f"                <td>0</td>")
            html_content.append(f"                <td class='failure'>N/A</td>")
            html_content.append(f"            </tr>")
            
            # Attack results
            for attack_name, result in attack_results.items():
                success_class = "success" if result['success'] else "failure"
                html_content.append(f"            <tr>")
                html_content.append(f"                <td>{attack_name}</td>")
                html_content.append(f"                <td>{self.class_names[result['prediction']]}</td>")
                html_content.append(f"                <td>{result['confidence']:.3f}</td>")
                html_content.append(f"                <td>{result['l2_norm']:.3f}</td>")
                html_content.append(f"                <td>{result['linf_norm']:.3f}</td>")
                html_content.append(f"                <td>{result['ssim']:.3f}</td>")
                html_content.append(f"                <td>{result['queries']:.1f}</td>")
                html_content.append(f"                <td class='{success_class}'>{result['success']}</td>")
                html_content.append(f"            </tr>")
                
            html_content.append(f"        </table>")
            html_content.append(f"    </div>")
            
        # Close HTML
        html_content.append("</body>")
        html_content.append("</html>")
        
        # Write HTML file
        report_path = os.path.join(self.save_dir, save_path)
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_content))
            
        print(f"Generated attack report at {report_path}")

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Callable
from PIL import Image

from ..attacks.base_attack import BaseAttack
from ..utils.frequency_utils import fft2d, visualize_spectrum
from ..utils.metrics import perturbation_norm, structural_similarity


class AttackComparison:
    """Visualization tool for comparing different adversarial attacks.
    
    This class provides visualization methods to compare the effects
    of different adversarial attacks on the same input images.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attacks: List[BaseAttack],
        class_names: List[str],
        device: torch.device = None,
        save_dir: str = "visualizations"
    ):
        """Initialize attack comparison visualizer.
        
        Args:
            model: Target model to visualize attacks against
            attacks: List of attack methods to compare
            class_names: List of class names for the model
            device: Device to run the visualizer on
            save_dir: Directory to save visualizations
        """
        self.model = model
        self.attacks = attacks
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def compare_single_image(
        self,
        image: torch.Tensor,
        label: int,
        filename: Optional[str] = None,
        show: bool = True,
        return_images: bool = False
    ) -> Optional[Dict[str, np.ndarray]]:
        """Compare different attacks on a single image.
        
        Args:
            image: Input image tensor [C, H, W] or [1, C, H, W]
            label: Ground truth label
            filename: Filename to save visualization (if None, doesn't save)
            show: Whether to display the visualization
            return_images: Whether to return the adversarial images
            
        Returns:
            Dictionary of adversarial images if return_images=True, otherwise None
        """
        # Ensure image is 4D
        if image.dim() == 3:
            image = image.unsqueeze(0)
            
        # Move to device
        image = image.to(self.device)
        label_tensor = torch.tensor([label], device=self.device)
        
        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(image)
            probs = torch.nn.functional.softmax(output, dim=1)
            _, pred = torch.max(output, dim=1)
            confidence = probs[0, pred].item()
            
        orig_pred = pred.item()
        orig_confidence = confidence
        
        # Generate adversarial examples for each attack
        adv_images = {}
        attack_results = {}
        
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            
            # Generate adversarial example
            adv_image, metadata = attack(image, label_tensor)
            
            # Get prediction on adversarial example
            with torch.no_grad():
                output = self.model(adv_image)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                confidence = probs[0, pred].item()
                
            # Calculate perturbation metrics
            l2_norm = perturbation_norm(image, adv_image, 'L2')[0].item()
            linf_norm = perturbation_norm(image, adv_image, 'Linf')[0].item()
            ssim = structural_similarity(image, adv_image)[0].item()
            
            # Store results
            adv_images[attack_name] = adv_image
            attack_results[attack_name] = {
                'prediction': pred.item(),
                'confidence': confidence,
                'l2_norm': l2_norm,
                'linf_norm': linf_norm,
                'ssim': ssim,
                'success': pred.item() != label,
                'queries': metadata.get('avg_queries', 0)
            }
            
        # Visualize comparison
        self._visualize_comparison(
            image, adv_images, attack_results, 
            label, orig_pred, orig_confidence,
            filename, show
        )
        
        if return_images:
            # Convert to numpy for easier handling
            np_images = {
                name: img[0].detach().cpu().numpy() 
                for name, img in adv_images.items()
            }
            np_images['original'] = image[0].detach().cpu().numpy()
            return np_images
            
        return None
    
    def _visualize_comparison(
        self,
        original: torch.Tensor,
        adversarial: Dict[str, torch.Tensor],
        results: Dict[str, Dict],
        true_label: int,
        orig_pred: int,
        orig_confidence: float,
        filename: Optional[str] = None,
        show: bool = True
    ):
        """Create visualization of attack comparison.
        
        Args:
            original: Original image tensor [1, C, H, W]
            adversarial: Dictionary of adversarial images
            results: Dictionary of attack results
            true_label: Ground truth label
            orig_pred: Original prediction
            orig_confidence: Original confidence
            filename: Filename to save visualization
            show: Whether to display the visualization
        """
        num_attacks = len(adversarial)
        fig, axes = plt.subplots(3, num_attacks + 1, figsize=(4 * (num_attacks + 1), 10))
        
        # Remove axes for all subplots
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
                
        # Original image
        orig_np = original[0].detach().cpu().permute(1, 2, 0).numpy()
        
        # Normalize to [0, 1] if needed
        if orig_np.max() > 1.0:
            orig_np = orig_np / 255.0
            
        # Handle grayscale images
        if orig_np.shape[2] == 1:
            orig_np = np.repeat(orig_np, 3, axis=2)
            
        axes[0, 0].imshow(np.clip(orig_np, 0, 1))
        axes[0, 0].set_title(f"Original\n{self.class_names[true_label]}\n"
                             f"Pred: {self.class_names[orig_pred]}\n"
                             f"Conf: {orig_confidence:.3f}")
        
        # Original frequency spectrum
        orig_freq = fft2d(np.clip(orig_np * 255, 0, 255).astype(np.uint8))
        axes[1, 0].imshow(visualize_spectrum(orig_freq))
        axes[1, 0].set_title("Original Spectrum")
        
        # Original perturbation (empty)
        axes[2, 0].imshow(np.zeros_like(orig_np))
        axes[2, 0].set_title("No Perturbation")
        
        # For each attack
        for i, (attack_name, adv_image) in enumerate(adversarial.items(), 1):
            # Get attack results
            result = results[attack_name]
            pred = result['prediction']
            conf = result['confidence']
            l2 = result['l2_norm']
            linf = result['linf_norm']
            ssim = result['ssim']
            queries = result.get('queries', 0)
            
            # Convert to numpy
            adv_np = adv_image[0].detach().cpu().permute(1, 2, 0).numpy()
            
            # Normalize to [0, 1] if needed
            if adv_np.max() > 1.0:
                adv_np = adv_np / 255.0
                
            # Handle grayscale images
            if adv_np.shape[2] == 1:
                adv_np = np.repeat(adv_np, 3, axis=2)
                
            # Adversarial image
            axes[0, i].imshow(np.clip(adv_np, 0, 1))
            axes[0, i].set_title(
                f"{attack_name}\n"
                f"Pred: {self.class_names[pred]}\n"
                f"Conf: {conf:.3f}\n"
                f"L2: {l2:.2f}, L∞: {linf:.2f}\n"
                f"SSIM: {ssim:.3f}, Q: {queries:.1f}"
            )
            
            # Color border based on attack success
            for spine in axes[0, i].spines.values():
                spine.set_visible(True)
                spine.set_color('red' if pred != true_label else 'blue')
                spine.set_linewidth(3)
                
            # Adversarial frequency spectrum
            adv_freq = fft2d(np.clip(adv_np * 255, 0, 255).astype(np.uint8))
            axes[1, i].imshow(visualize_spectrum(adv_freq))
            axes[1, i].set_title(f"{attack_name} Spectrum")
            
            # Perturbation visualization
            perturbation = adv_np - orig_np
            
            # Normalize perturbation for visibility
            abs_pert = np.abs(perturbation)
            max_pert = max(abs_pert.max(), 1e-8)
            norm_pert = abs_pert / max_pert
            
            # Use a heatmap colormap for perturbation
            axes[2, i].imshow(norm_pert, cmap='hot')
            axes[2, i].set_title(
                f"Perturbation\n"
                f"Max: {max_pert:.4f}"
            )
            
        # Add overall title
        plt.suptitle(f"Comparison of Adversarial Attacks", fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save if filename provided
        if filename:
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_multiple_images(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        base_filename: str = "comparison",
        show: bool = False
    ):
        """Compare attacks on multiple images.
        
        Args:
            images: Batch of images [B, C, H, W]
            labels: Ground truth labels [B]
            base_filename: Base filename for saved visualizations
            show: Whether to display each visualization
        """
        batch_size = images.shape[0]
        
        for i in range(batch_size):
            filename = f"{base_filename}_{i}.png"
            self.compare_single_image(
                images[i:i+1],
                labels[i].item(),
                filename=filename,
                show=show
            )
    
    def generate_attack_report(
        self,
        test_loader: torch.utils.data.DataLoader,
        num_samples: int = 10,
        save_path: str = "attack_report.html"
    ):
        """Generate an HTML report comparing attacks on multiple samples.
        
        Args:
            test_loader: DataLoader for test set
            num_samples: Number of samples to include in report
            save_path: Path to save HTML report
        """
        # Get samples from test loader
        images = []
        labels = []
        
        for data, target in test_loader:
            images.append(data)
            labels.append(target)
            
            if len(images) * data.shape[0] >= num_samples:
                break
                
        # Concatenate and limit to num_samples
        images = torch.cat(images)[:num_samples]
        labels = torch.cat(labels)[:num_samples]
        
        # Create HTML content
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>Adversarial Attack Comparison Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        h1 { text-align: center; }",
            "        .sample { margin-bottom: 40px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }",
            "        .sample-title { font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }",
            "        img { max-width: 100%; }",
            "        table { border-collapse: collapse; width: 100%; }",
            "        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "        th { background-color: #f2f2f2; }",
            "        .success { color: red; font-weight: bold; }",
            "        .failure { color: blue; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>Adversarial Attack Comparison Report</h1>"
        ]
        
        # Process each sample
        for i in range(num_samples):
            image = images[i:i+1]
            label = labels[i].item()
            
            # Save comparison visualization
            filename = f"comparison_{i}.png"
            adv_images = self.compare_single_image(
                image, label, filename=filename, show=False, return_images=True
            )
            
            # Generate adversarial examples for each attack
            attack_results = {}
            
            for attack in self.attacks:
                attack_name = attack.__class__.__name__
                
                # Generate adversarial example
                image = image.to(self.device)
                label_tensor = torch.tensor([label], device=self.device)
                adv_image, metadata = attack(image, label_tensor)
                
                # Get prediction on adversarial example
                with torch.no_grad():
                    output = self.model(adv_image)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    _, pred = torch.max(output, dim=1)
                    confidence = probs[0, pred].item()
                    
                # Calculate perturbation metrics
                l2_norm = perturbation_norm(image, adv_image, 'L2')[0].item()
                linf_norm = perturbation_norm(image, adv_image, 'Linf')[0].item()
                ssim = structural_similarity(image, adv_image)[0].item()
                
                # Store results
                attack_results[attack_name] = {
                    'prediction': pred.item(),
                    'confidence': confidence,
                    'l2_norm': l2_norm,
                    'linf_norm': linf_norm,
                    'ssim': ssim,
                    'success': pred.item() != label,
                    'queries': metadata.get('avg_queries', 0)
                }
            
            # Add sample to HTML report
            html_content.append(f"    <div class='sample'>")
            html_content.append(f"        <div class='sample-title'>Sample {i+1} - True Class: {self.class_names[label]}</div>")
            html_content.append(f"        <img src='{filename}' alt='Attack comparison for sample {i+1}'>")
            
            # Add table with metrics
            html_content.append(f"        <h3>Attack Metrics</h3>")
            html_content.append(f"        <table>")
            html_content.append(f"            <tr>")
            html_content.append(f"                <th>Attack</th>")
            html_content.append(f"                <th>Prediction</th>")
            html_content.append(f"                <th>Confidence</th>")
            html_content.append(f"                <th>L2 Norm</th>")
            html_content.append(f"                <th>L∞ Norm</th>")
            html_content.append(f"                <th>SSIM</th>")
            html_content.append(f"                <th>Queries</th>")
            html_content.append(f"                <th>Success</th>")
            html_content.append(f"            </tr>")
            
            # Original prediction
            with torch.no_grad():
                output = self.model(image)
                probs = torch.nn.functional.softmax(output, dim=1)
                _, pred = torch.max(output, dim=1)
                confidence = probs[0, pred].item()
                
            html_content.append(f"            <tr>")
            html_content.append(f"                <td>Original</td>")
            html_content.append(f"                <td>{self.class_names[pred.item()]}</td>")
            html_content.append(f"                <td>{confidence:.3f}</td>")
            html_content.append(f"                <td>0.0</td>")
            html_content.append(f"                <td>0.0</td>")
            html_content.append(f"                <td>1.0</td>")
            html_content.append(f"                <td>0</td>")
            html_content.append(f"                <td class='failure'>N/A</td>")
            html_content.append(f"            </tr>")
            
            # Attack results
            for attack_name, result in attack_results.items():
                success_class = "success" if result['success'] else "failure"
                html_content.append(f"            <tr>")
                html_content.append(f"                <td>{attack_name}</td>")
                html_content.append(f"                <td>{self.class_names[result['prediction']]}</td>")
                html_content.append(f"                <td>{result['confidence']:.3f}</td>")
                html_content.append(f"                <td>{result['l2_norm']:.3f}</td>")
                html_content.append(f"                <td>{result['linf_norm']:.3f}</td>")
                html_content.append(f"                <td>{result['ssim']:.3f}</td>")
                html_content.append(f"                <td>{result['queries']:.1f}</td>")
                html_content.append(f"                <td class='{success_class}'>{result['success']}</td>")
                html_content.append(f"            </tr>")
                
            html_content.append(f"        </table>")
            html_content.append(f"    </div>")
            
        # Close HTML
        html_content.append("</body>")
        html_content.append("</html>")
        
        # Write HTML file
        report_path = os.path.join(self.save_dir, save_path)
        with open(report_path, 'w') as f:
            f.write('\n'.join(html_content))
            
        print(f"Generated attack report at {report_path}")