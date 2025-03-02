"""Interactive visualization of frequency domain attacks."""
#-Interactive visualization of frequency domain attacks.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Callable

from ..attacks.fast_drop import FastDrop
from ..utils.frequency_utils import fft2d, ifft2d, visualize_spectrum, square_zero


class FrequencyVisualizer:
    """Interactive visualization of frequency domain attacks.
    
    This class provides an interactive matplotlib interface to visualize
    how frequency domain modifications affect images and model predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        image: np.ndarray,
        class_names: List[str],
        device: torch.device = None
    ):
        """Initialize the visualizer.
        
        Args:
            model: Target model to visualize attacks against
            image: Image to visualize (numpy array, [H, W, C])
            class_names: List of class names for the model
            device: Device to run the model on
        """
        self.model = model
        self.image = image.copy()
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to appropriate size based on image dimensions
        self.square_max_num = 224 if self.image.shape[0] == 224 else 32
        
        # Compute frequency representation
        self.freq = fft2d(self.image)
        self.freq_mag = np.abs(self.freq)
        self.freq_phase = np.angle(self.freq)
        
        # Create attack instance for analysis
        self.attack = FastDrop(
            model=self.model,
            device=self.device,
            square_max_num=self.square_max_num
        )
        
        # For tracking modifications
        self.modified_freq = self.freq.copy()
        self.modified_image = self.image.copy()
        self.block_mask = np.ones(self.square_max_num // 2, dtype=bool)
        
        # Track predictions
        self._update_predictions()
        
        # Set up the figure
        self._setup_figure()
        
    def _setup_figure(self):
        """Set up the matplotlib figure and widgets."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Original image
        self.axes[0, 0].set_title('Original Image')
        self.im_orig = self.axes[0, 0].imshow(self.image)
        self.axes[0, 0].axis('off')
        
        # Original spectrum
        self.axes[0, 1].set_title('Original Spectrum')
        self.im_orig_spec = self.axes[0, 1].imshow(visualize_spectrum(self.freq))
        self.axes[0, 1].axis('off')
        
        # Original prediction
        self.axes[0, 2].set_title('Original Prediction')
        self.axes[0, 2].axis('off')
        self.orig_pred_text = self.axes[0, 2].text(
            0.5, 0.5, 
            f"{self.class_names[self.orig_pred]}\n{self.orig_confidence:.3f}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16
        )
        
        # Modified image
        self.axes[1, 0].set_title('Modified Image')
        self.im_mod = self.axes[1, 0].imshow(self.modified_image)
        self.axes[1, 0].axis('off')
        
        # Modified spectrum
        self.axes[1, 1].set_title('Modified Spectrum')
        self.im_mod_spec = self.axes[1, 1].imshow(visualize_spectrum(self.modified_freq))
        self.axes[1, 1].axis('off')
        
        # Modified prediction
        self.axes[1, 2].set_title('Modified Prediction')
        self.axes[1, 2].axis('off')
        self.mod_pred_text = self.axes[1, 2].text(
            0.5, 0.5, 
            f"{self.class_names[self.mod_pred]}\n{self.mod_confidence:.3f}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            color='green' if self.mod_pred != self.orig_pred else 'black'
        )
        
        # Add slider for frequency block selection
        self.ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider = Slider(
            self.ax_slider, 'Block Index', 
            0, self.square_max_num // 2 - 1, 
            valinit=0, valstep=1
        )
        self.slider.on_changed(self._update_slider)
        
        # Add button for toggling frequency blocks
        self.ax_toggle = plt.axes([0.25, 0.05, 0.15, 0.05])
        self.toggle_button = Button(self.ax_toggle, 'Toggle Block')
        self.toggle_button.on_clicked(self._toggle_block)
        
        # Add button for resetting
        self.ax_reset = plt.axes([0.45, 0.05, 0.15, 0.05])
        self.reset_button = Button(self.ax_reset, 'Reset')
        self.reset_button.on_clicked(self._reset)
        
        # Add button for automatic attack
        self.ax_attack = plt.axes([0.65, 0.05, 0.15, 0.05])
        self.attack_button = Button(self.ax_attack, 'Auto Attack')
        self.attack_button.on_clicked(self._auto_attack)
        
        # Add text for difference metrics
        self.ax_metrics = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_metrics.axis('off')
        self.metrics_text = self.ax_metrics.text(
            0.5, 0.5, 
            f"L2 Difference: {self._compute_l2_diff():.2f} | Blocks modified: 0/{self.square_max_num//2}",
            horizontalalignment='center',
            verticalalignment='center'
        )
        
        plt.tight_layout()
        
    def _update_slider(self, val):
        """Handle slider value changes."""
        # Just update the display, don't modify the image
        block_index = int(self.slider.val)
        
        # Highlight the selected block in the original spectrum
        orig_spec = visualize_spectrum(self.freq)
        mod_spec = visualize_spectrum(self.modified_freq)
        
        # Create a copy of the visualizations
        orig_spec_highlight = orig_spec.copy()
        mod_spec_highlight = mod_spec.copy()
        
        # Add red highlight to the selected frequency block
        h, w = self.freq.shape[:2]
        
        # Top horizontal line
        orig_spec_highlight[block_index, block_index:h-block_index, :] = [255, 0, 0]
        orig_spec_highlight[h-1-block_index, block_index:h-block_index, :] = [255, 0, 0]
        orig_spec_highlight[block_index:h-block_index, block_index, :] = [255, 0, 0]
        orig_spec_highlight[block_index:h-block_index, h-1-block_index, :] = [255, 0, 0]
        
        # Do the same for modified spectrum
        mod_spec_highlight[block_index, block_index:h-block_index, :] = [255, 0, 0]
        mod_spec_highlight[h-1-block_index, block_index:h-block_index, :] = [255, 0, 0]
        mod_spec_highlight[block_index:h-block_index, block_index, :] = [255, 0, 0]
        mod_spec_highlight[block_index:h-block_index, h-1-block_index, :] = [255, 0, 0]
        
        # Update the images
        self.im_orig_spec.set_data(orig_spec_highlight)
        self.im_mod_spec.set_data(mod_spec_highlight)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        
    def _toggle_block(self, event):
        """Toggle the selected frequency block on/off."""
        block_index = int(self.slider.val)
        
        # Toggle the block in our tracking array
        self.block_mask[block_index] = not self.block_mask[block_index]
        
        # Recompute the modified frequency and image
        self._recompute_modified_image()
        
        # Update predictions
        self._update_predictions()
        
        # Update the display
        self._update_display()
        
    def _reset(self, event):
        """Reset to the original image."""
        self.modified_freq = self.freq.copy()
        self.modified_image = self.image.copy()
        self.block_mask = np.ones(self.square_max_num // 2, dtype=bool)
        
        # Update predictions
        self._update_predictions()
        
        # Update the display
        self._update_display()
        
    def _auto_attack(self, event):
        """Run automatic attack to find minimal blocks to modify."""
        # Reset first
        self._reset(None)
        
        # Convert image for attack
        img_tensor = torch.tensor(
            self.image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Run attack on a single image
        label_tensor = torch.tensor([self.orig_pred], dtype=torch.long)
        adv_images, metadata = self.attack(img_tensor, label_tensor)
        
        # Get the result
        if metadata['success'][0]:
            # Convert back to numpy
            adv_img = adv_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            adv_img = (adv_img * 255).astype(np.uint8)
            
            # Update modified image and compute its frequency
            self.modified_image = adv_img
            self.modified_freq = fft2d(self.modified_image)
            
            # Update block mask (approximate based on diff)
            freq_diff = np.abs(self.freq_mag - np.abs(self.modified_freq))
            for i in range(self.square_max_num // 2):
                diff_sum = np.sum(freq_diff[i, i:self.square_max_num-i, :]) + \
                           np.sum(freq_diff[self.square_max_num-1-i, i:self.square_max_num-i, :]) + \
                           np.sum(freq_diff[i+1:self.square_max_num-1-i, i, :]) + \
                           np.sum(freq_diff[i+1:self.square_max_num-1-i, self.square_max_num-1-i, :])
                
                self.block_mask[i] = (diff_sum < 1e-5)
            
            # Update predictions
            self._update_predictions()
            
            # Update display
            self._update_display()
        else:
            print("Attack failed!")
        
    def _recompute_modified_image(self):
        """Recompute the modified image based on block mask."""
        # Start with original frequency
        self.modified_freq = self.freq.copy()
        freq_mag = np.abs(self.modified_freq)
        freq_phase = np.angle(self.modified_freq)
        
        # Apply mask
        for i in range(len(self.block_mask)):
            if not self.block_mask[i]:
                freq_mag = square_zero(freq_mag, i, self.square_max_num)
        
        # Recombine magnitude and phase
        self.modified_freq = freq_mag * np.exp(1j * freq_phase)
        
        # Convert back to spatial domain
        self.modified_image = ifft2d(self.modified_freq)
        
    def _update_predictions(self):
        """Update model predictions for original and modified images."""
        # Prepare tensors
        orig_tensor = torch.tensor(
            self.image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        mod_tensor = torch.tensor(
            self.modified_image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Normalize
        orig_tensor = self.attack.preprocess_image(orig_tensor.to(self.device))
        mod_tensor = self.attack.preprocess_image(mod_tensor.to(self.device))
        
        # Get predictions
        with torch.no_grad():
            orig_output = self.model(orig_tensor)
            mod_output = self.model(mod_tensor)
            
            # Apply softmax
            orig_probs = torch.nn.functional.softmax(orig_output, dim=1)
            mod_probs = torch.nn.functional.softmax(mod_output, dim=1)
            
            # Get top prediction
            _, self.orig_pred = torch.max(orig_output, dim=1)
            _, self.mod_pred = torch.max(mod_output, dim=1)
            
            # Get confidence
            self.orig_confidence = orig_probs[0, self.orig_pred].item()
            self.mod_confidence = mod_probs[0, self.mod_pred].item()
            
    def _update_display(self):
        """Update the display with current state."""
        # Update images
        self.im_mod.set_data(self.modified_image)
        self.im_mod_spec.set_data(visualize_spectrum(self.modified_freq))
        
        # Update prediction text
        self.mod_pred_text.set_text(f"{self.class_names[self.mod_pred]}\n{self.mod_confidence:.3f}")
        self.mod_pred_text.set_color('red' if self.mod_pred != self.orig_pred else 'black')
        
        # Update metrics
        num_modified = np.sum(~self.block_mask)
        self.metrics_text.set_text(
            f"L2 Difference: {self._compute_l2_diff():.2f} | "
            f"Blocks modified: {num_modified}/{self.square_max_num//2}"
        )
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _compute_l2_diff(self):
        """Compute L2 difference between original and modified images."""
        orig_norm = self.image.astype(np.float32) / 255.0
        mod_norm = self.modified_image.astype(np.float32) / 255.0
        return np.sqrt(np.sum((orig_norm - mod_norm) ** 2))
    
    def show(self):
        """Show the visualization."""
        plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Callable

from ..attacks.fast_drop import FastDrop
from ..utils.frequency_utils import fft2d, ifft2d, visualize_spectrum, square_zero


class FrequencyVisualizer:
    """Interactive visualization of frequency domain attacks.
    
    This class provides an interactive matplotlib interface to visualize
    how frequency domain modifications affect images and model predictions.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        image: np.ndarray,
        class_names: List[str],
        device: torch.device = None
    ):
        """Initialize the visualizer.
        
        Args:
            model: Target model to visualize attacks against
            image: Image to visualize (numpy array, [H, W, C])
            class_names: List of class names for the model
            device: Device to run the model on
        """
        self.model = model
        self.image = image.copy()
        self.class_names = class_names
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to appropriate size based on image dimensions
        self.square_max_num = 224 if self.image.shape[0] == 224 else 32
        
        # Compute frequency representation
        self.freq = fft2d(self.image)
        self.freq_mag = np.abs(self.freq)
        self.freq_phase = np.angle(self.freq)
        
        # Create attack instance for analysis
        self.attack = FastDrop(
            model=self.model,
            device=self.device,
            square_max_num=self.square_max_num
        )
        
        # For tracking modifications
        self.modified_freq = self.freq.copy()
        self.modified_image = self.image.copy()
        self.block_mask = np.ones(self.square_max_num // 2, dtype=bool)
        
        # Track predictions
        self._update_predictions()
        
        # Set up the figure
        self._setup_figure()
        
    def _setup_figure(self):
        """Set up the matplotlib figure and widgets."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Original image
        self.axes[0, 0].set_title('Original Image')
        self.im_orig = self.axes[0, 0].imshow(self.image)
        self.axes[0, 0].axis('off')
        
        # Original spectrum
        self.axes[0, 1].set_title('Original Spectrum')
        self.im_orig_spec = self.axes[0, 1].imshow(visualize_spectrum(self.freq))
        self.axes[0, 1].axis('off')
        
        # Original prediction
        self.axes[0, 2].set_title('Original Prediction')
        self.axes[0, 2].axis('off')
        self.orig_pred_text = self.axes[0, 2].text(
            0.5, 0.5, 
            f"{self.class_names[self.orig_pred]}\n{self.orig_confidence:.3f}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16
        )
        
        # Modified image
        self.axes[1, 0].set_title('Modified Image')
        self.im_mod = self.axes[1, 0].imshow(self.modified_image)
        self.axes[1, 0].axis('off')
        
        # Modified spectrum
        self.axes[1, 1].set_title('Modified Spectrum')
        self.im_mod_spec = self.axes[1, 1].imshow(visualize_spectrum(self.modified_freq))
        self.axes[1, 1].axis('off')
        
        # Modified prediction
        self.axes[1, 2].set_title('Modified Prediction')
        self.axes[1, 2].axis('off')
        self.mod_pred_text = self.axes[1, 2].text(
            0.5, 0.5, 
            f"{self.class_names[self.mod_pred]}\n{self.mod_confidence:.3f}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            color='green' if self.mod_pred != self.orig_pred else 'black'
        )
        
        # Add slider for frequency block selection
        self.ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.slider = Slider(
            self.ax_slider, 'Block Index', 
            0, self.square_max_num // 2 - 1, 
            valinit=0, valstep=1
        )
        self.slider.on_changed(self._update_slider)
        
        # Add button for toggling frequency blocks
        self.ax_toggle = plt.axes([0.25, 0.05, 0.15, 0.05])
        self.toggle_button = Button(self.ax_toggle, 'Toggle Block')
        self.toggle_button.on_clicked(self._toggle_block)
        
        # Add button for resetting
        self.ax_reset = plt.axes([0.45, 0.05, 0.15, 0.05])
        self.reset_button = Button(self.ax_reset, 'Reset')
        self.reset_button.on_clicked(self._reset)
        
        # Add button for automatic attack
        self.ax_attack = plt.axes([0.65, 0.05, 0.15, 0.05])
        self.attack_button = Button(self.ax_attack, 'Auto Attack')
        self.attack_button.on_clicked(self._auto_attack)
        
        # Add text for difference metrics
        self.ax_metrics = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.ax_metrics.axis('off')
        self.metrics_text = self.ax_metrics.text(
            0.5, 0.5, 
            f"L2 Difference: {self._compute_l2_diff():.2f} | Blocks modified: 0/{self.square_max_num//2}",
            horizontalalignment='center',
            verticalalignment='center'
        )
        
        plt.tight_layout()
        
    def _update_slider(self, val):
        """Handle slider value changes."""
        # Just update the display, don't modify the image
        block_index = int(self.slider.val)
        
        # Highlight the selected block in the original spectrum
        orig_spec = visualize_spectrum(self.freq)
        mod_spec = visualize_spectrum(self.modified_freq)
        
        # Create a copy of the visualizations
        orig_spec_highlight = orig_spec.copy()
        mod_spec_highlight = mod_spec.copy()
        
        # Add red highlight to the selected frequency block
        h, w = self.freq.shape[:2]
        
        # Top horizontal line
        orig_spec_highlight[block_index, block_index:h-block_index, :] = [255, 0, 0]
        orig_spec_highlight[h-1-block_index, block_index:h-block_index, :] = [255, 0, 0]
        orig_spec_highlight[block_index:h-block_index, block_index, :] = [255, 0, 0]
        orig_spec_highlight[block_index:h-block_index, h-1-block_index, :] = [255, 0, 0]
        
        # Do the same for modified spectrum
        mod_spec_highlight[block_index, block_index:h-block_index, :] = [255, 0, 0]
        mod_spec_highlight[h-1-block_index, block_index:h-block_index, :] = [255, 0, 0]
        mod_spec_highlight[block_index:h-block_index, block_index, :] = [255, 0, 0]
        mod_spec_highlight[block_index:h-block_index, h-1-block_index, :] = [255, 0, 0]
        
        # Update the images
        self.im_orig_spec.set_data(orig_spec_highlight)
        self.im_mod_spec.set_data(mod_spec_highlight)
        
        # Redraw the figure
        self.fig.canvas.draw_idle()
        
    def _toggle_block(self, event):
        """Toggle the selected frequency block on/off."""
        block_index = int(self.slider.val)
        
        # Toggle the block in our tracking array
        self.block_mask[block_index] = not self.block_mask[block_index]
        
        # Recompute the modified frequency and image
        self._recompute_modified_image()
        
        # Update predictions
        self._update_predictions()
        
        # Update the display
        self._update_display()
        
    def _reset(self, event):
        """Reset to the original image."""
        self.modified_freq = self.freq.copy()
        self.modified_image = self.image.copy()
        self.block_mask = np.ones(self.square_max_num // 2, dtype=bool)
        
        # Update predictions
        self._update_predictions()
        
        # Update the display
        self._update_display()
        
    def _auto_attack(self, event):
        """Run automatic attack to find minimal blocks to modify."""
        # Reset first
        self._reset(None)
        
        # Convert image for attack
        img_tensor = torch.tensor(
            self.image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Run attack on a single image
        label_tensor = torch.tensor([self.orig_pred], dtype=torch.long)
        adv_images, metadata = self.attack(img_tensor, label_tensor)
        
        # Get the result
        if metadata['success'][0]:
            # Convert back to numpy
            adv_img = adv_images[0].detach().cpu().numpy().transpose(1, 2, 0)
            adv_img = (adv_img * 255).astype(np.uint8)
            
            # Update modified image and compute its frequency
            self.modified_image = adv_img
            self.modified_freq = fft2d(self.modified_image)
            
            # Update block mask (approximate based on diff)
            freq_diff = np.abs(self.freq_mag - np.abs(self.modified_freq))
            for i in range(self.square_max_num // 2):
                diff_sum = np.sum(freq_diff[i, i:self.square_max_num-i, :]) + \
                           np.sum(freq_diff[self.square_max_num-1-i, i:self.square_max_num-i, :]) + \
                           np.sum(freq_diff[i+1:self.square_max_num-1-i, i, :]) + \
                           np.sum(freq_diff[i+1:self.square_max_num-1-i, self.square_max_num-1-i, :])
                
                self.block_mask[i] = (diff_sum < 1e-5)
            
            # Update predictions
            self._update_predictions()
            
            # Update display
            self._update_display()
        else:
            print("Attack failed!")
        
    def _recompute_modified_image(self):
        """Recompute the modified image based on block mask."""
        # Start with original frequency
        self.modified_freq = self.freq.copy()
        freq_mag = np.abs(self.modified_freq)
        freq_phase = np.angle(self.modified_freq)
        
        # Apply mask
        for i in range(len(self.block_mask)):
            if not self.block_mask[i]:
                freq_mag = square_zero(freq_mag, i, self.square_max_num)
        
        # Recombine magnitude and phase
        self.modified_freq = freq_mag * np.exp(1j * freq_phase)
        
        # Convert back to spatial domain
        self.modified_image = ifft2d(self.modified_freq)
        
    def _update_predictions(self):
        """Update model predictions for original and modified images."""
        # Prepare tensors
        orig_tensor = torch.tensor(
            self.image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        mod_tensor = torch.tensor(
            self.modified_image.transpose(2, 0, 1) / 255.0,
            dtype=torch.float32
        ).unsqueeze(0)
        
        # Normalize
        orig_tensor = self.attack.preprocess_image(orig_tensor.to(self.device))
        mod_tensor = self.attack.preprocess_image(mod_tensor.to(self.device))
        
        # Get predictions
        with torch.no_grad():
            orig_output = self.model(orig_tensor)
            mod_output = self.model(mod_tensor)
            
            # Apply softmax
            orig_probs = torch.nn.functional.softmax(orig_output, dim=1)
            mod_probs = torch.nn.functional.softmax(mod_output, dim=1)
            
            # Get top prediction
            _, self.orig_pred = torch.max(orig_output, dim=1)
            _, self.mod_pred = torch.max(mod_output, dim=1)
            
            # Get confidence
            self.orig_confidence = orig_probs[0, self.orig_pred].item()
            self.mod_confidence = mod_probs[0, self.mod_pred].item()
            
    def _update_display(self):
        """Update the display with current state."""
        # Update images
        self.im_mod.set_data(self.modified_image)
        self.im_mod_spec.set_data(visualize_spectrum(self.modified_freq))
        
        # Update prediction text
        self.mod_pred_text.set_text(f"{self.class_names[self.mod_pred]}\n{self.mod_confidence:.3f}")
        self.mod_pred_text.set_color('red' if self.mod_pred != self.orig_pred else 'black')
        
        # Update metrics
        num_modified = np.sum(~self.block_mask)
        self.metrics_text.set_text(
            f"L2 Difference: {self._compute_l2_diff():.2f} | "
            f"Blocks modified: {num_modified}/{self.square_max_num//2}"
        )
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def _compute_l2_diff(self):
        """Compute L2 difference between original and modified images."""
        orig_norm = self.image.astype(np.float32) / 255.0
        mod_norm = self.modified_image.astype(np.float32) / 255.0
        return np.sqrt(np.sum((orig_norm - mod_norm) ** 2))
    
    def show(self):
        """Show the visualization."""
        plt.show()