"""Interactive demo of FastDrop attack visualization."""
#-Interactive demo of FastDrop attack visualization.
import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from adversarial_bench.models.model_loader import load_model
from adversarial_bench.utils.data_loader import get_dataset_info
from adversarial_bench.visualization.frequency_visualizer import FrequencyVisualizer


def main():
    """Run the interactive visualization demo."""
    parser = argparse.ArgumentParser(description="FastDrop Attack Visualization")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "imagenet"], 
                        help="Dataset to use")
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model architecture")
    parser.add_argument("--image", type=str, default=None, 
                        help="Path to a specific image (optional)")
    parser.add_argument("--random-seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    resolution = dataset_info["resolution"]
    mean = dataset_info["mean"]
    std = dataset_info["std"]
    class_names = dataset_info["class_names"]
    
    # Load model
    print(f"Loading {args.model} model trained on {args.dataset}...")
    model = load_model(args.model, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load or select an image
    if args.image:
        # Load user-provided image
        print(f"Loading image from {args.image}...")
        image = Image.open(args.image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        norm_transform = transforms.Normalize(mean, std)
        input_tensor = norm_transform(img_tensor.clone()).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = pred.item()
            
        print(f"Model predicts: {class_names[pred_class]}")
        
        # Convert to numpy for visualization
        img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
    else:
        # Use a sample from the dataset
        from adversarial_bench.utils.data_loader import load_dataset
        
        print(f"Loading a sample image from {args.dataset}...")
        dataloader = load_dataset(args.dataset, batch_size=1, train=False)
        img_tensor, label = next(iter(dataloader))
        
        # Get prediction
        input_tensor = img_tensor.clone().to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = pred.item()
            
        print(f"True class: {class_names[label.item()]}, Model predicts: {class_names[pred_class]}")
        
        # Convert to numpy for visualization
        img_tensor = img_tensor.clone()
        for i in range(3):
            img_tensor[0, i] = img_tensor[0, i] * std[i] + mean[i]
            
        img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create and show visualizer
    print("Launching interactive visualizer... (close the figure window to exit)")
    visualizer = FrequencyVisualizer(model, img_np, class_names, device)
    visualizer.show()


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from adversarial_bench.models.model_loader import load_model
from adversarial_bench.utils.data_loader import get_dataset_info
from adversarial_bench.visualization.frequency_visualizer import FrequencyVisualizer


def main():
    """Run the interactive visualization demo."""
    parser = argparse.ArgumentParser(description="FastDrop Attack Visualization")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["cifar10", "imagenet"], 
                        help="Dataset to use")
    parser.add_argument("--model", type=str, default="resnet18", 
                        help="Model architecture")
    parser.add_argument("--image", type=str, default=None, 
                        help="Path to a specific image (optional)")
    parser.add_argument("--random-seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    resolution = dataset_info["resolution"]
    mean = dataset_info["mean"]
    std = dataset_info["std"]
    class_names = dataset_info["class_names"]
    
    # Load model
    print(f"Loading {args.model} model trained on {args.dataset}...")
    model = load_model(args.model, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load or select an image
    if args.image:
        # Load user-provided image
        print(f"Loading image from {args.image}...")
        image = Image.open(args.image).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor()
        ])
        img_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        norm_transform = transforms.Normalize(mean, std)
        input_tensor = norm_transform(img_tensor.clone()).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = pred.item()
            
        print(f"Model predicts: {class_names[pred_class]}")
        
        # Convert to numpy for visualization
        img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
    else:
        # Use a sample from the dataset
        from adversarial_bench.utils.data_loader import load_dataset
        
        print(f"Loading a sample image from {args.dataset}...")
        dataloader = load_dataset(args.dataset, batch_size=1, train=False)
        img_tensor, label = next(iter(dataloader))
        
        # Get prediction
        input_tensor = img_tensor.clone().to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = pred.item()
            
        print(f"True class: {class_names[label.item()]}, Model predicts: {class_names[pred_class]}")
        
        # Convert to numpy for visualization
        img_tensor = img_tensor.clone()
        for i in range(3):
            img_tensor[0, i] = img_tensor[0, i] * std[i] + mean[i]
            
        img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
    
    # Create and show visualizer
    print("Launching interactive visualizer... (close the figure window to exit)")
    visualizer = FrequencyVisualizer(model, img_np, class_names, device)
    visualizer.show()


if __name__ == "__main__":
    main()