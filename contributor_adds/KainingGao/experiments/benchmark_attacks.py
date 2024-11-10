"""Benchmarking various attack methods."""
#-Benchmarking various attack methods
import os
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..attacks.base_attack import BaseAttack
from ..attacks.fast_drop import FastDrop
from ..models.model_loader import load_model
from ..utils.data_loader import load_dataset


class AttackBenchmark:
    """Benchmark for comparing different adversarial attacks.
    
    This class provides functionality to benchmark multiple attack methods
    against various models on different datasets.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attacks: List[BaseAttack],
        device: torch.device = None,
        save_dir: str = "results",
        verbose: bool = True
    ):
        """Initialize the benchmark.
        
        Args:
            model: Target model to attack
            attacks: List of attack methods to benchmark
            device: Device to run the benchmark on
            save_dir: Directory to save results
            verbose: Whether to print progress information
        """
        self.model = model
        self.attacks = attacks
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.verbose = verbose
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def run_on_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        batch_size: int = 10,
        targeted: bool = False,
        norm_bounds: List[float] = None
    ) -> Dict:
        """Run benchmark on a dataset.
        
        Args:
            dataset_name: Name of the dataset to use
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            targeted: Whether to use targeted attacks
            norm_bounds: List of norm bounds to evaluate
            
        Returns:
            Dictionary containing benchmark results
        """
        # Set default norm bounds if not provided
        if norm_bounds is None:
            if dataset_name.lower() == "cifar10":
                norm_bounds = [2.0, 5.0, 10.0]
            else:  # ImageNet
                norm_bounds = [10.0, 20.0, 40.0]
                
        # Load dataset
        test_loader = load_dataset(dataset_name, batch_size=batch_size)
        
        # Get class names
        if dataset_name.lower() == "cifar10":
            class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"]
        else:
            class_names = list(range(1000))  # ImageNet classes
            
        # Track overall results
        results = {
            "dataset": dataset_name,
            "num_samples": num_samples,
            "targeted": targeted,
            "norm_bounds": norm_bounds,
            "attacks": {},
            "runtime": {}
        }
        
        # Get samples from the dataset
        images = []
        labels = []
        
        for data, target in test_loader:
            images.append(data)
            labels.append(target)
            if len(images) * batch_size >= num_samples:
                break
                
        # Concatenate batches
        images = torch.cat(images, dim=0)[:num_samples]
        labels = torch.cat(labels, dim=0)[:num_samples]
        
        # Generate target labels for targeted attacks
        if targeted:
            target_labels = (labels + 1 + torch.randint(0, len(class_names) - 1, (num_samples,))) % len(class_names)
        else:
            target_labels = None
            
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
            
        # Evaluate model accuracy on clean samples
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
        clean_accuracy = correct / num_samples
        results["clean_accuracy"] = clean_accuracy
        
        if self.verbose:
            print(f"Clean accuracy: {clean_accuracy:.4f}")
            
        # Benchmark each attack
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            results["attacks"][attack_name] = {}
            results["runtime"][attack_name] = {}
            
            for norm_bound in norm_bounds:
                # Set norm bound for the attack
                attack.norm_bound = norm_bound
                
                if self.verbose:
                    print(f"Running {attack_name} with norm bound {norm_bound}...")
                    
                # Measure runtime
                start_time = time.time()
                
                # Generate adversarial examples
                adv_images, metadata = attack(images, labels, target_labels)
                
                # Measure runtime
                runtime = time.time() - start_time
                
                # Evaluate model on adversarial examples
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                    
                    if targeted:
                        success = (predicted == target_labels).sum().item() / num_samples
                    else:
                        success = (predicted != labels).sum().item() / num_samples
                        
                # Calculate average perturbation norm
                perturbation_norm = torch.norm(
                    adv_images.reshape(num_samples, -1) - images.reshape(num_samples, -1),
                    p=2, dim=1
                ).mean().item()
                
                # Store results
                results["attacks"][attack_name][str(norm_bound)] = {
                    "success_rate": success,
                    "avg_perturbation": perturbation_norm,
                    "avg_queries": metadata.get("avg_queries", 0),
                    "metadata": {k: v for k, v in metadata.items() if isinstance(v, (int, float, bool, str))}
                }
                
                results["runtime"][attack_name][str(norm_bound)] = runtime
                
                if self.verbose:
                    print(f"  Success rate: {success:.4f}")
                    print(f"  Avg perturbation: {perturbation_norm:.4f}")
                    print(f"  Avg queries: {metadata.get('avg_queries', 0):.1f}")
                    print(f"  Runtime: {runtime:.2f}s")
                    
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(
            self.save_dir,
            f"benchmark_{dataset_name}_{num_samples}_{timestamp}.json"
        )
        
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
            
        if self.verbose:
            print(f"Results saved to {save_path}")
            
        return results
    
    def visualize_results(self, results: Dict = None, save_path: Optional[str] = None):
        """Visualize benchmark results.
        
        Args:
            results: Results dictionary (if None, loads the latest results)
            save_path: Path to save the visualization (if None, displays it)
        """
        if results is None:
            # Get the latest results file
            result_files = [
                os.path.join(self.save_dir, f) 
                for f in os.listdir(self.save_dir) 
                if f.startswith("benchmark_") and f.endswith(".json")
            ]
            if not result_files:
                raise FileNotFoundError("No benchmark results found")
                
            latest_file = max(result_files, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                results = json.load(f)
                
        # Create figure for success rate vs. norm bound
        plt.figure(figsize=(12, 8))
        
        attacks = list(results["attacks"].keys())
        norm_bounds = [float(nb) for nb in results["norm_bounds"]]
        
        for attack_name in attacks:
            success_rates = [
                results["attacks"][attack_name][str(nb)]["success_rate"]
                for nb in norm_bounds
            ]
            plt.plot(norm_bounds, success_rates, marker='o', label=attack_name)
            
        plt.xlabel("Norm Bound")
        plt.ylabel("Attack Success Rate")
        plt.title(f"Attack Success Rate vs. Norm Bound ({results['dataset']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        # Create figure for query efficiency
        plt.figure(figsize=(12, 8))
        
        for attack_name in attacks:
            if "avg_queries" in results["attacks"][attack_name][str(norm_bounds[0])]:
                queries = [
                    results["attacks"][attack_name][str(nb)]["avg_queries"]
                    for nb in norm_bounds
                ]
                plt.plot(norm_bounds, queries, marker='o', label=attack_name)
                
        plt.xlabel("Norm Bound")
        plt.ylabel("Average Queries")
        plt.title(f"Average Queries vs. Norm Bound ({results['dataset']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path.replace(".png", "_queries.png"))
        else:
            plt.show()
            

def main():
    """Run attack benchmark from command line."""
    parser = argparse.ArgumentParser(description="Benchmark adversarial attacks")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--targeted", action="store_true", help="Use targeted attacks")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create attacks
    attacks = [
        FastDrop(model, device=device, max_queries=100)
    ]
    
    # Initialize benchmark
    benchmark = AttackBenchmark(model, attacks, device, args.save_dir)
    
    # Run benchmark
    results = benchmark.run_on_dataset(
        args.dataset,
        num_samples=args.samples,
        batch_size=args.batch_size,
        targeted=args.targeted
    )
    
    # Visualize results
    benchmark.visualize_results(results)
    

if __name__ == "__main__":
    main()

import os
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..attacks.base_attack import BaseAttack
from ..attacks.fast_drop import FastDrop
from ..models.model_loader import load_model
from ..utils.data_loader import load_dataset


class AttackBenchmark:
    """Benchmark for comparing different adversarial attacks.
    
    This class provides functionality to benchmark multiple attack methods
    against various models on different datasets.
    """
    
    def __init__(
        self,
        model: nn.Module,
        attacks: List[BaseAttack],
        device: torch.device = None,
        save_dir: str = "results",
        verbose: bool = True
    ):
        """Initialize the benchmark.
        
        Args:
            model: Target model to attack
            attacks: List of attack methods to benchmark
            device: Device to run the benchmark on
            save_dir: Directory to save results
            verbose: Whether to print progress information
        """
        self.model = model
        self.attacks = attacks
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.verbose = verbose
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def run_on_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        batch_size: int = 10,
        targeted: bool = False,
        norm_bounds: List[float] = None
    ) -> Dict:
        """Run benchmark on a dataset.
        
        Args:
            dataset_name: Name of the dataset to use
            num_samples: Number of samples to evaluate
            batch_size: Batch size for evaluation
            targeted: Whether to use targeted attacks
            norm_bounds: List of norm bounds to evaluate
            
        Returns:
            Dictionary containing benchmark results
        """
        # Set default norm bounds if not provided
        if norm_bounds is None:
            if dataset_name.lower() == "cifar10":
                norm_bounds = [2.0, 5.0, 10.0]
            else:  # ImageNet
                norm_bounds = [10.0, 20.0, 40.0]
                
        # Load dataset
        test_loader = load_dataset(dataset_name, batch_size=batch_size)
        
        # Get class names
        if dataset_name.lower() == "cifar10":
            class_names = ["airplane", "automobile", "bird", "cat", "deer", 
                           "dog", "frog", "horse", "ship", "truck"]
        else:
            class_names = list(range(1000))  # ImageNet classes
            
        # Track overall results
        results = {
            "dataset": dataset_name,
            "num_samples": num_samples,
            "targeted": targeted,
            "norm_bounds": norm_bounds,
            "attacks": {},
            "runtime": {}
        }
        
        # Get samples from the dataset
        images = []
        labels = []
        
        for data, target in test_loader:
            images.append(data)
            labels.append(target)
            if len(images) * batch_size >= num_samples:
                break
                
        # Concatenate batches
        images = torch.cat(images, dim=0)[:num_samples]
        labels = torch.cat(labels, dim=0)[:num_samples]
        
        # Generate target labels for targeted attacks
        if targeted:
            target_labels = (labels + 1 + torch.randint(0, len(class_names) - 1, (num_samples,))) % len(class_names)
        else:
            target_labels = None
            
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        if target_labels is not None:
            target_labels = target_labels.to(self.device)
            
        # Evaluate model accuracy on clean samples
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            
        clean_accuracy = correct / num_samples
        results["clean_accuracy"] = clean_accuracy
        
        if self.verbose:
            print(f"Clean accuracy: {clean_accuracy:.4f}")
            
        # Benchmark each attack
        for attack in self.attacks:
            attack_name = attack.__class__.__name__
            results["attacks"][attack_name] = {}
            results["runtime"][attack_name] = {}
            
            for norm_bound in norm_bounds:
                # Set norm bound for the attack
                attack.norm_bound = norm_bound
                
                if self.verbose:
                    print(f"Running {attack_name} with norm bound {norm_bound}...")
                    
                # Measure runtime
                start_time = time.time()
                
                # Generate adversarial examples
                adv_images, metadata = attack(images, labels, target_labels)
                
                # Measure runtime
                runtime = time.time() - start_time
                
                # Evaluate model on adversarial examples
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = torch.max(outputs, 1)
                    
                    if targeted:
                        success = (predicted == target_labels).sum().item() / num_samples
                    else:
                        success = (predicted != labels).sum().item() / num_samples
                        
                # Calculate average perturbation norm
                perturbation_norm = torch.norm(
                    adv_images.reshape(num_samples, -1) - images.reshape(num_samples, -1),
                    p=2, dim=1
                ).mean().item()
                
                # Store results
                results["attacks"][attack_name][str(norm_bound)] = {
                    "success_rate": success,
                    "avg_perturbation": perturbation_norm,
                    "avg_queries": metadata.get("avg_queries", 0),
                    "metadata": {k: v for k, v in metadata.items() if isinstance(v, (int, float, bool, str))}
                }
                
                results["runtime"][attack_name][str(norm_bound)] = runtime
                
                if self.verbose:
                    print(f"  Success rate: {success:.4f}")
                    print(f"  Avg perturbation: {perturbation_norm:.4f}")
                    print(f"  Avg queries: {metadata.get('avg_queries', 0):.1f}")
                    print(f"  Runtime: {runtime:.2f}s")
                    
        # Save results
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(
            self.save_dir,
            f"benchmark_{dataset_name}_{num_samples}_{timestamp}.json"
        )
        
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
            
        if self.verbose:
            print(f"Results saved to {save_path}")
            
        return results
    
    def visualize_results(self, results: Dict = None, save_path: Optional[str] = None):
        """Visualize benchmark results.
        
        Args:
            results: Results dictionary (if None, loads the latest results)
            save_path: Path to save the visualization (if None, displays it)
        """
        if results is None:
            # Get the latest results file
            result_files = [
                os.path.join(self.save_dir, f) 
                for f in os.listdir(self.save_dir) 
                if f.startswith("benchmark_") and f.endswith(".json")
            ]
            if not result_files:
                raise FileNotFoundError("No benchmark results found")
                
            latest_file = max(result_files, key=os.path.getmtime)
            with open(latest_file, "r") as f:
                results = json.load(f)
                
        # Create figure for success rate vs. norm bound
        plt.figure(figsize=(12, 8))
        
        attacks = list(results["attacks"].keys())
        norm_bounds = [float(nb) for nb in results["norm_bounds"]]
        
        for attack_name in attacks:
            success_rates = [
                results["attacks"][attack_name][str(nb)]["success_rate"]
                for nb in norm_bounds
            ]
            plt.plot(norm_bounds, success_rates, marker='o', label=attack_name)
            
        plt.xlabel("Norm Bound")
        plt.ylabel("Attack Success Rate")
        plt.title(f"Attack Success Rate vs. Norm Bound ({results['dataset']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        # Create figure for query efficiency
        plt.figure(figsize=(12, 8))
        
        for attack_name in attacks:
            if "avg_queries" in results["attacks"][attack_name][str(norm_bounds[0])]:
                queries = [
                    results["attacks"][attack_name][str(nb)]["avg_queries"]
                    for nb in norm_bounds
                ]
                plt.plot(norm_bounds, queries, marker='o', label=attack_name)
                
        plt.xlabel("Norm Bound")
        plt.ylabel("Average Queries")
        plt.title(f"Average Queries vs. Norm Bound ({results['dataset']})")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path.replace(".png", "_queries.png"))
        else:
            plt.show()
            

def main():
    """Run attack benchmark from command line."""
    parser = argparse.ArgumentParser(description="Benchmark adversarial attacks")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size")
    parser.add_argument("--targeted", action="store_true", help="Use targeted attacks")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model, args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create attacks
    attacks = [
        FastDrop(model, device=device, max_queries=100)
    ]
    
    # Initialize benchmark
    benchmark = AttackBenchmark(model, attacks, device, args.save_dir)
    
    # Run benchmark
    results = benchmark.run_on_dataset(
        args.dataset,
        num_samples=args.samples,
        batch_size=args.batch_size,
        targeted=args.targeted
    )
    
    # Visualize results
    benchmark.visualize_results(results)
    

if __name__ == "__main__":
    main()