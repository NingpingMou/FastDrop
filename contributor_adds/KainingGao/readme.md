#-AdversarialBench

A comprehensive framework for testing, visualizing, and comparing adversarial attacks and defenses on deep neural networks.

## Features

- **Multiple Attack Methods**: FastDrop (frequency-based), plus interfaces for other popular attacks
- **Defense Evaluation**: Test various defense mechanisms including frequency-domain defenses
- **Visualization Tools**: Interactive visualizations to understand attack effects in both pixel and frequency domains
- **Benchmarking**: Standardized evaluation metrics for comparing attack efficiency and efficacy
- **Model Zoo**: Support for popular model architectures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AdversarialBench.git
cd AdversarialBench

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from adversarial_bench import AttackBenchmark, FastDrop
from adversarial_bench.models import load_model
from adversarial_bench.visualization import FrequencyVisualizer

# Load a model
model = load_model("resnet18", "cifar10")

# Create an attack
attack = FastDrop(model)

# Run the attack on a single image
image = load_image("path/to/image.png")
adversarial_image, metadata = attack(image, label)

# Visualize the attack in frequency domain
visualizer = FrequencyVisualizer(model, image.numpy())
visualizer.show()

# Benchmark multiple attacks
benchmark = AttackBenchmark(model, [attack])
results = benchmark.run_on_dataset("cifar10", num_samples=100)
benchmark.visualize_results()
```

## Key Components

### Attack Methods

The core of this framework is the implementation of the FastDrop attack, which operates in the frequency domain using Fast Fourier Transform (FFT) to create adversarial examples with minimal queries.

- **FastDrop**: A novel decision-based adversarial attack that drops information in the frequency domain
- **Base Attack Interface**: Extensible interface for implementing other attacks

### Defense Methods

- **Frequency-Based Defenses**: Designed specifically to counter frequency-domain attacks
- **Input Transformations**: Various preprocessing techniques like JPEG compression, bit depth reduction, etc.

### Visualization Tools

- **Frequency Visualizer**: Interactive tool to visualize how frequency components affect model predictions
- **Attack Comparisons**: Side-by-side visual comparison of different attack methods

### Benchmarking

- **Standardized Metrics**: Success rate, perturbation size, query efficiency
- **Cross-Model Evaluation**: Test attacks across different model architectures
- **HTML Reports**: Generate comprehensive visual reports

## Documentation

For detailed documentation, see the `docs/` directory or visit our [documentation site](https://adversarial-bench.readthedocs.io/).

## Examples

Check out the `examples/` directory for more usage examples:

- `examples/fastdrop_demo.py`: Interactive demo of the FastDrop attack
- `examples/benchmark_attacks.py`: Compare multiple attack methods
- `examples/defense_evaluation.py`: Evaluate different defense mechanisms

## Citing

If you use this framework in your research, please cite:

```
@inproceedings{DBLP:conf/ijcai/MouZWGG22,
author    = {Ningping Mou and
            Baolin Zheng and
            Qian Wang and
            Yunjie Ge and
            Binqing Guo},
title     = {A Few Seconds Can Change Everything: Fast Decision-based Attacks against
            DNNs},
booktitle = {Proceedings of the Thirty-First International Joint Conference on
            Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
            2022},
year      = {2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


A comprehensive framework for testing, visualizing, and comparing adversarial attacks and defenses on deep neural networks.

## Features

- **Multiple Attack Methods**: FastDrop (frequency-based), plus interfaces for other popular attacks
- **Defense Evaluation**: Test various defense mechanisms including frequency-domain defenses
- **Visualization Tools**: Interactive visualizations to understand attack effects in both pixel and frequency domains
- **Benchmarking**: Standardized evaluation metrics for comparing attack efficiency and efficacy
- **Model Zoo**: Support for popular model architectures

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AdversarialBench.git
cd AdversarialBench

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from adversarial_bench import AttackBenchmark, FastDrop
from adversarial_bench.models import load_model
from adversarial_bench.visualization import FrequencyVisualizer

# Load a model
model = load_model("resnet18", "cifar10")

# Create an attack
attack = FastDrop(model)

# Run the attack on a single image
image = load_image("path/to/image.png")
adversarial_image, metadata = attack(image, label)

# Visualize the attack in frequency domain
visualizer = FrequencyVisualizer(model, image.numpy())
visualizer.show()

# Benchmark multiple attacks
benchmark = AttackBenchmark(model, [attack])
results = benchmark.run_on_dataset("cifar10", num_samples=100)
benchmark.visualize_results()
```

## Key Components

### Attack Methods

The core of this framework is the implementation of the FastDrop attack, which operates in the frequency domain using Fast Fourier Transform (FFT) to create adversarial examples with minimal queries.

- **FastDrop**: A novel decision-based adversarial attack that drops information in the frequency domain
- **Base Attack Interface**: Extensible interface for implementing other attacks

### Defense Methods

- **Frequency-Based Defenses**: Designed specifically to counter frequency-domain attacks
- **Input Transformations**: Various preprocessing techniques like JPEG compression, bit depth reduction, etc.

### Visualization Tools

- **Frequency Visualizer**: Interactive tool to visualize how frequency components affect model predictions
- **Attack Comparisons**: Side-by-side visual comparison of different attack methods

### Benchmarking

- **Standardized Metrics**: Success rate, perturbation size, query efficiency
- **Cross-Model Evaluation**: Test attacks across different model architectures
- **HTML Reports**: Generate comprehensive visual reports

## Documentation

For detailed documentation, see the `docs/` directory or visit our [documentation site](https://adversarial-bench.readthedocs.io/).

## Examples

Check out the `examples/` directory for more usage examples:

- `examples/fastdrop_demo.py`: Interactive demo of the FastDrop attack
- `examples/benchmark_attacks.py`: Compare multiple attack methods
- `examples/defense_evaluation.py`: Evaluate different defense mechanisms

## Citing

If you use this framework in your research, please cite:

```
@inproceedings{DBLP:conf/ijcai/MouZWGG22,
author    = {Ningping Mou and
            Baolin Zheng and
            Qian Wang and
            Yunjie Ge and
            Binqing Guo},
title     = {A Few Seconds Can Change Everything: Fast Decision-based Attacks against
            DNNs},
booktitle = {Proceedings of the Thirty-First International Joint Conference on
            Artificial Intelligence, {IJCAI} 2022, Vienna, Austria, 23-29 July
            2022},
year      = {2022}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request