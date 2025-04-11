# LDPC Neural Network Decoder

A PyTorch implementation of a neural network-based LDPC decoder with residual connections and weight sharing.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Training Parameters](#training-parameters)
- [Performance](#performance)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Residual Weight-Sharing Architecture**
  - Efficient parameter sharing across iterations
  - Residual connections for improved training
  - GPU-accelerated computations

- **Training Framework**
  - Custom data generation with configurable SNR
  - Flexible training parameters
  - Comprehensive evaluation metrics

- **Performance Optimizations**
  - Memory-efficient implementation
  - Vectorized operations
  - GPU acceleration

## Installation

```bash
git clone https://github.com/yourusername/LDPC-NeuralNetwork-Decoder.git
cd LDPC-NeuralNetwork-Decoder
pip install -r requirements.txt
```

## Usage

### Training

```python
from neural_decoder import ResidualWeightSharingDecoder
from training import train_model

# Initialize decoder
decoder = ResidualWeightSharingDecoder(
    n=128,  # codeword length
    k=64,   # message length
    iterations=10,
    residual_depth=2,
    expansion_factor=4
)

# Train model
train_model(
    decoder,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    momentum=0.9,
    weight_decay=1e-2,
    snr_range=(-1, 8)
)
```

### Evaluation

```python
from evaluation import evaluate_model

# Evaluate model
results = evaluate_model(
    decoder,
    test_data,
    snr_values=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
)

# Print results
print(f"BER: {results['ber']}")
print(f"FER: {results['fer']}")
```

## Training Parameters

- **Model Architecture**
  - Iterations: 10
  - Residual Depth: 2
  - Expansion Factor: 4

- **Training Configuration**
  - Epochs: 50
  - Batch Size: 64
  - Learning Rate: 1e-3
  - Momentum: 0.9
  - Weight Decay: 1e-2

- **Data Generation**
  - SNR Range: -1 to 8 dB
  - Data Split: 80/10/10 (train/val/test)
  - Total Samples: 320,000

## Performance

- **Memory Efficiency**
  - Parameter count: ~1.5M
  - Memory usage: ~6GB (training)
  - Batch processing: ~100ms

- **Training Stability**
  - Loss convergence: ~20 epochs
  - Gradient stability: RMSprop optimizer
  - Learning rate scheduling: Cosine annealing

## Documentation

For detailed documentation, see the [docs](docs/) directory:

- [Architecture Overview](docs/architecture/overview.md)
- [Implementation Details](docs/implementation/core_components.md)
- [Training Guide](docs/training/parameters.md)
- [Results Analysis](docs/results/metrics.md)
- [API Reference](docs/usage/api.md)

## Project Structure

```
LDPC-NeuralNetwork-Decoder/
├── docs/                    # Documentation
│   ├── architecture/        # Architecture details
│   ├── implementation/      # Implementation guides
│   ├── training/           # Training documentation
│   ├── results/            # Results and analysis
│   └── usage/              # Usage guides
├── src/                    # Source code
│   ├── neural_decoder.py   # Main decoder implementation
│   ├── training.py         # Training utilities
│   └── evaluation.py       # Evaluation tools
├── tests/                  # Test files
├── data/                   # Data and results
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 