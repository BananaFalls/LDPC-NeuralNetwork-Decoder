# LDPC Neural Network Decoder

A neural network-based LDPC decoder implementation using PyTorch, featuring weight sharing and residual connections.

## Overview

This project implements a neural LDPC decoder that combines traditional message passing with neural network components. The decoder uses weight sharing across circulant blocks and incorporates residual connections to improve training stability and performance.

## Key Features

- **Weight Sharing**: Weights are shared across circulant blocks corresponding to the same base matrix element
- **Residual Connections**: Combines outputs from previous iterations to improve gradient flow
- **Neural Message Processing**: Uses a neural network for message processing
- **Efficient Implementation**: Optimized for both training and inference
- **5G LDPC Code Support**: Compatible with 5G LDPC codes

## Architecture

The decoder consists of several key components:

1. **Base Matrix Expansion**
   - Expands base matrix into full H matrix using circulants
   - Supports various expansion factors (z)

2. **Message Passing System**
   - Variable node updates with shared weights
   - Check node updates with min-sum approximation
   - Residual connections for improved performance

3. **Neural Network Components**
   - Message processor with linear layers and ReLU activation
   - Shared weights for variable-to-check and check-to-variable messages
   - Residual connection weights

## Training Process

The training process includes:

1. **Data Generation**
   - Batch size: 64
   - SNR range: [-1, 8] dB
   - Total examples: 320,000 (training)
   - Split ratio: 80% train, 10% validation, 10% test

2. **Training Configuration**
   - Number of epochs: 50
   - Learning rate: 1e-3
   - Number of iterations: 10
   - Residual depth: 2

3. **Optimization**
   - SGD optimizer with momentum (0.9)
   - L2 regularization (weight decay: 1e-2)
   - Learning rate scheduling

## Implementation Details

### Message Passing

The decoder performs message passing in the following steps:

1. **Variable Node Update**
   - Combines channel LLR with messages from check nodes
   - Applies shared weights
   - Incorporates residual connections

2. **Check Node Update**
   - Uses min-sum approximation
   - Applies shared weights
   - Processes messages through neural network

3. **Posterior LLR Computation**
   - Combines channel LLR with weighted check node messages
   - Updates variable node values

### Weight Management

- Weights are shared based on base matrix structure
- Separate weights for variable-to-check and check-to-variable messages
- Residual connection weights for previous iterations

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/yourusername/LDPC-NeuralNetwork-Decoder.git
cd LDPC-NeuralNetwork-Decoder
pip install -r requirements.txt
```

## Usage

1. **Training**
```python
from ldpc_neural_decoder.models import ResidualWeightSharingDecoder

# Initialize decoder
decoder = ResidualWeightSharingDecoder(
    base_matrix=base_matrix,
    expansion_factor=4,
    num_iterations=10,
    residual_depth=2
)

# Train decoder
train_decoder(decoder, train_data, val_data)
```

2. **Inference**
```python
# Load trained model
decoder.load_state_dict(torch.load('checkpoints/best_model.pt'))

# Decode received LLRs
decoded_bits = decoder(input_llrs)
```

## Performance

The decoder achieves:
- Improved BER performance over traditional decoders
- Efficient memory usage through weight sharing
- Stable training with residual connections
- Fast convergence with optimized message passing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 