# LDPC Neural Network Decoder Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Implementation](#implementation)
4. [Training](#training)
5. [Results](#results)
6. [Usage](#usage)
7. [API Reference](#api-reference)

## Introduction

The LDPC Neural Network Decoder is a PyTorch implementation of a neural network-based LDPC decoder that combines traditional message passing with neural network components. The decoder features weight sharing and residual connections for improved performance.

## Architecture

### Core Components

1. **ResidualWeightSharingDecoder**
   - Base matrix expansion
   - Message passing system
   - Weight sharing mechanism
   - Residual connections

2. **Message Processing**
   - Variable node updates
   - Check node updates
   - Posterior LLR computation

3. **Neural Network Components**
   - Message processor
   - Shared weights
   - Residual connection weights

### Key Features

- Weight sharing across circulant blocks
- Residual connections for improved gradient flow
- Neural network-based message processing
- Efficient implementation
- 5G LDPC code support

## Implementation

### Message Passing System

1. **Variable Node Update**
   ```python
   def variable_node_update(self, var_values, check_to_var_messages, prev_var_messages, iteration, batch_idx):
       # Combines channel LLR with messages from check nodes
       # Applies shared weights
       # Incorporates residual connections
   ```

2. **Check Node Update**
   ```python
   def check_node_update(self, var_to_check_messages, batch_idx):
       # Uses min-sum approximation
       # Applies shared weights
       # Processes messages through neural network
   ```

3. **Posterior LLR Computation**
   ```python
   def _compute_posterior_llr(self, input_llr, check_to_var_messages, batch_idx):
       # Combines channel LLR with weighted check node messages
       # Updates variable node values
   ```

### Weight Management

- Shared weights based on base matrix structure
- Separate weights for variable-to-check and check-to-variable messages
- Residual connection weights for previous iterations

## Training

### Configuration

- Number of epochs: 50
- Batch size: 64
- Learning rate: 1e-3
- Number of iterations: 10
- Residual depth: 2

### Data Generation

- Total examples: 320,000
- SNR range: [-1, 8] dB
- Split ratio: 80% train, 10% validation, 10% test

### Optimization

- SGD optimizer with momentum (0.9)
- L2 regularization (weight decay: 1e-2)
- Learning rate scheduling

## Results

### Performance Metrics

- Bit Error Rate (BER)
- Frame Error Rate (FER)
- Training convergence
- Memory efficiency

### Visualization

- BER vs SNR plots
- FER vs SNR plots
- Training curves
- Loss curves

## Usage

### Basic Usage

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

# Inference
decoded_bits = decoder(input_llrs)
```

### Advanced Usage

- Custom base matrix
- Different expansion factors
- Variable number of iterations
- Configurable residual depth

## API Reference

### ResidualWeightSharingDecoder

```python
class ResidualWeightSharingDecoder(nn.Module):
    def __init__(self, base_matrix, expansion_factor, num_iterations=5, residual_depth=2):
        """
        Initialize the decoder.
        
        Args:
            base_matrix: Base parity check matrix
            expansion_factor: Size of circulant blocks
            num_iterations: Number of decoding iterations
            residual_depth: Number of previous iterations to consider
        """
```

### Key Methods

1. **forward**
   ```python
   def forward(self, input_llr):
       """
       Forward pass of the decoder.
       
       Args:
           input_llr: Input LLR values
       Returns:
           decoded_bits: Decoded bits
       """
   ```

2. **variable_node_update**
   ```python
   def variable_node_update(self, var_values, check_to_var_messages, prev_var_messages, iteration, batch_idx):
       """
       Update variable node messages.
       
       Args:
           var_values: Current variable node values
           check_to_var_messages: Messages from check nodes
           prev_var_messages: Previous iteration messages
           iteration: Current iteration
           batch_idx: Batch index
       """
   ```

3. **check_node_update**
   ```python
   def check_node_update(self, var_to_check_messages, batch_idx):
       """
       Update check node messages.
       
       Args:
           var_to_check_messages: Messages from variable nodes
           batch_idx: Batch index
       """
   ```

4. **_compute_posterior_llr**
   ```python
   def _compute_posterior_llr(self, input_llr, check_to_var_messages, batch_idx):
       """
       Compute posterior LLR values.
       
       Args:
           input_llr: Input LLR values
           check_to_var_messages: Messages from check nodes
           batch_idx: Batch index
       """
   ```

## 1. Architecture
- [Overview](architecture/overview.md)
- [Neural Network Structure](architecture/neural_network.md)
- [Residual Connections](architecture/residual_connections.md)
- [Weight Sharing](architecture/weight_sharing.md)
- [GPU Acceleration](architecture/gpu_acceleration.md)

## 2. Implementation
- [Core Components](implementation/core_components.md)
- [Training Process](implementation/training_process.md)
- [Performance Optimizations](implementation/optimizations.md)
- [Error Handling](implementation/error_handling.md)
- [Memory Management](implementation/memory_management.md)

## 3. Training
- [Data Generation](training/data_generation.md)
- [Training Parameters](training/parameters.md)
  - Epochs: 50
  - Batch Size: 64
  - Learning Rate: 1e-3
  - Momentum: 0.9
  - Weight Decay: 1e-2
  - Iterations: 10
  - Residual Depth: 2
  - Expansion Factor: 4
  - SNR Range: -1 to 8 dB
  - Data Split: 80/10/10
- [Loss Functions](training/loss_functions.md)
- [Optimization](training/optimization.md)
- [GPU Training](training/gpu_training.md)

## 4. Results
- [Performance Metrics](results/metrics.md)
- [Comparative Analysis](results/comparison.md)
- [Visualizations](results/visualizations.md)
- [Ablation Studies](results/ablation.md)

## 5. Usage
- [Installation](usage/installation.md)
- [Quick Start](usage/quick_start.md)
- [API Reference](usage/api.md)
- [Examples](usage/examples.md)
- [Troubleshooting](usage/troubleshooting.md)

## 6. Development
- [Code Structure](development/structure.md)
- [Testing](development/testing.md)
- [Contributing](development/contributing.md)
- [Version History](development/versions.md)

## 7. References
- [Papers](references/papers.md)
- [Documentation](references/documentation.md)
- [Resources](references/resources.md) 