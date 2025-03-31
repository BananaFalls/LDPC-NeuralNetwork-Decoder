# LDPC Neural Decoder Architecture

## Overview
The LDPC Neural Decoder implements a neural network-based approach to LDPC decoding, combining traditional belief propagation concepts with deep learning techniques. The architecture supports both soft and hard decoding approaches.

## Core Components

### 1. Residual Weight-Sharing Decoder
- **Base Structure**: Neural network implementation of LDPC decoding
- **Key Features**:
  - Weight sharing based on base matrix structure
  - Residual connections between iterations
  - Configurable number of iterations
  - Support for both soft and hard decoding

### 2. Message Passing Framework
- **Variable Node Updates**:
  - Processes incoming messages from check nodes
  - Combines with channel LLRs
  - Applies learned weights based on message types
  - Supports residual connections

- **Check Node Updates**:
  - Processes messages from variable nodes
  - Implements parity check constraints
  - Uses learned weights for message updates
  - Maintains code structure through weight sharing

### 3. Decoding Approaches

#### Soft Decoding
- **Probability-Based Processing**:
  - Works with continuous probabilities (0 to 1)
  - Uses tanh rule for check node updates
  - Maintains gradient information throughout
  - Better for training and low SNR conditions

- **Loss Functions**:
  - Binary Cross-Entropy (BCE) loss
  - Soft parity check loss using syndrome
  - Combined weighted loss function

#### Hard Decoding
- **Binary Decision Processing**:
  - Uses threshold-based decisions (0 or 1)
  - Traditional syndrome computation
  - Used for final evaluation
  - Better for high SNR conditions

## Training Process

### 1. Data Generation
- Zero codewords generation
- BPSK modulation
- AWGN noise addition
- LLR computation

### 2. Training Loop
- Forward pass with soft decoding
- Loss computation
- Backward pass
- Parameter updates

### 3. Evaluation
- Validation on separate dataset
- BER and FER computation
- Model checkpointing
- Performance visualization

## Implementation Details

### 1. Weight Sharing
```python
# Weight lookup based on base matrix position
weight_idx = self.base_matrix[row, col]
if weight_idx >= 0:
    return self.weights[weight_idx]
```

### 2. Residual Connections
```python
# Residual connection implementation
if self.residual_depth > 0:
    residual = self._get_residual_connection(iteration)
    message = message + residual
```

### 3. Soft Decoding
```python
# Soft syndrome computation
soft_bits = 2 * decoded_bits - 1  # Convert to [-1, 1] range
syndrome = torch.matmul(soft_bits, H.T)
parity_loss = torch.mean(torch.abs(syndrome))
```

## Performance Metrics

### 1. Training Metrics
- Loss values (total, BCE, parity)
- Learning rate changes
- Gradient statistics

### 2. Evaluation Metrics
- Bit Error Rate (BER)
- Frame Error Rate (FER)
- Validation loss

### 3. Visualization
- BER vs. Epoch plots
- FER vs. Epoch plots
- Learning rate schedules

## Usage Guidelines

### 1. Model Configuration
```python
model = ResidualWeightSharingDecoder(
    base_matrix=Hb,
    expansion_factor=z,
    num_iterations=num_iterations,
    residual_depth=residual_depth
)
```

### 2. Training Parameters
```python
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=1e-2
)
```

### 3. Evaluation
```python
# Compute metrics
val_loss, val_ber, val_fer = evaluate(model, val_loader, model.H, device)
```

## Best Practices

1. **Soft vs. Hard Decoding**:
   - Use soft decoding during training
   - Use hard decoding for final evaluation
   - Adjust threshold based on SNR

2. **Residual Connections**:
   - Start with small residual depth
   - Increase based on performance
   - Monitor memory usage

3. **Training Process**:
   - Use learning rate scheduling
   - Monitor validation metrics
   - Save best checkpoints
   - Visualize training progress 