# ResidualWeightSharingDecoder Class Documentation

## Overview
The `ResidualWeightSharingDecoder` class implements a neural network-based LDPC decoder with residual connections and weight sharing. This implementation extends the traditional weight-sharing decoder by adding skip connections between iterations to improve gradient flow and convergence.

## Class Structure

### Initialization
```python
def __init__(self, base_matrix, expansion_factor, num_iterations, residual_depth=3):
    """
    Initialize the residual weight-sharing decoder.
    
    Args:
        base_matrix (torch.Tensor): Base matrix of the LDPC code
        expansion_factor (int): Expansion factor z
        num_iterations (int): Number of decoding iterations
        residual_depth (int): Number of previous iterations to use for residual connections
    """
```

### Key Components

#### 1. Weight Management
```python
def _create_weight_indices(self):
    """
    Create indices for weight sharing based on base matrix structure.
    Returns:
        torch.Tensor: Matrix of weight indices
    """
```

#### 2. Message Indices
```python
def _create_message_indices(self):
    """
    Create indices for message passing between variable and check nodes.
    Returns:
        tuple: (var_to_check_indices, check_to_var_indices)
    """
```

#### 3. Residual Connections
```python
def _get_residual_connection(self, iteration):
    """
    Get residual connection from previous iterations.
    
    Args:
        iteration (int): Current iteration number
    Returns:
        torch.Tensor: Residual connection tensor
    """
```

### Core Methods

#### 1. Variable Node Update
```python
def variable_node_update(self, messages, channel_llr, iteration):
    """
    Update variable node messages.
    
    Args:
        messages (torch.Tensor): Messages from check nodes
        channel_llr (torch.Tensor): Channel LLRs
        iteration (int): Current iteration number
    Returns:
        torch.Tensor: Updated variable node messages
    """
```

#### 2. Check Node Update
```python
def check_node_update(self, messages, iteration):
    """
    Update check node messages.
    
    Args:
        messages (torch.Tensor): Messages from variable nodes
        iteration (int): Current iteration number
    Returns:
        torch.Tensor: Updated check node messages
    """
```

#### 3. Forward Pass
```python
def forward(self, channel_llr):
    """
    Forward pass through the decoder.
    
    Args:
        channel_llr (torch.Tensor): Channel LLRs
    Returns:
        torch.Tensor: Decoded bit probabilities
    """
```

## Implementation Details

### 1. Weight Sharing
- Weights are shared based on base matrix structure
- Each unique position in base matrix has its own weight
- Efficient lookup using pre-computed indices

### 2. Residual Connections
- Skip connections from previous iterations
- Configurable depth for residual connections
- Helps with gradient flow and convergence

### 3. Message Passing
- Alternating variable and check node updates
- Maintains code structure through weight sharing
- Supports both soft and hard decoding

## Usage Example

```python
# Initialize decoder
decoder = ResidualWeightSharingDecoder(
    base_matrix=Hb,
    expansion_factor=z,
    num_iterations=8,
    residual_depth=3
)

# Forward pass
decoded_bits = decoder(channel_llr)
```

## Performance Considerations

### 1. Memory Usage
- Residual connections increase memory requirements
- Trade-off between performance and memory usage
- Configurable through residual_depth parameter

### 2. Computational Efficiency
- Weight sharing reduces parameter count
- Efficient message passing using pre-computed indices
- Batch processing support

### 3. Training Stability
- Residual connections help with gradient flow
- Soft decoding improves training stability
- Configurable number of iterations

## Best Practices

1. **Initialization**:
   - Start with small residual depth
   - Adjust based on performance
   - Monitor memory usage

2. **Training**:
   - Use soft decoding during training
   - Monitor gradient flow
   - Adjust learning rate based on convergence

3. **Evaluation**:
   - Use hard decoding for final evaluation
   - Monitor BER and FER
   - Compare with traditional decoder 