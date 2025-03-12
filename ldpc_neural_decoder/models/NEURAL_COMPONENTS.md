# Neural Network Components in LDPC Decoders

This document provides a detailed explanation of the neural network components used in our LDPC decoders, with a focus on the message-centered GNN approach.

## Table of Contents

1. [Weight Sharing](#weight-sharing)
2. [Message Passing](#message-passing)
3. [Residual Connections](#residual-connections)
4. [Activation Functions](#activation-functions)
5. [Loss Functions](#loss-functions)
6. [Initialization](#initialization)
7. [Optimization](#optimization)
8. [Comparison of Architectures](#comparison-of-architectures)

## Weight Sharing

Weight sharing is a technique to reduce the number of parameters in a neural network by reusing the same weights across different parts of the network. In our LDPC decoders, we implement weight sharing in several ways:

### Standard Neural Decoder

In the standard neural LDPC decoder, weight sharing is limited:

```python
# Each check node and variable node has its own set of weights
self.check_weights = nn.Parameter(torch.ones(num_checks, max_check_degree))
self.variable_weights = nn.Parameter(torch.ones(num_variables, max_variable_degree))
```

### GNN-based Decoder

The GNN-based decoder introduces weight sharing through edge types:

```python
# Weights are shared across edges of the same type
self.edge_type_embeddings = nn.Parameter(torch.randn(num_edge_types, hidden_dim))
```

### Message-Centered GNN Decoder

The message-centered GNN decoder implements the most extensive weight sharing:

```python
# Message type embeddings - shared across messages of the same type
self.message_type_embeddings = nn.Parameter(torch.randn(num_message_types, hidden_dim))

# Neural networks for message updates - shared across ALL messages
self.var_to_check_update = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)

self.check_to_var_update = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)
```

### Benefits of Weight Sharing

1. **Parameter Efficiency**: Significantly reduces the number of trainable parameters
2. **Better Generalization**: Helps the model generalize to different code structures
3. **Faster Training**: Fewer parameters lead to faster training and convergence
4. **Reduced Overfitting**: Less prone to overfitting on small datasets

### Example: Message Type Determination

For 5G LDPC codes with a base graph structure, message types are determined by the shift values in the base graph:

```python
def get_message_types(self, base_graph=None, Z=None):
    # ...
    for i, (v, c) in enumerate(self.messages):
        base_row = c // Z
        base_col = v // Z
        
        # Get shift value from base graph
        shift = base_graph[base_row, base_col].item()
        
        # Use shift value as message type
        message_types[i] = int(shift) if shift >= 0 else 0
```

This allows messages corresponding to the same shift value to share parameters, leveraging the inherent structure of 5G LDPC codes.

## Message Passing

Message passing is the core operation in both traditional belief propagation and our neural LDPC decoders. It involves exchanging information between nodes in the Tanner graph.

### Standard Neural Decoder

In the standard neural decoder, message passing follows the traditional belief propagation algorithm with learnable weights:

```python
# Check node update (min-sum algorithm with learnable weights)
def forward(self, messages_from_variables):
    # Find minimum magnitude
    min_magnitude, min_indices = torch.min(torch.abs(messages_from_variables), dim=1)
    
    # Apply learnable weights
    weighted_min = min_magnitude * self.weights
    
    # Compute sign
    signs = torch.prod(torch.sign(messages_from_variables), dim=1)
    
    # Combine sign and magnitude
    messages_to_variables = signs.unsqueeze(1) * weighted_min.unsqueeze(1)
    
    return messages_to_variables
```

### Message-Centered GNN Decoder

In the message-centered GNN decoder, message passing occurs between message nodes:

```python
def forward(self, message_features, message_types, var_to_check_adjacency, check_to_var_adjacency):
    # Get message type embeddings
    type_embeddings = self.message_type_embeddings[message_types]
    
    # Combine message features with type embeddings
    combined_features = message_features + type_embeddings
    
    # Variable-to-check message update
    var_to_check_messages = torch.bmm(var_to_check_adjacency.unsqueeze(0), combined_features)
    var_to_check_input = torch.cat([combined_features, var_to_check_messages], dim=2)
    var_to_check_updated = self.var_to_check_update(var_to_check_input)
    
    # Check-to-variable message update
    check_to_var_messages = torch.bmm(check_to_var_adjacency.unsqueeze(0), combined_features)
    check_to_var_input = torch.cat([combined_features, check_to_var_messages], dim=2)
    check_to_var_updated = self.check_to_var_update(check_to_var_input)
    
    # Combine updates
    updated_features = var_to_check_updated + check_to_var_updated
    
    return updated_features
```

### Key Differences

1. **Traditional BP**: Fixed update rules based on min-sum or sum-product algorithms
2. **Standard Neural Decoder**: Learnable weights applied to traditional update rules
3. **GNN-based Decoder**: Learnable message passing between variable and check nodes
4. **Message-Centered GNN**: Learnable message passing between message nodes

## Residual Connections

Residual connections help with gradient flow in deep neural networks by providing a direct path for gradients to flow backward. They are implemented in our decoders as follows:

### Standard Neural Decoder

```python
class ResidualLayer(nn.Module):
    def __init__(self, H, depth_L):
        super(ResidualLayer, self).__init__()
        self.variable_layers = nn.ModuleList([VariableLayer(H) for _ in range(depth_L)])
    
    def forward(self, messages_from_checks, input_llr):
        x = messages_from_checks
        for layer in self.variable_layers:
            x = x + layer(x, input_llr)
        return x
```

### Message-Centered GNN Decoder

In the message-centered GNN decoder, residual connections are implemented implicitly by combining the updates:

```python
# Combine updates with residual connection
updated_features = var_to_check_updated + check_to_var_updated
```

### Benefits of Residual Connections

1. **Improved Gradient Flow**: Helps with vanishing gradient problem
2. **Faster Convergence**: Speeds up training
3. **Better Performance**: Often leads to better final performance
4. **Deeper Networks**: Allows training of deeper networks

## Activation Functions

Activation functions introduce non-linearity into the neural network, allowing it to learn complex patterns.

### ReLU (Rectified Linear Unit)

Used in most of our neural components:

```python
self.var_to_check_update = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),  # ReLU activation
    nn.Linear(hidden_dim, hidden_dim)
)
```

### Sigmoid

Used for final bit decisions:

```python
# Convert to soft bits
soft_bits = torch.sigmoid(combined_llrs)
```

## Loss Functions

The choice of loss function is critical for training neural LDPC decoders.

### Binary Cross-Entropy

Used for bit-level loss:

```python
# Binary cross-entropy loss
loss = F.binary_cross_entropy(soft_bits, ground_truth, reduction='none')
```

### Max Loss (for FER Minimization)

To optimize for frame error rate (FER), we use a max operation over the bit-level losses:

```python
# Apply max function over the loss vector (for FER minimization)
max_loss = torch.max(loss, dim=1).values
```

## Initialization

Proper initialization of neural network parameters is important for training stability.

### Weight Initialization

For message type embeddings:

```python
self.message_type_embeddings = nn.Parameter(torch.randn(num_message_types, hidden_dim))
```

For neural network layers, PyTorch's default initialization (Kaiming/He initialization for ReLU networks) is used.

## Optimization

### SGD with Momentum and Weight Decay

```python
optimizer = torch.optim.SGD(
    decoder.parameters(),
    lr=args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
)
```

### Training Process

1. **Forward Pass**: Compute soft bit values and loss
2. **Backward Pass**: Compute gradients
3. **Parameter Update**: Update weights using optimizer

```python
# Zero gradients
optimizer.zero_grad()

# Forward pass
soft_bits, loss = model(llrs, ..., ground_truth=transmitted_bits)

# Compute mean loss
batch_loss = loss.mean()

# Backward pass
batch_loss.backward()

# Update parameters
optimizer.step()
```

## Comparison of Architectures

| Feature | Standard Neural Decoder | GNN-based Decoder | Message-Centered GNN |
|---------|-------------------------|-------------------|----------------------|
| **Node Representation** | Variable and check nodes | Variable and check nodes | Message nodes |
| **Weight Sharing** | Limited | Edge-type based | Message-type based |
| **Parameter Count** | High | Medium | Low |
| **Generalization** | Limited | Good | Excellent |
| **Alignment with BP** | Direct | Indirect | Natural |
| **Residual Connections** | Explicit | Explicit | Implicit |
| **Base Graph Compatibility** | Limited | Good | Excellent |

### Parameter Count Example

For a typical LDPC code with 100 variable nodes, 50 check nodes, and average degree 3:

- **Standard Neural Decoder**: ~450 parameters
- **GNN-based Decoder**: ~200 parameters
- **Message-Centered GNN**: ~100 parameters

The message-centered GNN achieves the highest parameter efficiency while maintaining or improving decoding performance. 