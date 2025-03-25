# CheckGNNLayer Class Documentation

## Overview

The `CheckGNNLayer` class is a key component of the Message-Centered GNN LDPC Decoder architecture. It's responsible for updating messages based on their connections through check nodes in the Tanner graph.

## Class Definition

```python
class CheckGNNLayer(nn.Module):
    """
    Check-side GNN layer for alternating message passing in LDPC decoding.
    
    This layer updates messages based on their connections through check nodes.
    """
```

## Constructor Parameters

```python
def __init__(self, num_message_types=1, hidden_dim=64):
```

- `num_message_types` (int, default=1): Number of different message types for weight sharing
- `hidden_dim` (int, default=64): Dimension of the hidden message representations

## Module Components

### Message Type Embeddings

```python
self.message_type_embeddings = nn.Parameter(torch.randn(num_message_types, hidden_dim))
```

This parameter stores embeddings for each message type, allowing for weight sharing across similar message types.

### Check Update Network

```python
self.check_update = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim)
)
```

A neural network that processes and updates the message features. It consists of two linear layers with a ReLU activation in between.

### Output Projection

```python
self.output_projection = nn.Linear(hidden_dim, 1)
```

A linear layer that projects high-dimensional message features back to scalar LLR values.

## Forward Method

```python
def forward(self, check_messages, var_messages, message_types, check_to_var_adjacency):
```

### Parameters

- `check_messages` (torch.Tensor): Check-side message features
- `var_messages` (torch.Tensor): Variable-side message features
- `message_types` (torch.Tensor): Type indices for each message
- `check_to_var_adjacency` (torch.Tensor): Adjacency matrix for check connections

### Shape Information

- `check_messages`: Shape [batch_size, num_messages, hidden_dim]
- `var_messages`: Shape [batch_size, num_messages, hidden_dim]
- `message_types`: Shape [num_messages]
- `check_to_var_adjacency`: Shape [num_messages, num_messages]

### Process Flow

1. Embeddings for each message type are retrieved:
   ```python
   type_embeddings = self.message_type_embeddings[safe_message_types]
   ```

2. Type embeddings are added to message features:
   ```python
   messages_with_types = check_messages + type_embeddings.unsqueeze(0)
   ```

3. Messages are aggregated based on the adjacency matrix:
   ```python
   aggregated_messages = torch.matmul(check_to_var_adjacency, messages_with_types)
   ```

4. Aggregated messages are combined with variable messages:
   ```python
   update_input = torch.cat([messages_with_types, var_messages], dim=2)
   ```

5. Combined features are processed through the neural network:
   ```python
   updated_check_messages = self.check_update(update_input)
   ```

6. The result is updated check-side message features.

## Decode Messages Method

```python
def decode_messages(self, message_features):
    """Decode message features to LLR values."""
    return self.output_projection(message_features).squeeze(-1)
```

This method converts high-dimensional message features back to scalar LLR values.

## Typical Usage

This layer is typically used in alternating sequence with the `VariableGNNLayer` in the `MessageGNNDecoder` class:

```python
# Inside MessageGNNDecoder.forward():
for i in range(self.num_iterations):
    # Variable-to-Check Update
    updated_var_features = self.var_gnn_layers[i](
        var_input,
        check_message_features,
        message_types,
        var_to_check_adjacency
    )
    
    # Check-to-Variable Update (using updated variable messages)
    updated_check_features = self.check_gnn_layers[i](
        check_message_features,
        var_message_features,
        message_types,
        check_to_var_adjacency
    )
```

## Differences from VariableGNNLayer

While the `CheckGNNLayer` is structurally similar to the `VariableGNNLayer`, there are a few key differences:

1. **Connection Direction**: `CheckGNNLayer` operates on messages connected through check nodes, while `VariableGNNLayer` operates on messages connected through variable nodes.

2. **Aggregation**: The aggregation is performed using the `check_to_var_adjacency` matrix, which connects messages sharing the same check node.

3. **Feature Combination**: In the update step, the input features are combined in a slightly different order compared to the `VariableGNNLayer`.

4. **Residual Connection**: Unlike `VariableGNNLayer`, there is no explicit residual connection to the original LLR values.

## Neural Network Interpretation

The `CheckGNNLayer` implements a form of message passing in graph neural networks where:

1. Messages are nodes in the graph
2. The adjacency matrix defines edges between messages connected through check nodes
3. Message aggregation is performed through matrix multiplication
4. Message updates are learned through neural networks 