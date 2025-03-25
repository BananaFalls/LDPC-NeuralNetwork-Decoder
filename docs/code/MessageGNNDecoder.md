# MessageGNNDecoder Class Documentation

## Overview

The `MessageGNNDecoder` class is the main component of the Message-Centered GNN LDPC Decoder architecture. It implements a Graph Neural Network-based decoder that treats messages as nodes in a graph and updates them using alternating variable and check GNN layers.

## Class Definition

```python
class MessageGNNDecoder(nn.Module):
    """
    Message-centered GNN decoder for LDPC codes with alternating layers.
    
    This decoder treats messages as nodes in a graph and updates them using graph neural networks.
    It implements alternating variable and check layers, with residual connections from the previous "n" variable layer to the current variable layer.
    The graph structure is defined by the Tanner graph of the LDPC code.
    """
```

## Constructor Parameters

```python
def __init__(self, num_messages, num_iterations=5, hidden_dim=64, num_message_types=1, num_of_residual_layers=2):
```

- `num_messages` (int): Number of message nodes in the graph
- `num_iterations` (int, default=5): Number of decoding iterations
- `hidden_dim` (int, default=64): Dimension of hidden message representations
- `num_message_types` (int, default=1): Number of different message types for weight sharing
- `num_of_residual_layers` (int, default=2): Number of previous variable layers to use for residual connections

## Module Components

### Variable-side GNN Layers

```python
self.var_gnn_layers = nn.ModuleList([
    VariableGNNLayer(num_message_types, hidden_dim)
    for _ in range(num_iterations)
])
```

A list of `VariableGNNLayer` modules, one for each iteration of decoding.

### Check-side GNN Layers

```python
self.check_gnn_layers = nn.ModuleList([
    CheckGNNLayer(num_message_types, hidden_dim)
    for _ in range(num_iterations)
])
```

A list of `CheckGNNLayer` modules, one for each iteration of decoding.

### Layer Normalization

```python
self.var_layer_norms = nn.ModuleList([
    nn.LayerNorm(hidden_dim)
    for _ in range(num_iterations)
])

self.check_layer_norms = nn.ModuleList([
    nn.LayerNorm(hidden_dim)
    for _ in range(num_iterations)
])
```

Layer normalization modules to stabilize training, one for each iteration.

### Output Projection

```python
self.output_projection = nn.Linear(hidden_dim, 1, bias=False)
# Initialize to average over hidden dimensions
with torch.no_grad():
    self.output_projection.weight.fill_(1.0 / hidden_dim)
# Freeze the output projection
self.output_projection.weight.requires_grad = False
```

A parameter-free linear layer that projects high-dimensional message features back to scalar LLR values. This is initialized to average over hidden dimensions and then frozen.

## Method: count_parameters

```python
def count_parameters(self):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

Utility method to count the number of trainable parameters in the model.

## Method: output_mapping

```python
def output_mapping(self, final_messages, message_to_var_mapping, input_llr, batch_size, num_vars):
```

Maps final messages back to variable nodes and combines them with input LLRs.

### Parameters

- `final_messages` (torch.Tensor): Final message features from the decoder
- `message_to_var_mapping` (torch.Tensor): Mapping from messages to variable nodes
- `input_llr` (torch.Tensor): Original input LLR values
- `batch_size` (int): Number of codewords in the batch
- `num_vars` (int): Number of variable nodes

### Process Flow

1. Project messages to scalar LLRs:
   ```python
   message_llrs = self.output_projection(final_messages).squeeze(-1)
   ```

2. Initialize output probabilities:
   ```python
   output_probs = torch.zeros(batch_size, num_vars, device=input_llr.device)
   ```

3. Combine messages for each variable node:
   ```python
   for b in range(batch_size):
       var_llrs = torch.zeros(num_vars, device=input_llr.device)
       for msg_idx in range(self.num_messages):
           var_idx = message_to_var_mapping[msg_idx].item()
           var_llrs[var_idx] += message_llrs[b, msg_idx]
   ```

4. Add original input LLRs:
   ```python
   combined_llrs = var_llrs + input_llr[b]
   ```

5. Convert to probabilities using sigmoid:
   ```python
   output_probs[b] = torch.sigmoid(combined_llrs)
   ```

6. Return the output probabilities.

## Method: forward

```python
def forward(self, input_llr, message_to_var_mapping, message_types=None, 
            var_to_check_adjacency=None, check_to_var_adjacency=None, ground_truth=None):
```

The main forward pass of the Message GNN Decoder with alternating layers.

### Parameters

- `input_llr` (torch.Tensor): Input LLR values for each variable node
- `message_to_var_mapping` (torch.Tensor): Mapping from messages to variable nodes
- `message_types` (torch.Tensor, optional): Types of each message
- `var_to_check_adjacency` (torch.Tensor, optional): Adjacency matrix for var-to-check connections
- `check_to_var_adjacency` (torch.Tensor, optional): Adjacency matrix for check-to-var connections
- `ground_truth` (torch.Tensor, optional): Ground truth bits for potential auxiliary loss

### Process Flow

1. Validate input shape:
   ```python
   if len(input_llr.shape) != 2:
       raise ValueError(f"Expected input_llr to be a 2D tensor (batch_size × num_vars), got shape {input_llr.shape}")
   ```

2. Initialize message features from input LLRs:
   ```python
   message_llrs = torch.zeros(batch_size, self.num_messages, device=input_llr.device)
   for b in range(batch_size):
       var_indices = message_to_var_mapping
       message_llrs[b] = input_llr[b][var_indices]
   ```

3. Transform scalar LLRs into feature vectors:
   ```python
   var_message_features = self.input_embedding(message_llrs.unsqueeze(-1))
   check_message_features = torch.zeros_like(var_message_features)
   ```

4. Set default values if not provided:
   ```python
   if message_types is None:
       message_types = torch.zeros(self.num_messages, dtype=torch.long, device=input_llr.device)
   if var_to_check_adjacency is None:
       var_to_check_adjacency = torch.eye(self.num_messages, device=input_llr.device)
   if check_to_var_adjacency is None:
       check_to_var_adjacency = torch.eye(self.num_messages, device=input_llr.device)
   ```

5. Initialize queue for residual connections:
   ```python
   var_to_check_queue = []
   ```

6. Iterative message passing:
   ```python
   for i in range(self.num_iterations):
       # Variable-to-Check Update
       updated_var_features = self.var_gnn_layers[i](
           var_input,
           check_message_features,
           message_types,
           var_to_check_adjacency
       )
       
       # Add residual connections
       if i >= self.num_of_residual_layers and var_to_check_queue:
           for prev_var_to_check in var_to_check_queue:
               updated_var_features = updated_var_features + prev_var_to_check
       
       # Apply layer norm
       var_message_features = self.var_layer_norms[i](updated_var_features)
       
       # Update queue
       var_to_check_queue.append(var_message_features.clone())
       if len(var_to_check_queue) > self.num_of_residual_layers:
           var_to_check_queue.pop(0)
       
       # Check-to-Variable Update
       updated_check_features = self.check_gnn_layers[i](
           check_message_features,
           var_message_features,
           message_types,
           check_to_var_adjacency
       )
       
       # Apply layer norm
       check_message_features = self.check_layer_norms[i](updated_check_features)
   ```

7. Final decoding:
   ```python
   return self.output_mapping(
       final_messages=check_message_features,
       message_to_var_mapping=message_to_var_mapping,
       input_llr=input_llr,
       batch_size=batch_size,
       num_vars=num_vars
   )
   ```

## Typical Usage

The `MessageGNNDecoder` is typically used with a `TannerToMessageGraph` converter to set up the message graph:

```python
# Load LDPC code
H = load_ldpc_code('path/to/ldpc_code.txt')

# Create decoder and converter
decoder, converter = create_message_gnn_decoder(
    H, 
    num_iterations=5,
    hidden_dim=64,
    num_of_residual_layers=2
)

# Get required mappings for decoding
message_to_var_mapping = converter.get_message_to_var_mapping()
var_to_check_adjacency = converter.var_to_check_adjacency
check_to_var_adjacency = converter.check_to_var_adjacency
message_types = converter.message_types

# Decode input LLRs
output_probs = decoder(
    input_llr=llr_values,
    message_to_var_mapping=message_to_var_mapping,
    message_types=message_types,
    var_to_check_adjacency=var_to_check_adjacency,
    check_to_var_adjacency=check_to_var_adjacency
)

# Convert to hard decisions
decoded_bits = (output_probs >= 0.5).float()
```

## Key Features

1. **Message-Centered Representation**: Treats messages as nodes in a graph, enabling more sophisticated message passing.
   
2. **Alternating Layers**: Implements alternating variable and check GNN layers, similar to belief propagation.
   
3. **Residual Connections**: Uses residual connections across multiple iterations to improve gradient flow.
   
4. **Layer Normalization**: Applies layer normalization for more stable training.
   
5. **Parameter-Free Output**: Uses a parameter-free output mapping to convert back to LLR values.
   
6. **Weight Sharing**: Implements weight sharing through message type embeddings.

## Neural Network Interpretation

The `MessageGNNDecoder` implements a form of message passing in graph neural networks where:

1. Messages are nodes in the graph
2. Adjacency matrices define connections between messages
3. Alternating layers update messages from variable and check perspectives
4. Residual connections across iterations help maintain gradient flow
5. The final output combines all messages to produce bit probabilities 