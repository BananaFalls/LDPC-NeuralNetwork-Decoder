# Message-Centered GNN LDPC Decoder

## Overview

The Message-Centered Graph Neural Network (GNN) LDPC decoder represents a novel approach to neural LDPC decoding. Unlike traditional decoders that operate on the Tanner graph directly, this architecture transforms the decoding problem by treating messages as nodes in a graph, enabling more sophisticated message passing operations through learnable neural networks.

## Key Features

- **Message-Centered Graph Representation**: Transforms edges in Tanner graph to nodes in a new graph
- **Neural Message Passing**: Uses learned neural networks for message updates
- **Type-Specific Embeddings**: Enables weight sharing across similar message types
- **Multi-Layer Residual Connections**: Maintains gradient flow across iterations
- **Parameter-Free Output Mapping**: Efficiently maps high-dimensional features to LLRs

## Architecture Components

### 1. MessageGNNDecoder

The main decoder class that integrates all components:

```python
class MessageGNNDecoder(nn.Module):
    """
    Message-centered GNN decoder for LDPC codes with alternating layers.
    
    This decoder treats messages as nodes in a graph and updates them using graph neural networks.
    It implements alternating variable and check layers, with residual connections from the 
    previous "n" variable layer to the current variable layer.
    The graph structure is defined by the Tanner graph of the LDPC code.
    """
    
    def __init__(self, num_messages, num_iterations=5, hidden_dim=64, 
                 num_message_types=1, num_of_residual_layers=2):
        # ...
```

The `MessageGNNDecoder` manages:
- Multiple iterations of alternating variable and check GNN layers
- A queue for residual connections across iterations
- Layer normalization for training stability
- Final parameter-free output mapping

### 2. VariableGNNLayer

Updates messages based on their connections through variable nodes:

```python
class VariableGNNLayer(nn.Module):
    """
    Variable-side GNN layer for alternating message passing in LDPC decoding.
    
    This layer updates messages based on their connections through variable nodes.
    The incoming messages from CheckGNNLayer are "check-to-variable" messages.
    The outgoing messages from VariableGNNLayer are "variable-to-check" messages.
    """
    
    def __init__(self, num_message_types=1, hidden_dim=64):
        # ...
```

The `VariableGNNLayer` implements:
- Type-specific message embeddings
- Message aggregation through adjacency matrices
- Neural transformation of aggregated messages
- Residual connection to original LLR values

### 3. CheckGNNLayer

Updates messages based on their connections through check nodes:

```python
class CheckGNNLayer(nn.Module):
    """
    Check-side GNN layer for alternating message passing in LDPC decoding.
    
    This layer updates messages based on their connections through check nodes.
    """
    
    def __init__(self, num_message_types=1, hidden_dim=64):
        # ...
```

The `CheckGNNLayer` mirrors the structure of `VariableGNNLayer` but processes information in the opposite direction.

### 4. TannerToMessageGraph

Converts a Tanner graph to a message-centered graph:

```python
class TannerToMessageGraph:
    """
    Converter from a Tanner graph to a message-centered graph.
    
    This class converts a parity-check matrix (representing a Tanner graph) to a message-centered graph
    where messages (edges in the Tanner graph) become nodes in the new graph.
    """
    
    def __init__(self, H):
        # ...
```

The `TannerToMessageGraph` is responsible for:
- Creating message nodes from Tanner graph edges
- Establishing connections between messages
- Generating adjacency matrices for efficient message passing
- Handling message type assignments for weight sharing

## Message-Centered Graph Transformation

The key innovation is the transformation from a Tanner graph to a message-centered graph:

1. In the **Tanner graph**:
   - Vertices are variable nodes and check nodes
   - Edges connect variable nodes to check nodes
   - Messages flow along edges

2. In the **Message-Centered graph**:
   - Vertices are messages (previously edges in Tanner graph)
   - Edges connect messages that share the same variable or check node
   - Information flows between connected messages

This transformation allows for:
- More natural representation of belief propagation
- Enhanced parameter sharing across similar message types
- Improved generalization to different code structures
- Reduced parameter count compared to node-centered GNNs

## Message Passing Process

The message passing process follows these steps:

1. **Initialization**: Convert input LLRs to high-dimensional message features
2. **Iterative Decoding**:
   - Update variable-to-check messages using `VariableGNNLayer`
   - Update check-to-variable messages using `CheckGNNLayer`
   - Apply residual connections from previous iterations
   - Repeat for specified number of iterations
3. **Output Mapping**:
   - Project high-dimensional features to scalar LLRs
   - Combine messages for each variable node
   - Add original input LLRs
   - Apply sigmoid to get final bit probabilities

## Weight Sharing Through Message Types

To efficiently handle large LDPC codes, the architecture implements weight sharing based on message types:

1. For structured codes (e.g., 5G LDPC), message types can be derived from the base graph
2. Messages of the same type share weights in the neural network
3. This significantly reduces the parameter count while maintaining performance
4. The `create_message_type_mapping` function maps messages to their corresponding types

## Residual Connections

The architecture implements sophisticated residual connections:

1. A queue of previous variable-to-check messages is maintained
2. These messages are added to current variable-to-check messages
3. This helps maintain gradient flow and stabilize training
4. The number of residual layers is configurable via `num_of_residual_layers`

## Usage Example

```python
# Load LDPC code
H = load_ldpc_code('path/to/ldpc_code.txt')

# Optional: Load base graph for weight sharing
base_graph = load_base_graph('path/to/base_graph.txt')
lifting_factor = 4  # Expansion factor Z

# Create decoder and converter
decoder, converter = create_message_gnn_decoder(
    H, 
    base_graph=base_graph,
    lifting_factor=lifting_factor,
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

## Performance Considerations

### Advantages

1. **Learned Update Rules**: Instead of using fixed mathematical formulas, the decoder learns optimal message update functions from data
2. **Rich Message Representation**: High-dimensional vectors capture more complex dependencies than scalar LLRs
3. **Residual Connections**: Preserve information across iterations and improve gradient flow
4. **Flexible Graph Structure**: Naturally handles arbitrary code structures and irregular LDPC codes

### Computational Complexity

The neural network operations are more computationally intensive than traditional belief propagation, but the architecture mitigates this through:

1. **Weight Sharing**: Significant parameter reduction through message types
2. **Matrix Operations**: Efficient implementation using batch matrix multiplications
3. **Configurable Complexity**: Adjustable hidden dimensions and iterations
4. **Parameter-Free Output**: Final layer uses a fixed mapping without trainable parameters

## References

- "Neural Enhanced Belief Propagation on Factor Graphs" (Satorras & Welling, 2021)
- "Message Passing Neural Networks for LDPC Decoding" (Liu & Deng, 2019)
- "Beyond Belief Propagation for LDPC Codes: Improved Decoding Using Neural Message Passing" (Dai et al., 2020)
- 5G NR LDPC codes specification (3GPP TS 38.212) 