# GNN-based LDPC Decoder

This module implements a Graph Neural Network (GNN) based LDPC decoder with weight sharing across similar edge types. The implementation is based on the paper "Deep Neural Network Based Decoding of Short 5G LDPC Codes" but extends it with a GNN approach for more efficient parameter sharing.

## Overview

The GNN-based LDPC decoder treats the Tanner graph as a graph neural network, where:

1. **Nodes** correspond to variable nodes and check nodes in the Tanner graph
2. **Edges** correspond to connections between variable and check nodes
3. **Messages** are passed along the edges during the decoding process
4. **Edge types** allow for weight sharing across similar edges

## Key Components

### GNNCheckLayer

The `GNNCheckLayer` implements the check node update step using shared weights across similar edge types. It includes:

- Edge-specific weights and biases for different edge types
- Learnable parameters for the min-sum algorithm (scaling factor and bias)

### GNNVariableLayer

The `GNNVariableLayer` implements the variable node update step using shared weights across similar edge types. It includes:

- Edge-specific weights and biases for different edge types
- Learnable parameters for combining messages

### GNNResidualLayer

The `GNNResidualLayer` implements residual connections between variable node layers with shared weights. It includes:

- Shared weight for channel LLR
- Shared weights for residual connections
- Edge type specific weights

### GNNOutputLayer

The `GNNOutputLayer` maps the final LLRs to soft-bit values and computes the loss with shared weights. It includes:

- Learnable parameters for combining final and input LLRs

### GNNLDPCDecoder

The `GNNLDPCDecoder` combines all the GNN layers to implement a complete LDPC decoder with weight sharing. It performs iterative decoding with message passing between variable and check nodes.

### BaseGraphGNNDecoder

The `BaseGraphGNNDecoder` extends the GNN approach to exploit the structure of the 5G LDPC base graph. It:

- Identifies unique edge types based on the shift values in the base graph
- Creates edge type tensors for weight sharing
- Uses the GNNLDPCDecoder with these edge types

## Advantages

1. **Parameter Efficiency**: By sharing weights across similar edge types, the GNN-based decoder has significantly fewer parameters than the standard neural LDPC decoder.

2. **Improved Generalization**: Weight sharing helps the model generalize better to unseen data and different code rates.

3. **Scalability**: The GNN approach scales better to larger code lengths since the number of parameters depends on the number of edge types, not the code length.

4. **Interpretability**: The GNN structure aligns with the message-passing nature of belief propagation, making the model more interpretable.

## Usage

### Basic Usage

```python
from ldpc_neural_decoder.models.gnn_ldpc_decoder import GNNLDPCDecoder

# Create GNN-based LDPC decoder
model = GNNLDPCDecoder(
    num_nodes=num_nodes,
    num_iterations=5,
    depth_L=2,
    num_edge_types=1
)

# Forward pass
soft_bits, loss = model(
    input_llr, 
    check_index_tensor, 
    var_index_tensor, 
    ground_truth
)

# Decode
hard_bits = model.decode(input_llr, check_index_tensor, var_index_tensor)
```

### Base Graph Usage

```python
from ldpc_neural_decoder.models.gnn_ldpc_decoder import BaseGraphGNNDecoder

# Create Base Graph GNN decoder
model = BaseGraphGNNDecoder(
    base_graph=base_matrix,
    Z=lifting_factor,
    num_iterations=5,
    depth_L=2
)

# Forward pass
soft_bits, loss = model(input_llr, check_index_tensor, var_index_tensor, ground_truth)

# Decode
hard_bits = model.decode(input_llr, check_index_tensor, var_index_tensor)
```

### Example Script

An example script is provided in `ldpc_neural_decoder/examples/gnn_ldpc_example.py`. To run it:

```bash
# Training
python -m ldpc_neural_decoder.examples.gnn_ldpc_example --mode train

# Evaluation
python -m ldpc_neural_decoder.examples.gnn_ldpc_example --mode evaluate --model_path models/saved_models/gnn_model.pt
```

## Implementation Details

### Edge Types

Edge types are used to group similar edges in the Tanner graph. For the base graph implementation, edge types are derived from the shift values in the base graph. Edges with the same shift value share the same weights.

### Weight Sharing

Weight sharing is implemented by:

1. Assigning an edge type to each edge in the Tanner graph
2. Using these edge types to index into a small set of learnable parameters
3. Applying the appropriate weights and biases to each message based on its edge type

### Residual Connections

Residual connections are implemented with shared weights across iterations, which helps with gradient flow during training and improves convergence.

## References

- "Deep Neural Network Based Decoding of Short 5G LDPC Codes"
- "Neural Message Passing for LDPC Decoding"
- "Graph Neural Networks for Decoding LDPC Codes" 