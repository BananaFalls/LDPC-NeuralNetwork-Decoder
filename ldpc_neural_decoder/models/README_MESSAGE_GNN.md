# Message-Centered GNN LDPC Decoder

## Overview

The Message-Centered GNN LDPC Decoder implements a novel approach to LDPC decoding by treating the Tanner graph as a line graph (or message graph). In this representation, each edge in the original Tanner graph becomes a node in the message graph, and two message-nodes are connected if they share the same endpoint (variable or check node) in the original Tanner graph.

This approach aligns more closely with the principles of Graph Neural Networks (GNNs) by:
1. Treating messages as the primary entities (nodes) in the graph
2. Defining connections between messages based on their shared endpoints
3. Applying weight sharing across similar message types
4. Utilizing message passing between connected message nodes

## Key Components

### TannerToMessageGraph

This utility class converts a traditional Tanner graph representation (parity-check matrix) to a message-centered graph representation:

- **Messages**: Each edge in the Tanner graph becomes a message node
- **Message Types**: Messages are assigned types based on their position in the base graph
- **Adjacency Matrices**: Creates adjacency matrices for message interactions:
  - `var_to_check_adjacency`: Connects messages sharing the same variable node
  - `check_to_var_adjacency`: Connects messages sharing the same check node
- **Mappings**: Creates mappings between messages and variable/check nodes

### MessageGNNLayer

This is the core component that implements the message-centered GNN layer:

- **Message Feature Update**: Updates message features based on neighboring messages
- **Weight Sharing**: Shares weights across messages of the same type
- **Message Aggregation**: Aggregates messages from neighbors sharing the same variable or check node
- **Non-linear Transformations**: Applies learnable transformations to message features

### MessageGNNDecoder

This class implements the complete decoder:

- **Iterative Decoding**: Applies multiple iterations of message passing
- **LLR Integration**: Incorporates channel LLRs into the decoding process
- **Loss Computation**: Computes loss based on the difference between decoded and ground truth bits
- **Hard Decision**: Makes final bit decisions based on soft outputs

## Advantages

1. **Parameter Efficiency**: By sharing weights across messages of the same type, the model requires significantly fewer parameters than traditional neural decoders.

2. **Improved Generalization**: The weight sharing mechanism allows the decoder to generalize better to unseen code structures with similar base graphs.

3. **Scalability**: The message-centered approach scales well with increasing code lengths, as the number of parameters depends on the base graph size rather than the expanded code size.

4. **Alignment with GNN Theory**: This implementation aligns more closely with GNN theory, treating messages as nodes and defining clear message passing operations.

5. **Interpretability**: The message-centered approach provides better interpretability, as it directly models the message passing algorithm used in traditional LDPC decoders.

## Usage

### Basic Usage

```python
from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix

# Load base matrix and expand to parity-check matrix
base_matrix = load_base_matrix('path/to/base_matrix.txt')
lifting_factor = 4
H = expand_base_matrix(base_matrix, lifting_factor)

# Create message-centered GNN decoder
decoder, converter = create_message_gnn_decoder(
    H,
    num_iterations=5,
    hidden_dim=64,
    base_graph=base_matrix,
    Z=lifting_factor
)

# Forward pass (training)
soft_bits, loss = decoder(
    llrs,
    converter.message_to_var_mapping,
    message_types=converter.get_message_types(base_matrix, lifting_factor),
    var_to_check_adjacency=converter.var_to_check_adjacency,
    check_to_var_adjacency=converter.check_to_var_adjacency,
    ground_truth=transmitted_bits
)

# Decoding (inference)
hard_bits = decoder.decode(
    llrs,
    converter.message_to_var_mapping,
    message_types=converter.get_message_types(base_matrix, lifting_factor),
    var_to_check_adjacency=converter.var_to_check_adjacency,
    check_to_var_adjacency=converter.check_to_var_adjacency
)
```

### Example Script

A complete example script is provided at `ldpc_neural_decoder/examples/simple_test.py`, which demonstrates:
- Loading a small base graph
- Creating a message-centered GNN decoder
- Training the decoder
- Evaluating its performance

## Implementation Details

### Message Types

Messages are assigned types based on their position in the base graph. This allows for efficient weight sharing across similar edges in the expanded code.

### Message Passing

The message passing mechanism follows these steps:
1. Each message node aggregates information from neighboring messages that share the same variable node
2. Each message node aggregates information from neighboring messages that share the same check node
3. The aggregated information is transformed using learnable weights
4. The transformed information is used to update the message features

### Residual Connections

The implementation includes residual connections to facilitate gradient flow during training, which helps with convergence for deeper networks.

## References

1. Nachmani, E., Be'ery, Y., & Burshtein, D. (2016). Learning to decode linear codes using deep learning. 2016 54th Annual Allerton Conference on Communication, Control, and Computing.

2. Nachmani, E., Marciano, E., Lugosch, L., Gross, W. J., Burshtein, D., & Be'ery, Y. (2018). Deep learning methods for improved decoding of linear codes. IEEE Journal of Selected Topics in Signal Processing.

3. Satorras, V. G., & Welling, M. (2021). Neural enhanced belief propagation on factor graphs. International Conference on Machine Learning.

4. Zhang, J., Wang, X., Xu, H., Zhang, C., & You, X. (2019). A graph neural network decoder for LDPC codes. IEEE Transactions on Communications. 