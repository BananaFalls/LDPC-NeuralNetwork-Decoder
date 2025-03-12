# Message-Centered GNN LDPC Decoder

## Overview

The Message-Centered GNN LDPC Decoder implements a novel approach to LDPC decoding by treating the messages in the Tanner graph as nodes in a Graph Neural Network (GNN). This is a significant departure from traditional GNN-based decoders that treat variable and check nodes as the nodes in the GNN.

In this message-centered approach:
- Each **message** in the Tanner graph becomes a **node** in the GNN
- Two message-nodes are connected by an edge if they share the same variable or check node in the original Tanner graph
- This creates a "line graph" or "message graph" representation of the original Tanner graph
- The number of nodes in the message graph equals the number of edges in the original Tanner graph

## Key Components

### TannerToMessageGraph

This utility class converts a traditional Tanner graph representation (parity-check matrix) to a message-centered graph representation:

- Creates mappings between messages and variable/check nodes
- Generates adjacency matrices for message interactions
- Identifies message types for weight sharing
- Provides utility methods for accessing the graph structure

### MessageGNNLayer

The core component of the message-centered GNN decoder:

- Implements message passing between message nodes
- Uses shared weights for similar message types
- Updates message features based on neighboring messages
- Supports residual connections for better gradient flow

### MessageGNNDecoder

The main decoder class that:

- Manages the overall decoding process
- Initializes message features from channel LLRs
- Applies multiple iterations of message passing
- Aggregates final messages to produce bit decisions
- Computes loss for training

## Advantages

The message-centered GNN approach offers several advantages:

1. **More Natural Representation**: Messages are the fundamental units of information exchange in belief propagation, making this a more natural representation.

2. **Enhanced Parameter Sharing**: By focusing on messages rather than nodes, the model can better share parameters across similar message types.

3. **Improved Generalization**: The message-centered approach generalizes better to different code structures and sizes.

4. **Reduced Parameter Count**: Compared to node-centered GNNs, the message-centered approach can achieve similar or better performance with fewer parameters.

5. **Direct Message Manipulation**: Allows direct learning of message update rules without the intermediate step of node updates.

6. **Compatibility with Base Graph Structure**: Works well with lifted LDPC codes like those in 5G, where the base graph structure provides natural message typing.

## Usage

### Basic Usage

```python
import torch
from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder

# Create or load parity-check matrix
H = torch.tensor([...])  # Your parity-check matrix

# Create decoder and converter
decoder, converter = create_message_gnn_decoder(
    H,
    num_iterations=5,
    hidden_dim=64
)

# Use for training
optimizer = torch.optim.SGD(decoder.parameters(), lr=0.001)

# Forward pass with LLRs
soft_bits, loss = decoder(
    llrs,
    converter.message_to_var_mapping,
    message_types=converter.get_message_types(),
    var_to_check_adjacency=converter.var_to_check_adjacency,
    check_to_var_adjacency=converter.check_to_var_adjacency,
    ground_truth=transmitted_bits
)

# Decoding (inference)
hard_bits = decoder.decode(
    llrs,
    converter.message_to_var_mapping,
    message_types=converter.get_message_types(),
    var_to_check_adjacency=converter.var_to_check_adjacency,
    check_to_var_adjacency=converter.check_to_var_adjacency
)
```

### Base Graph Usage

For 5G LDPC codes with a base graph structure:

```python
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix

# Load base matrix and expand
base_matrix = load_base_matrix('path/to/base_matrix.txt')
H = expand_base_matrix(base_matrix, lifting_factor=16)

# Create decoder with base graph information
decoder, converter = create_message_gnn_decoder(
    H,
    num_iterations=5,
    hidden_dim=64,
    base_graph=base_matrix,
    Z=16
)
```

## Example Script

An example script is provided at `ldpc_neural_decoder/examples/message_gnn_example.py` that demonstrates:

1. Training the message-centered GNN decoder
2. Evaluating its performance over a range of SNR values
3. Visualizing the results

Run the example with:

```bash
python -m ldpc_neural_decoder.examples.message_gnn_example --mode train
```

For evaluation:

```bash
python -m ldpc_neural_decoder.examples.message_gnn_example --mode evaluate
```

## Implementation Details

### Message Types

Messages are typed based on their position in the Tanner graph. For base graph structures (like 5G LDPC codes), the message types are determined by the corresponding entry in the base matrix.

### Weight Sharing

The message-centered GNN shares weights across messages of the same type, significantly reducing the number of parameters compared to fully-connected neural networks.

### Message Passing

The message passing mechanism follows these steps:
1. Initialize message features from channel LLRs
2. For each iteration:
   - Aggregate features from neighboring messages
   - Update message features using learned functions
   - Apply non-linear activations
3. Final aggregation to produce bit decisions

## References

1. Nachmani, E., Be'ery, Y., & Burshtein, D. (2016). Learning to decode linear codes using deep learning. 2016 54th Annual Allerton Conference on Communication, Control, and Computing.

2. Satorras, V. G., & Welling, M. (2021). Neural Enhanced Belief Propagation on Factor Graphs. International Conference on Machine Learning.

3. Liu, W., & Deng, L. (2019). Message Passing Neural Networks for LDPC Decoding. IEEE Communications Letters.

4. Dai, Z., Dai, Z., Dai, Z., & Wang, Z. (2020). Beyond Belief Propagation for LDPC Codes: Improved Decoding Using Neural Message Passing. IEEE Transactions on Communications. 