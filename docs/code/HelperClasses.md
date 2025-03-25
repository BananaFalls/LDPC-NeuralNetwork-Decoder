# Helper Classes Documentation

This document provides documentation for additional helper classes and functions related to the Message-Centered GNN architecture.

## Message Type Management

### create_message_type_mapping Function

The `create_message_type_mapping` function is crucial for efficient weight sharing based on the base graph structure.

```python
def create_message_type_mapping(H, base_graph, lifting_factor):
    """
    Create a mapping of message types based on the base graph structure.
    Messages corresponding to the same edge in the base graph will have the same type.
    """
```

#### Process

1. Maps edges in the expanded graph to their corresponding edges in the base graph
2. Assigns a type ID to each unique edge in the base graph
3. Propagates these type IDs to all corresponding edges in the expanded graph
4. Returns a tensor mapping each message to its type

#### Weight Sharing Benefits

This approach enables significant parameter reduction:
- Instead of learning different weights for each message in the expanded graph
- Weights are shared among messages that correspond to the same edge in the base graph
- Particularly efficient for structured LDPC codes used in 5G NR

## Index Tensor Creation

### create_variable_index_tensor Function

```python
def create_variable_index_tensor(H, converter):
    """
    Create a tensor mapping variable nodes to message nodes.
    """
```

This function creates a padded tensor representation of the variable-to-message mapping, which is useful for batch processing in the neural network.

### create_check_index_tensor Function

```python
def create_check_index_tensor(H, message_type_map=None):
    """
    Create a tensor mapping check nodes to message nodes.
    """
```

This function creates a padded tensor representation of the check-to-message mapping, also useful for batch processing.

## Training and Evaluation

### LDPCDecoderTrainer Class (referenced in examples)

While not defined directly in message_gnn_decoder.py, the `LDPCDecoderTrainer` class is typically used to train the Message GNN Decoder:

```python
class LDPCDecoderTrainer:
    """
    Trainer class for LDPC decoders.
    """
```

#### Key Components

1. **Data Generation**: Creates synthetic training data by:
   - Generating random bits
   - Encoding using the LDPC encoder
   - Modulating (typically QPSK)
   - Adding AWGN noise at various SNR levels
   - Demodulating to LLRs

2. **Training Loop**: Implements:
   - Mini-batch training with SGD or Adam optimizer
   - Learning rate scheduling
   - Early stopping based on validation performance
   - Checkpoint saving

3. **Evaluation**: Measures:
   - Bit Error Rate (BER)
   - Frame Error Rate (FER)
   - Convergence speed
   - Generalization to different SNR levels

## Performance Benchmarking

### LDPC Benchmarking Tools

```python
class LDPCDecoderBenchmark:
    """
    Benchmarking suite for comparing different LDPC decoders.
    """
```

This class (typically part of the evaluation tools) provides:

1. **Comparative Analysis**: Compares against:
   - Traditional BP decoder
   - Min-Sum Scaled decoder
   - Neural BP decoder
   - Other GNN-based decoders
   - The Message-Centered GNN decoder

2. **Visualization**: Generates:
   - BER vs SNR curves
   - FER vs SNR curves
   - Iteration count comparisons
   - Complexity analysis

3. **Metrics**:
   - Error rate at different SNR levels
   - Computational complexity
   - Parameter count
   - Convergence speed

## Inference Optimization

### JIT Compilation Tools

For deployment scenarios, JIT compilation can be used to optimize the Message GNN Decoder:

```python
def compile_message_gnn_decoder(decoder, example_input):
    """
    Compile the Message GNN Decoder using torch.jit for faster inference.
    """
    return torch.jit.trace(decoder, example_input)
```

This approach enables:
1. Faster inference by removing Python overhead
2. Optimized execution on both CPU and GPU
3. Potential for deployment in production environments 