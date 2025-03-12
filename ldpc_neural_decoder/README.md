# Neural LDPC Decoder for 5G

This project implements a neural network-based decoder for 5G LDPC codes, based on the paper "Deep Neural Network Based Decoding of Short 5G LDPC Codes".

## Project Structure

```
ldpc_neural_decoder/
├── data/                  # Data storage
├── models/                # Neural network models
│   ├── decoder.py         # Main decoder models
│   ├── layers.py          # Neural network layers
│   ├── traditional_decoders.py # Traditional LDPC decoders
│   ├── gnn_ldpc_decoder.py # GNN-based LDPC decoder
│   ├── message_gnn_decoder.py # Message-centered GNN decoder
│   └── saved_models/      # Saved model weights
├── training/              # Training utilities
│   ├── trainer.py         # Trainer class
│   └── comparative_evaluation.py # Comparative evaluation
├── utils/                 # Utility functions
│   ├── channel.py         # Channel simulation
│   └── ldpc_utils.py      # LDPC code utilities
├── visualization/         # Visualization utilities
├── examples/              # Example scripts
│   ├── gnn_ldpc_example.py # Example for GNN-based decoder
│   └── message_gnn_example.py # Example for message-centered GNN decoder
├── results/               # Results storage
└── main.py                # Main script
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ldpc_neural_decoder
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the neural LDPC decoder using SGD optimizer:

```bash
python -m ldpc_neural_decoder.main --mode train --num_epochs 100 --batch_size 32 --learning_rate 0.001 --momentum 0.9 --weight_decay 0.0001
```

### Evaluation

To evaluate the trained decoder:

```bash
python -m ldpc_neural_decoder.main --mode evaluate --model_path ldpc_neural_decoder/models/saved_models/model.pt
```

### Comparative Evaluation

To compare the neural decoder with traditional decoders (Belief Propagation and Min-Sum Scaled):

```bash
python -m ldpc_neural_decoder.main --mode compare --model_path ldpc_neural_decoder/models/saved_models/model.pt --compare_with_traditional
```

### Complete Workflow

To run the complete workflow (training, evaluation, comparison, and visualization):

```bash
python -m ldpc_neural_decoder.run_workflow
```

### Visualization

To visualize the results:

```bash
python -m ldpc_neural_decoder.main --mode visualize
```

### GNN-based LDPC Decoder

To train the GNN-based LDPC decoder with weight sharing:

```bash
python -m ldpc_neural_decoder.examples.gnn_ldpc_example --mode train
```

### Message-Centered GNN LDPC Decoder

To train the message-centered GNN LDPC decoder:

```bash
python -m ldpc_neural_decoder.examples.message_gnn_example --mode train
```

## Command Line Arguments

- `--mode`: Mode of operation (`train`, `evaluate`, `compare`, or `visualize`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--model_type`: Type of model (`standard`, `tied`, `gnn`, or `message_gnn`)
- `--num_iterations`: Number of decoding iterations
- `--depth_L`: Depth of residual connections
- `--base_matrix_path`: Path to base matrix file
- `--lifting_factor`: Lifting factor Z
- `--num_epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--momentum`: Momentum factor for SGD optimizer (default: 0.9)
- `--weight_decay`: Weight decay (L2 penalty) for SGD optimizer (default: 0.0001)
- `--snr_min`: Minimum SNR for training/evaluation
- `--snr_max`: Maximum SNR for training/evaluation
- `--snr_step`: SNR step for training/evaluation
- `--num_trials`: Number of trials per SNR
- `--compare_with_traditional`: Compare with traditional decoders
- `--bp_max_iterations`: Maximum iterations for belief propagation decoder
- `--ms_scaling_factor`: Scaling factor for min-sum decoder
- `--model_path`: Path to save/load model
- `--results_dir`: Directory to save results

## Implementation Details

### LDPC Neural Decoder

The LDPC neural decoder is implemented as a neural network that performs belief propagation with trainable weights. The main components are:

1. **Check Layer**: Implements the check node update step using the min-sum algorithm.
2. **Variable Layer**: Implements the variable node update step.
3. **Residual Layer**: Implements residual connections between variable node layers.
4. **Output Layer**: Maps the final LLRs to soft-bit values and computes the loss.

### GNN-based LDPC Decoder

The GNN-based LDPC decoder treats the Tanner graph as a graph neural network with weight sharing:

1. **GNNCheckLayer**: Implements check node updates with shared weights.
2. **GNNVariableLayer**: Implements variable node updates with shared weights.
3. **GNNResidualLayer**: Implements residual connections with shared weights.
4. **GNNOutputLayer**: Maps final LLRs to soft-bit values and computes loss.
5. **GNNLDPCDecoder**: Combines all layers for a complete decoder.
6. **BaseGraphGNNDecoder**: Extends the GNN approach for 5G LDPC base graph structure.

For more details, see [GNN LDPC Decoder README](models/README_GNN_LDPC.md).

### Message-Centered GNN LDPC Decoder

The message-centered GNN LDPC decoder implements a novel approach by treating messages as nodes in the GNN:

1. **MessageGNNLayer**: Core component that implements message passing between message nodes.
2. **MessageGNNDecoder**: Main decoder class that manages the overall decoding process.
3. **TannerToMessageGraph**: Utility class to convert a Tanner graph to a message-centered graph.

Key advantages:
- More natural representation of belief propagation
- Enhanced parameter sharing across similar message types
- Improved generalization to different code structures
- Reduced parameter count compared to node-centered GNNs

For more details, see [Message GNN LDPC Decoder README](models/README_MESSAGE_GNN.md).

### Traditional Decoders

For comparison, the project includes implementations of traditional LDPC decoders:

1. **Belief Propagation Decoder**: Implements the sum-product algorithm.
2. **Min-Sum Scaled Decoder**: Implements the min-sum algorithm with a scaling factor.

### Training

The decoder is trained using the SGD optimizer with binary cross-entropy loss. The training process involves:

1. Generating random bits
2. Modulating using QPSK
3. Passing through an AWGN channel
4. Demodulating to LLRs
5. Decoding using the neural LDPC decoder
6. Computing the loss and updating the weights

The SGD optimizer is configured with momentum and weight decay to improve convergence and generalization.

### Evaluation

The decoder is evaluated by computing the Bit Error Rate (BER) and Frame Error Rate (FER) over a range of Signal-to-Noise Ratios (SNRs).

### Comparative Evaluation

The comparative evaluation compares the performance of:
- Neural LDPC Decoder
- GNN-based LDPC Decoder
- Message-Centered GNN LDPC Decoder
- Belief Propagation Decoder
- Min-Sum Scaled Decoder

The comparison includes:
- BER vs SNR curves
- FER vs SNR curves
- Average number of iterations for traditional decoders
- Parameter count comparison

## References

- "Deep Neural Network Based Decoding of Short 5G LDPC Codes"
- "Neural Enhanced Belief Propagation on Factor Graphs" (Satorras & Welling, 2021)
- "Message Passing Neural Networks for LDPC Decoding" (Liu & Deng, 2019)
- "Beyond Belief Propagation for LDPC Codes: Improved Decoding Using Neural Message Passing" (Dai et al., 2020)
- 5G NR LDPC codes specification (3GPP TS 38.212) 