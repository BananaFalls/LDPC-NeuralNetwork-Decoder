# Neural LDPC Decoder

A Graph Neural Network (GNN) based decoder for Low-Density Parity-Check (LDPC) codes with a focus on 5G NR LDPC codes.

## Overview

This project implements a neural network-based decoder for LDPC codes that focuses on a message-passing framework. The key approach treats messages in the Tanner graph as nodes in a GNN, which offers improved parameter efficiency and better generalization across different code structures and SNR environments.

## Features

- **Message GNN Decoder**: Novel GNN decoder treating messages as graph nodes
- **Weight Sharing**: Efficient parameter sharing based on message types in the base graph
- **Residual Connections**: Implementation of residual connections for improved gradient flow
- **QPSK Modulation**: Support for QPSK modulation and demodulation
- **Comprehensive Training**: Modular training framework with support for various SNR ranges
- **Performance Evaluation**: Tools for measuring BER (Bit Error Rate) and FER (Frame Error Rate)

## Message-Centered GNN Architecture

The core innovation in this decoder is the message-centered representation:

### Key Innovations

1. **Message-Centered Representation**:
   - Traditional Tanner graphs represent variable and check nodes as vertices
   - Our architecture treats messages (traditionally edges) as nodes in a new graph
   - New edges connect messages that share the same variable or check node
   - This enables more complex message representations and update rules

2. **Neural Message Passing**:
   - Updates to variable-to-check and check-to-variable messages are learned
   - High-dimensional message representations (configurable hidden dimensions)
   - Alternating layer architecture mimics belief propagation iterations
   - Weight sharing through message type embeddings reduces parameter count

3. **Multi-Layer Residual Connections**:
   - Multiple residual connections across iterations facilitate gradient flow
   - Original LLR values are preserved through direct connections
   - Enables training of deeper networks (more iterations)
   - A queue of previous variable-to-check messages is maintained for residual connections

### Architecture Components

1. **VariableGNNLayer**: Updates messages from variable nodes to check nodes
2. **CheckGNNLayer**: Updates messages from check nodes to variable nodes
3. **MessageGNNDecoder**: Main decoder class integrating multiple Variable and Check GNN layers
4. **TannerToMessageGraph**: Converts a traditional Tanner graph to a message-centered graph

For more detailed documentation on the architecture, refer to the following files:
- `MessageGNN_Architecture_Analysis.txt`: Comprehensive analysis of the architecture
- `MessageGNN_Architecture_Diagrams.md`: Visual representations using Mermaid diagrams
- `VariableGNNLayer_analysis.txt`: Detailed analysis of the Variable GNN Layer

## Repository Structure

```
LDPC-NeuralNetwork-Decoder/
├── 5G LDPC CODES/           # 5G NR LDPC code definitions
│   ├── NR_2_0_4.txt         # Base graph 2 with expansion factor 4
│   └── ...                  # Other LDPC code definitions
├── ldpc_neural_decoder/     # Main package
│   ├── models/              # Neural network models
│   │   ├── message_gnn_decoder.py  # Message-centered GNN decoder
│   │   └── ...              # Other model components
│   ├── training/            # Training utilities
│   │   ├── trainer.py       # Trainer implementation
│   │   └── ...              # Other training utilities
│   ├── utils/               # Utility functions
│   │   ├── channel.py       # Channel simulation (AWGN, QPSK)
│   │   ├── ldpc_utils.py    # LDPC code utilities
│   │   └── ...              # Other utilities
│   └── visualization/       # Visualization tools
├── saved_models/            # Saved model weights
│   └── message_gnn_nr_2_0_4.pth  # Trained model for NR_2_0_4 code
├── debug_scripts/           # Debug and testing scripts
│   ├── debug_variable_gnn_layer.py  # Test for variable GNN layer
│   └── ...                  # Other debug scripts
├── train_message_gnn_nr_2_0_4.py  # Training script for NR_2_0_4
└── ...                      # Other files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LDPC-NeuralNetwork-Decoder.git
cd LDPC-NeuralNetwork-Decoder
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

4. Install the package in development mode:
```bash
pip install -e .
```

## Usage

### Training a new model

To train a new Message GNN Decoder model on the NR_2_0_4 code:

```bash
python train_message_gnn_nr_2_0_4.py
```

### Evaluating performance

To evaluate the performance of a trained model:

```python
import torch
from ldpc_neural_decoder.training.trainer import LDPCDecoderTrainer

# Load a trained model
trainer = LDPCDecoderTrainer.load_from_checkpoint('saved_models/message_gnn_nr_2_0_4.pth')

# Evaluate over an SNR range
snr_range = np.arange(0, 5.5, 0.5)
ber_results, fer_results = trainer.evaluate_snr_range(snr_range, num_codewords=1000)

# Plot results
import matplotlib.pyplot as plt
plt.semilogy(snr_range, ber_results, 'o-', label='BER')
plt.semilogy(snr_range, fer_results, 's-', label='FER')
plt.grid(True)
plt.xlabel('SNR (dB)')
plt.ylabel('Error Rate')
plt.legend()
plt.show()
```

## Message Passing Architecture

The neural decoder implements a message-passing algorithm where:

1. **Message Representation**: Messages are represented as nodes in a graph
2. **Variable-Check Message Flow**: Messages flow between variable nodes and check nodes
3. **Residual Connections**: Previous variable-to-check messages are used as residual connections
4. **Type Embeddings**: Message types are embedded in a high-dimensional space for richer representation

The general flow follows:
- Input LLRs → Variable Layer → Check Layer → ... (repeat) → Output Probabilities

## Implementation Details

### Message Types

The decoder leverages the structure of the LDPC code's base graph to assign types to messages. Messages of the same type in the expanded graph share weights, which significantly reduces the number of parameters while maintaining or improving performance.

### Residual Connections

Residual connections are implemented by maintaining a queue of previous variable-to-check messages. These are added to current messages to improve gradient flow and help with convergence during training.

### Output Mapping

The final layer maps high-dimensional message features back to bit probabilities using:
1. Parameter-free projection to scalar LLRs
2. Combining messages connected to the same variable node
3. Adding original input LLRs from channel estimation
4. Converting to probabilities using sigmoid function

### Advantages Over Traditional Decoders

1. **Learned Update Rules**: Instead of using fixed mathematical formulas, our decoder learns optimal message update functions from data.
2. **Rich Message Representation**: High-dimensional message features capture more complex dependencies than scalar LLRs.
3. **Residual Connections**: Preserve information across iterations and improve gradient flow.
4. **Flexible Graph Structure**: Can represent arbitrary code structures and handle irregular LDPC codes naturally.

## References

- "Neural Enhanced Belief Propagation on Factor Graphs" (Satorras & Welling, 2021)
- "Learning to Decode: Deep Learning for LDPC Decoding" (Nachmani et al., 2016)
- "Improved Decoding of LDPC Codes Using Message-Passing Neural Networks" (Kim et al., 2020)
- 5G NR LDPC codes specification (3GPP TS 38.212)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 