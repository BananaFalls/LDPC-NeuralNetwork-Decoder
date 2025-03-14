# Message-Centered GNN LDPC Decoder

A novel approach to LDPC decoding using Graph Neural Networks (GNNs) where messages in the Tanner graph are treated as nodes in the GNN.

## Overview

This project implements several neural network-based decoders for Low-Density Parity-Check (LDPC) codes, with a focus on a novel message-centered GNN approach. The key innovation is treating messages (edges in the Tanner graph) as nodes in a GNN, which offers improved parameter efficiency and better generalization.

## Key Features

- **Message-Centered GNN Decoder**: A novel approach that treats messages as nodes in the GNN
- **Standard Neural LDPC Decoder**: Traditional neural decoder with learnable weights
- **GNN-based LDPC Decoder**: GNN decoder with variable and check nodes
- **Traditional Decoders**: Belief Propagation and Min-Sum Scaled decoders for comparison
- **Comprehensive Evaluation**: Tools for comparing all decoder types

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
git clone https://github.com/BananaFalls/LDPC-NeuralNetwork-Decoder.git
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

## Usage

### Message-Centered GNN LDPC Decoder

To train the message-centered GNN LDPC decoder:

```bash
python -m ldpc_neural_decoder.examples.message_gnn_example --mode train
```

For evaluation:

```bash
python -m ldpc_neural_decoder.examples.message_gnn_example --mode evaluate
```

### Complete Workflow

To run the complete workflow (training, evaluation, comparison, and visualization):

```bash
python -m ldpc_neural_decoder.run_workflow
```

### Comparison of All Decoders

To compare all decoder types:

```bash
python -m ldpc_neural_decoder.run_comparison_all
```

## Key Innovations

### Message-Centered Approach

The message-centered approach treats each message in the Tanner graph as a node in the GNN, which offers several advantages:

1. **More Natural Representation**: Messages are the fundamental units of information exchange in belief propagation.
2. **Enhanced Parameter Sharing**: Better sharing of parameters across similar message types.
3. **Improved Generalization**: Better generalization to different code structures and sizes.
4. **Reduced Parameter Count**: Achieves similar or better performance with fewer parameters.
5. **Direct Message Manipulation**: Allows direct learning of message update rules.

### Weight Sharing

The message-centered GNN implements weight sharing through:

1. **Message Type Embeddings**: Messages of the same type share the same embeddings.
2. **Shared Neural Networks**: The same neural networks are used for updating all messages.
3. **Base Graph Structure**: For 5G LDPC codes, the base graph structure provides natural message typing.

## Documentation

For more detailed documentation, see:

- [Neural Components](ldpc_neural_decoder/models/NEURAL_COMPONENTS.md): Detailed explanation of neural network components
- [Architecture Diagrams](ldpc_neural_decoder/models/ARCHITECTURE_DIAGRAM.md): Visual representations of decoder architectures
- [Message GNN README](ldpc_neural_decoder/models/README_MESSAGE_GNN.md): Comprehensive documentation of the message-centered GNN approach

## References

- "Neural Enhanced Belief Propagation on Factor Graphs" (Satorras & Welling, 2021)
- "Message Passing Neural Networks for LDPC Decoding" (Liu & Deng, 2019)
- "Beyond Belief Propagation for LDPC Codes: Improved Decoding Using Neural Message Passing" (Dai et al., 2020)
- 5G NR LDPC codes specification (3GPP TS 38.212)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 