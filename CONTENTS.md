# LDPC Neural Network Decoder Contents

This document provides quick navigation links to different parts of the repository.

## Documentation

### Main Documentation
- [Main README](README.md) - Overview and general information
- [Package README](ldpc_neural_decoder/README.md) - Package-specific details

### Architecture Documentation
- [Message GNN Architecture](ldpc_neural_decoder/models/README_MESSAGE_GNN.md) - Detailed explanation of Message-Centered GNN architecture
- [Architecture Analysis](docs/architecture/MessageGNN_Architecture_Analysis.txt) - Comprehensive analysis of the architecture
- [VariableGNN Analysis](docs/architecture/VariableGNNLayer_analysis.txt) - Detailed analysis of the Variable GNN Layer

### Diagrams and Visualizations
- [Architecture Diagrams](docs/diagrams/MessageGNN_Architecture_Diagrams.md) - Visual representations using Mermaid diagrams

### Report Information
- [Report Modification Suggestions](docs/report/Report_Modification_Suggestions.md) - Suggestions for modifying the report
- [Report Structure](docs/report/REPORT_STRUCTURE.md) - Structured outline for the report

### Code Documentation
- [VariableGNNLayer Class](docs/code/VariableGNNLayer.md) - Documentation for the VariableGNNLayer class
- [CheckGNNLayer Class](docs/code/CheckGNNLayer.md) - Documentation for the CheckGNNLayer class 
- [MessageGNNDecoder Class](docs/code/MessageGNNDecoder.md) - Documentation for the MessageGNNDecoder class
- [TannerToMessageGraph Class](docs/code/TannerToMessageGraph.md) - Documentation for the TannerToMessageGraph class
- [Utility Functions](docs/code/UtilityFunctions.md) - Documentation for utility functions

## Code

### Core Components

- [Message GNN Decoder](ldpc_neural_decoder/models/message_gnn_decoder.py) - Main implementation of Message-Centered GNN decoder
- [Trainer](ldpc_neural_decoder/training/trainer.py) - Training utilities for LDPC decoders
- [Channel Utilities](ldpc_neural_decoder/utils/channel.py) - Channel simulation (AWGN, QPSK)
- [LDPC Utilities](ldpc_neural_decoder/utils/ldpc_utils.py) - LDPC code utilities

### Examples & Scripts

- [Training Script](train_message_gnn_nr_2_0_4.py) - Script for training the Message GNN Decoder
- [Message GNN Example](ldpc_neural_decoder/examples/message_gnn_example.py) - Example usage of Message GNN Decoder
- [Debug Scripts](debug_scripts/) - Debugging and testing scripts

### Data & Models

- [5G LDPC Codes](5G%20LDPC%20CODES/) - 5G NR LDPC code definitions
  - [Base Graph 2 with Expansion Factor 4](5G%20LDPC%20CODES/NR_2_0_4.txt)
- [Saved Models](saved_models/) - Pre-trained model weights

## Architecture Overview

### Message-Centered GNN Components

1. **VariableGNNLayer** - Updates messages from variable nodes to check nodes
   - Takes LLR values, variable messages, check messages, message types, and adjacency information
   - Applies message type embeddings to enrich features
   - Gathers messages from variables sharing the same check node
   - Combines with check messages from previous iteration
   - Produces updated variable-side messages

2. **CheckGNNLayer** - Updates messages from check nodes to variable nodes
   - Updates messages based on connections through check nodes
   - Mirrors the Variable GNN Layer structure
   - Processes information from check to variable nodes

3. **MessageGNNDecoder** - Main decoder class
   - Integrates multiple Variable and Check GNN layers
   - Manages message queues for residual connections
   - Implements the alternating message-passing scheme
   - Includes a parameter-free output mapping layer

4. **TannerToMessageGraph** - Converter for graph transformation
   - Converts a traditional Tanner graph to a message-centered graph
   - Creates adjacency matrices for message passing
   - Maps between variable/check nodes and messages

## Quick Start Guides

### Installation

```bash
git clone https://github.com/yourusername/LDPC-NeuralNetwork-Decoder.git
cd LDPC-NeuralNetwork-Decoder
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Training a Model

```bash
python train_message_gnn_nr_2_0_4.py
```

### Evaluation

```bash
python -m ldpc_neural_decoder.main --mode evaluate --model_path saved_models/message_gnn_nr_2_0_4.pth
```

### Comparative Evaluation

```bash
python -m ldpc_neural_decoder.main --mode compare --model_path saved_models/message_gnn_nr_2_0_4.pth --compare_with_traditional
``` 