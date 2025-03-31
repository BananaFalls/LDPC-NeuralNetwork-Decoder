# Project Contents Overview

## Core Components

### 1. Decoder Models
- **Weight Sharing Decoder** (`models/weight_sharing_decoder.py`)
  - Traditional neural decoder with weight sharing
  - Efficient parameter sharing based on base matrix structure
  - Configurable number of iterations

- **Residual Weight Sharing Decoder** (`models/residual_weight_sharing_decoder.py`)
  - Enhanced decoder with residual connections
  - Improved gradient flow
  - Configurable residual depth
  - Better convergence properties

### 2. Training Utilities
- **LLR Generator** (`utils/llr_generator.py`)
  - Generates Log-Likelihood Ratios from zero codewords
  - Implements BPSK modulation
  - Adds AWGN noise
  - Calculates LLRs using formula: \( \frac{2y}{\sigma^2} \)

- **Training Data Generator** (`utils/training_data_generator.py`)
  - Generates training, validation, and test datasets
  - Configurable batch sizes and SNR ranges
  - Saves datasets in PyTorch format

### 3. Debug and Testing Scripts
- **Test Weight Sharing Decoder** (`debug_scripts/test_weight_sharing_decoder.py`)
  - Tests traditional decoder implementation
  - Verifies decoding performance
  - Uses small example matrix

- **Test Residual Decoder** (`debug_scripts/test_residual_decoder.py`)
  - Tests residual decoder implementation
  - Uses 5G LDPC base matrix
  - Verifies residual connections

- **Train Residual Decoder** (`debug_scripts/train_residual_decoder.py`)
  - Main training script
  - Implements SGD optimizer
  - Includes learning rate scheduling
  - Generates BER and FER plots
  - Saves model checkpoints

- **Generate Training Data** (`debug_scripts/generate_training_data.py`)
  - Creates training datasets
  - Configurable parameters
  - Splits data into train/val/test sets

## Data and Results

### 1. Training Data
- **Location**: `training_data/`
- **Contents**:
  - `train_data_z4.pt`: Training dataset
  - `val_data_z4.pt`: Validation dataset
  - `test_data_z4.pt`: Test dataset

### 2. Training Results
- **Location**: `training_results/`
- **Contents**:
  - `ber_plot.png`: Bit Error Rate plot
  - `fer_plot.png`: Frame Error Rate plot

### 3. Model Checkpoints
- **Location**: `checkpoints/residual_decoder_z4_[timestamp]/`
- **Contents**:
  - `best_model.pt`: Best model based on validation loss
  - `checkpoint_epoch_N.pt`: Regular checkpoints
  - `config.json`: Training configuration

## Configuration Files

### 1. Base Matrices
- **Location**: `5G LDPC CODES/`
- **Contents**:
  - `NR_2_0_4.txt`: 5G LDPC base matrix
  - Other code definitions

### 2. Requirements
- **Location**: `requirements.txt`
- **Contents**:
  - Python package dependencies
  - Version specifications

## Documentation

### 1. Main Documentation
- **README.md**: Project overview and setup instructions
- **CONTENTS.md**: This file, detailed component overview

### 2. Code Documentation
- Inline documentation in all Python files
- Detailed function and class descriptions
- Usage examples in docstrings

## Usage Examples

### 1. Training Process
```python
# Generate training data
python debug_scripts/generate_training_data.py

# Train the decoder
python debug_scripts/train_residual_decoder.py

# Test the decoder
python debug_scripts/test_residual_decoder.py
```

### 2. Model Parameters
```python
# Key parameters for residual decoder
z = 4              # Expansion factor
num_iterations = 8 # Number of iterations
residual_depth = 3 # Residual connection depth
learning_rate = 1e-3
```

### 3. Training Configuration
```python
# Training settings
num_epochs = 50
batch_size = 64
snr_range = [-1, 8]  # dB
snr_points = 10
```

## Directory Structure
```
LDPC-NeuralNetwork-Decoder/
├── ldpc_neural_decoder/     # Main package
├── debug_scripts/          # Testing and training scripts
├── training_data/         # Generated datasets
├── training_results/      # Training plots
├── checkpoints/          # Model checkpoints
└── 5G LDPC CODES/       # Code definitions
``` 