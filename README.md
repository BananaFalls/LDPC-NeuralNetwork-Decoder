# Neural LDPC Decoder

This project implements a neural network-based decoder for Low-Density Parity-Check (LDPC) codes, specifically focusing on 5G LDPC codes. The implementation includes both traditional and residual weight-sharing decoders.

## Features

- Implementation of 5G LDPC codes with configurable expansion factor
- Neural network-based decoder with weight sharing
- Residual connections for improved gradient flow
- Training data generation with BPSK modulation and AWGN noise
- Comprehensive training pipeline with metrics tracking
- Visualization of training progress (BER and FER plots)

## Project Structure

```
LDPC-NeuralNetwork-Decoder/
├── ldpc_neural_decoder/
│   ├── models/
│   │   ├── weight_sharing_decoder.py
│   │   └── residual_weight_sharing_decoder.py
│   └── utils/
│       ├── llr_generator.py
│       └── training_data_generator.py
├── debug_scripts/
│   ├── test_weight_sharing_decoder.py
│   ├── test_residual_decoder.py
│   ├── train_residual_decoder.py
│   └── generate_training_data.py
├── training_data/
│   ├── train_data_z4.pt
│   ├── val_data_z4.pt
│   └── test_data_z4.pt
├── training_results/
│   ├── ber_plot.png
│   └── fer_plot.png
└── checkpoints/
    └── residual_decoder_z4_[timestamp]/
        ├── best_model.pt
        ├── checkpoint_epoch_N.pt
        └── config.json
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LDPC-NeuralNetwork-Decoder.git
cd LDPC-NeuralNetwork-Decoder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Data Generation

Generate training data with specified parameters:
```bash
python debug_scripts/generate_training_data.py
```

This will create:
- Training set (80% of data)
- Validation set (10% of data)
- Test set (10% of data)

### Training the Residual Decoder

Train the residual weight-sharing decoder:
```bash
python debug_scripts/train_residual_decoder.py
```

The training script will:
- Create checkpoints directory with timestamp
- Save best model based on validation loss
- Generate BER and FER plots
- Track training progress

### Testing the Decoder

Test the decoder's performance:
```bash
python debug_scripts/test_residual_decoder.py
```

## Model Architecture

### Residual Weight-Sharing Decoder

The residual decoder extends the traditional weight-sharing decoder with:
- Residual connections between iterations
- Configurable residual depth
- Improved gradient flow
- Better convergence properties

Key parameters:
- Expansion factor (z)
- Number of iterations
- Residual depth
- Learning rate
- Batch size

## Training Process

The training process includes:
1. Data generation with BPSK modulation and AWGN noise
2. Training loop with SGD optimizer
3. Validation after each epoch
4. Learning rate scheduling
5. Checkpoint saving
6. Metrics visualization

## Results

Training results are stored in:
- `training_results/`: Contains BER and FER plots
- `checkpoints/`: Contains model checkpoints and configurations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 