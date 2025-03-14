#!/usr/bin/env python3
"""
Example script for using the CustomVariableMessageGNNDecoder with traditional min-sum variable layer update.

This script demonstrates how to:
1. Load a small LDPC code
2. Create a CustomVariableMessageGNNDecoder
3. Train and evaluate the decoder
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import create_custom_variable_message_gnn_decoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import AWGNChannel
from ldpc_neural_decoder.training.trainer import Trainer


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load a small base matrix
    base_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "small_base_graph.txt")
    base_matrix = load_base_matrix(base_matrix_path)
    
    # Expand the base matrix
    Z = 4  # Lifting factor
    H = expand_base_matrix(base_matrix, Z)
    
    # Convert to PyTorch tensor if it's not already
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
    
    # Create the decoder and converter using the helper function
    decoder, converter = create_custom_variable_message_gnn_decoder(
        H=H,
        num_iterations=5,
        hidden_dim=64,
        depth_L=3
    )
    
    # Create the channel
    channel = AWGNChannel()
    
    # Create the trainer
    trainer = Trainer(
        model=decoder,
        channel=channel,
        message_to_var_mapping=converter.message_to_var_mapping,
        var_to_check_adjacency=converter.var_to_check_adjacency,
        check_to_var_adjacency=converter.check_to_var_adjacency,
        learning_rate=0.001,
        weight_decay=1e-5
    )
    
    # Training parameters
    num_epochs = 50
    batch_size = 32
    snr_train = 2.0  # Training SNR in dB
    
    # Train the decoder
    print("Training the decoder...")
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(batch_size=batch_size, snr=snr_train)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}")
    
    # Evaluate the decoder
    print("\nEvaluating the decoder...")
    snr_range = np.arange(0, 6.1, 1.0)
    ber_results = []
    fer_results = []
    
    for snr in snr_range:
        ber, fer = trainer.evaluate(batch_size=1000, snr=snr)
        ber_results.append(ber)
        fer_results.append(fer)
        print(f"SNR: {snr:.1f} dB, BER: {ber:.6f}, FER: {fer:.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'o-', label='BER')
    plt.semilogy(snr_range, fer_results, 's-', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.legend()
    plt.title('Performance of CustomVariableMessageGNNDecoder with Min-Sum Variable Layer')
    plt.savefig('custom_variable_layer_performance.png')
    plt.show()


if __name__ == "__main__":
    main() 