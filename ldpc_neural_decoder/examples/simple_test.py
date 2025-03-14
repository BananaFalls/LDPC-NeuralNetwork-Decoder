"""
Simple test script for Message-Centered GNN LDPC decoder.

This script demonstrates how to use the Message-Centered GNN LDPC decoder
with a small base graph matrix.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer


def main():
    """Main function to test the Message-Centered GNN LDPC decoder."""
    print("Starting simple test for Message-Centered GNN LDPC decoder...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base matrix
    base_matrix_path = os.path.join(os.path.dirname(__file__), 'small_base_graph.txt')
    print(f"Loading base matrix from: {base_matrix_path}")
    base_matrix = load_base_matrix(base_matrix_path)
    print(f"Base matrix shape: {base_matrix.shape}")
    
    # Set lifting factor
    lifting_factor = 4
    print(f"Using lifting factor: {lifting_factor}")
    
    # Expand base matrix to parity-check matrix
    H = expand_base_matrix(base_matrix, lifting_factor)
    print(f"Parity-check matrix shape: {H.shape}")
    
    # Create message-centered GNN decoder
    num_iterations = 3
    hidden_dim = 32
    print(f"Creating decoder with {num_iterations} iterations and hidden dimension {hidden_dim}...")
    
    decoder, converter = create_message_gnn_decoder(
        H.to(device),
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
        base_graph=base_matrix.to(device),
        Z=lifting_factor
    )
    
    print(f"Decoder created. Number of parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")
    print(f"Number of messages: {len(converter.messages)}")
    print(f"Number of message types: {len(torch.unique(converter.get_message_types(base_matrix, lifting_factor)))}")
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        decoder.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Training parameters
    batch_size = 8
    num_epochs = 3
    snr_db = 2.0
    
    print(f"\nTraining for {num_epochs} epochs with batch size {batch_size} at SNR {snr_db} dB...")
    
    # Training loop
    train_losses = []
    ber_history = []
    fer_history = []
    
    for epoch in range(num_epochs):
        decoder.train()
        
        # Generate random bits
        transmitted_bits = torch.randint(0, 2, (batch_size, converter.num_variables), device=device).float()
        
        # Modulate using QPSK
        qpsk_symbols = qpsk_modulate(transmitted_bits)
        
        # Pass through AWGN channel
        received_signal = awgn_channel(qpsk_symbols, snr_db)
        
        # Demodulate to LLRs
        llrs = qpsk_demodulate(received_signal, snr_db)
        
        # Reshape LLRs to match the original shape of transmitted_bits
        llrs = llrs.reshape(transmitted_bits.shape)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        soft_bits, loss = decoder(
            llrs,
            converter.message_to_var_mapping,
            message_types=converter.get_message_types(base_matrix, lifting_factor),
            var_to_check_adjacency=converter.var_to_check_adjacency,
            check_to_var_adjacency=converter.check_to_var_adjacency,
            ground_truth=transmitted_bits
        )
        
        # Compute mean loss
        batch_loss = loss.mean()
        
        # Backward pass
        batch_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Compute BER and FER
        hard_bits = (soft_bits > 0.5).float()
        ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
        
        # Store history
        train_losses.append(batch_loss.item())
        ber_history.append(ber)
        fer_history.append(fer)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {batch_loss.item():.6f}, BER: {ber:.6f}, FER: {fer:.6f}")
    
    print("\nTraining completed.")
    
    # Evaluation
    print("\nEvaluating decoder...")
    decoder.eval()
    
    # Generate all-zero codeword
    transmitted_bits = torch.zeros((batch_size, converter.num_variables), device=device)
    
    # Modulate using QPSK
    qpsk_symbols = qpsk_modulate(transmitted_bits)
    
    # Pass through AWGN channel
    received_signal = awgn_channel(qpsk_symbols, snr_db)
    
    # Demodulate to LLRs
    llrs = qpsk_demodulate(received_signal, snr_db)
    
    # Reshape LLRs to match the original shape of transmitted_bits
    llrs = llrs.reshape(transmitted_bits.shape)
    
    # Decode
    with torch.no_grad():
        hard_bits = decoder.decode(
            llrs,
            converter.message_to_var_mapping,
            message_types=converter.get_message_types(base_matrix, lifting_factor),
            var_to_check_adjacency=converter.var_to_check_adjacency,
            check_to_var_adjacency=converter.check_to_var_adjacency
        )
    
    # Compute BER and FER
    ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
    
    print(f"Evaluation results - BER: {ber:.6f}, FER: {fer:.6f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    print(f"Training loss plot saved to training_loss.png")
    
    # Plot BER and FER
    plt.figure(figsize=(10, 6))
    plt.semilogy(ber_history, 'o-', label='BER')
    plt.semilogy(fer_history, 's-', label='FER')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('BER and FER History')
    plt.legend()
    plt.grid(True)
    plt.savefig('error_rates.png')
    print(f"Error rates plot saved to error_rates.png")
    
    print("\nSimple test completed successfully!")


if __name__ == '__main__':
    main() 