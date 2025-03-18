#!/usr/bin/env python3
"""
Run Message-Centered GNN LDPC Decoder with Enhanced Architecture

This script demonstrates the use of the enhanced Message-Centered GNN Decoder with alternating
variable and check layers, deep residual connections, and weight sharing from the base graph.

Key Steps:
1. Load a small base graph
2. Expand the base graph to a parity-check matrix
3. Create a message-centered GNN decoder with alternating layers
4. Generate random codewords
5. Add noise to the codewords
6. Run the decoder on the noisy codewords
7. Calculate bit error rates

The script uses a small base graph from 'small_base_graph.txt' and expands it
with a lifting factor of 4.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# Import the message GNN decoder
from ldpc_neural_decoder.models.message_gnn_decoder import (
    MessageGNNDecoder, 
    TannerToMessageGraph,
    create_message_gnn_decoder,
    create_variable_index_tensor,
    create_message_type_mapping
)

def load_base_graph(filename):
    """
    Load a base graph from a file.
    
    Args:
        filename (str): Path to the base graph file
        
    Returns:
        np.ndarray: Base graph matrix
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse dimensions from the first line
    m, n = map(int, lines[0].strip().split())
    
    # Create base graph matrix
    base_graph = np.zeros((m, n), dtype=int)
    
    # Fill the matrix with the shift values
    for i in range(m):
        row_values = list(map(int, lines[i+1].strip().split()))
        for j in range(n):
            base_graph[i, j] = row_values[j]
    
    print(f"Loaded base graph with shape {base_graph.shape}")
    return base_graph

def expand_base_graph(base_graph, Z):
    """
    Expand a base graph into a parity-check matrix using the lifting factor Z.
    
    Args:
        base_graph (np.ndarray): Base graph matrix
        Z (int): Lifting factor
        
    Returns:
        np.ndarray: Expanded parity-check matrix
    """
    m, n = base_graph.shape
    H = np.zeros((m * Z, n * Z), dtype=int)
    
    for i in range(m):
        for j in range(n):
            if base_graph[i, j] >= 0:  # -1 means no connection
                # Place a 1 in the corresponding submatrix
                row_offset = i * Z
                col_offset = j * Z
                shift = base_graph[i, j] % Z
                
                # Create identity matrix shifted by 'shift' positions
                for k in range(Z):
                    H[row_offset + k, col_offset + ((k + shift) % Z)] = 1
    
    print(f"Expanded base graph to parity-check matrix with shape {H.shape}")
    return H

def generate_zero_codewords(n, num_codewords):
    """
    Generate all-zero codewords for testing LDPC decoders.
    
    For linear codes like LDPC codes, the all-zero codeword is commonly used for testing
    since any codeword has the same error performance due to the linearity property.
    
    Args:
        n (int): Length of each codeword
        num_codewords (int): Number of codewords to generate
        
    Returns:
        np.ndarray: Generated zero codewords
    """
    return np.zeros((num_codewords, n), dtype=int)

def add_noise_qpsk(codewords, snr_db=None, min_snr=0.0, max_snr=10.0, variable_snr=False):
    """
    Add AWGN noise to QPSK-modulated codewords.
    
    Args:
        codewords (np.ndarray): Binary codewords
        snr_db (float, optional): Fixed SNR value in dB (used if variable_snr=False)
        min_snr (float): Minimum SNR value in dB (used if variable_snr=True)
        max_snr (float): Maximum SNR value in dB (used if variable_snr=True)
        variable_snr (bool): Whether to use variable SNR values for each codeword
        
    Returns:
        tuple: (noisy_llrs, symbols, noise, snr_values)
    """
    # Convert bits to QPSK symbols (using Gray coding)
    num_codewords, n = codewords.shape
    bits_reshaped = codewords.reshape(num_codewords, -1, 2)
    
    # Map bit pairs to QPSK symbols
    symbols = np.zeros((num_codewords, bits_reshaped.shape[1]), dtype=complex)
    for i in range(num_codewords):
        for j in range(bits_reshaped.shape[1]):
            bit_pair = bits_reshaped[i, j]
            if np.array_equal(bit_pair, [0, 0]):
                symbols[i, j] = (1 + 1j) / np.sqrt(2)  # First quadrant
            elif np.array_equal(bit_pair, [0, 1]):
                symbols[i, j] = (1 - 1j) / np.sqrt(2)  # Fourth quadrant
            elif np.array_equal(bit_pair, [1, 0]):
                symbols[i, j] = (-1 + 1j) / np.sqrt(2)  # Second quadrant
            else:  # [1, 1]
                symbols[i, j] = (-1 - 1j) / np.sqrt(2)  # Third quadrant
    
    # Generate SNR values for each codeword
    if variable_snr:
        print(f"Using variable SNR values within range [{min_snr:.1f}, {max_snr:.1f}] dB")
        snr_values = np.random.uniform(min_snr, max_snr, size=num_codewords)
    else:
        if snr_db is None:
            snr_db = 5.0  # Default value
        print(f"Using fixed SNR value: {snr_db} dB")
        snr_values = np.ones(num_codewords) * snr_db
    
    # Initialize arrays for noise and received symbols
    noise = np.zeros_like(symbols, dtype=complex)
    received_symbols = np.zeros_like(symbols, dtype=complex)
    noisy_llrs = np.zeros_like(codewords, dtype=float)
    
    # Apply noise with different SNR for each codeword
    for i in range(num_codewords):
        # Calculate noise standard deviation based on SNR for this codeword
        snr_linear = 10**(snr_values[i]/10)
        noise_std = 1 / np.sqrt(2 * snr_linear)
        
        # Add complex Gaussian noise to this codeword
        noise[i] = noise_std * (np.random.randn(symbols.shape[1]) + 1j * np.random.randn(symbols.shape[1]))
        received_symbols[i] = symbols[i] + noise[i]
        
        # Calculate LLRs for each bit in this codeword
        for j in range(bits_reshaped.shape[1]):
            # First bit LLR
            noisy_llrs[i, j*2] = -2 * np.sqrt(2) * np.real(received_symbols[i, j]) / (noise_std**2)
            
            # Second bit LLR
            noisy_llrs[i, j*2+1] = -2 * np.sqrt(2) * np.imag(received_symbols[i, j]) / (noise_std**2)
    
    # Plot the first QPSK constellation for visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(symbols[0]), np.imag(symbols[0]), c='blue', label='Original')
    plt.scatter(np.real(received_symbols[0]), np.imag(received_symbols[0]), c='red', label='Received')
    
    # Draw decision boundaries
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'QPSK Constellation (SNR = {snr_values[0]:.1f} dB)')
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.savefig('qpsk_constellation.png')
    
    return noisy_llrs, symbols, noise, snr_values

def plot_ber_comparison(snr_values, ber_standard, ber_custom):
    """
    Plot BER comparison between standard and custom decoders.
    
    Args:
        snr_values (list): SNR values in dB
        ber_standard (list): BER values for standard decoder
        ber_custom (list): BER values for custom decoder
    """
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, ber_standard, 'o-', label='Standard Decoder')
    plt.semilogy(snr_values, ber_custom, 's-', label='Enhanced GNN Decoder')
    
    plt.grid(True, which="both", ls="--")
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Decoder Performance Comparison')
    plt.legend()
    plt.savefig('ber_comparison.png')

def main():
    """Main function to run the Message GNN decoder demo."""
    print("\n===== Message GNN Decoder Demo with Enhanced Architecture =====\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load base graph
    base_graph_path = os.path.join(os.path.dirname(__file__), 'small_base_graph.txt')
    base_graph = load_base_graph(base_graph_path)
    
    # Lifting factor
    Z = 4
    
    # Expand base graph to parity-check matrix
    H = expand_base_graph(base_graph, Z)
    
    # Convert to PyTorch tensor
    H_tensor = torch.from_numpy(H).float()
    base_graph_tensor = torch.from_numpy(base_graph).float()
    
    # Settings for the decoder
    num_iterations = 5
    hidden_dim = 64
    
    print("\n----- Creating Message GNN Decoder -----")
    
    # Create message GNN decoder with weight sharing from the base graph
    print("Creating enhanced Message GNN Decoder with weight sharing from base graph")
    decoder, converter = create_message_gnn_decoder(
        H=H_tensor,
        base_graph=base_graph_tensor,
        lifting_factor=Z,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim
    )
    
    # Create standard decoder for comparison
    print("Creating standard decoder for comparison")
    # This would typically be another decoder like Sum-Product or Min-Sum
    # For this demo, we'll just use a simplified version
    
    # Print some info about the message graph
    num_messages = converter.num_messages
    message_types = converter.message_types
    
    print(f"Number of message nodes: {num_messages}")
    print(f"Number of unique message types: {len(torch.unique(message_types))}")
    print(f"Message type distribution: {torch.bincount(message_types)}")
    
    # Get adjacency matrices
    var_to_check_adjacency = converter.var_to_check_adjacency
    check_to_var_adjacency = converter.check_to_var_adjacency
    
    print(f"Variable-to-check adjacency matrix shape: {var_to_check_adjacency.shape}")
    print(f"Check-to-variable adjacency matrix shape: {check_to_var_adjacency.shape}")
    
    # Get mappings
    message_to_var_mapping = converter.get_message_to_var_mapping()
    
    print(f"Message-to-variable mapping shape: {message_to_var_mapping.shape}")
    
    # Generate zero codewords for testing
    print("\n----- Generating Zero Codewords -----")
    num_codewords = 10
    codewords = generate_zero_codewords(H.shape[1], num_codewords)
    codewords_tensor = torch.from_numpy(codewords).float()
    
    print(f"Generated {num_codewords} zero codewords")
    
    # Add noise using QPSK modulation with variable SNR values
    print("\n----- Adding Noise (QPSK Modulation) -----")
    
    # SNR range where FER is between 10^-1 and 10^-8 for increased observation range
    min_snr = 0.0   # Lower SNR for FER ~10^-1
    max_snr = 10.0  # Higher SNR for FER ~10^-8
    
    noisy_llrs, symbols, noise, snr_values = add_noise_qpsk(
        codewords, 
        variable_snr=True,
        min_snr=min_snr,
        max_snr=max_snr
    )
    noisy_llrs_tensor = torch.from_numpy(noisy_llrs).float()
    
    print(f"Added noise with variable SNR values: min={snr_values.min():.2f} dB, max={snr_values.max():.2f} dB")
    print(f"LLR values range: [{noisy_llrs.min():.4f}, {noisy_llrs.max():.4f}]")
    
    # Run decoders
    print("\n----- Running Decoders -----")
    with torch.no_grad():
        # Run enhanced Message GNN decoder
        decoder_output = decoder(
            input_llr=noisy_llrs_tensor,
            message_to_var_mapping=message_to_var_mapping,
            message_types=message_types,
            var_to_check_adjacency=var_to_check_adjacency,
            check_to_var_adjacency=check_to_var_adjacency
        )
        
        # Extract probabilities and convert to bits
        decoded_probs = decoder_output
        decoded_bits = (decoded_probs > 0.5).float().numpy()
    
    # Run "standard" decoder (simplified for this demo)
    standard_decoded_bits = (noisy_llrs < 0).astype(float)
    
    # Calculate bit error rates per codeword and overall
    ber_per_codeword_gnn = np.mean(np.abs(decoded_bits - codewords), axis=1)
    ber_per_codeword_standard = np.mean(np.abs(standard_decoded_bits - codewords), axis=1)
    
    ber_gnn = np.mean(ber_per_codeword_gnn)
    ber_standard = np.mean(ber_per_codeword_standard)
    
    # Calculate frame error rates (probability of at least one error in a codeword)
    frame_errors_gnn = np.any(decoded_bits != codewords, axis=1)
    frame_errors_standard = np.any(standard_decoded_bits != codewords, axis=1)
    fer_gnn = np.mean(frame_errors_gnn)
    fer_standard = np.mean(frame_errors_standard)
    
    print(f"\nPerformance Metrics:")
    print(f"Bit Error Rate (Enhanced GNN Decoder): {ber_gnn:.4f}")
    print(f"Bit Error Rate (Standard Decoder): {ber_standard:.4f}")
    print(f"Frame Error Rate (Enhanced GNN Decoder): {fer_gnn:.4f}")
    print(f"Frame Error Rate (Standard Decoder): {fer_standard:.4f}")
    
    # Print per-codeword results with SNRs to see correlation
    print("\n----- Per-Codeword Results -----")
    print(f"{'SNR (dB)':10s} | {'BER (GNN)':10s} | {'BER (Std)':10s} | {'GNN Status':10s} | {'Std Status':10s}")
    print("-" * 65)
    
    for i in range(num_codewords):
        gnn_status = "Error" if frame_errors_gnn[i] else "Correct"
        std_status = "Error" if frame_errors_standard[i] else "Correct"
        print(f"{snr_values[i]:10.2f} | {ber_per_codeword_gnn[i]:10.4f} | {ber_per_codeword_standard[i]:10.4f} | {gnn_status:10s} | {std_status:10s}")
    
    # Display sample results for the first codeword
    print("\n----- Sample Results (First Codeword) -----")
    print(f"Original:       {codewords[0][:20]}")
    print(f"Decoded (GNN):  {decoded_bits[0][:20]}")
    print(f"Decoded (Std):  {standard_decoded_bits[0][:20]}")
    print(f"SNR: {snr_values[0]:.2f} dB")
    print(f"Frame Status (GNN): {'Error' if frame_errors_gnn[0] else 'Correct'}")
    print(f"Frame Status (Std): {'Error' if frame_errors_standard[0] else 'Correct'}")
    
    print("\n===== Demo Completed =====")
    
    # Return results for further analysis if needed
    return {
        'snr_values': snr_values,
        'ber_gnn': ber_per_codeword_gnn,
        'ber_standard': ber_per_codeword_standard,
        'frame_errors_gnn': frame_errors_gnn,
        'frame_errors_standard': frame_errors_standard
    }

if __name__ == "__main__":
    main() 