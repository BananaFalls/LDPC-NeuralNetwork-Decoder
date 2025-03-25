#!/usr/bin/env python3
"""
Test script for MessageGNNDecoder class.

This script creates a simple test case for the MessageGNNDecoder class,
initializes the decoder with sample parameters, and runs a forward pass to verify its functionality.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import (
    MessageGNNDecoder, TannerToMessageGraph, create_message_gnn_decoder
)


def create_sample_H(m=4, n=8):
    """
    Create a sample parity-check matrix for testing.
    
    Args:
        m (int): Number of check nodes
        n (int): Number of variable nodes
        
    Returns:
        torch.Tensor: Parity-check matrix
    """
    # Create a fixed H matrix for tests
    H = torch.zeros(m, n)
    
    # Regular (3,6) LDPC code pattern (each var connected to 3 checks, each check to 6 vars)
    connections = [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],  # Check 0 connections
        [(1, 0), (1, 1), (1, 6), (1, 7), (1, 2), (1, 3)],  # Check 1 connections
        [(2, 4), (2, 5), (2, 6), (2, 7), (2, 0), (2, 2)],  # Check 2 connections
        [(3, 1), (3, 3), (3, 4), (3, 6), (3, 5), (3, 7)]   # Check 3 connections
    ]
    
    for check_connections in connections:
        for c, v in check_connections:
            H[c, v] = 1
            
    return H


def plot_decoding_results(input_llr, output_probs, title="Decoding Results"):
    """
    Plot input LLRs and output probabilities.
    
    Args:
        input_llr (torch.Tensor): Input LLR values
        output_probs (torch.Tensor): Output probabilities
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    for i in range(min(3, input_llr.shape[0])):
        plt.plot(input_llr[i].detach().numpy(), label=f"Sample {i}")
    plt.title("Input LLR Values")
    plt.xlabel("Variable node index")
    plt.ylabel("LLR")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for i in range(min(3, output_probs.shape[0])):
        plt.plot(output_probs[i].detach().numpy(), label=f"Sample {i}")
    plt.title("Output Probabilities")
    plt.xlabel("Variable node index")
    plt.ylabel("Probability")
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("message_gnn_decoder_test.png")
    print(f"Decoding results visualization saved to message_gnn_decoder_test.png")


def test_message_gnn_decoder():
    """
    Test the MessageGNNDecoder class with a simple example.
    """
    print("\n=== Testing MessageGNNDecoder ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a sample parity-check matrix
    H = create_sample_H(m=4, n=8)
    print(f"Created parity-check matrix with shape {H.shape}")
    
    # Parameters
    batch_size = 3
    num_iterations = 3
    hidden_dim = 32
    num_of_residual_layers = 1
    
    try:
        # Create decoder and converter using the utility function
        decoder, converter = create_message_gnn_decoder(
            H, 
            num_iterations=num_iterations,
            hidden_dim=hidden_dim,
            num_of_residual_layers=num_of_residual_layers
        )
        
        print(f"Created MessageGNNDecoder with parameters:")
        print(f"  num_messages: {converter.num_messages}")
        print(f"  num_iterations: {num_iterations}")
        print(f"  hidden_dim: {hidden_dim}")
        print(f"  num_message_types: {decoder.num_message_types}")
        print(f"  num_of_residual_layers: {num_of_residual_layers}")
        print(f"Trainable parameters: {decoder.count_parameters():,}")
        
        # Get required mappings
        message_to_var_mapping = converter.get_message_to_var_mapping()
        var_to_check_adjacency = converter.var_to_check_adjacency
        check_to_var_adjacency = converter.check_to_var_adjacency
        message_types = converter.message_types
        
        # Create dummy input LLRs (positive values for bit 0, negative for bit 1)
        bits = torch.randint(0, 2, (batch_size, H.shape[1]))
        input_llr = 3.0 * (1 - 2 * bits.float())  # LLR = +3 for bit 0, -3 for bit 1
        
        print(f"\nInput LLR shape: {input_llr.shape}")
        print(f"Input LLR values (first sample):\n{input_llr[0]}")
        
        # Forward pass
        output_probs = decoder(
            input_llr=input_llr,
            message_to_var_mapping=message_to_var_mapping,
            message_types=message_types,
            var_to_check_adjacency=var_to_check_adjacency,
            check_to_var_adjacency=check_to_var_adjacency
        )
        
        print(f"\nForward pass successful!")
        print(f"Output probabilities shape: {output_probs.shape}")
        print(f"Output probabilities (first sample):\n{output_probs[0]}")
        
        # Convert to hard decisions
        decoded_bits = (output_probs >= 0.5).float()
        print(f"\nDecoded bits (first sample):\n{decoded_bits[0]}")
        
        # Compare with original bits
        matches = (decoded_bits == bits).float().mean().item()
        print(f"\nBit match rate: {matches:.2%}")
        
        # Plot results
        plot_decoding_results(input_llr, output_probs)
        
        print("\nTest passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during decoder creation or forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the test."""
    success = test_message_gnn_decoder()
    print(f"\nTest {'succeeded' if success else 'failed'}!")
    
    
if __name__ == "__main__":
    main() 