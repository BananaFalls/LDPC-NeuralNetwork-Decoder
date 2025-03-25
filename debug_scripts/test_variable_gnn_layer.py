#!/usr/bin/env python3
"""
Test script for VariableGNNLayer class.

This script creates a simple test case for the VariableGNNLayer class,
initializes the layer, and runs a forward pass to verify its functionality.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import VariableGNNLayer


def test_variable_gnn_layer():
    """
    Test the VariableGNNLayer class with a simple example.
    """
    print("\n=== Testing VariableGNNLayer ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    batch_size = 2
    num_messages = 10
    hidden_dim = 16
    num_message_types = 2
    
    # Initialize layer
    var_gnn_layer = VariableGNNLayer(num_message_types=num_message_types, hidden_dim=hidden_dim)
    print(f"Initialized VariableGNNLayer with hidden_dim={hidden_dim}, num_message_types={num_message_types}")
    
    # Create dummy inputs
    copied_llr = torch.randn(batch_size, num_messages, hidden_dim)
    var_messages = torch.randn(batch_size, num_messages, hidden_dim)
    check_messages = torch.randn(batch_size, num_messages, hidden_dim)
    message_types = torch.randint(0, num_message_types, (num_messages,))
    
    # Create dummy adjacency matrix (each message connected to 3 others)
    var_to_check_adjacency = torch.zeros(num_messages, num_messages)
    for i in range(num_messages):
        # Connect each message to 3 random other messages
        connections = np.random.choice(
            [j for j in range(num_messages) if j != i], 
            size=min(3, num_messages-1), 
            replace=False
        )
        var_to_check_adjacency[i, connections] = 1.0
    
    print(f"Input shapes:")
    print(f"  copied_llr: {copied_llr.shape}")
    print(f"  var_messages: {var_messages.shape}")
    print(f"  check_messages: {check_messages.shape}")
    print(f"  message_types: {message_types.shape}")
    print(f"  var_to_check_adjacency: {var_to_check_adjacency.shape}")
    
    # Forward pass
    try:
        updated_var_messages = var_gnn_layer(
            copied_llr, 
            var_messages, 
            check_messages, 
            message_types, 
            var_to_check_adjacency
        )
        print(f"Forward pass successful!")
        print(f"Output shape: {updated_var_messages.shape}")
        
        # Test decode_messages
        decoded_llr = var_gnn_layer.decode_messages(updated_var_messages)
        print(f"Decoded LLR shape: {decoded_llr.shape}")
        
        # Plot some message features before and after update
        plt.figure(figsize=(12, 6))
        
        # Plot original message features
        plt.subplot(1, 2, 1)
        plt.imshow(var_messages[0, :5, :].detach().numpy())
        plt.colorbar()
        plt.title("Original message features (first 5 messages)")
        plt.xlabel("Hidden dimension")
        plt.ylabel("Message index")
        
        # Plot updated message features
        plt.subplot(1, 2, 2)
        plt.imshow(updated_var_messages[0, :5, :].detach().numpy())
        plt.colorbar()
        plt.title("Updated message features (first 5 messages)")
        plt.xlabel("Hidden dimension")
        plt.ylabel("Message index")
        
        plt.tight_layout()
        plt.savefig("variable_gnn_layer_test.png")
        print(f"Plot saved to variable_gnn_layer_test.png")
        
        print("\nTest passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False


def main():
    """Main function to run the test."""
    success = test_variable_gnn_layer()
    print(f"\nTest {'succeeded' if success else 'failed'}!")
    
    
if __name__ == "__main__":
    main() 