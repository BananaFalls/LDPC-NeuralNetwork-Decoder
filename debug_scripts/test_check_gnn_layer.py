#!/usr/bin/env python3
"""
Test script for CheckGNNLayer class.

This script creates a simple test case for the CheckGNNLayer class,
initializes the layer, and runs a forward pass to verify its functionality.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import CheckGNNLayer


def test_check_gnn_layer():
    """
    Test the CheckGNNLayer class with a simple example.
    """
    print("\n=== Testing CheckGNNLayer ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    batch_size = 2
    num_messages = 10
    num_message_types = 2
    
    # Initialize layer with scalar operations
    check_gnn_layer = CheckGNNLayer(num_message_types=num_message_types, hidden_dim=1, use_bp_op=True)
    print(f"Initialized CheckGNNLayer with scalar operations and {num_message_types} message types")
    
    # Create dummy inputs (all in scalar domain)
    check_messages = torch.randn(batch_size, num_messages, 1)  # Check messages
    var_messages = torch.randn(batch_size, num_messages, 1)  # Variable messages
    message_types = torch.randint(0, num_message_types, (num_messages,))  # Message types
    
    # Create dummy adjacency matrix (each message connected to 3 others)
    check_to_var_adjacency = torch.zeros(num_messages, num_messages)
    for i in range(num_messages):
        # Connect each message to 3 random other messages
        connections = np.random.choice(
            [j for j in range(num_messages) if j != i], 
            size=min(3, num_messages-1), 
            replace=False
        )
        check_to_var_adjacency[i, connections] = 1.0
    
    print(f"\nInput shapes:")
    print(f"  check_messages: {check_messages.shape}")
    print(f"  var_messages: {var_messages.shape}")
    print(f"  message_types: {message_types.shape}")
    print(f"  check_to_var_adjacency: {check_to_var_adjacency.shape}")
    
    # Forward pass
    try:
        updated_check_messages = check_gnn_layer(
            check_messages, 
            var_messages, 
            message_types, 
            check_to_var_adjacency
        )
        print(f"\nForward pass successful!")
        print(f"Output shape: {updated_check_messages.shape}")
        
        # Plot message values before and after update
        plt.figure(figsize=(12, 6))
        
        # Plot original messages
        plt.subplot(1, 2, 1)
        plt.plot(check_messages[0, :, 0].detach().numpy(), 'b-', label='Check')
        plt.plot(var_messages[0, :, 0].detach().numpy(), 'r--', label='Variable')
        plt.title("Original Messages (first batch)")
        plt.xlabel("Message index")
        plt.ylabel("LLR value")
        plt.legend()
        plt.grid(True)
        
        # Plot updated messages
        plt.subplot(1, 2, 2)
        plt.plot(updated_check_messages[0, :, 0].detach().numpy(), 'b-', label='Updated')
        plt.title("Updated Messages (first batch)")
        plt.xlabel("Message index")
        plt.ylabel("LLR value")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("check_gnn_layer_test.png")
        print(f"Plot saved to check_gnn_layer_test.png")
        
        # Print learned weights
        print(f"\nLearned message type weights:")
        print(check_gnn_layer.message_type_weights.data)
        
        print("\nTest passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False


def main():
    """Main function to run the test."""
    success = test_check_gnn_layer()
    print(f"\nTest {'succeeded' if success else 'failed'}!")
    
    
if __name__ == "__main__":
    main() 