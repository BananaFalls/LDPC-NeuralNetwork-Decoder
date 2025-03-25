#!/usr/bin/env python3
"""
Test script for Message Graph utility functions.

This script tests utility functions defined in message_gnn_decoder.py
for visualization, message passing, and conversions.
"""

import torch
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import (
    TannerToMessageGraph,
    visualize_tanner_graph,
    visualize_message_graph,
    get_llr_from_noise
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
    
    # Regular pattern for testing
    connections = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],  # Check 0 connections
        [(1, 2), (1, 3), (1, 4), (1, 5)],  # Check 1 connections
        [(2, 4), (2, 5), (2, 6), (2, 7)],  # Check 2 connections
        [(3, 0), (3, 1), (3, 6), (3, 7)]   # Check 3 connections
    ]
    
    for check_connections in connections:
        for c, v in check_connections:
            H[c, v] = 1
            
    return H


def test_visualize_tanner_graph():
    """Test the visualize_tanner_graph function."""
    print("\n=== Testing Tanner Graph Visualization ===")
    
    # Create a sample parity-check matrix
    H = create_sample_H()
    print(f"Created parity-check matrix with shape {H.shape}")
    
    try:
        # Visualize the Tanner graph
        G = visualize_tanner_graph(H, show=False)
        
        # Check if the graph has the correct number of nodes
        num_var_nodes = H.shape[1]
        num_check_nodes = H.shape[0]
        expected_nodes = num_var_nodes + num_check_nodes
        
        if len(G.nodes) == expected_nodes:
            print(f"Graph has the expected number of nodes: {expected_nodes}")
        else:
            print(f"Graph node count mismatch: expected {expected_nodes}, got {len(G.nodes)}")
        
        # Count the number of edges
        num_edges = H.sum().item()
        if len(G.edges) == num_edges:
            print(f"Graph has the expected number of edges: {num_edges}")
        else:
            print(f"Graph edge count mismatch: expected {num_edges}, got {len(G.edges)}")
        
        # Save the figure
        plt.figure(figsize=(10, 8))
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, node_size=500, node_color=nx.get_node_attributes(G, 'color').values())
        plt.savefig("tanner_graph_util_test.png")
        plt.close()
        print(f"Tanner graph visualization saved to tanner_graph_util_test.png")
        
        return True
        
    except Exception as e:
        print(f"Error in Tanner graph visualization: {e}")
        return False


def test_visualize_message_graph():
    """Test the visualize_message_graph function."""
    print("\n=== Testing Message Graph Visualization ===")
    
    # Create a sample parity-check matrix
    H = create_sample_H()
    
    try:
        # Create a message graph converter
        converter = TannerToMessageGraph(H)
        print(f"Created message graph with {converter.num_messages} messages")
        
        # Get the adjacency matrices
        var_to_check_adjacency = converter.var_to_check_adjacency
        check_to_var_adjacency = converter.check_to_var_adjacency
        
        # Get message types
        message_types = converter.message_types
        
        # Visualize the message graph
        G = visualize_message_graph(
            converter.message_mapping,
            var_to_check_adjacency,
            check_to_var_adjacency,
            message_types,
            show=False
        )
        
        # Check if the graph has the correct number of nodes
        expected_nodes = converter.num_messages
        if len(G.nodes) == expected_nodes:
            print(f"Message graph has the expected number of nodes: {expected_nodes}")
        else:
            print(f"Message graph node count mismatch: expected {expected_nodes}, got {len(G.nodes)}")
        
        # Save the figure
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=400, 
                node_color=nx.get_node_attributes(G, 'color').values(),
                font_size=8)
        plt.savefig("message_graph_util_test.png")
        plt.close()
        print(f"Message graph visualization saved to message_graph_util_test.png")
        
        return True
        
    except Exception as e:
        print(f"Error in message graph visualization: {e}")
        return False


def test_get_llr_from_noise():
    """Test the get_llr_from_noise function."""
    print("\n=== Testing LLR from Noise Utility ===")
    
    try:
        # Test parameters
        batch_size = 3
        codeword_len = 8
        snr_db = 2.0
        
        # Generate all-zero codewords
        codewords = torch.zeros(batch_size, codeword_len)
        
        # Get LLRs from noise
        llrs = get_llr_from_noise(codewords, snr_db)
        
        print(f"Generated LLRs with shape: {llrs.shape}")
        print(f"LLR sample (first batch):\n{llrs[0]}")
        
        # Validate shape
        if llrs.shape == (batch_size, codeword_len):
            print("LLR shape is correct")
        else:
            print(f"LLR shape mismatch: expected {(batch_size, codeword_len)}, got {llrs.shape}")
        
        # Plot the distribution of LLRs
        plt.figure(figsize=(8, 6))
        plt.hist(llrs.flatten().numpy(), bins=30, alpha=0.7)
        plt.title(f"LLR Distribution at SNR = {snr_db} dB")
        plt.xlabel("LLR Value")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig("llr_distribution_test.png")
        plt.close()
        print(f"LLR distribution saved to llr_distribution_test.png")
        
        # Test with different SNR values
        snr_values = [0.0, 2.0, 5.0]
        plt.figure(figsize=(10, 6))
        
        for snr in snr_values:
            llrs = get_llr_from_noise(codewords, snr)
            plt.hist(llrs.flatten().numpy(), bins=30, alpha=0.6, label=f"SNR = {snr} dB")
        
        plt.title("LLR Distributions at Different SNR Values")
        plt.xlabel("LLR Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("llr_snr_comparison_test.png")
        plt.close()
        print(f"LLR SNR comparison saved to llr_snr_comparison_test.png")
        
        return True
        
    except Exception as e:
        print(f"Error in LLR from noise test: {e}")
        return False


def main():
    """Main function to run all tests."""
    tanner_graph_test = test_visualize_tanner_graph()
    message_graph_test = test_visualize_message_graph()
    llr_test = test_get_llr_from_noise()
    
    if tanner_graph_test and message_graph_test and llr_test:
        print("\nAll utility function tests passed!")
    else:
        print("\nSome utility function tests failed.")


if __name__ == "__main__":
    main() 