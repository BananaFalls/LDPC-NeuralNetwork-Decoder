#!/usr/bin/env python3
"""
Test script for TannerToMessageGraph class.

This script creates a simple test case for the TannerToMessageGraph class,
initializes the converter with a sample parity-check matrix, and validates
the resulting message-centered graph.
"""

import torch
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.message_gnn_decoder import TannerToMessageGraph


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


def visualize_tanner_graph(H, title="Tanner Graph"):
    """
    Visualize the Tanner graph corresponding to H.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        title (str): Plot title
    """
    m, n = H.shape
    G = nx.Graph()
    
    # Add variable nodes
    for i in range(n):
        G.add_node(f"v{i}", bipartite=0, node_type="variable")
    
    # Add check nodes
    for j in range(m):
        G.add_node(f"c{j}", bipartite=1, node_type="check")
    
    # Add edges
    for j in range(m):
        for i in range(n):
            if H[j, i] == 1:
                G.add_edge(f"c{j}", f"v{i}")
    
    # Prepare node positions
    pos = {}
    var_nodes = [n for n in G.nodes() if n.startswith('v')]
    check_nodes = [n for n in G.nodes() if n.startswith('c')]
    
    # Position var nodes on top
    for i, node in enumerate(var_nodes):
        pos[node] = (i, 1)
    
    # Position check nodes on bottom
    for i, node in enumerate(check_nodes):
        pos[node] = (i * (len(var_nodes) / len(check_nodes)), 0)
    
    # Draw the graph
    plt.figure(figsize=(10, 6))
    
    # Draw var nodes as circles
    nx.draw_networkx_nodes(G, pos, nodelist=var_nodes, node_color='skyblue', 
                         node_size=500, alpha=0.8, label="Variable Nodes")
    
    # Draw check nodes as squares
    nx.draw_networkx_nodes(G, pos, nodelist=check_nodes, node_color='lightgreen', 
                         node_shape='s', node_size=500, alpha=0.8, label="Check Nodes")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    
    plt.axis('off')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tanner_graph_test.png")
    print(f"Tanner graph visualization saved to tanner_graph_test.png")


def visualize_message_graph(converter, title="Message-Centered Graph"):
    """
    Visualize the message-centered graph.
    
    Args:
        converter (TannerToMessageGraph): Converter instance
        title (str): Plot title
    """
    G = nx.Graph()
    
    # Add message nodes
    for msg_idx in range(converter.num_messages):
        check_idx = converter.message_to_check[msg_idx]
        var_idx = converter.message_to_var[msg_idx]
        G.add_node(msg_idx, label=f"m{msg_idx}\n(c{check_idx},v{var_idx})")
    
    # Add edges based on var_to_check_adjacency
    for i in range(converter.num_messages):
        for j in range(converter.num_messages):
            if converter.var_to_check_adjacency[i, j] == 1:
                G.add_edge(i, j, edge_type="var")
    
    # Add edges based on check_to_var_adjacency
    for i in range(converter.num_messages):
        for j in range(converter.num_messages):
            if converter.check_to_var_adjacency[i, j] == 1:
                # Check if edge already exists
                if G.has_edge(i, j):
                    # Edge already exists as var connection, mark as both
                    G[i][j]['edge_type'] = "both"
                else:
                    G.add_edge(i, j, edge_type="check")
    
    # Spring layout often works well for visualizing graphs
    pos = nx.spring_layout(G, seed=42)
    
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=700, alpha=0.8)
    
    # Draw different edge types with different colors and styles
    var_edges = [(i, j) for i, j in G.edges() if G[i][j]['edge_type'] == "var"]
    check_edges = [(i, j) for i, j in G.edges() if G[i][j]['edge_type'] == "check"]
    both_edges = [(i, j) for i, j in G.edges() if G[i][j]['edge_type'] == "both"]
    
    nx.draw_networkx_edges(G, pos, edgelist=var_edges, width=1.0, alpha=0.7, 
                          edge_color='blue', style='solid', label="Variable connection")
    nx.draw_networkx_edges(G, pos, edgelist=check_edges, width=1.0, alpha=0.7, 
                          edge_color='green', style='dashed', label="Check connection")
    nx.draw_networkx_edges(G, pos, edgelist=both_edges, width=1.5, alpha=0.7, 
                          edge_color='red', style='dashdot', label="Both connections")
    
    # Draw node labels
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("message_graph_test.png")
    print(f"Message graph visualization saved to message_graph_test.png")


def test_tanner_to_message_graph():
    """
    Test the TannerToMessageGraph class with a sample parity-check matrix.
    """
    print("\n=== Testing TannerToMessageGraph ===")
    
    # Create a sample parity-check matrix
    H = create_sample_H(m=4, n=8)
    print(f"Created parity-check matrix with shape {H.shape}")
    print(f"H matrix:\n{H}")
    
    # Visualize the Tanner graph
    visualize_tanner_graph(H)
    
    # Create converter
    try:
        converter = TannerToMessageGraph(H)
        print(f"Created TannerToMessageGraph converter")
        print(f"Number of messages: {converter.num_messages}")
        
        # Print sample of messages
        print(f"\nFirst few messages (check_idx, var_idx):")
        for i, msg in enumerate(converter.messages[:5]):
            print(f"  Message {i}: {msg}")
        
        # Print sample of message mappings
        print(f"\nFirst few message mappings:")
        for i in range(min(5, converter.num_messages)):
            print(f"  Message {i}: var={converter.message_to_var[i]}, check={converter.message_to_check[i]}")
        
        # Print sample of adjacency matrices
        print(f"\nSample of var_to_check_adjacency (5x5):")
        print(converter.var_to_check_adjacency[:5, :5])
        
        print(f"\nSample of check_to_var_adjacency (5x5):")
        print(converter.check_to_var_adjacency[:5, :5])
        
        # Get tensor mappings
        message_to_var_mapping = converter.get_message_to_var_mapping()
        message_to_check_mapping = converter.get_message_to_check_mapping()
        var_to_messages_indices, var_to_messages_offsets = converter.get_var_to_messages_mapping()
        check_to_messages_indices, check_to_messages_offsets = converter.get_check_to_messages_mapping()
        
        print(f"\nTensor mappings:")
        print(f"  message_to_var_mapping shape: {message_to_var_mapping.shape}")
        print(f"  message_to_check_mapping shape: {message_to_check_mapping.shape}")
        print(f"  var_to_messages_indices shape: {var_to_messages_indices.shape}")
        print(f"  var_to_messages_offsets shape: {var_to_messages_offsets.shape}")
        print(f"  check_to_messages_indices shape: {check_to_messages_indices.shape}")
        print(f"  check_to_messages_offsets shape: {check_to_messages_offsets.shape}")
        
        # Visualize the message-centered graph
        visualize_message_graph(converter)
        
        print("\nTest passed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during converter creation: {e}")
        return False


def main():
    """Main function to run the test."""
    success = test_tanner_to_message_graph()
    print(f"\nTest {'succeeded' if success else 'failed'}!")
    
    
if __name__ == "__main__":
    main() 