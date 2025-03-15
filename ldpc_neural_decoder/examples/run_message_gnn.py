"""
Example script to run the Message-Centered GNN LDPC Decoder using small_base_graph.txt.

This script demonstrates how to:
1. Load a small base graph
2. Expand it to a parity-check matrix
3. Create a message-centered GNN decoder
4. Run the decoder on random data with QPSK modulation
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import from ldpc_neural_decoder
# This ensures we can import the module regardless of how the script is run
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from ldpc_neural_decoder.models.message_gnn_decoder import (
    create_message_gnn_decoder, 
    create_variable_index_tensor,
    create_check_index_tensor,
    MessageGNNDecoder,
    TannerToMessageGraph
)


def load_base_graph(file_path):
    """
    Load a base graph from a text file.
    
    Args:
        file_path (str): Path to the base graph file
        
    Returns:
        torch.Tensor: Base graph matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse the base graph
    base_graph = []
    for line in lines:
        row = [int(val) for val in line.strip().split()]
        base_graph.append(row)
    
    return torch.tensor(base_graph, dtype=torch.float)


def expand_base_graph(base_graph, Z):
    """
    Expand a base graph to a parity-check matrix using lifting.
    
    Args:
        base_graph (torch.Tensor): Base graph matrix
        Z (int): Lifting factor
        
    Returns:
        torch.Tensor: Expanded parity-check matrix
    """
    num_check_blocks, num_var_blocks = base_graph.shape
    H = torch.zeros(num_check_blocks * Z, num_var_blocks * Z, dtype=torch.float)
    
    for i in range(num_check_blocks):
        for j in range(num_var_blocks):
            shift = int(base_graph[i, j].item())  # Convert to int
            if shift >= 0:  # -1 means no connection
                # Create a shifted identity matrix
                block = torch.zeros(Z, Z)
                for k in range(Z):
                    block[k, (k + shift) % Z] = 1
                
                # Place the block in the parity-check matrix
                H[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = block
    
    return H


def generate_random_codeword(H, batch_size=1):
    """
    Generate random codewords that satisfy the parity-check matrix.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        batch_size (int): Number of codewords to generate
        
    Returns:
        torch.Tensor: Random codewords of shape (batch_size, num_variables)
    """
    num_checks, num_variables = H.shape
    device = H.device
    
    # Generate random bits
    codewords = torch.randint(0, 2, (batch_size, num_variables), dtype=torch.float, device=device)
    
    # Ensure codewords satisfy parity checks (H * c^T = 0 mod 2)
    for b in range(batch_size):
        # Check if codeword satisfies parity checks
        syndrome = torch.matmul(H, codewords[b]) % 2
        
        # If not, flip bits until it does
        max_iterations = 100
        iteration = 0
        
        while torch.any(syndrome != 0) and iteration < max_iterations:
            # Find a check that is not satisfied
            unsatisfied_check = torch.where(syndrome != 0)[0][0].item()
            
            # Find variables connected to this check
            connected_vars = torch.where(H[unsatisfied_check] != 0)[0]
            
            # Randomly choose one variable to flip
            var_to_flip = connected_vars[torch.randint(0, len(connected_vars), (1,))].item()
            
            # Flip the bit
            codewords[b, var_to_flip] = 1 - codewords[b, var_to_flip]
            
            # Recompute syndrome
            syndrome = torch.matmul(H, codewords[b]) % 2
            
            iteration += 1
    
    return codewords


def add_noise_qpsk(codewords, snr_db):
    """
    Add AWGN noise to codewords using QPSK modulation.
    
    Args:
        codewords (torch.Tensor): Codewords of shape (batch_size, num_variables)
        snr_db (float): Signal-to-noise ratio in dB
        
    Returns:
        torch.Tensor: Noisy LLRs of shape (batch_size, num_variables)
    """
    batch_size, num_variables = codewords.shape
    
    # Ensure num_variables is even for QPSK (2 bits per symbol)
    if num_variables % 2 != 0:
        # Pad with an extra bit if needed
        padding = torch.zeros((batch_size, 1), device=codewords.device)
        codewords = torch.cat([codewords, padding], dim=1)
        num_variables += 1
    
    # Reshape to group bits in pairs for QPSK modulation
    bits_reshaped = codewords.reshape(batch_size, -1, 2)
    
    # QPSK modulation: map bit pairs to complex symbols
    # (0,0) -> (1+1j)/sqrt(2)
    # (0,1) -> (1-1j)/sqrt(2)
    # (1,0) -> (-1+1j)/sqrt(2)
    # (1,1) -> (-1-1j)/sqrt(2)
    
    # Create complex symbols
    real_part = 1 - 2 * bits_reshaped[:, :, 0]  # First bit determines real part
    imag_part = 1 - 2 * bits_reshaped[:, :, 1]  # Second bit determines imaginary part
    
    # Combine into complex symbols and normalize by 1/sqrt(2) for unit energy
    symbols = (real_part + 1j * imag_part) / np.sqrt(2)
    
    # Calculate noise standard deviation
    snr_linear = 10 ** (snr_db / 10)
    sigma = 1 / np.sqrt(snr_linear)
    
    # Generate complex Gaussian noise
    noise_real = torch.normal(0, sigma/np.sqrt(2), symbols.shape)
    noise_imag = torch.normal(0, sigma/np.sqrt(2), symbols.shape)
    noise = noise_real + 1j * noise_imag
    
    # Add noise to symbols
    received = symbols + noise
    
    # Demodulate to LLRs
    # For QPSK, we compute LLRs for each bit separately
    llr_real = 2 * received.real * np.sqrt(2) / (sigma**2)  # LLR for first bit
    llr_imag = 2 * received.imag * np.sqrt(2) / (sigma**2)  # LLR for second bit
    
    # Interleave the LLRs to match the original bit ordering
    llrs = torch.zeros((batch_size, num_variables), device=codewords.device)
    llrs[:, 0::2] = llr_real  # Even indices get real part LLRs
    llrs[:, 1::2] = llr_imag  # Odd indices get imaginary part LLRs
    
    return llrs


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load and expand base graph
    base_graph_path = os.path.join(os.path.dirname(__file__), 'small_base_graph.txt')
    base_graph = load_base_graph(base_graph_path)
    print(f"Base graph shape: {base_graph.shape}")
    
    # Lifting factor
    Z = 4
    
    # Expand base graph to parity-check matrix
    H = expand_base_graph(base_graph, Z)
    print(f"Expanded parity-check matrix shape: {H.shape}")
    
    # Create message-centered GNN decoder
    num_iterations = 3
    hidden_dim = 32  # Changed from 1 to 32 for better performance
    
    # Create converter first
    converter = TannerToMessageGraph(H)
    
    # Get number of messages
    num_messages = len(converter.messages)
    print(f"Number of messages: {num_messages}")
    
    # Get number of message types
    num_message_types = 1
    if base_graph is not None and Z is not None:
        # Count unique shift values in base graph (excluding -1)
        unique_shifts = set()
        for i in range(base_graph.shape[0]):
            for j in range(base_graph.shape[1]):
                shift = int(base_graph[i, j].item())
                if shift >= 0:
                    unique_shifts.add(shift)
        num_message_types = len(unique_shifts) if unique_shifts else 1
    print(f"Number of message types: {num_message_types}")
    
    # Create decoder
    decoder = MessageGNNDecoder(
        num_messages=num_messages,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,  # Using hidden_dim = 32
        num_message_types=num_message_types
    )
    
    # Create variable and check index tensors for custom decoder
    variable_index_tensor = create_variable_index_tensor(H, converter)
    check_index_tensor = create_check_index_tensor(H)
    
    # Create a second instance of the standard decoder for comparison
    custom_decoder = MessageGNNDecoder(
        num_messages=num_messages,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,  # Using hidden_dim = 32
        num_message_types=num_message_types
    )
    
    # Set variable and check index tensors
    # Note: These methods might not exist in the standard MessageGNNDecoder
    # Uncomment if they do exist
    # custom_decoder.set_variable_index_tensor(variable_index_tensor)
    # custom_decoder.set_check_index_tensor(check_index_tensor)
    
    # Generate zero codewords instead of random codewords
    batch_size = 10
    num_variables = H.shape[1]
    codewords = torch.zeros((batch_size, num_variables), dtype=torch.float)
    print(f"Generated {batch_size} zero codewords")
    
    # Add noise using QPSK modulation
    snr_db = 5.0
    noisy_llrs = add_noise_qpsk(codewords, snr_db)
    print(f"Added noise at SNR = {snr_db} dB using QPSK modulation")
    
    # Get message types
    message_types = torch.zeros(num_messages, dtype=torch.long)
    print(f"Message types shape: {message_types.shape}")
    
    # Create a var-to-message mapping for the decoder
    # This maps each variable node to the message nodes it's connected to
    var_to_message_mapping = torch.zeros(H.shape[1], dtype=torch.long)
    
    # For each message, find the variable node it's connected to
    for i, (var_idx, _) in enumerate(converter.messages):
        var_to_message_mapping[var_idx] = i
    
    print(f"Variable to message mapping shape: {var_to_message_mapping.shape}")
    print(f"Variable to message mapping: {var_to_message_mapping}")
    
    # Ensure adjacency matrices are float tensors
    var_to_check_adjacency = converter.var_to_check_adjacency.float()
    check_to_var_adjacency = converter.check_to_var_adjacency.float()
    print(f"Var to check adjacency shape: {var_to_check_adjacency.shape}")
    print(f"Check to var adjacency shape: {check_to_var_adjacency.shape}")
    
    # Create variable to message mapping for custom decoder
    variable_to_message_mapping = []
    for v in range(H.shape[1]):
        connected_messages = converter.var_to_messages.get(v, [])
        # Pad with -1 to handle variable node degrees
        max_degree = max(len(msgs) for msgs in converter.var_to_messages.values())
        padded_messages = connected_messages + [-1] * (max_degree - len(connected_messages))
        variable_to_message_mapping.append(padded_messages)
    
    variable_to_message_mapping = torch.tensor(variable_to_message_mapping, dtype=torch.long)
    print(f"Full variable to message mapping shape: {variable_to_message_mapping.shape}")
    
    # Run standard decoder
    print("\nRunning standard Message GNN Decoder...")
    with torch.no_grad():
        decoded_probs = decoder(
            noisy_llrs,
            converter.message_to_var_mapping.long(),  # Use the original mapping from the converter
            message_types,
            var_to_check_adjacency,
            check_to_var_adjacency
        )
    
    decoded_bits = (decoded_probs > 0.5).float()
    
    # Calculate bit error rate
    # If we padded for QPSK, make sure to only compare the original bits
    original_num_variables = H.shape[1]
    errors = torch.sum(decoded_bits[:, :original_num_variables] != codewords[:, :original_num_variables]).item()
    total_bits = codewords[:, :original_num_variables].numel()
    ber = errors / total_bits
    
    print(f"Standard Decoder - Bit Error Rate: {ber:.6f} ({errors}/{total_bits})")
    
    # Run second decoder instance
    print("\nRunning Second Message GNN Decoder Instance...")
    with torch.no_grad():
        # Note: The parameter order might be different for the standard MessageGNNDecoder
        # Adjust as needed based on the actual implementation
        custom_decoded_probs = custom_decoder(
            noisy_llrs,
            converter.message_to_var_mapping.long(),  # Use the original mapping from the converter
            message_types,
            var_to_check_adjacency,
            check_to_var_adjacency
        )
    
    custom_decoded_bits = (custom_decoded_probs > 0.5).float()
    
    # Calculate bit error rate for second decoder
    custom_errors = torch.sum(custom_decoded_bits[:, :original_num_variables] != codewords[:, :original_num_variables]).item()
    custom_ber = custom_errors / total_bits
    
    print(f"Second Decoder - Bit Error Rate: {custom_ber:.6f} ({custom_errors}/{total_bits})")
    
    # Print sample results
    print("\nSample Results (first codeword):")
    print(f"Original:  {codewords[0][:20].int().tolist()}")
    print(f"Decoded:   {decoded_bits[0][:20].int().tolist()}")
    print(f"Second:    {custom_decoded_bits[0][:20].int().tolist()}")
    
    # Print QPSK constellation diagram
    plt.figure(figsize=(8, 8))
    
    # Create QPSK constellation points
    qpsk_points = np.array([(1+1j), (1-1j), (-1+1j), (-1-1j)]) / np.sqrt(2)
    plt.scatter(qpsk_points.real, qpsk_points.imag, color='red', s=100, marker='o', label='QPSK Constellation')
    
    # Add labels for each point
    for i, point in enumerate(qpsk_points):
        bits = [(i >> 1) & 1, i & 1]  # Convert index to bit pair
        plt.annotate(f'{bits[0]}{bits[1]}', 
                    (point.real, point.imag),
                    xytext=(10, 10),
                    textcoords='offset points')
    
    # Plot axes and grid
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.title('QPSK Constellation Diagram')
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.axis('equal')
    
    # Save the constellation diagram
    plt.savefig('qpsk_constellation.png')
    print("\nQPSK constellation diagram saved as 'qpsk_constellation.png'")


if __name__ == "__main__":
    main() 