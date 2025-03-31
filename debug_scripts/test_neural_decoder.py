import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.neural_message_gnn_decoder import NeuralMessageGNNDecoder

def main():
    # Define the same H matrix as in test_simple_decoder.py
    H = torch.tensor([
        [1, 1, 0, 1],  # check0 connected to var0, var1, var3
        [0, 1, 1, 0],  # check1 connected to var1, var2
        [1, 0, 1, 1]   # check2 connected to var0, var2, var3
    ], dtype=torch.float)
    
    # Create input LLRs for multiple batches
    # Each row represents a different batch
    input_llr = torch.tensor([
        [2.0, -1.5, 0.5, -2.0],  # Batch 0
        [-1.0, 1.5, -0.5, 1.0],  # Batch 1
        [0.5, -0.5, 1.0, -1.5]   # Batch 2
    ], dtype=torch.float)
    
    print("\n=== Test Configuration ===")
    print(f"H matrix shape: {H.shape}")
    print(f"Input LLR shape: {input_llr.shape}")
    print(f"Number of batches: {input_llr.shape[0]}")
    
    # Create decoder instance
    decoder = NeuralMessageGNNDecoder(H_matrix=H, num_iterations=3)
    
    # Run decoder on batched input
    print("\n=== Running Neural Decoder ===")
    decoded_bits = decoder.forward(input_llr=input_llr)
    
    # Print results for each batch
    print("\n=== Decoding Results ===")
    for batch_idx in range(input_llr.shape[0]):
        print(f"\nBatch {batch_idx}:")
        # print(f"Input LLR: {input_llr[batch_idx]}")
        print(f"Decoded bits: {decoded_bits[batch_idx]}")
        
        # Verify parity check equations for this batch
        syndrome = torch.matmul(decoded_bits[batch_idx], H.t()) % 2
        is_valid = torch.all(syndrome == 0)
        print(f"Valid codeword: {is_valid}")
        if not is_valid:
            print(f"Syndrome: {syndrome}")

if __name__ == "__main__":
    main() 