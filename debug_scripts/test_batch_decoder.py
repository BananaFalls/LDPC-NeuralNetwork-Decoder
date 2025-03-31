import torch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.batch_message_gnn_decoder import BatchMessageGNNDecoder

def main():
    # Create H matrix (same as test_simple_decoder.py)
    H = torch.tensor([
        [1, 1, 0, 1],  # check0 connected to var0, var1, var3
        [0, 1, 1, 0],  # check1 connected to var1, var2
        [1, 0, 1, 1]   # check2 connected to var0, var2, var3
    ], dtype=torch.float)
    
    # Create input LLRs for multiple batches
    batch_size = 3
    input_llr = torch.tensor([
        [2.0, -1.5, 0.5, -2.0],  # batch 0 (same as test_simple_decoder.py)
        [-1.0, 2.0, -1.5, 1.0],  # batch 1
        [1.5, -2.0, 1.0, -1.5]   # batch 2
    ], dtype=torch.float)
    
    print("\n=== Test Setup ===")
    print(f"H matrix:\n{H}")
    print(f"\nInput LLRs:")
    for i in range(batch_size):
        print(f"Batch {i}: {input_llr[i]}")
    
    # Create and run decoder
    decoder = BatchMessageGNNDecoder(H, num_iterations=3)  # Three iterations for clarity
    decoded_bits = decoder.forward(input_llr=input_llr)
    
    # Print final results for each batch
    print("\n=== Final Results ===")
    for i in range(batch_size):
        print(f"\nBatch {i} decoded bits: {decoded_bits[i]}")

if __name__ == "__main__":
    main() 