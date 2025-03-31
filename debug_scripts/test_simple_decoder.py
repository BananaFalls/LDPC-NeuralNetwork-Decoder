import torch
import sys
import os
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.simple_message_gnn_decoder import SimpleMessageGNNDecoder

def main():
    # Create H matrix
    H = torch.tensor([
        [1, 1, 0, 1],  # check0 connected to var0, var1, var3
        [0, 1, 1, 0],  # check1 connected to var1, var2
        [1, 0, 1, 1]   # check2 connected to var0, var2, var3
    ], dtype=torch.float)
    
    # Create input LLRs
    input_llr = torch.tensor([[2.0, -1.5, 0.5, -2.0]], dtype=torch.float)
    
    print("\n=== Test Setup ===")
    print(f"H matrix:\n{H}")
    print(f"\nInput LLRs: {input_llr[0]}")
    
    # Create and run decoder
    decoder = SimpleMessageGNNDecoder(H, num_iterations=3)  # Two iterations for clarity
    output_probs = decoder.forward(input_llr=input_llr)
    
    # Convert probabilities to bits
    decoded_bits = (output_probs >= 0.5).float()
    print(f"Decoded bits: {decoded_bits[0]}")

if __name__ == "__main__":
    main() 