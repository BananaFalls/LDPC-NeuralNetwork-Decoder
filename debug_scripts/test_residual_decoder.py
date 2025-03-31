import torch
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.residual_weight_sharing_decoder import ResidualWeightSharingDecoder
from ldpc_neural_decoder.utils.llr_generator import generate_llrs, verify_llrs

def load_base_matrix(file_path):
    """Load base matrix from file"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert text to numpy array
    matrix = []
    for line in lines:
        row = [int(x) for x in line.strip().split()]
        matrix.append(row)
    
    # Convert to torch tensor
    return torch.tensor(matrix, dtype=torch.float)

def main():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Small example (commented out)
    """
    # Define a simple base matrix Hb
    # Convention:
    # -1: No connection (zero matrix)
    # 0: Non-shifted identity matrix
    # >0: Shifted identity matrix by that amount
    Hb_small = torch.tensor([
        [1, 2, -1],  # First row: shift by 1, shift by 2, no connection
        [-1, 0, 1],  # Second row: no connection, identity matrix, shift by 1
    ], dtype=torch.float)
    z_small = 2  # This will create a 4×6 expanded matrix
    """
    
    # Load the 5G LDPC base matrix
    base_matrix_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "5G LDPC CODES",
        "NR_2_0_4.txt"
    )
    Hb = load_base_matrix(base_matrix_path)
    
    # Define expansion factor (size of circulant blocks)
    z = 4  # Small expansion factor for testing
    # z = 2 # to be used with small 2x3 matrix
    
    print("\n=== Test Configuration ===")
    print(f"Base matrix Hb shape: {Hb.shape}")
    print(f"Expansion factor z: {z}")
    print(f"This will create a {Hb.shape[0]*z}×{Hb.shape[1]*z} expanded matrix")
    
    # Create decoder instance with residual connections
    decoder = ResidualWeightSharingDecoder(
        base_matrix=Hb, 
        expansion_factor=z, 
        num_iterations=5,  # Increased iterations for larger matrix
        residual_depth=2    # Increased residual depth
    )
    
    print("\nExpanded H matrix shape:")
    print(decoder.H.shape)
    
    # Settings for testing
    # Generate LLRs from zero codewords with AWGN
    batch_size = 2  # Reduced batch size for larger matrix
    code_length = Hb.shape[1] * z  # Number of variable nodes
    snr_db = 3  # Increased SNR for better initial performance
    
    print(f"\nGenerating LLRs:")
    print(f"Batch size: {batch_size}")
    print(f"Code length: {code_length}")
    print(f"SNR: {snr_db} dB")
    
    input_llr, transmitted_bits, received_symbols = generate_llrs(
        batch_size=batch_size,
        code_length=code_length,
        snr_db=snr_db
    )
    
    print("\nGenerated values for first batch:")
    print(f"Transmitted bits shape: {transmitted_bits.shape}")
    print(f"Received symbols shape: {received_symbols.shape}")
    print(f"Input LLRs shape: {input_llr.shape}")
    
    # Run decoder
    print("\n=== Running Residual Weight Sharing Decoder ===")
    decoded_bits = decoder.forward(input_llr)
    
    # Print results for each batch
    print("\n=== Decoding Results ===")
    for batch_idx in range(input_llr.shape[0]):
        print(f"\nBatch {batch_idx}:")
        print(f"First 10 transmitted bits: {transmitted_bits[batch_idx][:10]}")
        print(f"First 10 decoded bits: {decoded_bits[batch_idx][:10]}")
        
        # Verify parity check equations
        syndrome = torch.matmul(decoded_bits[batch_idx], decoder.H.t()) % 2
        is_valid = torch.all(syndrome == 0)
        print(f"Valid codeword: {is_valid}")
        if not is_valid:
            print(f"Number of unsatisfied checks: {torch.sum(syndrome != 0).item()}")
        
        # Calculate bit errors
        errors = torch.sum(decoded_bits[batch_idx] != transmitted_bits[batch_idx])
        ber = errors / code_length
        print(f"Bit errors: {errors}")
        print(f"BER: {ber:.4f}")
    
    # Print number of learnable parameters
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nNumber of learnable parameters: {num_params}")
    
    # Show residual weights
    # print("\nAverage residual weights:")
    # for t in range(1, decoder.residual_depth + 1):
    #     print(f"w_{t}: {decoder.residual_weights[f'w_{t}'].mean().item():.3f}")

if __name__ == "__main__":
    main() 