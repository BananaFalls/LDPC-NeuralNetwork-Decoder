import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.weight_sharing_decoder import WeightSharingDecoder
from ldpc_neural_decoder.utils.llr_generator import generate_llrs, verify_llrs

def main():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Define a simple base matrix Hb
    # Convention:
    # -1: No connection (zero matrix)
    # 0: Non-shifted identity matrix
    # >0: Shifted identity matrix by that amount
    Hb = torch.tensor([
        [1, 2, -1],  # First row: shift by 1, shift by 2, no connection
        [-1, 0, 1],  # Second row: no connection, identity matrix, shift by 1
    ], dtype=torch.float)
    
    # Define expansion factor (size of circulant blocks)
    z = 2  # This will create a 4×6 expanded matrix
    
    print("\n=== Test Configuration ===")
    print("Base matrix Hb (convention: -1=no connection, 0=identity, >0=shifted):")
    print(Hb)
    print(f"Expansion factor z: {z}")
    
    # Create decoder instance
    decoder = WeightSharingDecoder(
        base_matrix=Hb, 
        expansion_factor=z, 
        num_iterations=5
    )
    
    print("\nExpanded H matrix:")
    print(decoder.H)
    
    # Generate LLRs from zero codewords with AWGN
    batch_size = 3
    code_length = Hb.shape[1] * z  # Number of variable nodes
    snr_db = 1  # Try different SNR values
    
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
    print(f"Transmitted bits: {transmitted_bits[0]}")
    print(f"Received symbols: {received_symbols[0]}")
    print(f"Input LLRs: {input_llr[0]}")
    
    # Run decoder
    print("\n=== Running Weight Sharing Decoder ===")
    decoded_bits = decoder.forward(input_llr)
    
    # Print results for each batch
    print("\n=== Decoding Results ===")
    for batch_idx in range(input_llr.shape[0]):
        print(f"\nBatch {batch_idx}:")
        print(f"Transmitted bits: {transmitted_bits[batch_idx]}")
        print(f"Decoded bits: {decoded_bits[batch_idx]}")
        
        # Verify parity check equations
        syndrome = torch.matmul(decoded_bits[batch_idx], decoder.H.t()) % 2
        is_valid = torch.all(syndrome == 0)
        print(f"Valid codeword: {is_valid}")
        if not is_valid:
            print(f"Syndrome: {syndrome}")
        
        # Calculate bit errors
        errors = torch.sum(decoded_bits[batch_idx] != transmitted_bits[batch_idx])
        ber = errors / code_length
        print(f"Bit errors: {errors}")
        print(f"BER: {ber:.4f}")
    
    # Print number of learnable parameters
    num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"\nNumber of learnable parameters: {num_params}")
    
    # Show weight sharing structure
    # print("\nShared weights structure:")
    # for base_row in range(Hb.shape[0]):
    #     for base_col in range(Hb.shape[1]):
    #         if Hb[base_row, base_col] >= 0:  # Show weights for non-shifted and shifted identity matrices
    #             key = f"{base_row}_{base_col}"
    #             print(f"Block ({base_row}, {base_col}):")
    #             print(f"  Type: {'Identity' if Hb[base_row, base_col] == 0 else f'Shift by {int(Hb[base_row, base_col])}'}")
    #             print(f"  var->check weight: {decoder.shared_var_to_check_weights[key].item():.3f}")
    #             print(f"  check->var weight: {decoder.shared_check_to_var_weights[key].item():.3f}")

if __name__ == "__main__":
    main() 