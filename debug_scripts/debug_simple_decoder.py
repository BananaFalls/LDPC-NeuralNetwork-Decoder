import torch
import os
import sys

# Add the project root to the path to import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from ldpc_neural_decoder.models.simple_message_gnn_decoder import SimpleMessageGNNDecoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix, create_LLR_mapping

def create_small_example():
    """
    Create a small example parity-check matrix and corresponding structures
    """
    # Create a small 3x4 parity-check matrix
    H = torch.tensor([
        [1, 1, 0, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 0]
    ], dtype=torch.float32)
    
    # Create adjacency matrices
    var_to_check = torch.tensor([
        [1, 1, 0, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0]
    ], dtype=torch.float32)
    
    check_to_var = var_to_check.t()
    
    # Create message to variable mapping (identity in this case)
    message_to_var = torch.eye(4, dtype=torch.float32)
    
    # Create message types (not used in simplified version)
    message_types = torch.zeros(4, dtype=torch.float32)
    
    return H, var_to_check, check_to_var, message_to_var, message_types

def debug_decoder():
    """
    Debug the simplified decoder with a small example
    """
    print("Creating small example parity-check matrix and structures...")
    H, var_to_check, check_to_var, message_to_var, message_types = create_small_example()
    
    print("\nParity-check matrix H:")
    print(H)
    print("\nVariable to Check adjacency:")
    print(var_to_check)
    print("\nCheck to Variable adjacency:")
    print(check_to_var)
    
    # Create decoder
    decoder = SimpleMessageGNNDecoder(num_iterations=3)
    
    # Create example input LLRs (simulating received values)
    # Using small values for easier debugging
    input_llrs = torch.tensor([
        [1.0, -1.0, 0.5, -0.5]  # Example LLRs for 4 variable nodes
    ], dtype=torch.float32)
    
    print("\nInput LLRs:")
    print(input_llrs)
    
    # Run decoder
    print("\nRunning decoder...")
    decoded_probs = decoder(
        input_llr=input_llrs,
        message_to_var_mapping=message_to_var,
        message_types=message_types,
        var_to_check_adjacency=var_to_check,
        check_to_var_adjacency=check_to_var
    )
    
    print("\nDecoded probabilities:")
    print(decoded_probs)
    
    # Convert to hard decisions
    hard_decisions = (decoded_probs > 0.5).float()
    print("\nHard decisions:")
    print(hard_decisions)
    
    # Check if the decoded word satisfies all parity checks
    syndrome = torch.matmul(hard_decisions, H.t()) % 2
    print("\nSyndrome (should be all zeros for valid codeword):")
    print(syndrome)
    
    # Print final LLR values
    print("\nFinal LLR values (log(prob/(1-prob))):")
    final_llrs = torch.log(decoded_probs / (1 - decoded_probs))
    print(final_llrs)

def debug_real_code():
    """
    Debug the decoder with a real 5G LDPC code
    """
    print("\nTesting with real 5G LDPC code...")
    
    # Load and expand base matrix
    file_path = os.path.join(project_root, "5G LDPC CODES", "NR_2_0_4.txt")
    base_matrix = load_base_matrix(file_path)
    H = expand_base_matrix(base_matrix, Z=4)
    
    print(f"\nBase matrix shape: {base_matrix.shape}")
    print(f"Expanded matrix shape: {H.shape}")
    
    # Create necessary structures
    H_T = H.t()
    message_to_var, check_index_tensor, var_index_tensor, output_index_tensor = create_LLR_mapping(H_T)
    
    # Create decoder
    decoder = SimpleMessageGNNDecoder(num_iterations=3)
    
    # Create example input LLRs (simulating received values)
    batch_size = 1
    num_vars = H.shape[1]
    input_llrs = torch.randn(batch_size, num_vars)  # Random LLRs for testing
    
    print("\nRunning decoder on real code...")
    decoded_probs = decoder(
        input_llr=input_llrs,
        message_to_var_mapping=message_to_var,
        message_types=None,  # Not used in simplified version
        var_to_check_adjacency=var_index_tensor,
        check_to_var_adjacency=check_index_tensor
    )
    
    # Convert to hard decisions
    hard_decisions = (decoded_probs > 0.5).float()
    
    # Check syndrome
    syndrome = torch.matmul(hard_decisions, H.t()) % 2
    print("\nSyndrome (should be all zeros for valid codeword):")
    print(syndrome[:5])  # Print first 5 check equations
    
    print("\nNumber of unsatisfied check equations:", torch.sum(syndrome).item())

def main():
    print("Starting decoder debug...")
    
    # First test with small example
    print("\n=== Testing with small example ===")
    debug_decoder()
    
    # Then test with real code
    print("\n=== Testing with real 5G LDPC code ===")
    debug_real_code()

if __name__ == "__main__":
    main() 