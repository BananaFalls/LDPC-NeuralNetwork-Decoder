import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.utils.training_data_generator import generate_training_data, save_training_data
from debug_scripts.test_residual_decoder import load_base_matrix

def main():
    # Set fixed seed for reproducibility
    torch.manual_seed(42)
    
    # Load the 5G LDPC base matrix
    base_matrix_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "5G LDPC CODES",
        "NR_2_0_4.txt"
    )
    Hb = load_base_matrix(base_matrix_path)
    
    # Define expansion factor and code length
    z = 4  # Small expansion factor for testing
    code_length = Hb.shape[1] * z
    
    # Common parameters
    common_params = {
        'code_length': code_length,
        'batch_size': 64,       # 64 examples per batch
        'snr_range': (-1, 8),   # Train from -1dB to 8dB
        'snr_points': 10        # 10 different SNR points
    }
    
    # Training data parameters (80% of total)
    train_params = {
        **common_params,
        'num_batches': 500,     # 500 batches per SNR point
    }
    
    # Validation data parameters (10% of total)
    val_params = {
        **common_params,
        'num_batches': 63,      # 63 batches ≈ 10% of training
    }
    
    # Test data parameters (10% of total)
    test_params = {
        **common_params,
        'num_batches': 63,      # 63 batches ≈ 10% of training
    }
    
    # Calculate total examples
    train_examples = train_params['num_batches'] * train_params['batch_size'] * train_params['snr_points']
    val_examples = val_params['num_batches'] * val_params['batch_size'] * val_params['snr_points']
    test_examples = test_params['num_batches'] * test_params['batch_size'] * test_params['snr_points']
    total_examples = train_examples + val_examples + test_examples
    
    print("\n=== Generating Datasets ===")
    print(f"Base matrix shape: {Hb.shape}")
    print(f"Expansion factor z: {z}")
    print(f"Code length: {code_length}")
    
    print(f"\nDataset sizes:")
    print(f"Training:")
    print(f"- {train_params['num_batches']} batches per SNR point")
    print(f"- {train_params['batch_size']} examples per batch")
    print(f"- {train_params['snr_points']} SNR points from {train_params['snr_range'][0]}dB to {train_params['snr_range'][1]}dB")
    print(f"Total training examples: {train_examples}")
    print(f"Training examples per parameter: {train_examples / 1983:.1f}")
    
    print(f"\nValidation:")
    print(f"- {val_params['num_batches']} batches per SNR point")
    print(f"Total validation examples: {val_examples}")
    
    print(f"\nTesting:")
    print(f"- {test_params['num_batches']} batches per SNR point")
    print(f"Total test examples: {test_examples}")
    
    print(f"\nTotal examples across all sets: {total_examples}")
    
    # Create directory for data if it doesn't exist
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training_data"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate and save training data
    print("\nGenerating training data...")
    train_data = generate_training_data(**train_params)
    train_path = os.path.join(save_dir, f"train_data_z{z}.pt")
    save_training_data(train_data, train_path)
    print(f"Training data saved to: {train_path}")
    
    # Generate and save validation data
    print("\nGenerating validation data...")
    val_data = generate_training_data(**val_params)
    val_path = os.path.join(save_dir, f"val_data_z{z}.pt")
    save_training_data(val_data, val_path)
    print(f"Validation data saved to: {val_path}")
    
    # Generate and save test data
    print("\nGenerating test data...")
    test_data = generate_training_data(**test_params)
    test_path = os.path.join(save_dir, f"test_data_z{z}.pt")
    save_training_data(test_data, test_path)
    print(f"Test data saved to: {test_path}")

if __name__ == "__main__":
    main() 