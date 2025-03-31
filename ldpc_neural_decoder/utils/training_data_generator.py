import torch
import numpy as np
from typing import Tuple, List

def generate_training_data(
    code_length: int,
    num_batches: int = 1000,
    batch_size: int = 32,
    snr_range: Tuple[float, float] = (-1.0, 8.0),
    snr_points: int = 10,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Generate training data for LDPC decoder across different SNR points.
    
    Args:
        code_length: Length of the codeword (number of variable nodes)
        num_batches: Number of batches to generate per SNR point
        batch_size: Number of examples per batch
        snr_range: (min_snr, max_snr) in dB
        snr_points: Number of SNR points to sample from snr_range
        
    Returns:
        List of (input_llr, transmitted_bits, received_symbols) tuples,
        one for each batch at each SNR point
    """
    # Generate SNR points
    snr_values = np.linspace(snr_range[0], snr_range[1], snr_points)
    training_data = []
    
    print(f"Generating training data:")
    print(f"- {num_batches} batches per SNR point")
    print(f"- {batch_size} examples per batch")
    print(f"- {snr_points} SNR points from {snr_range[0]}dB to {snr_range[1]}dB")
    print(f"Total examples: {num_batches * batch_size * snr_points}")
    
    for snr_db in snr_values:
        print(f"\nGenerating data for SNR = {snr_db:.1f}dB")
        
        # Generate batches for this SNR point
        for _ in range(num_batches):
            # Generate random zero codewords
            transmitted_bits = torch.zeros(batch_size, code_length)
            
            # BPSK modulation: 0 -> +1, 1 -> -1
            transmitted_symbols = 1 - 2 * transmitted_bits
            
            # Add AWGN noise
            snr_linear = 10 ** (snr_db / 10)
            noise_std = 1 / np.sqrt(2 * snr_linear)
            noise = torch.normal(0, noise_std, transmitted_symbols.shape)
            received_symbols = transmitted_symbols + noise
            
            # Calculate LLRs: LLR = 2y/σ²
            input_llr = 2 * received_symbols / (noise_std ** 2)
            
            training_data.append((input_llr, transmitted_bits, received_symbols))
    
    return training_data

def save_training_data(
    training_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    save_path: str
) -> None:
    """
    Save training data to disk.
    
    Args:
        training_data: List of (input_llr, transmitted_bits, received_symbols) tuples
        save_path: Path to save the training data
    """
    # Convert list of tuples to tensors
    input_llrs = torch.stack([x[0] for x in training_data])
    transmitted_bits = torch.stack([x[1] for x in training_data])
    received_symbols = torch.stack([x[2] for x in training_data])
    
    # Save to disk
    torch.save({
        'input_llrs': input_llrs,
        'transmitted_bits': transmitted_bits,
        'received_symbols': received_symbols
    }, save_path)
    
def load_training_data(load_path: str) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Load training data from disk.
    
    Args:
        load_path: Path to load the training data from
        
    Returns:
        List of (input_llr, transmitted_bits, received_symbols) tuples
    """
    # Load data
    data = torch.load(load_path)
    
    # Convert back to list of tuples
    training_data = []
    for i in range(len(data['input_llrs'])):
        training_data.append((
            data['input_llrs'][i],
            data['transmitted_bits'][i],
            data['received_symbols'][i]
        ))
    
    return training_data 