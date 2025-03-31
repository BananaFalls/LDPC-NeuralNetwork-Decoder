import torch

def generate_llrs(batch_size, code_length, snr_db):
    """
    Generate LLRs from zero codewords using BPSK modulation and AWGN channel.
    
    Args:
        batch_size (int): Number of codewords to generate
        code_length (int): Length of each codeword
        snr_db (float): Signal-to-Noise Ratio in dB
        
    Returns:
        tuple: (llrs, transmitted_bits, received_symbols)
            - llrs: Log-Likelihood Ratios [batch_size, code_length]
            - transmitted_bits: Original zero codewords [batch_size, code_length]
            - received_symbols: Received symbols after AWGN [batch_size, code_length]
    """
    # 1. Generate zero codewords
    transmitted_bits = torch.zeros((batch_size, code_length))
    
    # 2. BPSK modulation (0 → +1, 1 → -1)
    transmitted_symbols = 1 - 2 * transmitted_bits  # All symbols will be +1
    
    # 3. Add AWGN noise
    snr_linear = 10 ** (snr_db / 10)
    noise_variance = 1 / (2 * snr_linear)  # Noise variance (σ²) for given SNR
    noise = torch.normal(0, torch.sqrt(torch.tensor(noise_variance)), 
                        size=transmitted_symbols.shape)
    received_symbols = transmitted_symbols + noise
    
    # 4. Calculate LLRs using 2y/σ²
    llrs = 2 * received_symbols / noise_variance
    
    return llrs, transmitted_bits, received_symbols

def verify_llrs(llrs, transmitted_bits, H):
    """
    Verify the generated LLRs by decoding and checking if we recover the original codeword.
    
    Args:
        llrs (torch.Tensor): Generated LLRs [batch_size, code_length]
        transmitted_bits (torch.Tensor): Original transmitted bits [batch_size, code_length]
        H (torch.Tensor): Parity check matrix
        
    Returns:
        tuple: (decoded_bits, is_valid, ber)
            - decoded_bits: Hard decisions from LLRs [batch_size, code_length]
            - is_valid: Whether decoded bits satisfy parity checks [batch_size]
            - ber: Bit Error Rate compared to transmitted bits
    """
    # Make hard decisions
    decoded_bits = (llrs < 0).float()
    
    # Verify parity checks
    syndromes = torch.matmul(decoded_bits, H.t()) % 2
    is_valid = torch.all(syndromes == 0, dim=1)
    
    # Calculate BER
    errors = torch.sum(decoded_bits != transmitted_bits, dim=1)
    ber = errors / transmitted_bits.shape[1]
    
    return decoded_bits, is_valid, ber 