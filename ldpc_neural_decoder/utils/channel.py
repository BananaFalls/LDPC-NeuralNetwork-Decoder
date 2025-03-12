import torch
import numpy as np

def qpsk_modulate(bits):
    """
    Modulate binary bits using QPSK.
    
    Args:
        bits (torch.Tensor): Binary bits (0s and 1s) to modulate
        
    Returns:
        torch.Tensor: Complex QPSK symbols
    """
    # Map bits to QPSK constellation points
    # 00 -> (1+j)/sqrt(2), 01 -> (1-j)/sqrt(2), 10 -> (-1+j)/sqrt(2), 11 -> (-1-j)/sqrt(2)
    # For simplicity, we'll map 0 -> 1/sqrt(2) and 1 -> -1/sqrt(2) for both I and Q
    
    # Ensure bits is a 1D tensor
    bits = bits.view(-1)
    
    # Map 0 -> 1/sqrt(2), 1 -> -1/sqrt(2)
    symbols = 1/np.sqrt(2) - bits.float() * np.sqrt(2)
    
    # If bits length is odd, add a padding bit
    if bits.size(0) % 2 == 1:
        symbols = torch.cat([symbols, torch.tensor([1/np.sqrt(2)], device=bits.device)])
    
    # Reshape to pairs for I and Q
    I = symbols[0::2]
    Q = symbols[1::2]
    
    # Create complex symbols
    qpsk_symbols = torch.complex(I, Q)
    
    return qpsk_symbols

def awgn_channel(symbols, snr_db):
    """
    Pass symbols through an AWGN channel with specified SNR.
    
    Args:
        symbols (torch.Tensor): Complex symbols to transmit
        snr_db (float): Signal-to-Noise Ratio in dB
        
    Returns:
        torch.Tensor: Received noisy symbols
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise power (assuming signal power = 1)
    noise_power = 1 / snr_linear
    
    # Generate complex Gaussian noise
    noise_real = torch.randn(symbols.size(), device=symbols.device) * np.sqrt(noise_power/2)
    noise_imag = torch.randn(symbols.size(), device=symbols.device) * np.sqrt(noise_power/2)
    noise = torch.complex(noise_real, noise_imag)
    
    # Add noise to symbols
    received_symbols = symbols + noise
    
    return received_symbols

def qpsk_demodulate(received_symbols, snr_db):
    """
    Demodulate received QPSK symbols to LLRs.
    
    Args:
        received_symbols (torch.Tensor): Received complex symbols
        snr_db (float): Signal-to-Noise Ratio in dB
        
    Returns:
        torch.Tensor: Log-Likelihood Ratios (LLRs)
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise variance
    noise_var = 1 / snr_linear
    
    # Extract real and imaginary parts
    I = received_symbols.real
    Q = received_symbols.imag
    
    # Calculate LLRs for I and Q components
    # LLR = 2*r/sigma^2 where r is the received signal and sigma^2 is the noise variance
    llr_I = 2 * I / noise_var
    llr_Q = 2 * Q / noise_var
    
    # Interleave I and Q LLRs
    llrs = torch.zeros(2 * len(received_symbols), device=received_symbols.device)
    llrs[0::2] = llr_I
    llrs[1::2] = llr_Q
    
    return llrs

def compute_ber_fer(transmitted_bits, decoded_bits):
    """
    Compute Bit Error Rate (BER) and Frame Error Rate (FER).
    
    Args:
        transmitted_bits (torch.Tensor): Original transmitted bits
        decoded_bits (torch.Tensor): Decoded bits
        
    Returns:
        tuple: (BER, FER)
            - BER: Bit Error Rate
            - FER: Frame Error Rate (1 if any bit is wrong, 0 otherwise)
    """
    # Ensure both tensors have the same shape
    assert transmitted_bits.shape == decoded_bits.shape, "Transmitted and decoded bits must have the same shape"
    
    # Calculate bit errors
    bit_errors = (transmitted_bits != decoded_bits).float()
    
    # BER: Number of bit errors divided by total number of bits
    ber = bit_errors.mean().item()
    
    # FER: 1 if any bit in the frame is wrong, 0 otherwise
    fer = (bit_errors.sum(dim=1) > 0).float().mean().item()
    
    return ber, fer 