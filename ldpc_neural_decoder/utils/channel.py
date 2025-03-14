import torch
import numpy as np

def qpsk_modulate(bits):
    """
    Modulate binary bits using QPSK.
    
    Args:
        bits (torch.Tensor): Binary bits (0s and 1s) to modulate. Can be of shape (batch_size, n_bits)
                            or just (n_bits,)
        
    Returns:
        torch.Tensor: Complex QPSK symbols of shape (batch_size, n_symbols) or (n_symbols,)
    """
    # Save original shape and device
    original_shape = bits.shape
    device = bits.device
    is_batched = len(original_shape) > 1
    
    if is_batched:
        batch_size = original_shape[0]
        n_bits = original_shape[1]
        # Reshape to (batch_size, n_bits)
        bits = bits.reshape(batch_size, n_bits)
    else:
        batch_size = 1
        n_bits = original_shape[0]
        # Add batch dimension
        bits = bits.reshape(1, n_bits)
    
    # Process each batch
    qpsk_symbols_list = []
    
    for i in range(batch_size):
        # Get bits for this batch
        batch_bits = bits[i]
        
        # Map 0 -> 1/sqrt(2), 1 -> -1/sqrt(2)
        symbols = 1/np.sqrt(2) - batch_bits.float() * np.sqrt(2)
        
        # If bits length is odd, add a padding bit
        if len(batch_bits) % 2 == 1:
            symbols = torch.cat([symbols, torch.tensor([1/np.sqrt(2)], device=device)])
        
        # Reshape to pairs for I and Q
        I = symbols[0::2]
        Q = symbols[1::2]
        
        # Create complex symbols
        batch_qpsk_symbols = torch.complex(I, Q)
        qpsk_symbols_list.append(batch_qpsk_symbols)
    
    # Stack batches
    qpsk_symbols = torch.stack(qpsk_symbols_list)
    
    # Remove batch dimension if input wasn't batched
    if not is_batched:
        qpsk_symbols = qpsk_symbols.squeeze(0)
    
    return qpsk_symbols

def awgn_channel(symbols, snr_db):
    """
    Pass symbols through an AWGN channel with specified SNR.
    
    Args:
        symbols (torch.Tensor): Complex symbols to transmit. Can be of shape (batch_size, n_symbols)
                               or just (n_symbols,)
        snr_db (float): Signal-to-Noise Ratio in dB
        
    Returns:
        torch.Tensor: Received noisy symbols with same shape as input
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
        received_symbols (torch.Tensor): Received complex symbols. Can be of shape (batch_size, n_symbols)
                                        or just (n_symbols,)
        snr_db (float): Signal-to-Noise Ratio in dB
        
    Returns:
        torch.Tensor: Log-Likelihood Ratios (LLRs) with shape (batch_size, 2*n_symbols) or (2*n_symbols,)
    """
    # Save original shape and device
    original_shape = received_symbols.shape
    device = received_symbols.device
    is_batched = len(original_shape) > 1
    
    if is_batched:
        batch_size = original_shape[0]
        n_symbols = original_shape[1]
        # Ensure shape is (batch_size, n_symbols)
        received_symbols = received_symbols.reshape(batch_size, n_symbols)
    else:
        batch_size = 1
        n_symbols = original_shape[0]
        # Add batch dimension
        received_symbols = received_symbols.reshape(1, n_symbols)
    
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate noise variance
    noise_var = 1 / snr_linear
    
    # Process each batch
    llrs_list = []
    
    for i in range(batch_size):
        # Get symbols for this batch
        batch_symbols = received_symbols[i]
        
        # Extract real and imaginary parts
        I = batch_symbols.real
        Q = batch_symbols.imag
        
        # Calculate LLRs for I and Q components
        # LLR = 2*r/sigma^2 where r is the received signal and sigma^2 is the noise variance
        llr_I = 2 * I / noise_var
        llr_Q = 2 * Q / noise_var
        
        # Interleave I and Q LLRs
        batch_llrs = torch.zeros(2 * len(batch_symbols), device=device)
        batch_llrs[0::2] = llr_I
        batch_llrs[1::2] = llr_Q
        
        llrs_list.append(batch_llrs)
    
    # Stack batches
    llrs = torch.stack(llrs_list)
    
    # Remove batch dimension if input wasn't batched
    if not is_batched:
        llrs = llrs.squeeze(0)
    
    return llrs

def compute_ber_fer(transmitted_bits, decoded_bits):
    """
    Compute Bit Error Rate (BER) and Frame Error Rate (FER).
    
    Args:
        transmitted_bits (torch.Tensor): Original transmitted bits. Can be of shape (batch_size, n_bits)
                                        or just (n_bits,)
        decoded_bits (torch.Tensor): Decoded bits with same shape as transmitted_bits
        
    Returns:
        tuple: (BER, FER)
            - BER: Bit Error Rate
            - FER: Frame Error Rate (1 if any bit is wrong, 0 otherwise)
    """
    # Ensure both tensors have the same shape
    assert transmitted_bits.shape == decoded_bits.shape, "Transmitted and decoded bits must have the same shape"
    
    # Check if input is batched
    is_batched = len(transmitted_bits.shape) > 1
    
    # If not batched, add a batch dimension
    if not is_batched:
        transmitted_bits = transmitted_bits.unsqueeze(0)
        decoded_bits = decoded_bits.unsqueeze(0)
    
    # Calculate bit errors
    bit_errors = (transmitted_bits != decoded_bits).float()
    
    # BER: Number of bit errors divided by total number of bits
    ber = bit_errors.mean().item()
    
    # FER: 1 if any bit in the frame is wrong, 0 otherwise
    fer = (bit_errors.sum(dim=1) > 0).float().mean().item()
    
    return ber, fer


class AWGNChannel:
    """
    Additive White Gaussian Noise (AWGN) channel.
    
    This class simulates an AWGN channel for binary transmission.
    """
    def __init__(self):
        """
        Initialize the AWGN channel.
        """
        pass
    
    def transmit(self, bits, snr_db):
        """
        Transmit bits through the AWGN channel.
        
        Args:
            bits (torch.Tensor): Binary bits (0s and 1s) to transmit. Shape (batch_size, n_bits)
            snr_db (float): Signal-to-Noise Ratio in dB
            
        Returns:
            torch.Tensor: Log-Likelihood Ratios (LLRs) of shape (batch_size, n_bits)
        """
        # Convert bits to BPSK symbols: 0 -> +1, 1 -> -1
        symbols = 1.0 - 2.0 * bits
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate noise standard deviation
        noise_std = 1.0 / np.sqrt(snr_linear)
        
        # Add Gaussian noise
        noise = torch.randn_like(symbols) * noise_std
        received_symbols = symbols + noise
        
        # Convert to LLRs: LLR = 2*r/sigma^2 where r is the received signal
        llrs = 2.0 * received_symbols / (noise_std ** 2)
        
        return llrs 