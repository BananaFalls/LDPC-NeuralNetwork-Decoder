import torch
import numpy as np
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate

# Set random seed
torch.manual_seed(42)

# 1. Create test data
print("=== QPSK Modulation/Demodulation Debug ===")
print("\n1. Creating test data")

# Create 4 test patterns to test all possible bit combinations (00, 01, 10, 11)
bits = torch.tensor([
    [0, 0, 0, 1, 1, 0, 1, 1],  # Contains all 4 possible bit pairs
    [0, 0, 0, 0, 0, 0, 0, 0],  # All zeros (all 00 pairs)
], dtype=torch.float)

print(f"Bits shape: {bits.shape}")
print(f"Bits (row 1): {bits[0]}")
print(f"Bits (row 2): {bits[1]}")

# 2. QPSK Modulation
print("\n2. QPSK Modulation")
symbols = qpsk_modulate(bits)
print(f"Symbols shape: {symbols.shape}")
print(f"Symbols (row 1): {symbols[0]}")
print(f"Symbols (row 2): {symbols[1]}")

# Check mapping for each possible bit pair
print("\nVerifying bit pair to symbol mapping:")
norm_factor = 1/np.sqrt(2)
expected_mapping = {
    (0, 0): complex(norm_factor, norm_factor),    # 00 -> (1+j)/√2
    (0, 1): complex(norm_factor, -norm_factor),   # 01 -> (1-j)/√2
    (1, 0): complex(-norm_factor, norm_factor),   # 10 -> (-1+j)/√2
    (1, 1): complex(-norm_factor, -norm_factor),  # 11 -> (-1-j)/√2
}

for i in range(4):  # Check first 4 bit pairs in row 1
    bit_pair = (int(bits[0, i*2]), int(bits[0, i*2+1]))
    symbol = symbols[0, i]
    expected = expected_mapping.get(bit_pair, "unknown")
    match = "✓" if abs(symbol - expected) < 1e-6 else "✗"
    print(f"Bit pair {bit_pair} -> Symbol {symbol:.6f} (Expected: {expected:.6f}) {match}")

# Verify symbol power (should be 1.0)
symbol_power = torch.mean(torch.abs(symbols)**2).item()
print(f"\nAverage symbol power: {symbol_power:.6f} (expected: 1.0)")

# 3. AWGN Channel - Test different SNRs
snr_values = [0, 5, 10]

for snr in snr_values:
    print(f"\n3. AWGN Channel (SNR = {snr} dB)")
    
    # Run multiple trials to verify noise statistics
    num_trials = 10
    noise_powers = []
    
    for trial in range(num_trials):
        noisy_symbols = awgn_channel(symbols, snr)
        noise = noisy_symbols - symbols
        noise_power = torch.mean(torch.abs(noise)**2).item()
        noise_powers.append(noise_power)
    
    avg_noise_power = sum(noise_powers) / num_trials
    expected_noise_power = 1 / (10**(snr/10))
    
    print(f"Average measured noise power over {num_trials} trials: {avg_noise_power:.6f}")
    print(f"Expected noise power: {expected_noise_power:.6f}")
    print(f"Ratio (measured/expected): {avg_noise_power/expected_noise_power:.4f}")
    
    # Use last generated noisy symbols for demodulation
    print(f"Sample noisy symbols: {noisy_symbols[0, :2]}")
    
    # 4. QPSK Demodulation
    print(f"\n4. QPSK Demodulation (SNR = {snr} dB)")
    llrs = qpsk_demodulate(noisy_symbols, snr)
    print(f"LLR shape: {llrs.shape}")
    print(f"First few LLRs (row 1): {llrs[0, :8]}")
    
    # Verify LLR sign matches expected bit value
    print("\nVerifying LLR polarity:")
    for i in range(min(8, bits.shape[1])):
        bit = bits[0, i]
        llr = llrs[0, i]
        expected_sign = "negative" if bit == 1 else "positive"
        actual_sign = "negative" if llr < 0 else "positive"
        correct = "✓" if (bit == 1 and llr < 0) or (bit == 0 and llr > 0) else "✗"
        print(f"Bit {i}: Value={int(bit)}, LLR={llr:.2f} (expected {expected_sign}, got {actual_sign}) {correct}")
    
    # 5. Hard decisions and error rates
    print(f"\n5. Hard decisions (SNR = {snr} dB)")
    hard_bits = (llrs < 0).float()
    print(f"Hard bits (row 1): {hard_bits[0]}")
    
    # Count errors 
    bit_errors = (hard_bits != bits).sum().item()
    ber = bit_errors / bits.numel()
    print(f"Bit errors: {bit_errors} out of {bits.numel()}")
    print(f"BER: {ber:.6f}")
    
    # Print error locations
    if bit_errors > 0:
        print("\nError locations:")
        for i in range(bits.shape[0]):
            for j in range(bits.shape[1]):
                if bits[i, j] != hard_bits[i, j]:
                    print(f"Error at row {i}, bit {j}: Original {int(bits[i,j])}, Decoded {int(hard_bits[i,j])}")
                    if j % 2 == 0 and j+1 < bits.shape[1]:  # If this is the first bit of a pair
                        bit_pair = (int(bits[i, j]), int(bits[i, j+1]))
                        symbol_idx = j // 2
                        print(f"  Bit pair: {bit_pair}, Symbol: {symbols[i, symbol_idx]}")
                        print(f"  Noisy symbol: {noisy_symbols[i, symbol_idx]}")
                        print(f"  LLRs: [{llrs[i, j]:.2f}, {llrs[i, j+1]:.2f}]")

# Test with a very high SNR (should have no errors)
print("\n6. High SNR Test (SNR = 20 dB)")
high_snr = 20
noisy_symbols_clean = awgn_channel(symbols, high_snr)
llrs_clean = qpsk_demodulate(noisy_symbols_clean, high_snr)
hard_bits_clean = (llrs_clean < 0).float()
bit_errors_clean = (hard_bits_clean != bits).sum().item()
print(f"Bit errors at {high_snr}dB: {bit_errors_clean} out of {bits.numel()}")

print("\nDebug complete!") 