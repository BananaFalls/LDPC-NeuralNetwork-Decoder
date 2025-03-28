import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder, get_llr_from_noise
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix, create_LLR_mapping
from ldpc_neural_decoder.utils.channel import bpsk_modulate, awgn_channel, bpsk_demodulate, compute_ber_fer

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # File path to the parity-check matrix
    file_path = "5G LDPC CODES/NR_2_0_4.txt"
    print(f"Loading parity-check matrix from {file_path}")
    
    # Load the base matrix
    H_base = load_base_matrix(file_path)
    
    # Expand the base matrix to create the parity-check matrix (specified for NR_2_0_4)
    H = expand_base_matrix(H_base, Z=4)
    print(f"H matrix shape: {H.shape}")
    
    # Move tensor to the device
    H_tensor = H.to(device)
    
    # Create LLR mapping needed for traditional decoders
    # Not directly used by MessageGNNDecoder, but kept for compatibility
    H_T = H_tensor.T
    _, check_index_tensor, var_index_tensor, output_index_tensor = create_LLR_mapping(H_T)
    
    # Decoder settings
    num_iterations = 3  # Number of decoding iterations
    hidden_dim = 1      # Hidden dimension for message features (scalar operation)
    num_of_residual_layers = 2
    
    # Training settings
    num_epochs = 2
    batch_size = 64
    learning_rate = 1e-4
    momentum = 0.9
    weight_decay = 1e-4
    snr_range = [0.0, 8.0]  # SNR range for training
    batches_per_epoch = 10  # Multiple batches per epoch
    
    # Create message GNN decoder
    print("Creating Message GNN Decoder")
    decoder, converter = create_message_gnn_decoder(
        H=H_tensor,
        base_graph=H_base,
        lifting_factor=4,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
        num_of_residual_layers=num_of_residual_layers
    )
    decoder.to(device)
    
    # Get necessary structures from the converter
    message_to_var_mapping = converter.get_message_to_var_mapping().to(device)
    message_types = converter.message_types.to(device) if hasattr(converter, 'message_types') else None
    var_to_check_adjacency = converter.var_to_check_adjacency.to(device)
    check_to_var_adjacency = converter.check_to_var_adjacency.to(device)
    
    print(f"Message to variable mapping shape: {message_to_var_mapping.shape}")
    print(f"Message types shape: {message_types.shape if message_types is not None else 'None'}")
    print(f"Variable to check adjacency shape: {var_to_check_adjacency.shape}")
    print(f"Check to variable adjacency shape: {check_to_var_adjacency.shape}")
    
    # Create directories for saving models if they do not exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Train the decoder
    print("\nStarting training...")
    print(f"Using learning rate: {learning_rate}, SNR range: {snr_range}, Iterations: {num_iterations}")
    print(f"Hidden dimension: {hidden_dim}, Batches per epoch: {batches_per_epoch}")
    print(f"Using BPSK modulation")
    
    # Set up optimizer
    optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    # Training history
    train_losses = []
    ber_history = []
    fer_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        decoder.train()
        epoch_loss = 0.0
        epoch_ber = 0.0 
        epoch_fer = 0.0
        
        # Multiple batches per epoch
        for batch in range(batches_per_epoch):
            # Generate training data
            # Using zero codewords as per paper recommendation
            variable_bit_length = H.shape[1]
            transmitted_bits = torch.zeros((batch_size, variable_bit_length), device=device)

            # BPSK modulation
            bpsk_symbols = bpsk_modulate(transmitted_bits)
            
            # Generate random SNR values within the range
            snr_values = torch.FloatTensor(batch_size).uniform_(snr_range[0], snr_range[1]).to(device)
            
            # Add noise with different SNR for each sample
            noisy_symbols = torch.zeros_like(bpsk_symbols, device=device, dtype=torch.complex64)
            for i in range(batch_size):
                # Pass through AWGN channel with this sample's SNR
                noisy_symbols[i] = awgn_channel(bpsk_symbols[i].unsqueeze(0), snr_values[i].item()).squeeze(0)
            
            # Get LLRs for each codeword with its corresponding SNR
            input_llrs = torch.zeros((batch_size, variable_bit_length), device=device)
            for i in range(batch_size):
                # Calculate noise standard deviation from SNR
                snr_linear = 10 ** (snr_values[i].item() / 10)
                noise_std = 1.0 / np.sqrt(2 * snr_linear)
                
                # Use bpsk_demodulate to get LLRs
                llr = bpsk_demodulate(noisy_symbols[i].unsqueeze(0), snr_values[i].item()).squeeze(0)
                input_llrs[i] = llr
            
            # Clear any previously accumulated gradients
            optimizer.zero_grad()
            
            # Forward pass through the decoder
            decoder_output = decoder(
                input_llr=input_llrs,
                message_to_var_mapping=message_to_var_mapping,
                message_types=message_types,
                var_to_check_adjacency=var_to_check_adjacency,
                check_to_var_adjacency=check_to_var_adjacency,
                ground_truth=transmitted_bits
            )
            
            # Compute binary cross entropy loss
            decoded_probs = decoder_output
            loss = torch.nn.functional.binary_cross_entropy(decoded_probs, transmitted_bits)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate BER and FER
            hard_bits = (decoded_probs > 0.5).float()
            ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_ber += ber
            epoch_fer += fer
            
            # Print batch progress occasionally
            if batch % 5 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch+1}/{batches_per_epoch} - Loss: {loss.item():.6f}, BER: {ber:.6f}, FER: {fer:.6f}")
        
        # Average metrics across batches
        epoch_loss /= batches_per_epoch
        epoch_ber /= batches_per_epoch
        epoch_fer /= batches_per_epoch
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.6f}, BER: {epoch_ber:.6f}, FER: {epoch_fer:.6f}")
        
        # Save history
        train_losses.append(epoch_loss)
        ber_history.append(epoch_ber)
        fer_history.append(epoch_fer)
    
    # Save the trained model
    torch.save(decoder.state_dict(), "saved_models/message_gnn_nr_2_0_4_bpsk.pth")
    print("Model saved to 'saved_models/message_gnn_nr_2_0_4_bpsk.pth'")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss (BPSK)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_nr_2_0_4_bpsk.png")
    
    # Evaluate the decoder on a range of SNR values
    print("\nEvaluating decoder over SNR range...")
    test_snr_range = np.arange(0.0, 8.1, 1.0)  # Updated evaluation range
    ber_results = []
    fer_results = []
    
    # Evaluation
    decoder.eval()
    eval_batch_size = 200  # Increased from 100 to 200
    num_eval_batches = 10
    
    for snr in test_snr_range:
        print(f"Evaluating at SNR = {snr} dB")
        batch_ber_sum = 0.0
        batch_fer_sum = 0.0
        
        for batch_idx in range(num_eval_batches):
            with torch.no_grad():
                # Generate evaluation data
                transmitted_bits = torch.zeros((eval_batch_size, variable_bit_length), device=device)
                
                # BPSK modulation
                bpsk_symbols = bpsk_modulate(transmitted_bits)
                
                # Add noise with fixed SNR
                noisy_symbols = awgn_channel(bpsk_symbols, snr)
                
                # Demodulate to get LLRs
                input_llrs = bpsk_demodulate(noisy_symbols, snr).to(device)
                
                # Forward pass through the decoder
                decoder_output = decoder(
                    input_llr=input_llrs,
                    message_to_var_mapping=message_to_var_mapping,
                    message_types=message_types,
                    var_to_check_adjacency=var_to_check_adjacency,
                    check_to_var_adjacency=check_to_var_adjacency
                )
                
                # Calculate BER and FER
                decoded_probs = decoder_output
                hard_bits = (decoded_probs > 0.5).float()
                ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
                
                batch_ber_sum += ber
                batch_fer_sum += fer
                
                # Print progress for the first batch
                if batch_idx == 0:
                    print(f"  Batch 1/{num_eval_batches} - BER: {ber:.6f}, FER: {fer:.6f}")
        
        # Calculate average BER and FER for this SNR
        avg_ber = batch_ber_sum / num_eval_batches
        avg_fer = batch_fer_sum / num_eval_batches
        
        ber_results.append(avg_ber)
        fer_results.append(avg_fer)
        
        print(f"SNR: {snr} dB, Avg BER: {avg_ber:.6f}, Avg FER: {avg_fer:.6f}")
    
    # Save evaluation results
    np.savez(
        "evaluation_results_nr_2_0_4_bpsk.npz",
        snr_range=test_snr_range,
        ber=ber_results,
        fer=fer_results
    )
    
    # Plot error rates
    plt.figure(figsize=(10, 6))
    plt.semilogy(test_snr_range, ber_results, "o-", label="BER")
    plt.semilogy(test_snr_range, fer_results, "s-", label="FER")
    plt.grid(True)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Error Rate")
    plt.legend()
    plt.title("Performance of Message GNN Decoder on NR_2_0_4 with BPSK")
    plt.savefig("error_rates_nr_2_0_4_bpsk.png")

if __name__ == "__main__":
    main() 