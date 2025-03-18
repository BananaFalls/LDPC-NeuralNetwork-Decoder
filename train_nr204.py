import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix, create_LLR_mapping
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer

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
    num_iterations = 15  # Increased from 10 to 15
    hidden_dim = 64      # Increased from 32 to 64
    num_of_residual_layers = 2
    
    # Training settings
    num_epochs = 50
    batch_size = 128
    learning_rate = 1e-4  # Reduced from 1e-3 to 1e-4
    momentum = 0.9
    weight_decay = 1e-5
    snr_range = [0.0, 8.0]  # Widened from [1.0, 5.0] to [0.0, 8.0]
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
    message_to_var_mapping = converter.get_message_to_var_mapping()
    message_types = converter.message_types if hasattr(converter, 'message_types') else None
    var_to_check_adjacency = converter.var_to_check_adjacency
    check_to_var_adjacency = converter.check_to_var_adjacency
    
    # Create directories for saving models if they do not exist
    os.makedirs("saved_models", exist_ok=True)
    
    # Train the decoder
    print("Starting training...")
    print(f"Using learning rate: {learning_rate}, SNR range: {snr_range}, Iterations: {num_iterations}")
    print(f"Hidden dimension: {hidden_dim}, Batches per epoch: {batches_per_epoch}")
    
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
        for batch_idx in range(batches_per_epoch):
            # Generate training data
            # Using zero codewords as per paper recommendation
            variable_bit_length = H.shape[1]
            transmitted_bits = torch.zeros((batch_size, variable_bit_length), device=device)
            
            # Generate random SNR values within the range
            snr_values = torch.FloatTensor(batch_size).uniform_(snr_range[0], snr_range[1]).to(device)
            
            # QPSK modulation
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Add noise with different SNR for each sample
            noisy_symbols = torch.zeros_like(qpsk_symbols, dtype=torch.complex64)
            for i in range(batch_size):
                # Pass through AWGN channel with this sample's SNR
                noisy_symbols[i] = awgn_channel(qpsk_symbols[i].unsqueeze(0), snr_values[i].item()).squeeze(0)
            
            # Get LLRs for each codeword with its corresponding SNR
            llrs_list = []
            for i in range(batch_size):
                llr = qpsk_demodulate(noisy_symbols[i].unsqueeze(0), snr_values[i].item()).squeeze(0)
                llrs_list.append(llr)
            
            # Stack LLRs into a batch
            input_llrs = torch.stack(llrs_list).to(device)
            
            # Forward pass with custom message GNN decoder
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
    torch.save(decoder.state_dict(), "saved_models/message_gnn_nr_2_0_4.pth")
    print("Model saved to 'saved_models/message_gnn_nr_2_0_4.pth'")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_nr_2_0_4.png")
    
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
        
        for _ in range(num_eval_batches):
            with torch.no_grad():
                # Generate evaluation data
                transmitted_bits = torch.zeros((eval_batch_size, variable_bit_length), device=device)
                
                # QPSK modulation
                qpsk_symbols = qpsk_modulate(transmitted_bits)
                
                # Add noise with fixed SNR
                noisy_symbols = awgn_channel(qpsk_symbols, snr)
                
                # Demodulate to get LLRs
                input_llrs = qpsk_demodulate(noisy_symbols, snr).to(device)
                
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
        
        # Calculate average BER and FER for this SNR
        avg_ber = batch_ber_sum / num_eval_batches
        avg_fer = batch_fer_sum / num_eval_batches
        
        ber_results.append(avg_ber)
        fer_results.append(avg_fer)
        
        print(f"SNR: {snr} dB, BER: {avg_ber:.6f}, FER: {avg_fer:.6f}")
    
    # Save evaluation results
    np.savez(
        "evaluation_results_nr_2_0_4.npz",
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
    plt.title("Performance of Message GNN Decoder on NR_2_0_4")
    plt.savefig("error_rates_nr_2_0_4.png")
    
    print("Evaluation complete. Results saved to 'evaluation_results_nr_2_0_4.npz'")
    return {
        "final_loss": train_losses[-1],
        "ber": ber_results,
        "fer": fer_results,
        "snr_range": test_snr_range
    }

if __name__ == "__main__":
    main() 