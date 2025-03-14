import torch
import numpy as np
import matplotlib.pyplot as plt
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import AWGNChannel
from ldpc_neural_decoder.models.message_gnn_decoder import create_custom_minsum_message_gnn_decoder
from ldpc_neural_decoder.utils.training import train_decoder, evaluate_decoder

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load a small base matrix for demonstration
    base_matrix = load_base_matrix("ldpc_neural_decoder/data/base_matrices/ieee80211/N648_R12.alist")
    
    # Expand the base matrix with a lifting factor Z=4 for a small example
    H = expand_base_matrix(base_matrix, Z=4)
    print(f"Parity-check matrix shape: {H.shape}")
    
    # Create the custom min-sum decoder and converter
    num_iterations = 5
    hidden_dim = 8
    depth = 2
    dropout = 0.1
    
    decoder, converter = create_custom_minsum_message_gnn_decoder(
        H, num_iterations, hidden_dim, depth, dropout
    )
    
    # Print decoder information
    print(f"Decoder created with {num_iterations} iterations")
    print(f"Number of messages: {decoder.num_messages}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # Create AWGN channel
    channel = AWGNChannel()
    
    # Generate training data
    batch_size = 32
    num_batches = 10
    snr_db = 2.0  # Signal-to-noise ratio in dB
    
    # Convert the parity-check matrix to adjacency matrices
    variable_adjacency, check_adjacency, message_types, variable_to_message_mapping = converter.convert(H)
    
    # Training parameters
    num_epochs = 3
    learning_rate = 0.001
    
    # Train the decoder
    print("Training the decoder...")
    train_decoder(
        decoder,
        H,
        channel,
        variable_adjacency,
        check_adjacency,
        message_types,
        variable_to_message_mapping,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_batches=num_batches,
        snr_db=snr_db,
        learning_rate=learning_rate
    )
    
    # Evaluate the decoder at different SNRs
    print("Evaluating the decoder...")
    snr_range = np.arange(0, 6, 1.0)
    ber_values = []
    fer_values = []
    
    for snr in snr_range:
        ber, fer = evaluate_decoder(
            decoder,
            H,
            channel,
            variable_adjacency,
            check_adjacency,
            message_types,
            variable_to_message_mapping,
            batch_size=100,
            snr_db=snr
        )
        ber_values.append(ber)
        fer_values.append(fer)
        print(f"SNR: {snr} dB, BER: {ber:.6f}, FER: {fer:.6f}")
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_values, 'o-', label='BER')
    plt.semilogy(snr_range, fer_values, 's-', label='FER')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Error Rate')
    plt.grid(True)
    plt.legend()
    plt.title('Performance of Custom Min-Sum Message GNN Decoder')
    plt.savefig('custom_minsum_performance.png')
    plt.show()
    
    # Print the learned scaling factor
    print(f"Learned scaling factor (alpha): {decoder.gnn_layers[0].alpha.item():.4f}")
    
    # Save the trained model
    torch.save(decoder.state_dict(), 'custom_minsum_decoder.pt')
    print("Model saved to 'custom_minsum_decoder.pt'")

if __name__ == "__main__":
    main() 