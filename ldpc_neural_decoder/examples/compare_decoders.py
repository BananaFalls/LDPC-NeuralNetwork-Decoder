"""
Comparison script for traditional and message-centered GNN LDPC decoders.

This script compares the performance of traditional decoders (Belief Propagation
and Min-Sum) with the message-centered GNN LDPC decoder using a small base graph.
"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

from ldpc_neural_decoder.models.message_gnn_decoder import create_message_gnn_decoder
from ldpc_neural_decoder.models.traditional_decoders import BeliefPropagationDecoder, MinSumScaledDecoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare traditional and message-centered GNN LDPC decoders')
    
    # Device and random seed
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'small_base_graph.txt'),
                        help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=4, help='Lifting factor')
    
    # Decoder parameters
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension for GNN decoder')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for SGD optimizer')
    
    # Evaluation parameters
    parser.add_argument('--snr_min', type=float, default=0.0, help='Minimum SNR for evaluation')
    parser.add_argument('--snr_max', type=float, default=6.0, help='Maximum SNR for evaluation')
    parser.add_argument('--snr_step', type=float, default=1.0, help='SNR step for evaluation')
    parser.add_argument('--num_trials', type=int, default=1000, help='Number of trials per SNR')
    
    # I/O parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/message_gnn_model.pt',
                        help='Path to save/load model')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    
    # Mode parameters
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use existing model')
    
    return parser.parse_args()


def train_message_gnn_decoder(args, H, base_matrix):
    """Train the message-centered GNN LDPC decoder."""
    print("\nTraining message-centered GNN LDPC decoder...")
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create message-centered GNN decoder
    device = torch.device(args.device)
    decoder, converter = create_message_gnn_decoder(
        H.to(device),
        num_iterations=args.num_iterations,
        hidden_dim=args.hidden_dim,
        base_graph=base_matrix.to(device),
        Z=args.lifting_factor
    )
    
    print(f"Decoder created. Number of parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        decoder.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Training parameters
    snr_db = 2.0  # Fixed SNR for training
    
    print(f"Training for {args.num_epochs} epochs with batch size {args.batch_size} at SNR {snr_db} dB...")
    
    # Training loop
    train_losses = []
    ber_history = []
    fer_history = []
    
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        decoder.train()
        
        # Generate random bits
        transmitted_bits = torch.randint(0, 2, (args.batch_size, converter.num_variables), device=device).float()
        
        # Modulate using QPSK
        qpsk_symbols = qpsk_modulate(transmitted_bits)
        
        # Pass through AWGN channel
        received_signal = awgn_channel(qpsk_symbols, snr_db)
        
        # Demodulate to LLRs
        llrs = qpsk_demodulate(received_signal, snr_db)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        soft_bits, loss = decoder(
            llrs,
            converter.message_to_var_mapping,
            message_types=converter.get_message_types(base_matrix, args.lifting_factor),
            var_to_check_adjacency=converter.var_to_check_adjacency,
            check_to_var_adjacency=converter.check_to_var_adjacency,
            ground_truth=transmitted_bits
        )
        
        # Compute mean loss
        batch_loss = loss.mean()
        
        # Backward pass
        batch_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Compute BER and FER
        hard_bits = (soft_bits > 0.5).float()
        ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
        
        # Store history
        train_losses.append(batch_loss.item())
        ber_history.append(ber)
        fer_history.append(fer)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {batch_loss.item():.6f}, BER: {ber:.6f}, FER: {fer:.6f}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds.")
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save({
        'decoder_state_dict': decoder.state_dict(),
        'converter': converter,
        'args': args
    }, args.model_path)
    print(f"Model saved to {args.model_path}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    os.makedirs(args.results_dir, exist_ok=True)
    plt.savefig(os.path.join(args.results_dir, 'message_gnn_training_loss.png'))
    
    return decoder, converter


def load_message_gnn_decoder(args, H, base_matrix):
    """Load a trained message-centered GNN LDPC decoder."""
    print(f"\nLoading message-centered GNN LDPC decoder from {args.model_path}...")
    
    device = torch.device(args.device)
    
    # Create decoder and converter
    decoder, converter = create_message_gnn_decoder(
        H.to(device),
        num_iterations=args.num_iterations,
        hidden_dim=args.hidden_dim,
        base_graph=base_matrix.to(device),
        Z=args.lifting_factor
    )
    
    # Load saved state
    checkpoint = torch.load(args.model_path, map_location=device)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print(f"Decoder loaded. Number of parameters: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")
    
    return decoder, converter


def evaluate_decoders(args, H, base_matrix):
    """Evaluate and compare all decoders."""
    print("\nEvaluating decoders...")
    
    device = torch.device(args.device)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create traditional decoders
    bp_decoder = BeliefPropagationDecoder(H.to(device), args.num_iterations)
    ms_decoder = MinSumScaledDecoder(H.to(device), args.num_iterations, alpha=0.8)
    
    # Load or train message-centered GNN decoder
    if args.skip_training and os.path.exists(args.model_path):
        message_gnn_decoder, converter = load_message_gnn_decoder(args, H, base_matrix)
    else:
        message_gnn_decoder, converter = train_message_gnn_decoder(args, H, base_matrix)
    
    # Set evaluation mode
    message_gnn_decoder.eval()
    
    # SNR range for evaluation
    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
    
    # Results storage
    bp_ber = []
    bp_fer = []
    ms_ber = []
    ms_fer = []
    gnn_ber = []
    gnn_fer = []
    
    # Timing storage
    bp_times = []
    ms_times = []
    gnn_times = []
    
    for snr_db in snr_range:
        print(f"\nEvaluating at SNR = {snr_db} dB...")
        
        # BP decoder results
        bp_ber_trials = []
        bp_fer_trials = []
        bp_time_trials = []
        
        # MS decoder results
        ms_ber_trials = []
        ms_fer_trials = []
        ms_time_trials = []
        
        # GNN decoder results
        gnn_ber_trials = []
        gnn_fer_trials = []
        gnn_time_trials = []
        
        for trial in range(args.num_trials):
            # Generate all-zero codeword (due to linearity of the code)
            transmitted_bits = torch.zeros((1, H.shape[1]), device=device)
            
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Pass through AWGN channel
            received_signal = awgn_channel(qpsk_symbols, snr_db)
            
            # Demodulate to LLRs
            llrs = qpsk_demodulate(received_signal, snr_db)
            
            # Decode using BP decoder
            start_time = time.time()
            bp_decoded = bp_decoder.decode(llrs)
            bp_time = time.time() - start_time
            bp_ber_trial, bp_fer_trial = compute_ber_fer(transmitted_bits, bp_decoded)
            bp_ber_trials.append(bp_ber_trial)
            bp_fer_trials.append(bp_fer_trial)
            bp_time_trials.append(bp_time)
            
            # Decode using MS decoder
            start_time = time.time()
            ms_decoded = ms_decoder.decode(llrs)
            ms_time = time.time() - start_time
            ms_ber_trial, ms_fer_trial = compute_ber_fer(transmitted_bits, ms_decoded)
            ms_ber_trials.append(ms_ber_trial)
            ms_fer_trials.append(ms_fer_trial)
            ms_time_trials.append(ms_time)
            
            # Decode using message-centered GNN decoder
            start_time = time.time()
            with torch.no_grad():
                gnn_decoded = message_gnn_decoder.decode(
                    llrs,
                    converter.message_to_var_mapping,
                    message_types=converter.get_message_types(base_matrix, args.lifting_factor),
                    var_to_check_adjacency=converter.var_to_check_adjacency,
                    check_to_var_adjacency=converter.check_to_var_adjacency
                )
            gnn_time = time.time() - start_time
            gnn_ber_trial, gnn_fer_trial = compute_ber_fer(transmitted_bits, gnn_decoded)
            gnn_ber_trials.append(gnn_ber_trial)
            gnn_fer_trials.append(gnn_fer_trial)
            gnn_time_trials.append(gnn_time)
            
            if (trial + 1) % 100 == 0:
                print(f"  Completed {trial + 1}/{args.num_trials} trials")
        
        # Compute average results
        bp_ber.append(np.mean(bp_ber_trials))
        bp_fer.append(np.mean(bp_fer_trials))
        bp_times.append(np.mean(bp_time_trials))
        
        ms_ber.append(np.mean(ms_ber_trials))
        ms_fer.append(np.mean(ms_fer_trials))
        ms_times.append(np.mean(ms_time_trials))
        
        gnn_ber.append(np.mean(gnn_ber_trials))
        gnn_fer.append(np.mean(gnn_fer_trials))
        gnn_times.append(np.mean(gnn_time_trials))
        
        print(f"  BP Decoder - BER: {bp_ber[-1]:.6f}, FER: {bp_fer[-1]:.6f}, Avg Time: {bp_times[-1]*1000:.2f} ms")
        print(f"  MS Decoder - BER: {ms_ber[-1]:.6f}, FER: {ms_fer[-1]:.6f}, Avg Time: {ms_times[-1]*1000:.2f} ms")
        print(f"  GNN Decoder - BER: {gnn_ber[-1]:.6f}, FER: {gnn_fer[-1]:.6f}, Avg Time: {gnn_times[-1]*1000:.2f} ms")
    
    # Save results
    os.makedirs(args.results_dir, exist_ok=True)
    results = {
        'snr_range': snr_range,
        'bp_ber': bp_ber,
        'bp_fer': bp_fer,
        'bp_times': bp_times,
        'ms_ber': ms_ber,
        'ms_fer': ms_fer,
        'ms_times': ms_times,
        'gnn_ber': gnn_ber,
        'gnn_fer': gnn_fer,
        'gnn_times': gnn_times
    }
    np.save(os.path.join(args.results_dir, 'decoder_comparison_results.npy'), results)
    
    # Plot BER results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, bp_ber, 'o-', label='BP Decoder')
    plt.semilogy(snr_range, ms_ber, 's-', label='MS Decoder')
    plt.semilogy(snr_range, gnn_ber, '^-', label='Message-GNN Decoder')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Comparison of LDPC Decoders')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'ber_comparison.png'))
    
    # Plot FER results
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, bp_fer, 'o-', label='BP Decoder')
    plt.semilogy(snr_range, ms_fer, 's-', label='MS Decoder')
    plt.semilogy(snr_range, gnn_fer, '^-', label='Message-GNN Decoder')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.title('FER Comparison of LDPC Decoders')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'fer_comparison.png'))
    
    # Plot decoding time
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, [t*1000 for t in bp_times], 'o-', label='BP Decoder')
    plt.plot(snr_range, [t*1000 for t in ms_times], 's-', label='MS Decoder')
    plt.plot(snr_range, [t*1000 for t in gnn_times], '^-', label='Message-GNN Decoder')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Decoding Time (ms)')
    plt.title('Decoding Time Comparison')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(args.results_dir, 'time_comparison.png'))
    
    print(f"\nResults and plots saved to {args.results_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    print("Starting decoder comparison...")
    print(f"Using device: {args.device}")
    print(f"Base matrix path: {args.base_matrix_path}")
    print(f"Lifting factor: {args.lifting_factor}")
    
    # Load base matrix
    base_matrix = load_base_matrix(args.base_matrix_path)
    print(f"Base matrix shape: {base_matrix.shape}")
    
    # Expand base matrix to parity-check matrix
    H = expand_base_matrix(base_matrix, args.lifting_factor)
    print(f"Parity-check matrix shape: {H.shape}")
    
    # Evaluate decoders
    evaluate_decoders(args, H, base_matrix)
    
    print("\nDecoder comparison completed successfully!")


if __name__ == '__main__':
    main() 