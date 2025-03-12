"""
Example script for Message-Centered GNN LDPC decoder.

This script demonstrates how to use the Message-Centered GNN LDPC decoder
where messages are represented as nodes in the GNN, and edges connect
messages that share the same variable or check node in the Tanner graph.
"""

import torch
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

from ldpc_neural_decoder.models.message_gnn_decoder import (
    MessageGNNDecoder, 
    TannerToMessageGraph,
    create_message_gnn_decoder
)
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Message-Centered GNN LDPC decoder example')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
    # Model parameters
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Dimension of hidden representations')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty) for SGD optimizer')
    parser.add_argument('--snr_min', type=int, default=-2, help='Minimum SNR for training')
    parser.add_argument('--snr_max', type=int, default=6, help='Maximum SNR for training')
    parser.add_argument('--snr_step', type=int, default=2, help='SNR step for training')
    
    # Evaluation parameters
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per SNR')
    
    # I/O parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/message_gnn_model.pt',
                        help='Path to save/load model')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    
    return parser.parse_args()


def create_simple_parity_check_matrix():
    """Create a simple parity-check matrix for demonstration."""
    # Simple (3,4)-regular LDPC code
    H = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1]
    ], dtype=torch.float32)
    
    return H


def train_step(model, converter, optimizer, batch_size, snr_db, device):
    """Perform a single training step."""
    # Generate random bits
    transmitted_bits = torch.randint(0, 2, (batch_size, converter.num_variables), device=device).float()
    
    # Modulate using QPSK
    qpsk_symbols = qpsk_modulate(transmitted_bits)
    
    # Pass through AWGN channel
    received_signal = awgn_channel(qpsk_symbols, snr_db)
    
    # Demodulate to LLRs
    llrs = qpsk_demodulate(received_signal, snr_db)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    soft_bits, loss = model(
        llrs,
        converter.message_to_var_mapping,
        message_types=converter.get_message_types(),
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
    
    return batch_loss.item(), ber, fer


def evaluate(model, converter, snr_range, batch_size, num_trials, device):
    """Evaluate the model over a range of SNR values."""
    model.eval()
    ber_results = []
    fer_results = []
    
    for snr_db in snr_range:
        total_ber = 0.0
        total_fer = 0.0
        
        for _ in range(num_trials):
            # Generate all-zero codeword
            transmitted_bits = torch.zeros((batch_size, converter.num_variables), device=device)
            
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Pass through AWGN channel
            received_signal = awgn_channel(qpsk_symbols, snr_db)
            
            # Demodulate to LLRs
            llrs = qpsk_demodulate(received_signal, snr_db)
            
            # Decode
            with torch.no_grad():
                hard_bits = model.decode(
                    llrs,
                    converter.message_to_var_mapping,
                    message_types=converter.get_message_types(),
                    var_to_check_adjacency=converter.var_to_check_adjacency,
                    check_to_var_adjacency=converter.check_to_var_adjacency
                )
            
            # Compute BER and FER
            ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
            
            # Accumulate metrics
            total_ber += ber
            total_fer += fer
        
        # Compute averages
        avg_ber = total_ber / num_trials
        avg_fer = total_fer / num_trials
        
        # Store results
        ber_results.append(avg_ber)
        fer_results.append(avg_fer)
        
        print(f"SNR: {snr_db} dB, BER: {avg_ber:.6f}, FER: {avg_fer:.6f}")
    
    return ber_results, fer_results


def train(args):
    """Train the Message-Centered GNN LDPC decoder."""
    # Set random seed
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Create or load parity-check matrix
    if args.base_matrix_path:
        base_matrix = load_base_matrix(args.base_matrix_path)
        H = expand_base_matrix(base_matrix, args.lifting_factor)
        
        # Create decoder with base graph information
        decoder, converter = create_message_gnn_decoder(
            H.to(device),
            num_iterations=args.num_iterations,
            hidden_dim=args.hidden_dim,
            base_graph=base_matrix.to(device),
            Z=args.lifting_factor
        )
    else:
        # Use a simple parity-check matrix
        H = create_simple_parity_check_matrix().to(device)
        
        # Create decoder
        decoder, converter = create_message_gnn_decoder(
            H,
            num_iterations=args.num_iterations,
            hidden_dim=args.hidden_dim
        )
    
    # Create optimizer
    optimizer = torch.optim.SGD(
        decoder.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Training loop
    print(f"Training Message-Centered GNN LDPC decoder with {args.num_epochs} epochs...")
    
    train_losses = []
    ber_history = []
    fer_history = []
    
    for epoch in range(args.num_epochs):
        decoder.train()
        epoch_loss = 0.0
        epoch_ber = 0.0
        epoch_fer = 0.0
        num_batches = 0
        
        # Train on different SNR values
        for snr_db in snr_range:
            batch_loss, batch_ber, batch_fer = train_step(
                decoder, converter, optimizer, args.batch_size, snr_db, device
            )
            
            # Accumulate metrics
            epoch_loss += batch_loss
            epoch_ber += batch_ber
            epoch_fer += batch_fer
            num_batches += 1
        
        # Compute averages
        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_ber = epoch_ber / num_batches
        avg_epoch_fer = epoch_fer / num_batches
        
        # Store history
        train_losses.append(avg_epoch_loss)
        ber_history.append(avg_epoch_ber)
        fer_history.append(avg_epoch_fer)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_epoch_loss:.6f}, BER: {avg_epoch_ber:.6f}, FER: {avg_epoch_fer:.6f}")
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'train_losses': train_losses,
        'ber_history': ber_history,
        'fer_history': fer_history,
        'H': H,
        'num_iterations': args.num_iterations,
        'hidden_dim': args.hidden_dim
    }, args.model_path)
    print(f"Model saved to {args.model_path}")
    
    # Plot training history
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(args.results_dir, 'message_gnn_training_loss.png'))
    
    # Plot BER and FER
    plt.figure(figsize=(10, 6))
    plt.semilogy(ber_history, 'o-', label='BER')
    plt.semilogy(fer_history, 's-', label='FER')
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('BER and FER History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.results_dir, 'message_gnn_error_rates.png'))
    
    print("Training completed.")


def evaluate_model(args):
    """Evaluate the Message-Centered GNN LDPC decoder."""
    # Set random seed
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    H = checkpoint['H'].to(device)
    
    # Create decoder
    decoder, converter = create_message_gnn_decoder(
        H,
        num_iterations=checkpoint['num_iterations'],
        hidden_dim=checkpoint['hidden_dim']
    )
    
    # Load model weights
    decoder.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Evaluate model
    print(f"Evaluating Message-Centered GNN LDPC decoder over SNR range {snr_range}...")
    ber_results, fer_results = evaluate(
        decoder, converter, snr_range, args.batch_size, args.num_trials, device
    )
    
    # Plot results
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Plot BER vs SNR
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_results, 'o-', label='Message GNN')
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Bit Error Rate vs SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.results_dir, 'message_gnn_ber_vs_snr.png'))
    
    # Plot FER vs SNR
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, fer_results, 'o-', label='Message GNN')
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.title('Frame Error Rate vs SNR')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.results_dir, 'message_gnn_fer_vs_snr.png'))
    
    # Save results
    results = {
        'snr_range': snr_range,
        'ber_results': ber_results,
        'fer_results': fer_results
    }
    torch.save(results, os.path.join(args.results_dir, 'message_gnn_evaluation_results.pt'))
    
    print("Evaluation completed.")


def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    main() 