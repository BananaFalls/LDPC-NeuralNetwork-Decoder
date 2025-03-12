"""
Example script for GNN-based LDPC decoder.

This script demonstrates how to use the GNN-based LDPC decoder with weight sharing.
"""

import torch
import matplotlib.pyplot as plt
import os
import argparse

from ldpc_neural_decoder.models.gnn_ldpc_decoder import GNNLDPCDecoder, BaseGraphGNNDecoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, create_LLR_mapping, expand_base_matrix
from ldpc_neural_decoder.training.trainer import LDPCDecoderTrainer
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GNN-based LDPC decoder example')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
    # Model parameters
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--depth_L', type=int, default=2, help='Depth of residual connections')
    parser.add_argument('--num_edge_types', type=int, default=1, help='Number of edge types for weight sharing')
    
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
    parser.add_argument('--model_path', type=str, default='models/saved_models/gnn_model.pt',
                        help='Path to save/load model')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Mode parameters
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    
    return parser.parse_args()


def create_model(args, H):
    """Create GNN-based LDPC decoder model."""
    # Count number of 1s in H
    num_nodes = torch.sum(H == 1).item()
    
    # Create GNN-based LDPC decoder
    model = GNNLDPCDecoder(
        num_nodes=num_nodes,
        num_iterations=args.num_iterations,
        depth_L=args.depth_L,
        num_edge_types=args.num_edge_types
    )
    
    return model


def create_base_graph_model(args):
    """Create Base Graph GNN LDPC decoder model."""
    # Load base matrix
    base_matrix = load_base_matrix(args.base_matrix_path)
    
    # Create Base Graph GNN decoder
    model = BaseGraphGNNDecoder(
        base_graph=base_matrix,
        Z=args.lifting_factor,
        num_iterations=args.num_iterations,
        depth_L=args.depth_L
    )
    
    return model


def train(args):
    """Train the GNN-based LDPC decoder."""
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create or load parity-check matrix
    if args.base_matrix_path:
        base_matrix = load_base_matrix(args.base_matrix_path)
        H = expand_base_matrix(base_matrix, args.lifting_factor)
        
        # Create Base Graph GNN decoder
        model = create_base_graph_model(args)
    else:
        # Example small parity-check matrix
        H = torch.tensor([
            [1, 1, 0, 0],  # Check 1
            [0, 1, 1, 1],  # Check 2
            [1, 0, 0, 1]   # Check 3
        ], dtype=torch.float32)
        
        # Create GNN-based LDPC decoder
        model = create_model(args, H)
    
    # Create LLR mapping
    H_T = H.T
    _, check_index_tensor, var_index_tensor, output_index_tensor = create_LLR_mapping(H_T)
    
    # Create trainer
    trainer = LDPCDecoderTrainer(model, device=args.device)
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Train model
    print(f"Training GNN-based LDPC decoder with {args.num_epochs} epochs...")
    history = trainer.train(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        check_index_tensor=check_index_tensor,
        var_index_tensor=var_index_tensor,
        snr_range=snr_range,
        variable_bit_length=H.shape[1],
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    trainer.save_model(args.model_path)
    print(f"Model saved to {args.model_path}")
    
    # Plot training history
    fig1, fig2 = trainer.plot_training_history()
    
    # Save plots
    os.makedirs(args.results_dir, exist_ok=True)
    fig1.savefig(os.path.join(args.results_dir, 'gnn_training_loss.png'))
    if fig2:
        fig2.savefig(os.path.join(args.results_dir, 'gnn_error_rates.png'))
    
    print("Training completed.")


def evaluate(args):
    """Evaluate the GNN-based LDPC decoder."""
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create or load parity-check matrix
    if args.base_matrix_path:
        base_matrix = load_base_matrix(args.base_matrix_path)
        H = expand_base_matrix(base_matrix, args.lifting_factor)
        
        # Create Base Graph GNN decoder
        model = create_base_graph_model(args)
    else:
        # Example small parity-check matrix
        H = torch.tensor([
            [1, 1, 0, 0],  # Check 1
            [0, 1, 1, 1],  # Check 2
            [1, 0, 0, 1]   # Check 3
        ], dtype=torch.float32)
        
        # Create GNN-based LDPC decoder
        model = create_model(args, H)
    
    # Create LLR mapping
    H_T = H.T
    _, check_index_tensor, var_index_tensor, output_index_tensor = create_LLR_mapping(H_T)
    
    # Create trainer
    trainer = LDPCDecoderTrainer(model, device=args.device)
    
    # Load model
    trainer.load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Evaluate model
    print(f"Evaluating GNN-based LDPC decoder over SNR range {snr_range}...")
    ber_results, fer_results = trainer.evaluate_snr_range(
        snr_range=snr_range,
        batch_size=args.batch_size,
        num_trials=args.num_trials,
        check_index_tensor=check_index_tensor,
        var_index_tensor=var_index_tensor,
        variable_bit_length=H.shape[1]
    )
    
    # Plot results
    fig1, fig2 = trainer.plot_snr_performance(snr_range, ber_results, fer_results)
    
    # Save plots
    os.makedirs(args.results_dir, exist_ok=True)
    fig1.savefig(os.path.join(args.results_dir, 'gnn_ber_vs_snr.png'))
    fig2.savefig(os.path.join(args.results_dir, 'gnn_fer_vs_snr.png'))
    
    # Save results
    results = {
        'snr_range': snr_range,
        'ber_results': ber_results,
        'fer_results': fer_results
    }
    torch.save(results, os.path.join(args.results_dir, 'gnn_evaluation_results.pt'))
    
    print("Evaluation completed.")


def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == '__main__':
    main() 