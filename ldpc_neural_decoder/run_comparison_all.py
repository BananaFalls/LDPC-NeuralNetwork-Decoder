"""
Script to compare the performance of all decoder types:
- Standard Neural LDPC Decoder
- GNN-based LDPC Decoder
- Message-centered GNN LDPC Decoder
- Belief Propagation Decoder
- Min-Sum Scaled Decoder

This script loads pre-trained models for each neural decoder type and
compares their performance in terms of BER, FER, and parameter count.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from ldpc_neural_decoder.models.decoder import LDPCNeuralDecoder, TiedNeuralLDPCDecoder
from ldpc_neural_decoder.models.gnn_ldpc_decoder import GNNLDPCDecoder, BaseGraphGNNDecoder
from ldpc_neural_decoder.models.message_gnn_decoder import MessageGNNDecoder, create_message_gnn_decoder
from ldpc_neural_decoder.models.traditional_decoders import BeliefPropagationDecoder, MinSumScaledDecoder
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Compare all LDPC decoder types')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
    # Evaluation parameters
    parser.add_argument('--snr_min', type=int, default=-2, help='Minimum SNR for evaluation')
    parser.add_argument('--snr_max', type=int, default=6, help='Maximum SNR for evaluation')
    parser.add_argument('--snr_step', type=int, default=1, help='SNR step for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per SNR')
    
    # Model paths
    parser.add_argument('--standard_model_path', type=str, 
                        default='models/saved_models/standard_model.pt',
                        help='Path to standard neural decoder model')
    parser.add_argument('--gnn_model_path', type=str, 
                        default='models/saved_models/gnn_model.pt',
                        help='Path to GNN-based decoder model')
    parser.add_argument('--message_gnn_model_path', type=str, 
                        default='models/saved_models/message_gnn_model.pt',
                        help='Path to message-centered GNN decoder model')
    
    # Traditional decoder parameters
    parser.add_argument('--bp_max_iterations', type=int, default=20,
                        help='Maximum iterations for belief propagation decoder')
    parser.add_argument('--ms_scaling_factor', type=float, default=0.8,
                        help='Scaling factor for min-sum decoder')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--paper_style', action='store_true',
                        help='Use paper-style formatting for plots')
    
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


def load_standard_model(model_path, H, device):
    """Load the standard neural LDPC decoder model."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        num_iterations = checkpoint.get('num_iterations', 5)
        depth_L = checkpoint.get('depth_L', 2)
        
        model = LDPCNeuralDecoder(H, num_iterations=num_iterations, depth_L=depth_L).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Standard neural decoder loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading standard model: {e}")
        print("Creating a new standard neural decoder with default parameters")
        return LDPCNeuralDecoder(H, num_iterations=5, depth_L=2).to(device)


def load_gnn_model(model_path, H, device):
    """Load the GNN-based LDPC decoder model."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        num_iterations = checkpoint.get('num_iterations', 5)
        depth_L = checkpoint.get('depth_L', 2)
        num_edge_types = checkpoint.get('num_edge_types', 1)
        
        model = GNNLDPCDecoder(H, num_iterations=num_iterations, depth_L=depth_L, 
                               num_edge_types=num_edge_types).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"GNN-based decoder loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading GNN model: {e}")
        print("Creating a new GNN-based decoder with default parameters")
        return GNNLDPCDecoder(H, num_iterations=5, depth_L=2, num_edge_types=1).to(device)


def load_message_gnn_model(model_path, H, device):
    """Load the message-centered GNN LDPC decoder model."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        num_iterations = checkpoint.get('num_iterations', 5)
        hidden_dim = checkpoint.get('hidden_dim', 64)
        
        decoder, converter = create_message_gnn_decoder(
            H,
            num_iterations=num_iterations,
            hidden_dim=hidden_dim
        )
        decoder = decoder.to(device)
        decoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Message-centered GNN decoder loaded from {model_path}")
        return decoder, converter
    except Exception as e:
        print(f"Error loading message GNN model: {e}")
        print("Creating a new message-centered GNN decoder with default parameters")
        return create_message_gnn_decoder(H, num_iterations=5, hidden_dim=64)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_standard_neural(model, H, snr_range, batch_size, num_trials, device):
    """Evaluate the standard neural LDPC decoder."""
    model.eval()
    ber_results = []
    fer_results = []
    
    for snr_db in tqdm(snr_range, desc="Evaluating standard neural decoder"):
        total_ber = 0.0
        total_fer = 0.0
        
        for _ in range(num_trials):
            # Generate all-zero codeword
            transmitted_bits = torch.zeros((batch_size, H.shape[1]), device=device)
            
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Pass through AWGN channel
            received_signal = awgn_channel(qpsk_symbols, snr_db)
            
            # Demodulate to LLRs
            llrs = qpsk_demodulate(received_signal, snr_db)
            
            # Decode
            with torch.no_grad():
                hard_bits = model.decode(llrs)
            
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


def evaluate_gnn(model, H, snr_range, batch_size, num_trials, device):
    """Evaluate the GNN-based LDPC decoder."""
    model.eval()
    ber_results = []
    fer_results = []
    
    for snr_db in tqdm(snr_range, desc="Evaluating GNN-based decoder"):
        total_ber = 0.0
        total_fer = 0.0
        
        for _ in range(num_trials):
            # Generate all-zero codeword
            transmitted_bits = torch.zeros((batch_size, H.shape[1]), device=device)
            
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Pass through AWGN channel
            received_signal = awgn_channel(qpsk_symbols, snr_db)
            
            # Demodulate to LLRs
            llrs = qpsk_demodulate(received_signal, snr_db)
            
            # Decode
            with torch.no_grad():
                hard_bits = model.decode(llrs)
            
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


def evaluate_message_gnn(decoder, converter, snr_range, batch_size, num_trials, device):
    """Evaluate the message-centered GNN LDPC decoder."""
    decoder.eval()
    ber_results = []
    fer_results = []
    
    for snr_db in tqdm(snr_range, desc="Evaluating message-centered GNN decoder"):
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
                hard_bits = decoder.decode(
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


def evaluate_traditional(H, snr_range, batch_size, num_trials, device, bp_max_iterations, ms_scaling_factor):
    """Evaluate traditional LDPC decoders."""
    bp_decoder = BeliefPropagationDecoder(H, max_iterations=bp_max_iterations).to(device)
    ms_decoder = MinSumScaledDecoder(H, max_iterations=bp_max_iterations, scaling_factor=ms_scaling_factor).to(device)
    
    bp_ber_results = []
    bp_fer_results = []
    bp_iterations_results = []
    
    ms_ber_results = []
    ms_fer_results = []
    ms_iterations_results = []
    
    for snr_db in tqdm(snr_range, desc="Evaluating traditional decoders"):
        bp_total_ber = 0.0
        bp_total_fer = 0.0
        bp_total_iterations = 0.0
        
        ms_total_ber = 0.0
        ms_total_fer = 0.0
        ms_total_iterations = 0.0
        
        for _ in range(num_trials):
            # Generate all-zero codeword
            transmitted_bits = torch.zeros((batch_size, H.shape[1]), device=device)
            
            # Modulate using QPSK
            qpsk_symbols = qpsk_modulate(transmitted_bits)
            
            # Pass through AWGN channel
            received_signal = awgn_channel(qpsk_symbols, snr_db)
            
            # Demodulate to LLRs
            llrs = qpsk_demodulate(received_signal, snr_db)
            
            # Decode with BP
            with torch.no_grad():
                bp_hard_bits, bp_iterations = bp_decoder.decode_with_iterations(llrs)
            
            # Decode with MS
            with torch.no_grad():
                ms_hard_bits, ms_iterations = ms_decoder.decode_with_iterations(llrs)
            
            # Compute BER and FER for BP
            bp_ber, bp_fer = compute_ber_fer(transmitted_bits, bp_hard_bits)
            
            # Compute BER and FER for MS
            ms_ber, ms_fer = compute_ber_fer(transmitted_bits, ms_hard_bits)
            
            # Accumulate metrics
            bp_total_ber += bp_ber
            bp_total_fer += bp_fer
            bp_total_iterations += bp_iterations.float().mean().item()
            
            ms_total_ber += ms_ber
            ms_total_fer += ms_fer
            ms_total_iterations += ms_iterations.float().mean().item()
        
        # Compute averages for BP
        bp_avg_ber = bp_total_ber / num_trials
        bp_avg_fer = bp_total_fer / num_trials
        bp_avg_iterations = bp_total_iterations / num_trials
        
        # Compute averages for MS
        ms_avg_ber = ms_total_ber / num_trials
        ms_avg_fer = ms_total_fer / num_trials
        ms_avg_iterations = ms_total_iterations / num_trials
        
        # Store results for BP
        bp_ber_results.append(bp_avg_ber)
        bp_fer_results.append(bp_avg_fer)
        bp_iterations_results.append(bp_avg_iterations)
        
        # Store results for MS
        ms_ber_results.append(ms_avg_ber)
        ms_fer_results.append(ms_avg_fer)
        ms_iterations_results.append(ms_avg_iterations)
        
        print(f"SNR: {snr_db} dB, BP BER: {bp_avg_ber:.6f}, BP FER: {bp_avg_fer:.6f}, BP Iterations: {bp_avg_iterations:.2f}")
        print(f"SNR: {snr_db} dB, MS BER: {ms_avg_ber:.6f}, MS FER: {ms_avg_fer:.6f}, MS Iterations: {ms_avg_iterations:.2f}")
    
    return {
        'bp_ber': bp_ber_results,
        'bp_fer': bp_fer_results,
        'bp_iterations': bp_iterations_results,
        'ms_ber': ms_ber_results,
        'ms_fer': ms_fer_results,
        'ms_iterations': ms_iterations_results
    }


def plot_ber_comparison(snr_range, standard_ber, gnn_ber, message_gnn_ber, trad_results, 
                        output_dir, paper_style=False):
    """Plot BER comparison for all decoders."""
    plt.figure(figsize=(10, 6))
    
    if paper_style:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': True,
            'lines.linewidth': 2,
            'axes.grid': True
        })
    
    plt.semilogy(snr_range, standard_ber, 'o-', label='Standard Neural Decoder')
    plt.semilogy(snr_range, gnn_ber, 's-', label='GNN-based Decoder')
    plt.semilogy(snr_range, message_gnn_ber, '^-', label='Message-centered GNN')
    plt.semilogy(snr_range, trad_results['bp_ber'], 'x-', label='Belief Propagation')
    plt.semilogy(snr_range, trad_results['ms_ber'], 'd-', label='Min-Sum Scaled')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Bit Error Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'ber_comparison_all.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'ber_comparison_all.pdf'))
    plt.close()


def plot_fer_comparison(snr_range, standard_fer, gnn_fer, message_gnn_fer, trad_results, 
                        output_dir, paper_style=False):
    """Plot FER comparison for all decoders."""
    plt.figure(figsize=(10, 6))
    
    if paper_style:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': True,
            'lines.linewidth': 2,
            'axes.grid': True
        })
    
    plt.semilogy(snr_range, standard_fer, 'o-', label='Standard Neural Decoder')
    plt.semilogy(snr_range, gnn_fer, 's-', label='GNN-based Decoder')
    plt.semilogy(snr_range, message_gnn_fer, '^-', label='Message-centered GNN')
    plt.semilogy(snr_range, trad_results['bp_fer'], 'x-', label='Belief Propagation')
    plt.semilogy(snr_range, trad_results['ms_fer'], 'd-', label='Min-Sum Scaled')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('FER')
    plt.title('Frame Error Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'fer_comparison_all.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fer_comparison_all.pdf'))
    plt.close()


def plot_iterations_comparison(snr_range, trad_results, output_dir, paper_style=False):
    """Plot iterations comparison for traditional decoders."""
    plt.figure(figsize=(10, 6))
    
    if paper_style:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': True,
            'lines.linewidth': 2,
            'axes.grid': True
        })
    
    plt.plot(snr_range, trad_results['bp_iterations'], 'x-', label='Belief Propagation')
    plt.plot(snr_range, trad_results['ms_iterations'], 'd-', label='Min-Sum Scaled')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Number of Iterations')
    plt.title('Average Iterations Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(output_dir, 'iterations_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'iterations_comparison.pdf'))
    plt.close()


def plot_parameter_comparison(param_counts, output_dir, paper_style=False):
    """Plot parameter count comparison for all neural decoders."""
    plt.figure(figsize=(10, 6))
    
    if paper_style:
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'text.usetex': True,
            'axes.grid': True
        })
    
    models = list(param_counts.keys())
    counts = list(param_counts.values())
    
    plt.bar(models, counts)
    
    plt.xlabel('Model Type')
    plt.ylabel('Number of Parameters')
    plt.title('Parameter Count Comparison')
    plt.yscale('log')
    plt.grid(True, axis='y')
    
    # Add parameter count labels on top of bars
    for i, count in enumerate(counts):
        plt.text(i, count * 1.1, f'{count:,}', ha='center')
    
    plt.savefig(os.path.join(output_dir, 'parameter_comparison.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'parameter_comparison.pdf'))
    plt.close()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    
    # Create or load parity-check matrix
    if args.base_matrix_path:
        base_matrix = load_base_matrix(args.base_matrix_path)
        H = expand_base_matrix(base_matrix, args.lifting_factor)
    else:
        # Use a simple parity-check matrix
        H = create_simple_parity_check_matrix()
    
    H = H.to(device)
    
    # Load models
    standard_model = load_standard_model(args.standard_model_path, H, device)
    gnn_model = load_gnn_model(args.gnn_model_path, H, device)
    message_gnn_model, converter = load_message_gnn_model(args.message_gnn_model_path, H, device)
    
    # Count parameters
    standard_params = count_parameters(standard_model)
    gnn_params = count_parameters(gnn_model)
    message_gnn_params = count_parameters(message_gnn_model)
    
    param_counts = {
        'Standard Neural': standard_params,
        'GNN-based': gnn_params,
        'Message-centered GNN': message_gnn_params
    }
    
    print("\nParameter counts:")
    for model_name, param_count in param_counts.items():
        print(f"{model_name}: {param_count:,} parameters")
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Evaluate models
    print("\nEvaluating models...")
    
    # Standard neural decoder
    standard_ber, standard_fer = evaluate_standard_neural(
        standard_model, H, snr_range, args.batch_size, args.num_trials, device
    )
    
    # GNN-based decoder
    gnn_ber, gnn_fer = evaluate_gnn(
        gnn_model, H, snr_range, args.batch_size, args.num_trials, device
    )
    
    # Message-centered GNN decoder
    message_gnn_ber, message_gnn_fer = evaluate_message_gnn(
        message_gnn_model, converter, snr_range, args.batch_size, args.num_trials, device
    )
    
    # Traditional decoders
    trad_results = evaluate_traditional(
        H, snr_range, args.batch_size, args.num_trials, device, 
        args.bp_max_iterations, args.ms_scaling_factor
    )
    
    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Plot results
    print("\nPlotting results...")
    plot_ber_comparison(
        snr_range, standard_ber, gnn_ber, message_gnn_ber, trad_results, 
        args.results_dir, args.paper_style
    )
    
    plot_fer_comparison(
        snr_range, standard_fer, gnn_fer, message_gnn_fer, trad_results, 
        args.results_dir, args.paper_style
    )
    
    plot_iterations_comparison(
        snr_range, trad_results, args.results_dir, args.paper_style
    )
    
    plot_parameter_comparison(
        param_counts, args.results_dir, args.paper_style
    )
    
    # Save results
    results = {
        'snr_range': snr_range,
        'standard_ber': standard_ber,
        'standard_fer': standard_fer,
        'gnn_ber': gnn_ber,
        'gnn_fer': gnn_fer,
        'message_gnn_ber': message_gnn_ber,
        'message_gnn_fer': message_gnn_fer,
        'bp_ber': trad_results['bp_ber'],
        'bp_fer': trad_results['bp_fer'],
        'bp_iterations': trad_results['bp_iterations'],
        'ms_ber': trad_results['ms_ber'],
        'ms_fer': trad_results['ms_fer'],
        'ms_iterations': trad_results['ms_iterations'],
        'param_counts': param_counts
    }
    
    torch.save(results, os.path.join(args.results_dir, 'comparison_all_results.pt'))
    
    print(f"\nComparison completed. Results saved to {args.results_dir}")


if __name__ == '__main__':
    main()