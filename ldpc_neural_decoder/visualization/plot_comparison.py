"""
Plot comparative results of LDPC decoders.

This script loads and visualizes the comparative results of different LDPC decoders.

Usage:
    python -m ldpc_neural_decoder.visualization.plot_comparison --results_path results/comparison_results.pt
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot comparative results of LDPC decoders')
    
    parser.add_argument('--results_path', type=str, default='results/comparison_results.pt',
                        help='Path to comparison results file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save plots')
    parser.add_argument('--paper_style', action='store_true',
                        help='Use paper-style formatting for plots')
    
    return parser.parse_args()

def plot_ber_comparison(results, output_path, paper_style=False):
    """
    Plot BER comparison of different decoders.
    
    Args:
        results (dict): Comparison results
        output_path (str): Path to save the plot
        paper_style (bool): Whether to use paper-style formatting
    """
    snr_range = results['snr_range']
    
    if paper_style:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'text.usetex': True if plt.rcParams.get('text.usetex', False) else False
        })
    else:
        plt.figure(figsize=(10, 6))
    
    # Plot BER for belief propagation
    plt.semilogy(snr_range, results['belief_propagation']['ber'], 'o-', label='Belief Propagation', linewidth=2)
    
    # Plot BER for min-sum scaled
    plt.semilogy(snr_range, results['min_sum_scaled']['ber'], 's-', label='Min-Sum Scaled', linewidth=2)
    
    # Plot BER for neural decoder if available
    if 'neural_decoder' in results:
        plt.semilogy(snr_range, results['neural_decoder']['ber'], '^-', label='Neural Decoder', linewidth=2)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Performance Comparison of LDPC Decoders')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    plt.ylim(bottom=1e-6, top=1)
    
    # Add theoretical AWGN bound if using paper style
    if paper_style:
        # Theoretical AWGN bound for QPSK (approximate)
        snr_linear = 10 ** (np.array(snr_range) / 10)
        theoretical_ber = 0.5 * np.exp(-snr_linear)
        plt.semilogy(snr_range, theoretical_ber, 'k--', label='QPSK Theoretical', linewidth=1.5)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_fer_comparison(results, output_path, paper_style=False):
    """
    Plot FER comparison of different decoders.
    
    Args:
        results (dict): Comparison results
        output_path (str): Path to save the plot
        paper_style (bool): Whether to use paper-style formatting
    """
    snr_range = results['snr_range']
    
    if paper_style:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'text.usetex': True if plt.rcParams.get('text.usetex', False) else False
        })
    else:
        plt.figure(figsize=(10, 6))
    
    # Plot FER for belief propagation
    plt.semilogy(snr_range, results['belief_propagation']['fer'], 'o-', label='Belief Propagation', linewidth=2)
    
    # Plot FER for min-sum scaled
    plt.semilogy(snr_range, results['min_sum_scaled']['fer'], 's-', label='Min-Sum Scaled', linewidth=2)
    
    # Plot FER for neural decoder if available
    if 'neural_decoder' in results:
        plt.semilogy(snr_range, results['neural_decoder']['fer'], '^-', label='Neural Decoder', linewidth=2)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.title('FER Performance Comparison of LDPC Decoders')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Set y-axis limits
    plt.ylim(bottom=1e-6, top=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_iterations_comparison(results, output_path, paper_style=False):
    """
    Plot average iterations comparison of traditional decoders.
    
    Args:
        results (dict): Comparison results
        output_path (str): Path to save the plot
        paper_style (bool): Whether to use paper-style formatting
    """
    snr_range = results['snr_range']
    
    if paper_style:
        plt.figure(figsize=(8, 6))
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'text.usetex': True if plt.rcParams.get('text.usetex', False) else False
        })
    else:
        plt.figure(figsize=(10, 6))
    
    # Plot average iterations for belief propagation
    plt.plot(snr_range, results['belief_propagation']['avg_iterations'], 'o-', label='Belief Propagation', linewidth=2)
    
    # Plot average iterations for min-sum scaled
    plt.plot(snr_range, results['min_sum_scaled']['avg_iterations'], 's-', label='Min-Sum Scaled', linewidth=2)
    
    # Add horizontal line for neural decoder if available
    if 'neural_decoder' in results:
        plt.axhline(y=5, color='r', linestyle='--', label='Neural Decoder (Fixed)')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Average Number of Iterations')
    plt.title('Average Iterations Comparison of LDPC Decoders')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_combined_performance(results, output_path, paper_style=False):
    """
    Plot combined BER and FER performance.
    
    Args:
        results (dict): Comparison results
        output_path (str): Path to save the plot
        paper_style (bool): Whether to use paper-style formatting
    """
    snr_range = results['snr_range']
    
    if paper_style:
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'text.usetex': True if plt.rcParams.get('text.usetex', False) else False
        })
    else:
        plt.figure(figsize=(12, 10))
    
    # BER subplot
    plt.subplot(2, 1, 1)
    plt.semilogy(snr_range, results['belief_propagation']['ber'], 'o-', label='Belief Propagation', linewidth=2)
    plt.semilogy(snr_range, results['min_sum_scaled']['ber'], 's-', label='Min-Sum Scaled', linewidth=2)
    if 'neural_decoder' in results:
        plt.semilogy(snr_range, results['neural_decoder']['ber'], '^-', label='Neural Decoder', linewidth=2)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('BER Performance Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(bottom=1e-6, top=1)
    
    # FER subplot
    plt.subplot(2, 1, 2)
    plt.semilogy(snr_range, results['belief_propagation']['fer'], 'o-', label='Belief Propagation', linewidth=2)
    plt.semilogy(snr_range, results['min_sum_scaled']['fer'], 's-', label='Min-Sum Scaled', linewidth=2)
    if 'neural_decoder' in results:
        plt.semilogy(snr_range, results['neural_decoder']['fer'], '^-', label='Neural Decoder', linewidth=2)
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('Frame Error Rate (FER)')
    plt.title('FER Performance Comparison')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.ylim(bottom=1e-6, top=1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Load results
    if not os.path.exists(args.results_path):
        print(f"Results file not found: {args.results_path}")
        return
    
    results = torch.load(args.results_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot BER comparison
    plot_ber_comparison(
        results,
        os.path.join(args.output_dir, 'ber_comparison.png'),
        paper_style=args.paper_style
    )
    
    # Plot FER comparison
    plot_fer_comparison(
        results,
        os.path.join(args.output_dir, 'fer_comparison.png'),
        paper_style=args.paper_style
    )
    
    # Plot iterations comparison
    plot_iterations_comparison(
        results,
        os.path.join(args.output_dir, 'iterations_comparison.png'),
        paper_style=args.paper_style
    )
    
    # Plot combined performance
    plot_combined_performance(
        results,
        os.path.join(args.output_dir, 'combined_performance.png'),
        paper_style=args.paper_style
    )
    
    print(f"Plots saved to {args.output_dir}")

if __name__ == '__main__':
    main() 