"""
Run comparative evaluation of LDPC decoders.

This script runs a comparative evaluation of different LDPC decoders:
- Neural LDPC Decoder
- Belief Propagation Decoder
- Min-Sum Scaled Decoder

Usage:
    python run_comparison.py --model_path models/saved_models/model.pt
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt

from ldpc_neural_decoder.models.decoder import LDPCNeuralDecoder
from ldpc_neural_decoder.training.comparative_evaluation import ComparativeEvaluator
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, create_LLR_mapping, expand_base_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run comparative evaluation of LDPC decoders')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
    # Neural decoder parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/model.pt',
                        help='Path to load neural decoder model')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--depth_L', type=int, default=2, help='Depth of residual connections')
    
    # Traditional decoder parameters
    parser.add_argument('--bp_max_iterations', type=int, default=50,
                        help='Maximum iterations for belief propagation decoder')
    parser.add_argument('--ms_scaling_factor', type=float, default=0.75,
                        help='Scaling factor for min-sum decoder')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per SNR')
    parser.add_argument('--snr_min', type=int, default=-2, help='Minimum SNR for evaluation')
    parser.add_argument('--snr_max', type=int, default=6, help='Maximum SNR for evaluation')
    parser.add_argument('--snr_step', type=int, default=1, help='SNR step for evaluation')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create or load parity-check matrix
    if args.base_matrix_path:
        base_matrix = load_base_matrix(args.base_matrix_path)
        H = expand_base_matrix(base_matrix, args.lifting_factor)
    else:
        # Example small parity-check matrix
        H = torch.tensor([
            [1, 1, 0, 0],  # Check 1
            [0, 1, 1, 1],  # Check 2
            [1, 0, 0, 1]   # Check 3
        ], dtype=torch.float32)
    
    # Create LLR mapping
    H_T = H.T
    _, check_index_tensor, var_index_tensor, output_index_tensor = create_LLR_mapping(H_T)
    
    # Load neural decoder
    num_nodes = torch.sum(H == 1).item()
    neural_decoder = LDPCNeuralDecoder(
        num_nodes=num_nodes,
        num_iterations=args.num_iterations,
        depth_L=args.depth_L
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=args.device)
    neural_decoder.load_state_dict(checkpoint['model_state_dict'])
    neural_decoder.eval()
    print(f"Neural model loaded from {args.model_path}")
    
    # Create comparative evaluator
    evaluator = ComparativeEvaluator(H, neural_decoder=neural_decoder, device=args.device)
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Evaluate all decoders
    print(f"Comparing decoders over SNR range {snr_range}...")
    results = evaluator.evaluate_all(
        snr_range=snr_range,
        batch_size=args.batch_size,
        num_trials=args.num_trials,
        variable_bit_length=H.shape[1],
        check_index_tensor=check_index_tensor,
        var_index_tensor=var_index_tensor
    )
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Plot BER comparison
    ber_fig = evaluator.plot_ber_comparison(
        save_path=os.path.join(args.results_dir, 'ber_comparison.png')
    )
    
    # Plot FER comparison
    fer_fig = evaluator.plot_fer_comparison(
        save_path=os.path.join(args.results_dir, 'fer_comparison.png')
    )
    
    # Plot iterations comparison
    iter_fig = evaluator.plot_iterations_comparison(
        save_path=os.path.join(args.results_dir, 'iterations_comparison.png')
    )
    
    # Save results
    evaluator.save_results(os.path.join(args.results_dir, 'comparison_results.pt'))
    
    # Print summary
    evaluator.print_summary()
    
    print("Comparison completed.")
    print(f"Results saved to {args.results_dir}")

if __name__ == '__main__':
    main() 