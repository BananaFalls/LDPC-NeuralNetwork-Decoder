"""
Run complete workflow for LDPC neural decoder.

This script runs a complete workflow for the LDPC neural decoder:
1. Train the neural LDPC decoder
2. Evaluate the neural decoder
3. Compare with traditional decoders
4. Visualize the results

Usage:
    python -m ldpc_neural_decoder.run_workflow
"""

import os
import argparse
import subprocess
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run complete workflow for LDPC neural decoder')
    
    # General parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay (L2 penalty) for SGD optimizer')
    
    # Model parameters
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--depth_L', type=int, default=2, help='Depth of residual connections')
    
    # Evaluation parameters
    parser.add_argument('--snr_min', type=int, default=-2, help='Minimum SNR for evaluation')
    parser.add_argument('--snr_max', type=int, default=6, help='Maximum SNR for evaluation')
    parser.add_argument('--snr_step', type=int, default=1, help='SNR step for evaluation')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of trials per SNR')
    
    # I/O parameters
    parser.add_argument('--model_path', type=str, default='models/saved_models/model.pt',
                        help='Path to save/load model')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save results')
    
    # Workflow control
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and use existing model')
    
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and print its output in real-time."""
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n{'='*80}")
    print(f"{description} completed in {elapsed_time:.2f} seconds.")
    print(f"{'='*80}\n")
    
    return process.returncode

def main():
    """Main function."""
    import torch  # Import here to avoid circular import
    
    args = parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Build base command with common arguments
    base_cmd = (
        f"python -m ldpc_neural_decoder.main "
        f"--device {args.device} "
        f"--seed {args.seed} "
        f"--num_iterations {args.num_iterations} "
        f"--depth_L {args.depth_L} "
        f"--batch_size {args.batch_size} "
        f"--snr_min {args.snr_min} "
        f"--snr_max {args.snr_max} "
        f"--snr_step {args.snr_step} "
        f"--num_trials {args.num_trials} "
        f"--model_path {args.model_path} "
        f"--results_dir {args.results_dir} "
    )
    
    if args.base_matrix_path:
        base_cmd += f"--base_matrix_path {args.base_matrix_path} "
        base_cmd += f"--lifting_factor {args.lifting_factor} "
    
    # 1. Train the neural LDPC decoder
    if not args.skip_training:
        train_cmd = (
            f"{base_cmd} "
            f"--mode train "
            f"--num_epochs {args.num_epochs} "
            f"--learning_rate {args.learning_rate} "
            f"--momentum {args.momentum} "
            f"--weight_decay {args.weight_decay} "
        )
        
        run_command(train_cmd, "Training neural LDPC decoder")
    
    # 2. Evaluate the neural decoder
    eval_cmd = f"{base_cmd} --mode evaluate"
    run_command(eval_cmd, "Evaluating neural LDPC decoder")
    
    # 3. Compare with traditional decoders
    compare_cmd = f"{base_cmd} --mode compare --compare_with_traditional"
    run_command(compare_cmd, "Comparing with traditional decoders")
    
    # 4. Visualize the results with paper-style formatting
    visualize_cmd = (
        f"python -m ldpc_neural_decoder.visualization.plot_comparison "
        f"--results_path {args.results_dir}/comparison_results.pt "
        f"--output_dir {args.results_dir} "
        f"--paper_style"
    )
    run_command(visualize_cmd, "Visualizing results")
    
    print("\nWorkflow completed successfully!")
    print(f"Results saved to {args.results_dir}")

if __name__ == '__main__':
    main() 