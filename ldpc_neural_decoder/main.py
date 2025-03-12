import torch
import argparse
import os
import matplotlib.pyplot as plt

from ldpc_neural_decoder.models.decoder import LDPCNeuralDecoder, TiedNeuralLDPCDecoder
from ldpc_neural_decoder.training.trainer import LDPCDecoderTrainer
from ldpc_neural_decoder.training.comparative_evaluation import ComparativeEvaluator
from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, create_LLR_mapping, expand_base_matrix

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train and evaluate LDPC neural decoder')
    
    # General parameters
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'evaluate', 'visualize', 'compare'],
                        help='Mode: train, evaluate, visualize, or compare')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'tied'],
                        help='Model type: standard or tied')
    parser.add_argument('--num_iterations', type=int, default=5, help='Number of decoding iterations')
    parser.add_argument('--depth_L', type=int, default=2, help='Depth of residual connections')
    
    # LDPC code parameters
    parser.add_argument('--base_matrix_path', type=str, default=None, help='Path to base matrix file')
    parser.add_argument('--lifting_factor', type=int, default=16, help='Lifting factor Z')
    
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
    
    # Comparison parameters
    parser.add_argument('--compare_with_traditional', action='store_true',
                        help='Compare with traditional decoders')
    parser.add_argument('--bp_max_iterations', type=int, default=50,
                        help='Maximum iterations for belief propagation decoder')
    parser.add_argument('--ms_scaling_factor', type=float, default=0.75,
                        help='Scaling factor for min-sum decoder')
    
    # I/O parameters
    parser.add_argument('--model_path', type=str, default='ldpc_neural_decoder/models/saved_models/model.pt',
                        help='Path to save/load model')
    parser.add_argument('--results_dir', type=str, default='ldpc_neural_decoder/results',
                        help='Directory to save results')
    
    return parser.parse_args()

def create_model(args, H):
    """Create LDPC neural decoder model."""
    # Count number of 1s in H
    num_nodes = torch.sum(H == 1).item()
    
    if args.model_type == 'standard':
        model = LDPCNeuralDecoder(
            num_nodes=num_nodes,
            num_iterations=args.num_iterations,
            depth_L=args.depth_L
        )
    else:  # tied
        base_matrix = load_base_matrix(args.base_matrix_path)
        model = TiedNeuralLDPCDecoder(
            base_graph=base_matrix,
            Z=args.lifting_factor,
            num_iterations=args.num_iterations,
            depth_L=args.depth_L
        )
    
    return model

def train(args):
    """Train the LDPC neural decoder."""
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
    
    # Create model
    model = create_model(args, H)
    
    # Create trainer
    trainer = LDPCDecoderTrainer(model, device=args.device)
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Train model
    print(f"Training model with {args.num_epochs} epochs...")
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
    fig1.savefig(os.path.join(args.results_dir, 'training_loss.png'))
    if fig2:
        fig2.savefig(os.path.join(args.results_dir, 'error_rates.png'))
    
    print("Training completed.")

def evaluate(args):
    """Evaluate the LDPC neural decoder."""
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
    
    # Create model
    model = create_model(args, H)
    
    # Create trainer
    trainer = LDPCDecoderTrainer(model, device=args.device)
    
    # Load model
    trainer.load_model(args.model_path)
    print(f"Model loaded from {args.model_path}")
    
    # Create SNR range
    snr_range = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    # Evaluate model
    print(f"Evaluating model over SNR range {snr_range}...")
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
    fig1.savefig(os.path.join(args.results_dir, 'ber_vs_snr.png'))
    fig2.savefig(os.path.join(args.results_dir, 'fer_vs_snr.png'))
    
    # Save results
    results = {
        'snr_range': snr_range,
        'ber_results': ber_results,
        'fer_results': fer_results
    }
    torch.save(results, os.path.join(args.results_dir, 'evaluation_results.pt'))
    
    print("Evaluation completed.")

def compare(args):
    """Compare different LDPC decoders."""
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
    
    # Create neural model if needed
    neural_decoder = None
    if args.compare_with_traditional:
        neural_decoder = create_model(args, H)
        # Load model
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

def visualize(args):
    """Visualize the LDPC neural decoder results."""
    # Load results
    results_path = os.path.join(args.results_dir, 'evaluation_results.pt')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    
    results = torch.load(results_path)
    snr_range = results['snr_range']
    ber_results = results['ber_results']
    fer_results = results['fer_results']
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # BER plot
    plt.subplot(1, 2, 1)
    plt.semilogy(snr_range, ber_results, 'o-', label="Neural LDPC")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True)
    plt.legend()
    plt.title("Bit Error Rate")
    
    # FER plot
    plt.subplot(1, 2, 2)
    plt.semilogy(snr_range, fer_results, 'o-', label="Neural LDPC")
    plt.xlabel("SNR (dB)")
    plt.ylabel("FER")
    plt.grid(True)
    plt.legend()
    plt.title("Frame Error Rate")
    
    plt.suptitle("LDPC Neural Decoder Performance")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(args.results_dir, 'performance.png'))
    plt.show()
    
    print("Visualization completed.")

def main():
    """Main function."""
    args = parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    elif args.mode == 'visualize':
        visualize(args)
    elif args.mode == 'compare':
        compare(args)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main() 