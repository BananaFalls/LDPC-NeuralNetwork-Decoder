import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ldpc_neural_decoder.models.decoder import LDPCNeuralDecoder
from ldpc_neural_decoder.models.traditional_decoders import BeliefPropagationDecoder, MinSumScaledDecoder
from ldpc_neural_decoder.utils.channel import qpsk_modulate, awgn_channel, qpsk_demodulate, compute_ber_fer

class ComparativeEvaluator:
    """
    Comparative evaluator for different LDPC decoders.
    
    Evaluates and compares the performance of different LDPC decoders.
    """
    def __init__(self, H, neural_decoder=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the comparative evaluator.
        
        Args:
            H (torch.Tensor): Parity-check matrix
            neural_decoder (nn.Module, optional): Pre-trained neural decoder
            device (str): Device to use for evaluation
        """
        self.H = H.to(device)
        self.device = device
        
        # Initialize decoders
        self.neural_decoder = neural_decoder
        if neural_decoder is not None:
            self.neural_decoder.to(device)
            self.neural_decoder.eval()
        
        self.bp_decoder = BeliefPropagationDecoder(H, max_iterations=50, early_stopping=True)
        self.ms_decoder = MinSumScaledDecoder(H, max_iterations=50, scaling_factor=0.75, early_stopping=True)
        
        # Results storage
        self.results = {}
    
    def evaluate_all(self, snr_range, batch_size=32, num_trials=100, variable_bit_length=None, check_index_tensor=None, var_index_tensor=None):
        """
        Evaluate all decoders over a range of SNR values.
        
        Args:
            snr_range (list): Range of SNR values to evaluate on
            batch_size (int): Batch size
            num_trials (int): Number of trials per SNR value
            variable_bit_length (int, optional): Length of variable bits
            check_index_tensor (torch.Tensor, optional): Check node index mapping for neural decoder
            var_index_tensor (torch.Tensor, optional): Variable node index mapping for neural decoder
            
        Returns:
            dict: Evaluation results
        """
        if variable_bit_length is None:
            variable_bit_length = self.H.shape[1]
        
        # Move tensors to device
        if check_index_tensor is not None:
            check_index_tensor = check_index_tensor.to(self.device)
        if var_index_tensor is not None:
            var_index_tensor = var_index_tensor.to(self.device)
        
        # Evaluate belief propagation decoder
        print("Evaluating Belief Propagation decoder...")
        bp_ber, bp_fer, bp_avg_iters = self._evaluate_traditional_decoder(
            self.bp_decoder, snr_range, batch_size, num_trials, variable_bit_length
        )
        
        # Evaluate min-sum scaled decoder
        print("Evaluating Min-Sum Scaled decoder...")
        ms_ber, ms_fer, ms_avg_iters = self._evaluate_traditional_decoder(
            self.ms_decoder, snr_range, batch_size, num_trials, variable_bit_length
        )
        
        # Evaluate neural decoder if provided
        if self.neural_decoder is not None and check_index_tensor is not None and var_index_tensor is not None:
            print("Evaluating Neural decoder...")
            neural_ber, neural_fer = self._evaluate_neural_decoder(
                snr_range, batch_size, num_trials, variable_bit_length, check_index_tensor, var_index_tensor
            )
        else:
            neural_ber, neural_fer = None, None
        
        # Store results
        self.results = {
            'snr_range': snr_range,
            'belief_propagation': {
                'ber': bp_ber,
                'fer': bp_fer,
                'avg_iterations': bp_avg_iters
            },
            'min_sum_scaled': {
                'ber': ms_ber,
                'fer': ms_fer,
                'avg_iterations': ms_avg_iters
            }
        }
        
        if neural_ber is not None:
            self.results['neural_decoder'] = {
                'ber': neural_ber,
                'fer': neural_fer
            }
        
        return self.results
    
    def _evaluate_traditional_decoder(self, decoder, snr_range, batch_size, num_trials, variable_bit_length):
        """
        Evaluate a traditional decoder over a range of SNR values.
        
        Args:
            decoder: Traditional decoder to evaluate
            snr_range (list): Range of SNR values to evaluate on
            batch_size (int): Batch size
            num_trials (int): Number of trials per SNR value
            variable_bit_length (int): Length of variable bits
            
        Returns:
            tuple: (ber_results, fer_results, avg_iterations)
        """
        ber_results = []
        fer_results = []
        avg_iterations = []
        
        for snr_db in tqdm(snr_range, desc=f"Evaluating {decoder.__class__.__name__}"):
            total_ber = 0.0
            total_fer = 0.0
            total_iterations = 0
            
            for _ in range(num_trials):
                # Generate all-zero codeword
                transmitted_bits = torch.zeros((batch_size, variable_bit_length), device=self.device)
                
                # Modulate using QPSK
                qpsk_symbols = qpsk_modulate(transmitted_bits.view(batch_size, -1))
                
                # Pass through AWGN channel
                received_signal = awgn_channel(qpsk_symbols, snr_db)
                
                # Demodulate to LLRs
                llrs = qpsk_demodulate(received_signal, snr_db)
                llrs = llrs.view(batch_size, -1)
                
                # Decode
                decoded_bits, iterations = decoder.decode(llrs)
                
                # Compute BER and FER
                ber, fer = compute_ber_fer(transmitted_bits, decoded_bits)
                
                # Accumulate metrics
                total_ber += ber
                total_fer += fer
                total_iterations += iterations
            
            # Compute averages
            avg_ber = total_ber / num_trials
            avg_fer = total_fer / num_trials
            avg_iter = total_iterations / num_trials
            
            # Store results
            ber_results.append(avg_ber)
            fer_results.append(avg_fer)
            avg_iterations.append(avg_iter)
        
        return ber_results, fer_results, avg_iterations
    
    def _evaluate_neural_decoder(self, snr_range, batch_size, num_trials, variable_bit_length, check_index_tensor, var_index_tensor):
        """
        Evaluate the neural decoder over a range of SNR values.
        
        Args:
            snr_range (list): Range of SNR values to evaluate on
            batch_size (int): Batch size
            num_trials (int): Number of trials per SNR value
            variable_bit_length (int): Length of variable bits
            check_index_tensor (torch.Tensor): Check node index mapping
            var_index_tensor (torch.Tensor): Variable node index mapping
            
        Returns:
            tuple: (ber_results, fer_results)
        """
        ber_results = []
        fer_results = []
        
        with torch.no_grad():
            for snr_db in tqdm(snr_range, desc="Evaluating Neural Decoder"):
                total_ber = 0.0
                total_fer = 0.0
                
                for _ in range(num_trials):
                    # Generate all-zero codeword
                    transmitted_bits = torch.zeros((batch_size, variable_bit_length), device=self.device)
                    
                    # Modulate using QPSK
                    qpsk_symbols = qpsk_modulate(transmitted_bits.view(batch_size, -1))
                    
                    # Pass through AWGN channel
                    received_signal = awgn_channel(qpsk_symbols, snr_db)
                    
                    # Demodulate to LLRs
                    llrs = qpsk_demodulate(received_signal, snr_db)
                    llrs = llrs.view(batch_size, -1)
                    
                    # Decode
                    hard_bits = self.neural_decoder.decode(llrs, check_index_tensor, var_index_tensor)
                    
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
        
        return ber_results, fer_results
    
    def plot_ber_comparison(self, save_path=None):
        """
        Plot BER comparison of different decoders.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: BER comparison plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_all() first.")
        
        snr_range = self.results['snr_range']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot BER for belief propagation
        ax.semilogy(snr_range, self.results['belief_propagation']['ber'], 'o-', label='Belief Propagation')
        
        # Plot BER for min-sum scaled
        ax.semilogy(snr_range, self.results['min_sum_scaled']['ber'], 's-', label='Min-Sum Scaled')
        
        # Plot BER for neural decoder if available
        if 'neural_decoder' in self.results:
            ax.semilogy(snr_range, self.results['neural_decoder']['ber'], '^-', label='Neural Decoder')
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.set_title('Bit Error Rate Comparison')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_fer_comparison(self, save_path=None):
        """
        Plot FER comparison of different decoders.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: FER comparison plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_all() first.")
        
        snr_range = self.results['snr_range']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot FER for belief propagation
        ax.semilogy(snr_range, self.results['belief_propagation']['fer'], 'o-', label='Belief Propagation')
        
        # Plot FER for min-sum scaled
        ax.semilogy(snr_range, self.results['min_sum_scaled']['fer'], 's-', label='Min-Sum Scaled')
        
        # Plot FER for neural decoder if available
        if 'neural_decoder' in self.results:
            ax.semilogy(snr_range, self.results['neural_decoder']['fer'], '^-', label='Neural Decoder')
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('FER')
        ax.set_title('Frame Error Rate Comparison')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_iterations_comparison(self, save_path=None):
        """
        Plot average iterations comparison of traditional decoders.
        
        Args:
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Iterations comparison plot
        """
        if not self.results:
            raise ValueError("No results to plot. Run evaluate_all() first.")
        
        snr_range = self.results['snr_range']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot average iterations for belief propagation
        ax.plot(snr_range, self.results['belief_propagation']['avg_iterations'], 'o-', label='Belief Propagation')
        
        # Plot average iterations for min-sum scaled
        ax.plot(snr_range, self.results['min_sum_scaled']['avg_iterations'], 's-', label='Min-Sum Scaled')
        
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Average Iterations')
        ax.set_title('Average Iterations Comparison')
        ax.legend()
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def save_results(self, path):
        """
        Save evaluation results to a file.
        
        Args:
            path (str): Path to save the results
        """
        if not self.results:
            raise ValueError("No results to save. Run evaluate_all() first.")
        
        torch.save(self.results, path)
    
    def load_results(self, path):
        """
        Load evaluation results from a file.
        
        Args:
            path (str): Path to load the results from
        """
        self.results = torch.load(path)
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.results:
            raise ValueError("No results to summarize. Run evaluate_all() first.")
        
        snr_range = self.results['snr_range']
        
        print("Evaluation Summary")
        print("=================")
        print(f"SNR Range: {snr_range}")
        print()
        
        print("Belief Propagation Decoder")
        print("--------------------------")
        print(f"BER: {self.results['belief_propagation']['ber']}")
        print(f"FER: {self.results['belief_propagation']['fer']}")
        print(f"Average Iterations: {self.results['belief_propagation']['avg_iterations']}")
        print()
        
        print("Min-Sum Scaled Decoder")
        print("---------------------")
        print(f"BER: {self.results['min_sum_scaled']['ber']}")
        print(f"FER: {self.results['min_sum_scaled']['fer']}")
        print(f"Average Iterations: {self.results['min_sum_scaled']['avg_iterations']}")
        print()
        
        if 'neural_decoder' in self.results:
            print("Neural Decoder")
            print("-------------")
            print(f"BER: {self.results['neural_decoder']['ber']}")
            print(f"FER: {self.results['neural_decoder']['fer']}")
            print() 