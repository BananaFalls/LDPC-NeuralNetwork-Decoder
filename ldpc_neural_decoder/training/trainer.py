"""
Training Module for LDPC Neural Decoder

This module provides functions for training and evaluating the LDPC neural decoder.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ldpc_neural_decoder.utils.channel import (
    qpsk_modulate,
    awgn_channel,
    qpsk_demodulate,
    compute_ber_fer,
    generate_variable_snr_values
)

class LDPCDecoderTrainer:
    """
    Trainer for LDPC Neural Decoder.
    
    Handles training, validation, and evaluation of the LDPC neural decoder.
    """
    def __init__(self, decoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            decoder (nn.Module): LDPC neural decoder model
            device (str): Device to use for training ('cuda' or 'cpu')
        """
        self.decoder = decoder
        self.device = device
        self.decoder.to(device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.ber_history = []
        self.fer_history = []
    
    def train(self, num_epochs, batch_size, learning_rate, check_index_tensor, var_index_tensor, 
              snr_range=None, variable_bit_length=None, validation_interval=5, momentum=0.9, weight_decay=0.0001):
        """
        Train the LDPC neural decoder.
        
        Args:
            num_epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            check_index_tensor (torch.Tensor): Check node index mapping
            var_index_tensor (torch.Tensor): Variable node index mapping
            snr_range (list, optional): Range of SNR values to train on [min_snr, max_snr]
            variable_bit_length (int, optional): Length of variable bits
            validation_interval (int): Interval for validation
            momentum (float): Momentum factor for SGD optimizer
            weight_decay (float): Weight decay (L2 penalty) for SGD optimizer
            
        Returns:
            dict: Training history
        """
        # Move tensors to device
        check_index_tensor = check_index_tensor.to(self.device)
        var_index_tensor = var_index_tensor.to(self.device)
        
        # Set up optimizer
        optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        
        # Set up SNR range if not provided
        if snr_range is None:
            # Default range covering FER from 10^-1 to 10^-8 (extended range)
            snr_range = [0.0, 12.0]
        
        print(f"Training with SNR values varying randomly between {snr_range[0]} and {snr_range[1]} dB")
        print(f"Using SGD optimizer with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
        
        # Training loop
        for epoch in range(num_epochs):
            self.decoder.train()
            epoch_loss = 0.0
            epoch_ber = 0.0
            epoch_fer = 0.0
            num_batches = 0
            
            # Generate random SNR values for each sample in the batch
            snr_values = generate_variable_snr_values(
                batch_size=batch_size, 
                min_snr=snr_range[0], 
                max_snr=snr_range[1]
            ).to(self.device)
            
            # Generate zero codewords (as per paper's recommendation)
            transmitted_bits = torch.zeros((batch_size, variable_bit_length), 
                                         device=self.device).float()
            
            # Generate modulated symbols (using BPSK for simplicity: 0->+1, 1->-1)
            modulated_symbols = 1.0 - 2.0 * transmitted_bits
            
            # Add noise with different SNR for each sample
            noisy_symbols = torch.zeros_like(modulated_symbols)
            llrs = torch.zeros_like(modulated_symbols)
            
            for i in range(batch_size):
                # Calculate noise standard deviation for this sample
                snr_linear = 10 ** (snr_values[i].item() / 10)
                noise_std = 1.0 / np.sqrt(snr_linear)
                
                # Add Gaussian noise
                noise = torch.randn_like(modulated_symbols[i]) * noise_std
                noisy_symbols[i] = modulated_symbols[i] + noise
                
                # Convert to LLRs: LLR = 2*r/sigma^2 where r is the received signal
                llrs[i] = 2.0 * noisy_symbols[i] / (noise_std ** 2)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            soft_bits, loss = self.decoder(llrs, check_index_tensor, var_index_tensor, transmitted_bits)
            
            # Compute mean loss
            batch_loss = loss.mean()
            
            # Backward pass
            batch_loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Calculate BER and FER metrics
            hard_bits = (soft_bits > 0.5).float()
            ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
            
            # Accumulate loss and metrics
            epoch_loss += batch_loss.item()
            epoch_ber += ber
            epoch_fer += fer
            num_batches += 1
            
            # Compute average metrics for the epoch
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_ber = epoch_ber / num_batches
            avg_epoch_fer = epoch_fer / num_batches
            
            self.train_losses.append(avg_epoch_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.6f}, BER: {avg_epoch_ber:.6f}, FER: {avg_epoch_fer:.6f}")
            
            # Validation
            if (epoch + 1) % validation_interval == 0:
                val_loss, ber, fer = self.validate(batch_size, check_index_tensor, var_index_tensor, 
                                                  snr_range, variable_bit_length)
                self.val_losses.append(val_loss)
                self.ber_history.append(ber)
                self.fer_history.append(fer)
                
                print(f"Validation - Loss: {val_loss:.6f}, BER: {ber:.6f}, FER: {fer:.6f}")
        
        # Return training history
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ber_history': self.ber_history,
            'fer_history': self.fer_history
        }
    
    def validate(self, batch_size, check_index_tensor, var_index_tensor, snr_range, variable_bit_length):
        """
        Validate the LDPC neural decoder.
        
        Args:
            batch_size (int): Batch size
            check_index_tensor (torch.Tensor): Check node index mapping
            var_index_tensor (torch.Tensor): Variable node index mapping
            snr_range (list): Range of SNR values to validate on [min_snr, max_snr]
            variable_bit_length (int): Length of variable bits
            
        Returns:
            tuple: (avg_loss, avg_ber, avg_fer)
        """
        self.decoder.eval()
        total_loss = 0.0
        total_ber = 0.0
        total_fer = 0.0
        num_batches = 10  # Run multiple validation batches
        
        with torch.no_grad():
            for _ in range(num_batches):
                # Generate random SNR values for each sample in the batch
                snr_values = generate_variable_snr_values(
                    batch_size=batch_size, 
                    min_snr=snr_range[0], 
                    max_snr=snr_range[1]
                ).to(self.device)
                
                # Generate zero codewords
                transmitted_bits = torch.zeros((batch_size, variable_bit_length), 
                                              device=self.device).float()
                
                # Generate modulated symbols (BPSK: 0->+1, 1->-1)
                modulated_symbols = 1.0 - 2.0 * transmitted_bits
                
                # Add noise with different SNR for each sample
                noisy_symbols = torch.zeros_like(modulated_symbols)
                llrs = torch.zeros_like(modulated_symbols)
                
                for i in range(batch_size):
                    # Calculate noise standard deviation for this sample
                    snr_linear = 10 ** (snr_values[i].item() / 10)
                    noise_std = 1.0 / np.sqrt(snr_linear)
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(modulated_symbols[i]) * noise_std
                    noisy_symbols[i] = modulated_symbols[i] + noise
                    
                    # Convert to LLRs: LLR = 2*r/sigma^2 where r is the received signal
                    llrs[i] = 2.0 * noisy_symbols[i] / (noise_std ** 2)
                
                # Forward pass
                soft_bits, loss = self.decoder(llrs, check_index_tensor, var_index_tensor, transmitted_bits)
                
                # Compute mean loss
                batch_loss = loss.mean().item()
                
                # Compute BER and FER
                hard_bits = (soft_bits > 0.5).float()
                ber, fer = compute_ber_fer(transmitted_bits, hard_bits)
                
                # Accumulate metrics
                total_loss += batch_loss
                total_ber += ber
                total_fer += fer
        
        # Compute averages
        avg_loss = total_loss / num_batches
        avg_ber = total_ber / num_batches
        avg_fer = total_fer / num_batches
        
        return avg_loss, avg_ber, avg_fer
    
    def evaluate_snr_range(self, snr_range, batch_size, num_trials, check_index_tensor, var_index_tensor, variable_bit_length):
        """
        Evaluate the LDPC neural decoder over a range of SNR values.
        
        Args:
            snr_range (list): Range of SNR values to evaluate on
            batch_size (int): Batch size
            num_trials (int): Number of trials per SNR value
            check_index_tensor (torch.Tensor): Check node index mapping
            var_index_tensor (torch.Tensor): Variable node index mapping
            variable_bit_length (int): Length of variable bits
            
        Returns:
            tuple: (ber_results, fer_results)
        """
        self.decoder.eval()
        ber_results = []
        fer_results = []
        
        # Move tensors to device
        check_index_tensor = check_index_tensor.to(self.device)
        var_index_tensor = var_index_tensor.to(self.device)
        
        for snr_db in tqdm(snr_range, desc="Evaluating SNR"):
            total_ber = 0.0
            total_fer = 0.0
            
            for _ in range(num_trials):
                # Generate all-zero codeword (as per paper's recommendation)
                transmitted_bits = torch.zeros((batch_size, variable_bit_length), device=self.device)
                
                # BPSK modulation (0 -> +1, 1 -> -1)
                modulated_symbols = 1.0 - 2.0 * transmitted_bits
                
                # Add AWGN noise
                snr_linear = 10 ** (snr_db / 10)
                noise_std = 1.0 / np.sqrt(snr_linear)
                noise = torch.randn_like(modulated_symbols) * noise_std
                noisy_symbols = modulated_symbols + noise
                
                # Convert to LLRs
                llrs = 2.0 * noisy_symbols / (noise_std ** 2)
                
                # Decode
                with torch.no_grad():
                    # For inference, use the decode method which returns hard bits
                    hard_bits = self.decoder.decode(llrs, check_index_tensor, var_index_tensor)
                
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
            
            # Print progress at key SNR points
            print(f"SNR: {snr_db} dB - BER: {avg_ber:.6f}, FER: {avg_fer:.6f}")
        
        return ber_results, fer_results
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            tuple: (fig1, fig2) - Training loss and BER/FER figures
        """
        # Plot training and validation loss
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            val_epochs = list(range(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses)))[:len(self.val_losses)]
            ax1.plot(val_epochs, self.val_losses, 'o-', label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot BER and FER history
        if self.ber_history:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(self.ber_history, 'o-', label='BER')
            ax2.plot(self.fer_history, 's-', label='FER')
            ax2.set_xlabel('Validation Step')
            ax2.set_ylabel('Error Rate')
            ax2.set_title('BER and FER History')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True)
        else:
            fig2 = None
        
        return fig1, fig2
    
    def plot_snr_performance(self, snr_range, ber_results, fer_results, comparison_ber=None, comparison_fer=None):
        """
        Plot the performance over SNR range.
        
        Args:
            snr_range (list): Range of SNR values
            ber_results (list): BER results for each SNR value
            fer_results (list): FER results for each SNR value
            comparison_ber (list, optional): BER results for comparison
            comparison_fer (list, optional): FER results for comparison
            
        Returns:
            tuple: (fig1, fig2) - BER and FER figures
        """
        # Plot BER vs SNR
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.semilogy(snr_range, ber_results, 'o-', label='Neural Decoder')
        if comparison_ber is not None:
            ax1.semilogy(snr_range, comparison_ber, 's-', label='Conventional Decoder')
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('BER')
        ax1.set_title('Bit Error Rate vs SNR')
        ax1.legend()
        ax1.grid(True)
        
        # Plot FER vs SNR
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.semilogy(snr_range, fer_results, 'o-', label='Neural Decoder')
        if comparison_fer is not None:
            ax2.semilogy(snr_range, comparison_fer, 's-', label='Conventional Decoder')
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('FER')
        ax2.set_title('Frame Error Rate vs SNR')
        ax2.legend()
        ax2.grid(True)
        
        return fig1, fig2
    
    def save_model(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.decoder.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'ber_history': self.ber_history,
            'fer_history': self.fer_history
        }, path)
    
    def load_model(self, path):
        """
        Load the model from a file.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.decoder.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.ber_history = checkpoint.get('ber_history', [])
        self.fer_history = checkpoint.get('fer_history', []) 