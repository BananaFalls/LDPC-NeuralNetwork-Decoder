import torch
import numpy as np

class BeliefPropagationDecoder:
    """
    Classic Belief Propagation decoder for LDPC codes.
    
    Implements the sum-product algorithm for LDPC decoding.
    """
    def __init__(self, H, max_iterations=50, early_stopping=True):
        """
        Initialize the Belief Propagation decoder.
        
        Args:
            H (torch.Tensor): Parity-check matrix
            max_iterations (int): Maximum number of decoding iterations
            early_stopping (bool): Whether to stop early if a valid codeword is found
        """
        self.H = H
        self.max_iterations = max_iterations
        self.early_stopping = early_stopping
        
        # Precompute indices for efficient message passing
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Precompute indices for efficient message passing."""
        H = self.H
        m, n = H.shape  # m: number of check nodes, n: number of variable nodes
        
        # For each check node, find connected variable nodes
        self.check_to_var = [[] for _ in range(m)]
        # For each variable node, find connected check nodes
        self.var_to_check = [[] for _ in range(n)]
        
        for i in range(m):
            for j in range(n):
                if H[i, j] == 1:
                    self.check_to_var[i].append(j)
                    self.var_to_check[j].append(i)
    
    def decode(self, llr):
        """
        Decode using the sum-product algorithm.
        
        Args:
            llr (torch.Tensor): Log-likelihood ratios of shape (batch_size, n)
            
        Returns:
            tuple: (decoded_bits, num_iterations)
                - decoded_bits (torch.Tensor): Decoded bits of shape (batch_size, n)
                - num_iterations (int): Number of iterations performed
        """
        batch_size, n = llr.shape
        m = len(self.check_to_var)
        device = llr.device
        
        # Initialize messages
        # Variable-to-check messages (initialized with LLRs)
        v2c = torch.zeros((batch_size, m, n), device=device)
        # Check-to-variable messages
        c2v = torch.zeros((batch_size, m, n), device=device)
        
        # Initialize v2c with LLRs
        for j in range(n):
            for i in self.var_to_check[j]:
                v2c[:, i, j] = llr[:, j]
        
        # Iterative decoding
        for iter_idx in range(self.max_iterations):
            # Check node update (horizontal step)
            for i in range(m):
                for j in self.check_to_var[i]:
                    # Compute product of tanh(v2c/2) for all connected variable nodes except j
                    prod = torch.ones(batch_size, device=device)
                    for j_prime in self.check_to_var[i]:
                        if j_prime != j:
                            prod = prod * torch.tanh(v2c[:, i, j_prime] / 2)
                    
                    # Update check-to-variable message
                    c2v[:, i, j] = 2 * torch.atanh(prod)
            
            # Variable node update (vertical step)
            for j in range(n):
                for i in self.var_to_check[j]:
                    # Sum all incoming check-to-variable messages except from check node i
                    sum_msgs = llr[:, j].clone()
                    for i_prime in self.var_to_check[j]:
                        if i_prime != i:
                            sum_msgs = sum_msgs + c2v[:, i_prime, j]
                    
                    # Update variable-to-check message
                    v2c[:, i, j] = sum_msgs
            
            # Compute current decoding
            var_beliefs = llr.clone()
            for j in range(n):
                for i in self.var_to_check[j]:
                    var_beliefs[:, j] = var_beliefs[:, j] + c2v[:, i, j]
            
            decoded_bits = (var_beliefs < 0).float()
            
            # Check if valid codeword (all parity checks satisfied)
            if self.early_stopping:
                valid = self._check_valid_codeword(decoded_bits)
                if valid.all():
                    return decoded_bits, iter_idx + 1
        
        return decoded_bits, self.max_iterations
    
    def _check_valid_codeword(self, decoded_bits):
        """
        Check if decoded bits form a valid codeword.
        
        Args:
            decoded_bits (torch.Tensor): Decoded bits of shape (batch_size, n)
            
        Returns:
            torch.Tensor: Boolean tensor of shape (batch_size,) indicating if each decoded word is valid
        """
        batch_size = decoded_bits.shape[0]
        valid = torch.ones(batch_size, dtype=torch.bool, device=decoded_bits.device)
        
        # Check each parity constraint
        for i in range(len(self.check_to_var)):
            # XOR of connected variable nodes should be 0
            for b in range(batch_size):
                parity = 0
                for j in self.check_to_var[i]:
                    parity = parity ^ int(decoded_bits[b, j].item())
                if parity != 0:
                    valid[b] = False
        
        return valid


class MinSumScaledDecoder:
    """
    Min-Sum Scaled decoder for LDPC codes.
    
    Implements the min-sum algorithm with scaling factor for LDPC decoding.
    """
    def __init__(self, H, max_iterations=50, scaling_factor=0.75, early_stopping=True):
        """
        Initialize the Min-Sum Scaled decoder.
        
        Args:
            H (torch.Tensor): Parity-check matrix
            max_iterations (int): Maximum number of decoding iterations
            scaling_factor (float): Scaling factor for min-sum algorithm (typically 0.75-0.9)
            early_stopping (bool): Whether to stop early if a valid codeword is found
        """
        self.H = H
        self.max_iterations = max_iterations
        self.scaling_factor = scaling_factor
        self.early_stopping = early_stopping
        
        # Precompute indices for efficient message passing
        self._precompute_indices()
    
    def _precompute_indices(self):
        """Precompute indices for efficient message passing."""
        H = self.H
        m, n = H.shape  # m: number of check nodes, n: number of variable nodes
        
        # For each check node, find connected variable nodes
        self.check_to_var = [[] for _ in range(m)]
        # For each variable node, find connected check nodes
        self.var_to_check = [[] for _ in range(n)]
        
        for i in range(m):
            for j in range(n):
                if H[i, j] == 1:
                    self.check_to_var[i].append(j)
                    self.var_to_check[j].append(i)
    
    def decode(self, llr):
        """
        Decode using the min-sum scaled algorithm.
        
        Args:
            llr (torch.Tensor): Log-likelihood ratios of shape (batch_size, n)
            
        Returns:
            tuple: (decoded_bits, num_iterations)
                - decoded_bits (torch.Tensor): Decoded bits of shape (batch_size, n)
                - num_iterations (int): Number of iterations performed
        """
        batch_size, n = llr.shape
        m = len(self.check_to_var)
        device = llr.device
        
        # Initialize messages
        # Variable-to-check messages (initialized with LLRs)
        v2c = torch.zeros((batch_size, m, n), device=device)
        # Check-to-variable messages
        c2v = torch.zeros((batch_size, m, n), device=device)
        
        # Initialize v2c with LLRs
        for j in range(n):
            for i in self.var_to_check[j]:
                v2c[:, i, j] = llr[:, j]
        
        # Iterative decoding
        for iter_idx in range(self.max_iterations):
            # Check node update (horizontal step)
            for i in range(m):
                for j in self.check_to_var[i]:
                    # Compute sign and minimum magnitude
                    signs = torch.ones(batch_size, device=device)
                    min_mag = torch.full((batch_size,), float('inf'), device=device)
                    second_min_mag = torch.full((batch_size,), float('inf'), device=device)
                    
                    for j_prime in self.check_to_var[i]:
                        if j_prime != j:
                            # Compute sign product
                            signs = signs * torch.sign(v2c[:, i, j_prime])
                            
                            # Update minimum magnitudes
                            mag = torch.abs(v2c[:, i, j_prime])
                            mask = mag < min_mag
                            second_min_mag[mask] = min_mag[mask]
                            min_mag[mask] = mag[mask]
                            
                            mask = (mag >= min_mag) & (mag < second_min_mag)
                            second_min_mag[mask] = mag[mask]
                    
                    # Apply scaling factor to min magnitude
                    scaled_min = self.scaling_factor * min_mag
                    
                    # Update check-to-variable message
                    c2v[:, i, j] = signs * scaled_min
            
            # Variable node update (vertical step)
            for j in range(n):
                for i in self.var_to_check[j]:
                    # Sum all incoming check-to-variable messages except from check node i
                    sum_msgs = llr[:, j].clone()
                    for i_prime in self.var_to_check[j]:
                        if i_prime != i:
                            sum_msgs = sum_msgs + c2v[:, i_prime, j]
                    
                    # Update variable-to-check message
                    v2c[:, i, j] = sum_msgs
            
            # Compute current decoding
            var_beliefs = llr.clone()
            for j in range(n):
                for i in self.var_to_check[j]:
                    var_beliefs[:, j] = var_beliefs[:, j] + c2v[:, i, j]
            
            decoded_bits = (var_beliefs < 0).float()
            
            # Check if valid codeword (all parity checks satisfied)
            if self.early_stopping:
                valid = self._check_valid_codeword(decoded_bits)
                if valid.all():
                    return decoded_bits, iter_idx + 1
        
        return decoded_bits, self.max_iterations
    
    def _check_valid_codeword(self, decoded_bits):
        """
        Check if decoded bits form a valid codeword.
        
        Args:
            decoded_bits (torch.Tensor): Decoded bits of shape (batch_size, n)
            
        Returns:
            torch.Tensor: Boolean tensor of shape (batch_size,) indicating if each decoded word is valid
        """
        batch_size = decoded_bits.shape[0]
        valid = torch.ones(batch_size, dtype=torch.bool, device=decoded_bits.device)
        
        # Check each parity constraint
        for i in range(len(self.check_to_var)):
            # XOR of connected variable nodes should be 0
            for b in range(batch_size):
                parity = 0
                for j in self.check_to_var[i]:
                    parity = parity ^ int(decoded_bits[b, j].item())
                if parity != 0:
                    valid[b] = False
        
        return valid 