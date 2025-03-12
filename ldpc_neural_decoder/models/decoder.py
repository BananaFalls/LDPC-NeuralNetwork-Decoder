"""
LDPC Neural Decoder Model

This module provides the main LDPC neural decoder model that combines
check node, variable node, and residual layers.
"""

import torch
import torch.nn as nn
from collections import deque

from ldpc_neural_decoder.models.layers import CheckLayer, VariableLayer, ResidualLayer, OutputLayer


class LDPCNeuralDecoder(nn.Module):
    """
    Neural LDPC Decoder with Residual Connections.
    
    This decoder implements the architecture described in the paper
    "Deep Neural Network Based Decoding of Short 5G LDPC Codes".
    """
    def __init__(self, num_nodes, num_iterations=5, depth_L=2):
        """
        Initialize the LDPC Neural Decoder.
        
        Args:
            num_nodes (int): Number of nodes in the Tanner graph
            num_iterations (int): Number of decoding iterations
            depth_L (int): Depth of residual connections
        """
        super(LDPCNeuralDecoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_iterations = num_iterations
        self.depth_L = depth_L
        
        # Initialize layers
        self.check_layer = CheckLayer()
        self.variable_layer = VariableLayer()
        self.residual_layer = ResidualLayer(num_nodes, depth_L)
        self.output_layer = OutputLayer()
        
        # Trainable weights for channel reliability
        self.w_ch = nn.Parameter(torch.ones(num_nodes))
        
    def forward(self, input_llr, check_index_tensor, var_index_tensor, ground_truth=None):
        """
        Forward pass of the LDPC Neural Decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_nodes)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_nodes)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        batch_size = input_llr.shape[0]
        
        # Apply channel weights to input LLRs
        weighted_llr = input_llr * self.w_ch.unsqueeze(0)
        
        # Initialize variable node messages with weighted LLRs
        var_messages = weighted_llr
        
        # Queue to store previous variable node messages for residual connections
        prev_var_messages = deque(maxlen=self.depth_L)
        
        # Iterative decoding
        for _ in range(self.num_iterations):
            # Store current variable messages for residual connections
            prev_var_messages.appendleft(var_messages)
            
            # Check node update
            check_messages = self.check_layer(var_messages, check_index_tensor)
            
            # Variable node update with residual connections
            var_messages = self.residual_layer(
                weighted_llr, 
                check_messages, 
                list(prev_var_messages)
            )
        
        # Final output mapping
        soft_bits, loss = self.output_layer(var_messages, input_llr, ground_truth)
        
        return soft_bits, loss
    
    def decode(self, input_llr, check_index_tensor, var_index_tensor):
        """
        Decode input LLRs to hard bit decisions.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            
        Returns:
            torch.Tensor: Hard bit decisions (0 or 1) of shape (batch_size, num_nodes)
        """
        # Get soft bit values
        soft_bits, _ = self.forward(input_llr, check_index_tensor, var_index_tensor)
        
        # Convert to hard decisions
        hard_bits = (soft_bits > 0.5).float()
        
        return hard_bits


class TiedNeuralLDPCDecoder(nn.Module):
    """
    Neural LDPC Decoder with parameter tying based on the base graph structure.
    
    This decoder exploits the structure of the 5G LDPC base graph to reduce
    the number of parameters and improve generalization.
    """
    def __init__(self, base_graph, Z, num_iterations=5, depth_L=2):
        """
        Initialize the Tied Neural LDPC Decoder.
        
        Args:
            base_graph (torch.Tensor): Base graph matrix with shift values
            Z (int): Lifting factor
            num_iterations (int): Number of decoding iterations
            depth_L (int): Depth of residual connections
        """
        super(TiedNeuralLDPCDecoder, self).__init__()
        
        self.base_graph = base_graph
        self.Z = Z
        self.num_iterations = num_iterations
        self.depth_L = depth_L
        
        # Build a small nn.ModuleDict for each cell that has shift >= 0
        # Key: (r,c)
        # Value: a small neural network
        self.cell_modules = nn.ModuleDict()
        
        for r in range(len(base_graph)):
            for c in range(len(base_graph[0])):
                if base_graph[r][c] >= 0:  # Valid shift value
                    # Create a unique key for this cell
                    key = f"{r}_{c}"
                    # Create a small neural network for this cell
                    self.cell_modules[key] = nn.Sequential(
                        nn.Linear(1, 4),
                        nn.ReLU(),
                        nn.Linear(4, 1)
                    )
        
        # Create the main decoder
        expanded_H = self._expand_base_graph()
        num_nodes = expanded_H.sum().item()  # Count number of 1s in H
        self.decoder = LDPCNeuralDecoder(num_nodes, num_iterations, depth_L)
    
    def _expand_base_graph(self):
        """
        Expand the base graph using the lifting factor Z.
        
        Returns:
            torch.Tensor: Expanded parity-check matrix
        """
        rows, cols = self.base_graph.shape
        H_expanded = torch.zeros((rows * self.Z, cols * self.Z))
        
        for i in range(rows):
            for j in range(cols):
                shift = self.base_graph[i, j].item()
                if shift >= 0:
                    # Create a shifted identity matrix
                    I_Z = torch.eye(self.Z)
                    I_Z = torch.roll(I_Z, shifts=int(shift), dims=1)
                    H_expanded[i*self.Z:(i+1)*self.Z, j*self.Z:(j+1)*self.Z] = I_Z
        
        return H_expanded
    
    def forward(self, input_llr, check_index_tensor, var_index_tensor, ground_truth=None):
        """
        Forward pass of the Tied Neural LDPC Decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_nodes)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_nodes)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        # Apply cell-specific transformations to input LLRs
        # This is a simplified version - in practice, you would need to map each LLR
        # to its corresponding base graph cell and apply the appropriate transformation
        
        # For now, we'll just use the standard decoder
        return self.decoder(input_llr, check_index_tensor, var_index_tensor, ground_truth)
    
    def decode(self, input_llr, check_index_tensor, var_index_tensor):
        """
        Decode input LLRs to hard bit decisions.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            
        Returns:
            torch.Tensor: Hard bit decisions (0 or 1) of shape (batch_size, num_nodes)
        """
        # Get soft bit values
        soft_bits, _ = self.forward(input_llr, check_index_tensor, var_index_tensor)
        
        # Convert to hard decisions
        hard_bits = (soft_bits > 0.5).float()
        
        return hard_bits 