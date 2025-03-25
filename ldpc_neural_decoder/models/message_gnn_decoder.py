"""
Message-Centered GNN LDPC Decoder

This module implements a Graph Neural Network (GNN) based LDPC decoder
where messages are represented as nodes in the GNN, and edges connect
messages that share the same variable or check node in the Tanner graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def safe_index(tensor, idx):
    """
    Helper function for safe indexing when some indices might be -1 (indicating padding)
    """
    mask = idx != -1
    safe_idx = idx.clone()
    safe_idx[~mask] = 0  # Set invalid indices to 0 temporarily
    result = tensor[safe_idx]
    return result

class VariableGNNLayer(nn.Module):
    """
    Variable-side GNN layer for alternating message passing in LDPC decoding.
    Combines scalar LLR operations with message type-specific weights.
    """
    
    def __init__(self, num_message_types=1, hidden_dim=16, use_bp_op=True):
        super().__init__()
        
        self.use_bp_op = use_bp_op
        self.hidden_dim = hidden_dim
        
        # Message type specific weights for scalar operations
        self.message_type_weights = nn.Parameter(torch.ones(num_message_types))
        
        # Output projection (for backward compatibility)
        self.output_projection = nn.Linear(hidden_dim, 1)
    
    def bp_variable_update(self, copied_llr, check_messages, message_types, var_to_check_adjacency):
        """Traditional BP variable node operation with type-specific weights"""
        # Project high-dim features back to LLR domain if needed
        if check_messages.size(-1) > 1:
            check_llrs = self.output_projection(check_messages).squeeze(-1)  # (batch_size, num_messages)
        else:
            check_llrs = check_messages.squeeze(-1)
        
        # Get type-specific weights for each message
        safe_message_types = torch.clamp(message_types, 0, len(self.message_type_weights) - 1)
        type_weights = self.message_type_weights[safe_message_types]  # (num_messages,)
        
        # Apply type weights to messages
        weighted_check_llrs = check_llrs * type_weights.unsqueeze(0)  # (batch_size, num_messages)
        
        # Expand adjacency matrix for batch processing
        # Shape: (batch_size, num_messages, num_messages)
        batch_adj = var_to_check_adjacency.unsqueeze(0).expand(weighted_check_llrs.size(0), -1, -1)
        
        # Sum messages from connected check nodes
        # Shape: (batch_size, num_messages)
        aggregated_llrs = torch.bmm(batch_adj, weighted_check_llrs.unsqueeze(-1)).squeeze(-1)
        
        # Add channel LLR
        updated_llrs = copied_llr.squeeze(-1) + aggregated_llrs
        
        # Keep in scalar domain
        return updated_llrs.unsqueeze(-1)
    
    def forward(self, copied_llr, var_messages, check_messages, message_types, var_to_check_adjacency):
        """Forward pass using scalar operations with type-specific weights"""
        return self.bp_variable_update(copied_llr, check_messages, message_types, var_to_check_adjacency)


class CheckGNNLayer(nn.Module):
    """
    Check-side GNN layer for alternating message passing in LDPC decoding.
    Combines min-sum operations with message type-specific weights.
    """
    
    def __init__(self, num_message_types=1, hidden_dim=16, use_bp_op=True):
        super().__init__()
        
        self.use_bp_op = use_bp_op
        self.hidden_dim = hidden_dim
        
        # Message type specific weights for scalar operations
        self.message_type_weights = nn.Parameter(torch.ones(num_message_types))
        
        # Output projection (for backward compatibility)
        self.output_projection = nn.Linear(hidden_dim, 1)
    
    def min_sum_update(self, var_messages, message_types, check_to_var_adjacency):
        """Min-sum check node operation with type-specific weights"""
        # Project high-dim features back to LLR domain if needed
        if var_messages.size(-1) > 1:
            var_llrs = self.output_projection(var_messages).squeeze(-1)  # (batch_size, num_messages)
        else:
            var_llrs = var_messages.squeeze(-1)
        
        # Get type-specific weights for each message
        safe_message_types = torch.clamp(message_types, 0, len(self.message_type_weights) - 1)
        type_weights = self.message_type_weights[safe_message_types]  # (num_messages,)
        
        # Apply type weights to messages
        weighted_var_llrs = var_llrs * type_weights.unsqueeze(0)  # (batch_size, num_messages)
        
        # Get signs and magnitudes
        signs = torch.sign(weighted_var_llrs)  # (batch_size, num_messages)
        magnitudes = torch.abs(weighted_var_llrs)  # (batch_size, num_messages)
        
        # For each check node, gather connected messages and find minimum magnitude
        # Using matrix multiplication for batch processing
        # check_to_var_adjacency shape: (num_messages, num_messages)
        # magnitudes shape: (batch_size, num_messages)
        
        # Expand adjacency matrix for batch processing
        # Shape: (batch_size, num_messages, num_messages)
        batch_adj = check_to_var_adjacency.unsqueeze(0).expand(weighted_var_llrs.size(0), -1, -1)
        
        # Get connected magnitudes for each message in the batch
        # Shape: (batch_size, num_messages, num_messages)
        connected_magnitudes = magnitudes.unsqueeze(2) * batch_adj
        
        # Replace zeros with large value for min operation
        masked_magnitudes = torch.where(connected_magnitudes == 0, 
                                      torch.tensor(1e10, device=var_llrs.device), 
                                      connected_magnitudes)
        
        # Find minimum magnitude for each message (excluding self)
        # Shape: (batch_size, num_messages)
        min_magnitudes = masked_magnitudes.min(dim=1)[0]
        
        # Compute sign products for each message in the batch
        # Shape: (batch_size, num_messages, num_messages)
        sign_matrix = signs.unsqueeze(2) * batch_adj
        # Replace zeros with ones so they don't affect the product
        sign_matrix = torch.where(batch_adj == 0, torch.tensor(1.0, device=var_llrs.device), sign_matrix)
        # Compute product along the message dimension
        # Shape: (batch_size, num_messages)
        sign_products = torch.prod(sign_matrix, dim=1)
        
        # Combine signs and magnitudes
        # Shape: (batch_size, num_messages)
        updated_llrs = sign_products * min_magnitudes
        
        # Keep in scalar domain
        # Shape: (batch_size, num_messages, 1)
        return updated_llrs.unsqueeze(-1)
    
    def forward(self, check_messages, var_messages, message_types, check_to_var_adjacency):
        """Forward pass using scalar operations with type-specific weights"""
        return self.min_sum_update(var_messages, message_types, check_to_var_adjacency)


class MessageGNNDecoder(nn.Module):
    """
    Message-centered GNN decoder for LDPC codes with alternating layers.
    
    This decoder treats messages as nodes in a graph and updates them using graph neural networks.
    It implements alternating variable and check layers, with residual connections from the previous "n" variable layer to the current variable layer.
    The graph structure is defined by the Tanner graph of the LDPC code.
    """
    
    def __init__(self, num_messages, num_iterations=5, hidden_dim=16, num_message_types=1, num_of_residual_layers=2):
        super().__init__()
        
        self.num_messages = num_messages
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        self.num_message_types = num_message_types
        self.num_of_residual_layers = num_of_residual_layers
        
        # Variable-side GNN layers (scalar operations)
        self.var_gnn_layers = nn.ModuleList([
            VariableGNNLayer(num_message_types, hidden_dim=1, use_bp_op=True)  # Use hidden_dim=1 for scalar operations
            for _ in range(num_iterations)
        ])
        
        # Check-side GNN layers (scalar operations)
        self.check_gnn_layers = nn.ModuleList([
            CheckGNNLayer(num_message_types, hidden_dim=1, use_bp_op=True)  # Use hidden_dim=1 for scalar operations
            for _ in range(num_iterations)
        ])
        
        # Layer normalization for stabilizing training (operating on scalar values)
        self.var_layer_norms = nn.ModuleList([
            nn.LayerNorm(1)  # Normalize scalar values
            for _ in range(num_iterations)
        ])
        
        self.check_layer_norms = nn.ModuleList([
            nn.LayerNorm(1)  # Normalize scalar values
            for _ in range(num_iterations)
        ])
        
        print(f"\nModel Configuration:")
        print(f"Number of messages: {num_messages}")
        print(f"Number of iterations: {num_iterations}")
        print(f"Using scalar operations with type-specific weights")
        print(f"Number of message types: {num_message_types}")
        print(f"Number of residual layers: {num_of_residual_layers}")
        print(f"Total trainable parameters: {self.count_parameters():,}")
    
    def count_parameters(self):
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def output_mapping(self, final_messages, message_to_var_mapping, input_llr, batch_size, num_vars):
        """
        Maps final messages back to variable nodes and combines them with input LLRs.
        This is a parameter-free output layer that:
        1. Projects high-dimensional messages back to scalar LLRs
        2. Combines messages that share the same variable node
        3. Adds the original input LLR from QPSK demodulation
        4. Applies sigmoid to get soft bit values
        
        Args:
            final_messages: Tensor of shape [batch_size, num_messages, hidden_dim]
            message_to_var_mapping: Mapping from messages to variable nodes
            input_llr: Original input LLRs from QPSK demodulation
            batch_size: Number of codewords in batch
            num_vars: Number of variable nodes
            
        Returns:
            Tensor of shape [batch_size, num_vars] containing probabilities for each bit
        """
        # Messages are already in scalar form, no need for projection
        message_llrs = final_messages.squeeze(-1)  # (batch_size, num_messages)
        
        # Initialize output probabilities
        output_probs = torch.zeros(batch_size, num_vars, device=input_llr.device)
        
        # Combine messages for each variable node
        for b in range(batch_size):
            var_llrs = torch.zeros(num_vars, device=input_llr.device)
            
            # Sum messages going to each variable node
            for msg_idx in range(self.num_messages):
                var_idx = message_to_var_mapping[msg_idx, 0].item() if len(message_to_var_mapping.shape) > 1 else message_to_var_mapping[msg_idx].item()
                var_llrs[var_idx] += message_llrs[b, msg_idx]
            
            # Add original input LLRs
            combined_llrs = var_llrs + input_llr[b]
            
            # Convert to probabilities using sigmoid
            output_probs[b] = torch.sigmoid(combined_llrs)
        
        return output_probs

    def forward(self, input_llr, message_to_var_mapping, message_types=None, 
                var_to_check_adjacency=None, check_to_var_adjacency=None, ground_truth=None):
        """
        Forward pass of the Message GNN Decoder with alternating layers.
        Uses residual connections in variable layers by combining:
        1. check-to-var messages from previous check layer
        2. previous var-to-check messages from the queue
        All operations are performed in scalar domain with type-specific weights.
        """
        # Validate input shape
        if len(input_llr.shape) != 2:
            raise ValueError(f"Expected input_llr to be a 2D tensor (batch_size × num_vars), got shape {input_llr.shape}")
        
        batch_size = input_llr.shape[0]
        num_vars = input_llr.shape[1]

        # Initialize message features from input LLRs
        message_llrs = torch.zeros(batch_size, self.num_messages, device=input_llr.device)
        
        # Map input LLRs to messages
        for b in range(batch_size):
            var_indices = message_to_var_mapping[:, 0] if len(message_to_var_mapping.shape) > 1 else message_to_var_mapping[b]
            message_llrs[b] = input_llr[b][var_indices]
        
        # Keep messages in scalar domain
        var_message_features = message_llrs.unsqueeze(-1)  # shape: (batch_size, num_messages, 1)
        check_message_features = torch.zeros_like(var_message_features)  # shape: (batch_size, num_messages, 1)
        
        # Default message types if not provided
        if message_types is None:
            message_types = torch.zeros(self.num_messages, dtype=torch.long, device=input_llr.device)
        
        # Default adjacency matrices if not provided
        if var_to_check_adjacency is None:
            var_to_check_adjacency = torch.eye(self.num_messages, device=input_llr.device)
        if check_to_var_adjacency is None:
            check_to_var_adjacency = torch.eye(self.num_messages, device=input_llr.device)
        
        # Queue to store previous var-to-check messages
        residual_queue = []

        # Iterative message passing
        for i in range(self.num_iterations):
            # 1. Variable-to-Check Update
            var_input = var_message_features
            
            # Apply variable GNN layer (scalar operations)
            updated_var_features = self.var_gnn_layers[i](
                var_input,
                var_message_features,
                check_message_features,
                message_types,
                var_to_check_adjacency
            )
            
            # Add residual connections from previous var-to-check messages
            if i >= self.num_of_residual_layers and residual_queue:
                for prev_var_to_check in residual_queue:
                    updated_var_features = updated_var_features + prev_var_to_check
            
            # Apply layer norm (on scalar values)
            var_message_features = self.var_layer_norms[i](updated_var_features)
            
            # Update queue with current var-to-check messages
            residual_queue.append(var_message_features.clone())
            if len(residual_queue) > self.num_of_residual_layers:
                residual_queue.pop(0)
            
            # 2. Check-to-Variable Update (scalar operations)
            updated_check_features = self.check_gnn_layers[i](
                check_message_features,
                var_message_features,
                message_types,
                check_to_var_adjacency
            )
            
            # Apply layer norm (on scalar values)
            check_message_features = self.check_layer_norms[i](updated_check_features)
        
        # Final decoding using scalar messages
        return self.output_mapping(
            final_messages=check_message_features,
            message_to_var_mapping=message_to_var_mapping,
            input_llr=input_llr,
            batch_size=batch_size,
            num_vars=num_vars
        )
    
    # def decode(self, input_llr, message_to_var_mapping, message_types=None,
    #           var_to_check_adjacency=None, check_to_var_adjacency=None):
    #     """
    #     Convenience method for decoding without training.
        
    #     Args:
    #         input_llr (torch.Tensor): Input LLR values for each variable node
    #         message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
    #         message_types (torch.Tensor, optional): Types of each message for weight sharing
    #         var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for var-to-check messages
    #         check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-var messages
            
    #     Returns:
    #         torch.Tensor: Hard decoded bits (0 or 1)
    #     """
    #     with torch.no_grad():
    #         output_probs = self.forward(
    #             input_llr, 
    #             message_to_var_mapping, 
    #             message_types, 
    #             var_to_check_adjacency,
    #             check_to_var_adjacency
    #         )
            
    #         # Convert probabilities to bits (0 or 1)
    #         decoded_bits = (output_probs >= 0.5).float()
            
    #         return decoded_bits


class TannerToMessageGraph:
    """
    Converter from a Tanner graph to a message-centered graph.
    
    This class converts a parity-check matrix (representing a Tanner graph) to a message-centered graph
    where messages (edges in the Tanner graph) become nodes in the new graph.
    """
    
    def __init__(self, H):
        """
        Initialize the converter with a parity-check matrix.
        
        Args:
            H (torch.Tensor): Parity-check matrix
        """
        if isinstance(H, np.ndarray):
            H = torch.from_numpy(H).float()
        
        self.H = H
        self.m, self.n = H.shape
        
        # Map variables to their check neighbors and vice versa
        self.var_to_checks = {}  # var_idx -> list of check_idx
        self.check_to_vars = {}  # check_idx -> list of var_idx
        
        for var_idx in range(self.n):
            self.var_to_checks[var_idx] = []
            for check_idx in range(self.m):
                if H[check_idx, var_idx] == 1:
                    self.var_to_checks[var_idx].append(check_idx)
        
        for check_idx in range(self.m):
            self.check_to_vars[check_idx] = []
            for var_idx in range(self.n):
                if H[check_idx, var_idx] == 1:
                    self.check_to_vars[check_idx].append(var_idx)
        
        # Create message nodes
        self.messages = []  # list of (check_idx, var_idx) tuples
        self.message_to_var = {}  # message_idx -> var_idx
        self.message_to_check = {}  # message_idx -> check_idx
        self.var_to_messages = {}  # var_idx -> list of message_idx
        self.check_to_messages = {}  # check_idx -> list of message_idx
        
        # Initialize empty lists
        for var_idx in range(self.n):
            self.var_to_messages[var_idx] = []
        
        for check_idx in range(self.m):
            self.check_to_messages[check_idx] = []
        
        # Create message nodes and their mappings
        msg_idx = 0
        for check_idx in range(self.m):
            for var_idx in range(self.n):
                if H[check_idx, var_idx] == 1:
                    # Add message
                    self.messages.append((check_idx, var_idx))
                    
                    # Map message to its variable and check
                    self.message_to_var[msg_idx] = var_idx
                    self.message_to_check[msg_idx] = check_idx
                    
                    # Map variable and check to their messages
                    self.var_to_messages[var_idx].append(msg_idx)
                    self.check_to_messages[check_idx].append(msg_idx)
                    
                    msg_idx += 1
        # Print the first five message nodes
        print("[debug] First five message nodes (check_idx, var_idx):", self.messages[:5])

        # Print the first five variable-to-messages mappings
        print("[debug] First five variable-to-messages matrix:", {var_idx: self.var_to_messages[var_idx][:5] for var_idx in range(self.n)})

        # Print the first five check-to-messages mappings
        print("[debug] First five check-to-messages matrix:", {check_idx: self.check_to_messages[check_idx][:5] for check_idx in range(self.m)})

        self.num_messages = len(self.messages)
        
        # Default message types (all same type)
        self.message_types = torch.zeros(self.num_messages, dtype=torch.long)
        
        # Create adjacency matrices for message passing
        self.create_adjacency_matrices()
    
    def create_adjacency_matrices(self):
        """
        Create adjacency matrices for message passing in the message-centered graph.
        
        Creates two adjacency matrices:
        1. var_to_check_adjacency: For messages connected through the same variable node
        2. check_to_var_adjacency: For messages connected through the same check node
        """
        # Create adjacency matrices for message passing
        self.var_to_check_adjacency = torch.zeros((self.num_messages, self.num_messages))
        self.check_to_var_adjacency = torch.zeros((self.num_messages, self.num_messages))
        
        # Set up var-to-check adjacency (messages connected through same variable)
        for var_idx in range(self.n):
            messages_index = self.var_to_messages[var_idx]
            for i, msg1 in enumerate(messages_index):
                for j, msg2 in enumerate(messages_index):
                    if i != j:  # Don't connect message to itself
                        self.var_to_check_adjacency[msg1, msg2] = 1.0 # this means that message with index msg1 is connected to the message with index msg2 through the same variable
        print(f"[debug] var_to_check_adjacency: {self.var_to_check_adjacency}")

        # Set up check-to-var adjacency (messages connected through same check)
        for check_idx in range(self.m):
            messages_index = self.check_to_messages[check_idx]
            for i, msg1 in enumerate(messages_index):
                for j, msg2 in enumerate(messages_index):
                    if i != j:  # Don't connect message to itself
                        self.check_to_var_adjacency[msg1, msg2] = 1.0 # this means that message with index msg1 is connected to the message with index msg2 through the same check
        print(f"[debug] check_to_var_adjacency: {self.check_to_var_adjacency}")

    def get_message_to_var_mapping(self):
        """
        Get a tensor mapping each message to its variable node.
        
        Returns:
            torch.Tensor: Tensor with shape (num_messages,) where each entry is the variable index
        """
        mapping = torch.zeros(self.num_messages, dtype=torch.long)
        for msg_idx in range(self.num_messages):
            mapping[msg_idx] = self.message_to_var[msg_idx]
        return mapping
    
    def get_message_to_check_mapping(self):
        """
        Get a tensor mapping each message to its check node.
        
        Returns:
            torch.Tensor: Tensor with shape (num_messages,) where each entry is the check index
        """
        mapping = torch.zeros(self.num_messages, dtype=torch.long)
        for msg_idx in range(self.num_messages):
            mapping[msg_idx] = self.message_to_check[msg_idx]
        return mapping
    
    def get_var_to_messages_mapping(self):
        """
        Get a packed tensor mapping each variable to its connected messages.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Variable-to-message indices and a tensor indicating where each variable's list starts
        """
        all_indices = []
        offsets = [0]
        
        for var_idx in range(self.n):
            all_indices.extend(self.var_to_messages[var_idx])
            offsets.append(len(all_indices))
        
        return torch.tensor(all_indices, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)
    
    def get_check_to_messages_mapping(self):
        """
        Get a packed tensor mapping each check to its connected messages.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Check-to-message indices and a tensor indicating where each check's list starts
        """
        all_indices = []
        offsets = [0]
        
        for check_idx in range(self.m):
            all_indices.extend(self.check_to_messages[check_idx])
            offsets.append(len(all_indices))
        
        return torch.tensor(all_indices, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def create_message_gnn_decoder(H, base_graph=None, lifting_factor=None, num_iterations=5, hidden_dim=16, num_of_residual_layers=2):
    """
    Create a Message GNN Decoder based on a parity-check matrix and optionally a base graph.
    
    Args:
        H (torch.Tensor or np.ndarray): Parity-check matrix
        base_graph (torch.Tensor or np.ndarray, optional): Base graph for weight sharing
        lifting_factor (int, optional): Lifting factor used to create H from base_graph
        num_iterations (int, optional): Number of decoding iterations
        hidden_dim (int, optional): Hidden dimension for message features
        num_of_residual_layers (int, optional): Number of previous layers to use for residual connections
        
    Returns:
        MessageGNNDecoder: The decoder model
        TannerToMessageGraph: A converter for setting up the message graph
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float()
    
    if base_graph is not None and isinstance(base_graph, np.ndarray):
        base_graph = torch.from_numpy(base_graph).float()
    
    # Convert Tanner graph to message graph
    converter = TannerToMessageGraph(H)
    
    # Get the number of message nodes (edges in the Tanner graph)
    num_messages = converter.num_messages
    print(f"Number of message nodes: {num_messages}")
    
    # Get the number of message types based on the base graph
    num_message_types = 1  # Default: all messages have the same type
    
    if base_graph is not None and lifting_factor is not None:
        # Create message type mapping based on the base graph
        message_types = create_message_type_mapping(H, base_graph, lifting_factor)
        num_message_types = int(message_types.max().item()) + 1
        print(f"Number of message types from base graph: {num_message_types}")
        
        # Store message types in the converter for easy access
        converter.message_types = message_types
    else:
        print("No base graph provided, all messages will have the same type.")
        converter.message_types = torch.zeros(num_messages, dtype=torch.long)
    
    # Create the decoder with the appropriate number of message types
    decoder = MessageGNNDecoder(
        num_messages=num_messages,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
        num_message_types=num_message_types,
        num_of_residual_layers=num_of_residual_layers
    )
    
    print(f"Created Message GNN Decoder with {num_iterations} iterations and {hidden_dim} hidden dimensions")
    print(f"Using weight sharing with {num_message_types} different message types based on the base graph")
    print(f"Using {num_of_residual_layers} previous layers for residual connections")
    
    return decoder, converter


def create_message_type_mapping(H, base_graph, lifting_factor):
    """
    Create a mapping of message types based on the base graph structure.
    Messages corresponding to the same edge in the base graph will have the same type.
    
    Args:
        H (torch.Tensor): Expanded parity-check matrix
        base_graph (torch.Tensor): Base graph
        lifting_factor (int): Lifting factor used to expand the base graph
        
    Returns:
        torch.Tensor: Message type indices for each message in the expanded graph
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float()
    
    if isinstance(base_graph, np.ndarray):
        base_graph = torch.from_numpy(base_graph).float()
    
    # Get dimensions
    base_m, base_n = base_graph.shape
    m, n = H.shape
    
    # Check if dimensions are consistent with lifting
    if m != base_m * lifting_factor or n != base_n * lifting_factor:
        raise ValueError(f"Expanded matrix dimensions {m}x{n} don't match expected dimensions from base graph {base_m * lifting_factor}x{base_n * lifting_factor}")
    
    # Create converter for the expanded graph
    converter = TannerToMessageGraph(H)
    num_messages = converter.num_messages
    
    # Initialize message types
    message_types = torch.zeros(num_messages, dtype=torch.long)
    
    # Create a mapping from the expanded graph to the base graph
    # For each edge (message) in the expanded graph, find which edge in the base graph it corresponds to
    message_idx = 0
    type_idx = 0
    type_mapping = {}  # Maps (base_row, base_col) to a type index
    
    for row in range(m):
        base_row = row // lifting_factor
        
        for col in range(n):
            if H[row, col] == 1:
                base_col = col // lifting_factor
                
                # The type of this message is determined by its position in the base graph
                base_edge = (base_row, base_col)
                
                # If this base edge hasn't been seen before, assign a new type
                if base_edge not in type_mapping and base_graph[base_row, base_col] >= 0:
                    type_mapping[base_edge] = base_graph[base_row, base_col]
                    type_idx += 1
                
                # Assign the type to this message if the corresponding base graph entry greater than 0
                if base_graph[base_row, base_col] >= 0:
                    message_types[message_idx] = type_mapping[base_edge]
                
                message_idx += 1
    
    print(f"[debug] base graph: {base_graph.shape}")
    print(f"[debug] Created {type_idx} message types from base graph with {(base_graph != -1).sum().item()} edges")
    return message_types


def create_variable_index_tensor(H, converter):
    """
    Create a tensor mapping variable nodes to message nodes.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        converter (TannerToMessageGraph): Converter object
        
    Returns:
        torch.Tensor: Variable index tensor
    """
    num_variables = H.shape[1]
    device = H.device
    
    # Find max number of messages per variable
    max_messages_per_var = max(len(msgs) for var_idx, msgs in converter.var_to_messages.items())
    
    # Create variable-to-message mapping
    variable_index_tensor = torch.full((num_variables, max_messages_per_var), -1, dtype=torch.long, device=device)
    
    # Fill the tensor
    for var_idx, msg_indices in converter.var_to_messages.items():
        for i, msg_idx in enumerate(msg_indices):
            variable_index_tensor[var_idx, i] = msg_idx
    
    return variable_index_tensor


def create_check_index_tensor(H, message_type_map=None):
    """
    Create a tensor mapping check nodes to message nodes.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        message_type_map (dict, optional): Mapping from messages to types
        
    Returns:
        torch.Tensor: Check index tensor
    """
    num_checks, num_variables = H.shape
    device = H.device
    
    # Create check-to-variable mapping
    check_to_var = [[] for _ in range(num_checks)]
    
    # Fill the mapping
    for check_idx in range(num_checks):
        for var_idx in range(num_variables):
            if H[check_idx, var_idx] != 0:
                check_to_var[check_idx].append(var_idx)
    
    # Find max number of variables per check
    max_vars_per_check = max(len(vars) for vars in check_to_var)
    
    # Create check index tensor
    check_index_tensor = torch.full((num_checks, max_vars_per_check), -1, dtype=torch.long, device=device)
    
    # Fill the tensor
    for check_idx, var_indices in enumerate(check_to_var):
        for i, var_idx in enumerate(var_indices):
            check_index_tensor[check_idx, i] = var_idx
    
    return check_index_tensor 