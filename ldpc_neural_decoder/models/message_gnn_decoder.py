"""
Message-Centered GNN LDPC Decoder

This module implements a Graph Neural Network (GNN) based LDPC decoder
where messages are represented as nodes in the GNN, and edges connect
messages that share the same variable or check node in the Tanner graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class MessageGNNLayer(nn.Module):
    """
    Message-Centered GNN Layer for LDPC decoding.
    
    In this layer, each node represents a message (edge in the Tanner graph),
    and edges connect messages that share the same variable or check node.
    """
    def __init__(self, num_message_types=1, hidden_dim=64):
        """
        Initialize the Message GNN Layer.
        
        Args:
            num_message_types (int): Number of different message types for weight sharing
            hidden_dim (int): Dimension of hidden representations
        """
        super(MessageGNNLayer, self).__init__()
        
        # Message type specific embeddings
        self.message_type_embeddings = nn.Parameter(torch.randn(num_message_types, hidden_dim))
        
        # Neural networks for message updates
        self.var_to_check_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.check_to_var_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, 1)
    
    def forward(self, message_features, message_types, var_to_check_adjacency, check_to_var_adjacency):
        """
        Perform message update in the GNN.
        
        Args:
            message_features (torch.Tensor): Features of each message node of shape (batch_size, num_messages, feature_dim)
            message_types (torch.Tensor): Type indices for each message of shape (num_messages,)
            var_to_check_adjacency (torch.Tensor): Adjacency matrix for variable-to-check messages of shape (num_messages, num_messages)
            check_to_var_adjacency (torch.Tensor): Adjacency matrix for check-to-variable messages of shape (num_messages, num_messages)
            
        Returns:
            torch.Tensor: Updated message features of shape (batch_size, num_messages, feature_dim)
        """
        batch_size, num_messages, feature_dim = message_features.shape
        
        # Get message type embeddings
        type_embeddings = self.message_type_embeddings[message_types]  # (num_messages, hidden_dim)
        
        # Expand type embeddings to match batch dimension
        type_embeddings = type_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_messages, hidden_dim)
        
        # Combine message features with type embeddings
        combined_features = message_features + type_embeddings  # (batch_size, num_messages, hidden_dim)
        
        # Variable-to-check message update
        # Aggregate messages from neighboring variable nodes
        var_to_check_messages = torch.bmm(var_to_check_adjacency.unsqueeze(0).expand(batch_size, -1, -1), combined_features)
        
        # Concatenate with current features
        var_to_check_input = torch.cat([combined_features, var_to_check_messages], dim=2)
        
        # Apply update network
        var_to_check_updated = self.var_to_check_update(var_to_check_input)
        
        # Check-to-variable message update
        # Aggregate messages from neighboring check nodes
        check_to_var_messages = torch.bmm(check_to_var_adjacency.unsqueeze(0).expand(batch_size, -1, -1), combined_features)
        
        # Concatenate with current features
        check_to_var_input = torch.cat([combined_features, check_to_var_messages], dim=2)
        
        # Apply update network
        check_to_var_updated = self.check_to_var_update(check_to_var_input)
        
        # Combine updates
        updated_features = var_to_check_updated + check_to_var_updated
        
        return updated_features
    
    def decode_messages(self, message_features):
        """
        Decode message features to LLR values.
        
        Args:
            message_features (torch.Tensor): Features of each message node of shape (batch_size, num_messages, feature_dim)
            
        Returns:
            torch.Tensor: Decoded LLR values of shape (batch_size, num_messages)
        """
        # Project to scalar values
        llr_values = self.output_projection(message_features).squeeze(-1)
        
        return llr_values


class MessageGNNDecoder(nn.Module):
    """
    Message-Centered GNN LDPC Decoder.
    
    This decoder represents messages as nodes in a GNN, where edges connect
    messages that share the same variable or check node in the Tanner graph.
    """
    def __init__(self, num_messages, num_iterations=5, hidden_dim=64, num_message_types=1):
        """
        Initialize the Message GNN Decoder.
        
        Args:
            num_messages (int): Number of messages (edges) in the Tanner graph
            num_iterations (int): Number of decoding iterations
            hidden_dim (int): Dimension of hidden representations
            num_message_types (int): Number of different message types for weight sharing
        """
        super(MessageGNNDecoder, self).__init__()
        
        self.num_messages = num_messages
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_embedding = nn.Linear(1, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            MessageGNNLayer(num_message_types, hidden_dim)
            for _ in range(num_iterations)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, input_llr, message_to_var_mapping, message_types=None, 
                var_to_check_adjacency=None, check_to_var_adjacency=None, ground_truth=None):
        """
        Forward pass of the Message GNN Decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_variables)
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes of shape (num_messages, num_variables)
            message_types (torch.Tensor, optional): Type indices for each message of shape (num_messages,)
            var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for variable-to-check messages of shape (num_messages, num_messages)
            check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-variable messages of shape (num_messages, num_messages)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_variables)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_variables)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        batch_size, num_variables = input_llr.shape
        device = input_llr.device
        
        # If message types not provided, assume all messages are of the same type
        if message_types is None:
            message_types = torch.zeros(self.num_messages, dtype=torch.long, device=device)
        
        # If adjacency matrices not provided, create dummy ones
        if var_to_check_adjacency is None:
            var_to_check_adjacency = torch.eye(self.num_messages, device=device)
        
        if check_to_var_adjacency is None:
            check_to_var_adjacency = torch.eye(self.num_messages, device=device)
        
        # Initialize message features from input LLRs
        # Map variable node LLRs to message nodes
        message_llrs = torch.matmul(message_to_var_mapping, input_llr.unsqueeze(-1)).squeeze(-1)  # (num_messages, batch_size)
        message_llrs = message_llrs.transpose(0, 1)  # (batch_size, num_messages)
        
        # Embed message LLRs
        message_features = self.input_embedding(message_llrs.unsqueeze(-1))  # (batch_size, num_messages, hidden_dim)
        
        # Iterative GNN decoding
        for i in range(self.num_iterations):
            # Update message features using GNN layer
            message_features = self.gnn_layers[i](
                message_features, 
                message_types, 
                var_to_check_adjacency, 
                check_to_var_adjacency
            )
        
        # Decode final message features to LLR values
        final_message_llrs = self.gnn_layers[-1].decode_messages(message_features)  # (batch_size, num_messages)
        
        # Aggregate message LLRs to variable nodes
        # Transpose message_to_var_mapping for aggregation
        var_to_message_mapping = message_to_var_mapping.transpose(0, 1)  # (num_variables, num_messages)
        
        # Normalize the mapping for weighted average
        var_to_message_mapping = var_to_message_mapping / (var_to_message_mapping.sum(dim=1, keepdim=True) + 1e-10)
        
        # Aggregate messages to variables
        variable_llrs = torch.matmul(final_message_llrs, var_to_message_mapping.t())  # (batch_size, num_variables)
        
        # Add input LLRs
        combined_llrs = variable_llrs + input_llr
        
        # Convert to soft bits
        soft_bits = torch.sigmoid(combined_llrs)
        
        # Compute loss if ground truth is provided
        if ground_truth is not None:
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy(soft_bits, ground_truth, reduction='none')
            
            # Apply max function over the loss vector (for FER minimization)
            max_loss = torch.max(loss, dim=1).values
            
            return soft_bits, max_loss
        
        return soft_bits, None
    
    def decode(self, input_llr, message_to_var_mapping, message_types=None, 
               var_to_check_adjacency=None, check_to_var_adjacency=None):
        """
        Decode input LLRs to hard bit decisions.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_variables)
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes of shape (num_messages, num_variables)
            message_types (torch.Tensor, optional): Type indices for each message of shape (num_messages,)
            var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for variable-to-check messages of shape (num_messages, num_messages)
            check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-variable messages of shape (num_messages, num_messages)
            
        Returns:
            torch.Tensor: Hard bit decisions (0 or 1) of shape (batch_size, num_variables)
        """
        # Get soft bit values
        soft_bits, _ = self.forward(
            input_llr, 
            message_to_var_mapping, 
            message_types, 
            var_to_check_adjacency, 
            check_to_var_adjacency
        )
        
        # Convert to hard decisions
        hard_bits = (soft_bits > 0.5).float()
        
        return hard_bits


class TannerToMessageGraph:
    """
    Utility class to convert a Tanner graph to a message-centered graph.
    
    This class creates the necessary mappings and adjacency matrices for
    the message-centered GNN decoder.
    """
    def __init__(self, H):
        """
        Initialize the Tanner to Message Graph converter.
        
        Args:
            H (torch.Tensor): Parity-check matrix of shape (num_checks, num_variables)
        """
        self.H = H
        self.num_checks, self.num_variables = H.shape
        
        # Create message list and mappings
        self.messages, self.var_to_messages, self.check_to_messages = self._create_message_mappings()
        
        # Create adjacency matrices
        self.var_to_check_adjacency, self.check_to_var_adjacency = self._create_adjacency_matrices()
        
        # Create message to variable mapping
        self.message_to_var_mapping = self._create_message_to_var_mapping()
    
    def _create_message_mappings(self):
        """
        Create mappings between variables/checks and messages.
        
        Returns:
            tuple: (messages, var_to_messages, check_to_messages)
                - messages: List of (var_idx, check_idx) tuples
                - var_to_messages: Dictionary mapping variable indices to message indices
                - check_to_messages: Dictionary mapping check indices to message indices
        """
        messages = []
        var_to_messages = {i: [] for i in range(self.num_variables)}
        check_to_messages = {i: [] for i in range(self.num_checks)}
        
        # Iterate through the parity-check matrix
        for c in range(self.num_checks):
            for v in range(self.num_variables):
                if self.H[c, v] == 1:
                    # Create a message (edge in the Tanner graph)
                    message_idx = len(messages)
                    messages.append((v, c))
                    
                    # Update mappings
                    var_to_messages[v].append(message_idx)
                    check_to_messages[c].append(message_idx)
        
        return messages, var_to_messages, check_to_messages
    
    def _create_adjacency_matrices(self):
        """
        Create adjacency matrices for the message-centered graph.
        
        Returns:
            tuple: (var_to_check_adjacency, check_to_var_adjacency)
                - var_to_check_adjacency: Adjacency matrix for variable-to-check messages
                - check_to_var_adjacency: Adjacency matrix for check-to-variable messages
        """
        num_messages = len(self.messages)
        device = self.H.device
        
        # Initialize adjacency matrices
        var_to_check_adjacency = torch.zeros((num_messages, num_messages), device=device)
        check_to_var_adjacency = torch.zeros((num_messages, num_messages), device=device)
        
        # Fill adjacency matrices
        for v in range(self.num_variables):
            # Messages that share the same variable node
            var_messages = self.var_to_messages[v]
            for i in var_messages:
                for j in var_messages:
                    if i != j:
                        var_to_check_adjacency[i, j] = 1.0
        
        for c in range(self.num_checks):
            # Messages that share the same check node
            check_messages = self.check_to_messages[c]
            for i in check_messages:
                for j in check_messages:
                    if i != j:
                        check_to_var_adjacency[i, j] = 1.0
        
        # Normalize adjacency matrices
        var_to_check_adjacency = self._normalize_adjacency(var_to_check_adjacency)
        check_to_var_adjacency = self._normalize_adjacency(check_to_var_adjacency)
        
        return var_to_check_adjacency, check_to_var_adjacency
    
    def _normalize_adjacency(self, adjacency):
        """
        Normalize adjacency matrix for graph convolution.
        
        Args:
            adjacency (torch.Tensor): Adjacency matrix
            
        Returns:
            torch.Tensor: Normalized adjacency matrix
        """
        # Add self-loops
        adjacency = adjacency + torch.eye(adjacency.shape[0], device=adjacency.device)
        
        # Compute degree matrix
        degree = adjacency.sum(dim=1)
        
        # Compute D^(-1/2) A D^(-1/2)
        degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        normalized_adjacency = torch.mm(torch.mm(degree_inv_sqrt, adjacency), degree_inv_sqrt)
        
        return normalized_adjacency
    
    def _create_message_to_var_mapping(self):
        """
        Create mapping from messages to variable nodes.
        
        Returns:
            torch.Tensor: Mapping matrix of shape (num_messages, num_variables)
        """
        num_messages = len(self.messages)
        device = self.H.device
        
        # Initialize mapping matrix
        message_to_var_mapping = torch.zeros((num_messages, self.num_variables), device=device)
        
        # Fill mapping matrix
        for i, (v, _) in enumerate(self.messages):
            message_to_var_mapping[i, v] = 1.0
        
        return message_to_var_mapping
    
    def get_message_types(self, base_graph=None, Z=None):
        """
        Get message type indices based on the base graph structure.
        
        Args:
            base_graph (torch.Tensor, optional): Base graph matrix with shift values
            Z (int, optional): Lifting factor
            
        Returns:
            torch.Tensor: Message type indices of shape (num_messages,)
        """
        num_messages = len(self.messages)
        device = self.H.device
        
        if base_graph is None or Z is None:
            # If base graph not provided, all messages are of the same type
            return torch.zeros(num_messages, dtype=torch.long, device=device)
        
        # Map each message to its corresponding cell in the base graph
        message_types = torch.zeros(num_messages, dtype=torch.long, device=device)
        
        # Create mapping from (row, col) in expanded graph to (base_row, base_col)
        for i, (v, c) in enumerate(self.messages):
            base_row = c // Z
            base_col = v // Z
            
            # Get shift value from base graph
            shift = base_graph[base_row, base_col].item()
            
            # Use shift value as message type
            message_types[i] = int(shift) if shift >= 0 else 0
        
        return message_types


def create_message_gnn_decoder(H, num_iterations=5, hidden_dim=64, base_graph=None, Z=None):
    """
    Create a Message GNN Decoder from a parity-check matrix.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        num_iterations (int): Number of decoding iterations
        hidden_dim (int): Dimension of hidden representations
        base_graph (torch.Tensor, optional): Base graph matrix with shift values
        Z (int, optional): Lifting factor
        
    Returns:
        tuple: (decoder, converter)
            - decoder: MessageGNNDecoder instance
            - converter: TannerToMessageGraph instance
    """
    # Convert Tanner graph to message graph
    converter = TannerToMessageGraph(H)
    
    # Get number of messages
    num_messages = len(converter.messages)
    
    # Get number of message types
    if base_graph is not None and Z is not None:
        # Count unique shift values in base graph
        unique_shifts = set()
        for i in range(base_graph.shape[0]):
            for j in range(base_graph.shape[1]):
                shift = base_graph[i, j].item()
                if shift >= 0:
                    unique_shifts.add(int(shift))
        num_message_types = len(unique_shifts) if unique_shifts else 1
    else:
        num_message_types = 1
    
    # Create decoder
    decoder = MessageGNNDecoder(
        num_messages=num_messages,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
        num_message_types=num_message_types
    )
    
    return decoder, converter 