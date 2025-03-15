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
        device = message_features.device
        
        # Ensure message_types has the right length
        if len(message_types) != num_messages:
            # If message_types has a different length than num_messages,
            # we need to pad or truncate it
            if len(message_types) < num_messages:
                # Pad with zeros
                padded_message_types = torch.zeros(num_messages, dtype=torch.long, device=device)
                padded_message_types[:len(message_types)] = message_types
                message_types = padded_message_types
            else:
                # Truncate
                message_types = message_types[:num_messages]
        
        # Ensure message_types doesn't exceed the number of message types we have embeddings for
        safe_message_types = torch.clamp(message_types, 0, self.message_type_embeddings.size(0) - 1)
        
        # Get embeddings for each message type
        type_embeddings = self.message_type_embeddings[safe_message_types]  # (num_messages, hidden_dim)
        
        # Expand type embeddings to match batch dimension
        type_embeddings = type_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_messages, hidden_dim)
        
        # Combine message features with type embeddings
        combined_features = message_features + type_embeddings  # (batch_size, num_messages, hidden_dim)
        
        # Check if adjacency matrices match the number of messages
        if var_to_check_adjacency.size(0) != num_messages or var_to_check_adjacency.size(1) != num_messages:
            # Resize adjacency matrices to match num_messages
            new_var_to_check = torch.zeros((num_messages, num_messages), device=device)
            new_check_to_var = torch.zeros((num_messages, num_messages), device=device)
            
            # Copy as much as possible from the original matrices
            min_size = min(var_to_check_adjacency.size(0), num_messages)
            new_var_to_check[:min_size, :min_size] = var_to_check_adjacency[:min_size, :min_size]
            new_check_to_var[:min_size, :min_size] = check_to_var_adjacency[:min_size, :min_size]
            
            var_to_check_adjacency = new_var_to_check
            check_to_var_adjacency = new_check_to_var
        
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
        
        # Ensure the output has shape (batch_size, num_messages)
        if len(llr_values.shape) == 1:
            # If we have a single batch, add batch dimension
            llr_values = llr_values.unsqueeze(0)
        elif llr_values.shape[0] != message_features.shape[0]:
            # If dimensions are transposed, fix them
            llr_values = llr_values.transpose(0, 1)
            
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
            input_llr (torch.Tensor): Input LLR values for each variable node
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
            message_types (torch.Tensor, optional): Types of each message for weight sharing
            var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for var-to-check messages
            check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-var messages
            ground_truth (torch.Tensor, optional): Ground truth codeword for training
            
        Returns:
            torch.Tensor: Decoded codeword probabilities
        """
        print("\n===== Starting Message GNN Decoder Forward Pass =====")
        print(f"Input LLR shape: {input_llr.shape}")
        print(f"Message to var mapping shape: {message_to_var_mapping.shape}")
        
        batch_size = input_llr.shape[0]
        num_vars = input_llr.shape[1]
        
        # Initialize message features directly from input LLRs without embedding
        # First create a tensor to hold the raw LLR values for each message
        message_llrs = torch.zeros(batch_size, self.num_messages, device=input_llr.device)
        
        # Map input LLRs to messages directly
        for b in range(batch_size):
            # Check if message_to_var_mapping is 2D
            if len(message_to_var_mapping.shape) > 1:
                # If it's 2D, we need to extract just the first column
                # This assumes the first column contains the variable indices
                var_indices = message_to_var_mapping[:, 0]
            else:
                # If it's already 1D, use it directly
                var_indices = message_to_var_mapping
            
            # Copy the LLR values directly from variable nodes to message nodes
            message_llrs[b] = input_llr[b][var_indices]
        
        print(f"Initialized message LLRs shape: {message_llrs.shape}")
        
        # Since we're using a higher hidden_dim, we'll use the input_embedding to transform
        # the scalar LLRs into higher-dimensional feature vectors
        message_features = self.input_embedding(message_llrs.unsqueeze(-1))
        
        print(f"Embedded message features shape: {message_features.shape}")
        
        # If message types not provided, use default (all same type)
        if message_types is None:
            message_types = torch.zeros(self.num_messages, dtype=torch.long, device=input_llr.device)
        
        # Store all intermediate message features for residual connections
        all_message_features = [message_features]
        
        # Iterative message passing
        for i in range(self.num_iterations):
            print(f"\n----- Iteration {i+1}/{self.num_iterations} -----")
            
            # Apply GNN layer
            new_message_features = self.gnn_layers[i](
                all_message_features[-1], 
                message_types,
                var_to_check_adjacency, 
                check_to_var_adjacency
            )
            
            print(f"Updated message features shape: {new_message_features.shape}")
            
            # Apply residual connection if not the first layer
            if i > 0:
                new_message_features = new_message_features + all_message_features[-1]
                print("Applied residual connection")
            
            all_message_features.append(new_message_features)
        
        # Decode final messages to get codeword probabilities
        print("\n----- Final Decoding Stage -----")
        final_messages = all_message_features[-1]
        decoded_llrs = self.gnn_layers[-1].decode_messages(final_messages)
        
        # Aggregate messages for each variable node
        output_probs = torch.zeros(batch_size, num_vars, device=input_llr.device)
        
        print("Using SUM aggregation for mapping message nodes to variable bits")
        
        for b in range(batch_size):
            # Initialize tensor to store sum of messages for each variable node
            var_llrs = torch.zeros(num_vars, device=input_llr.device)
            
            # Count number of messages per variable node for potential normalization
            message_counts = torch.zeros(num_vars, device=input_llr.device)
            
            # Sum all messages going to each variable node
            for msg_idx in range(self.num_messages):
                # Get the variable index for this message
                if len(message_to_var_mapping.shape) > 1:
                    # If message_to_var_mapping is 2D, get the first column
                    var_idx = message_to_var_mapping[msg_idx, 0].item()
                else:
                    # If it's 1D, use it directly
                    var_idx = message_to_var_mapping[msg_idx].item()
                
                var_llrs[var_idx] += decoded_llrs[b, msg_idx]
                message_counts[var_idx] += 1
            
            # Add input LLRs to the summed messages
            combined_llrs = var_llrs + input_llr[b]
            
            # Convert to probabilities
            output_probs[b] = combined_llrs
        
        print(f"Output LLRs shape: {output_probs.shape}")
        print(f"Output LLRs range: [{output_probs.min().item():.4f}, {output_probs.max().item():.4f}]")
        
        # Apply sigmoid to get probabilities
        output_probs = torch.sigmoid(output_probs)
        
        print(f"Final output probabilities range: [{output_probs.min().item():.4f}, {output_probs.max().item():.4f}]")
        print("===== Completed Message GNN Decoder Forward Pass =====\n")
        
        # If ground truth is provided, compute loss for training
        if ground_truth is not None:
            loss = F.binary_cross_entropy(output_probs, ground_truth.float())
            return output_probs, loss
        
        return output_probs
    
    def decode(self, input_llr, message_to_var_mapping, message_types=None, 
               var_to_check_adjacency=None, check_to_var_adjacency=None):
        """
        Decode the input LLRs to get the estimated codeword.
        
        Args:
            input_llr (torch.Tensor): Input LLR values for each variable node
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
            message_types (torch.Tensor, optional): Types of each message for weight sharing
            var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for var-to-check messages
            check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-var messages
            
        Returns:
            torch.Tensor: Hard-decision decoded codeword
        """
        print("\n===== Starting Message GNN Decoder Decoding =====")
        print(f"Input LLR shape: {input_llr.shape}")
        
        # Get soft bit probabilities
        soft_bits = self.forward(
            input_llr, 
            message_to_var_mapping, 
            message_types, 
            var_to_check_adjacency, 
            check_to_var_adjacency
        )
        
        # Hard decision decoding
        hard_bits = (soft_bits > 0.5).float()
        
        print(f"Hard decision bits shape: {hard_bits.shape}")
        print(f"Sample of decoded bits: {hard_bits[0, :10]}")
        print("===== Completed Message GNN Decoder Decoding =====\n")
        
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
        
        # Get unique shift values for mapping to indices
        unique_shifts = []
        for i in range(base_graph.shape[0]):
            for j in range(base_graph.shape[1]):
                shift = base_graph[i, j].item()
                if shift >= 0 and int(shift) not in unique_shifts:
                    unique_shifts.append(int(shift))
        
        # Create mapping from shift values to indices (starting from 0)
        shift_to_index = {shift: idx for idx, shift in enumerate(sorted(unique_shifts))}
        
        # Map each message to its corresponding cell in the base graph
        message_types = torch.zeros(num_messages, dtype=torch.long, device=device)
        
        # Create mapping from (row, col) in expanded graph to (base_row, base_col)
        for i, (v, c) in enumerate(self.messages):
            base_row = c // Z
            base_col = v // Z
            
            # Get shift value from base graph
            shift = base_graph[base_row, base_col].item()
            
            # Map shift value to index
            if shift >= 0:
                message_types[i] = shift_to_index[int(shift)]
            else:
                message_types[i] = 0  # Default to 0 for -1 shifts
        
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
        # Count unique shift values in base graph (excluding -1)
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


class CustomVariableMessageGNNLayer(MessageGNNLayer):
    """
    Custom Message-Centered GNN Layer for LDPC decoding with traditional min-sum variable update.
    
    This layer replaces the neural network for variable node updates with a traditional
    min-sum algorithm that includes residual connections.
    """
    def __init__(self, num_message_types=1, hidden_dim=64, depth_L=3):
        """
        Initialize the Custom Variable Message GNN Layer.
        
        Args:
            num_message_types (int): Number of different message types for weight sharing
            hidden_dim (int): Dimension of hidden representations
            depth_L (int): Depth of residual connections
        """
        super(CustomVariableMessageGNNLayer, self).__init__(num_message_types, hidden_dim)
        
        # Parameters for variable layer update with residual connections
        self.depth_L = depth_L
        self.w_ch = nn.Parameter(torch.ones(1))
        self.w_res = nn.Parameter(torch.ones(depth_L))
        
        # Storage for previous variable layer outputs for residual connections
        self.previous_VL_storage = deque(maxlen=depth_L + 1)
        
    def variable_layer_update(self, input_mapping_LLR, check_to_variable_messages, variable_index_tensor, iteration):
        """
        Update variable node messages using min-sum algorithm.
        
        Args:
            input_mapping_LLR: LLR values from the channel
            check_to_variable_messages: Messages from check nodes to variable nodes
            variable_index_tensor: Tensor mapping messages to variable nodes
            iteration: Current iteration number
            
        Returns:
            Updated variable-to-check messages
        """
        print(f"\n----- Variable Layer Update (Iteration {iteration+1}) -----")
        print(f"Input LLR shape: {input_mapping_LLR.shape}")
        print(f"Check-to-variable messages shape: {check_to_variable_messages.shape}")
        print(f"Variable index tensor shape: {variable_index_tensor.shape}")
        
        batch_size = input_mapping_LLR.shape[0]
        num_messages = variable_index_tensor.shape[0]
        
        # Initialize output messages
        variable_to_check_messages = torch.zeros(batch_size, num_messages, device=input_mapping_LLR.device)
        
        # Process each message
        for msg_idx in range(num_messages):
            var_idx = variable_index_tensor[msg_idx, 0].item()
            incoming_msg_indices = variable_index_tensor[msg_idx, 1:].long()
            
            # Get channel LLR for this variable
            channel_llr = input_mapping_LLR[:, var_idx]
            
            # Get incoming messages from check nodes (excluding the current message)
            valid_indices = incoming_msg_indices[incoming_msg_indices >= 0]
            
            if len(valid_indices) > 0:
                incoming_messages = check_to_variable_messages[:, valid_indices]
                
                # Sum all incoming messages and channel LLR
                total_sum = channel_llr.unsqueeze(1) + incoming_messages.sum(dim=1)
                
                # For each outgoing message, subtract the corresponding incoming message
                for i, idx in enumerate(valid_indices):
                    variable_to_check_messages[:, msg_idx] = total_sum - incoming_messages[:, i]
            else:
                # If no incoming messages, just use channel LLR
                variable_to_check_messages[:, msg_idx] = channel_llr
        
        # Apply damping if not the first iteration
        if iteration > 0:
            # Damping factor (can be made learnable)
            alpha = 0.5
            variable_to_check_messages = alpha * variable_to_check_messages + (1 - alpha) * check_to_variable_messages
            print(f"Applied damping with factor {alpha}")
        
        print(f"Output variable-to-check messages shape: {variable_to_check_messages.shape}")
        print(f"Variable-to-check messages range: [{variable_to_check_messages.min().item():.4f}, {variable_to_check_messages.max().item():.4f}]")
        print("----- Completed Variable Layer Update -----\n")
        
        return variable_to_check_messages
    
    def forward(self, message_features, message_types, var_to_check_adjacency, check_to_var_adjacency, 
                input_llr=None, message_to_var_mapping=None, variable_index_tensor=None, iteration=0):
        """
        Perform message update in the GNN with custom variable layer update.
        
        Args:
            message_features (torch.Tensor): Features of each message node of shape (batch_size, num_messages, feature_dim)
            message_types (torch.Tensor): Type indices for each message of shape (num_messages,)
            var_to_check_adjacency (torch.Tensor): Adjacency matrix for variable-to-check messages
            check_to_var_adjacency (torch.Tensor): Adjacency matrix for check-to-variable messages
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_variables)
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
            variable_index_tensor (torch.Tensor): Variable node index mapping
            iteration (int): Current iteration number
            
        Returns:
            torch.Tensor: Updated message features of shape (batch_size, num_messages, feature_dim)
        """
        batch_size, num_messages, feature_dim = message_features.shape
        device = message_features.device
        
        # Ensure message_types has the right length and doesn't exceed the number of message types
        if len(message_types) != num_messages:
            if len(message_types) < num_messages:
                padded_message_types = torch.zeros(num_messages, dtype=torch.long, device=device)
                padded_message_types[:len(message_types)] = message_types
                message_types = padded_message_types
            else:
                message_types = message_types[:num_messages]
        
        safe_message_types = torch.clamp(message_types, 0, self.message_type_embeddings.size(0) - 1)
        
        # Get embeddings for each message type
        type_embeddings = self.message_type_embeddings[safe_message_types]
        type_embeddings = type_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine message features with type embeddings
        combined_features = message_features + type_embeddings
        
        # Check if adjacency matrices match the number of messages
        if var_to_check_adjacency.size(0) != num_messages or var_to_check_adjacency.size(1) != num_messages:
            new_var_to_check = torch.zeros((num_messages, num_messages), device=device)
            new_check_to_var = torch.zeros((num_messages, num_messages), device=device)
            
            min_size = min(var_to_check_adjacency.size(0), num_messages)
            new_var_to_check[:min_size, :min_size] = var_to_check_adjacency[:min_size, :min_size]
            new_check_to_var[:min_size, :min_size] = check_to_var_adjacency[:min_size, :min_size]
            
            var_to_check_adjacency = new_var_to_check
            check_to_var_adjacency = new_check_to_var
        
        # Check-to-variable message update (using the original neural network approach)
        check_to_var_messages = torch.bmm(check_to_var_adjacency.unsqueeze(0).expand(batch_size, -1, -1), combined_features)
        check_to_var_input = torch.cat([combined_features, check_to_var_messages], dim=2)
        check_to_var_updated = self.check_to_var_update(check_to_var_input)
        
        # Decode check-to-variable messages to LLR values
        check_to_var_llrs = self.output_projection(check_to_var_updated).squeeze(-1)
        
        # If we have the necessary inputs for the custom variable layer update, use it
        if input_llr is not None and message_to_var_mapping is not None and variable_index_tensor is not None:
            # Map input LLRs to message nodes
            input_mapping_LLR = torch.matmul(message_to_var_mapping, input_llr.unsqueeze(-1)).squeeze(-1).transpose(0, 1)
            
            # Use custom variable layer update
            var_updated_llrs = self.variable_layer_update(
                input_mapping_LLR, 
                check_to_var_llrs, 
                variable_index_tensor, 
                iteration
            )
            
            # Convert back to feature space
            var_to_check_updated = self.input_embedding(var_updated_llrs.unsqueeze(-1))
        else:
            # Fallback to original neural network approach if necessary inputs are missing
            var_to_check_messages = torch.bmm(var_to_check_adjacency.unsqueeze(0).expand(batch_size, -1, -1), combined_features)
            var_to_check_input = torch.cat([combined_features, var_to_check_messages], dim=2)
            var_to_check_updated = self.var_to_check_update(var_to_check_input)
        
        # Combine updates
        updated_features = var_to_check_updated + check_to_var_updated
        
        return updated_features


class CustomVariableMessageGNNDecoder(MessageGNNDecoder):
    """
    Message-Centered GNN LDPC Decoder with custom variable layer update.
    
    This decoder uses a traditional min-sum algorithm with residual connections
    for variable node updates instead of neural networks.
    """
    def __init__(self, num_messages, num_iterations=5, hidden_dim=64, num_message_types=1, depth_L=3):
        """
        Initialize the Custom Variable Message GNN Decoder.
        
        Args:
            num_messages (int): Number of messages (edges) in the Tanner graph
            num_iterations (int): Number of decoding iterations
            hidden_dim (int): Dimension of hidden representations
            num_message_types (int): Number of different message types for weight sharing
            depth_L (int): Depth of residual connections
        """
        super(CustomVariableMessageGNNDecoder, self).__init__(
            num_messages, num_iterations, hidden_dim, num_message_types
        )
        
        # Replace GNN layers with custom ones
        self.gnn_layers = nn.ModuleList([
            CustomVariableMessageGNNLayer(num_message_types, hidden_dim, depth_L)
            for _ in range(num_iterations)
        ])
        
        # Create variable index tensor storage
        self.variable_index_tensor = None
    
    def set_variable_index_tensor(self, variable_index_tensor):
        """
        Set the variable index tensor for variable layer updates.
        
        Args:
            variable_index_tensor (torch.Tensor): Variable node index mapping
        """
        self.variable_index_tensor = variable_index_tensor
    
    def forward(self, input_llr, message_to_var_mapping, message_types=None, 
                var_to_check_adjacency=None, check_to_var_adjacency=None, ground_truth=None):
        """
        Forward pass of the Custom Variable Message GNN Decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLR values for each variable node
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
            message_types (torch.Tensor, optional): Type indices for each message
            var_to_check_adjacency (torch.Tensor, optional): Adjacency matrix for variable-to-check messages
            check_to_var_adjacency (torch.Tensor, optional): Adjacency matrix for check-to-variable messages
            ground_truth (torch.Tensor, optional): True bits (0 or 1)
            
        Returns:
            tuple: (soft_bits, loss)
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
        message_llrs = torch.matmul(message_to_var_mapping, input_llr.unsqueeze(-1)).squeeze(-1)
        message_llrs = message_llrs.transpose(0, 1)
        
        # Since we're using a higher hidden_dim, we'll use the input_embedding to transform
        # the scalar LLRs into higher-dimensional feature vectors
        message_features = self.input_embedding(message_llrs.unsqueeze(-1))
        
        # Clear previous variable layer storage for all GNN layers
        for layer in self.gnn_layers:
            layer.previous_VL_storage.clear()
        
        # Iterative GNN decoding
        for i in range(self.num_iterations):
            # Update message features using custom GNN layer
            message_features = self.gnn_layers[i](
                message_features, 
                message_types, 
                var_to_check_adjacency, 
                check_to_var_adjacency,
                input_llr,
                message_to_var_mapping,
                self.variable_index_tensor,
                i + 1  # Iteration starts from 1
            )
        
        # Decode final message features to LLR values
        final_message_llrs = self.gnn_layers[-1].decode_messages(message_features)
        
        # Aggregate message LLRs to variable nodes
        var_to_message_mapping = message_to_var_mapping.transpose(0, 1)
        var_to_message_mapping = var_to_message_mapping / (var_to_message_mapping.sum(dim=1, keepdim=True) + 1e-10)
        
        if final_message_llrs.shape[0] != batch_size:
            final_message_llrs = final_message_llrs.transpose(0, 1)
        
        variable_llrs = torch.matmul(final_message_llrs, var_to_message_mapping.t())
        
        # Add input LLRs
        combined_llrs = variable_llrs + input_llr
        
        # Convert to soft bits
        soft_bits = torch.sigmoid(combined_llrs)
        
        # Compute loss if ground truth is provided
        loss = None
        if ground_truth is not None:
            loss = F.binary_cross_entropy(soft_bits, ground_truth, reduction='none')
            max_loss = torch.max(loss, dim=1).values
            return soft_bits, max_loss
        
        return soft_bits, None 


def create_custom_variable_message_gnn_decoder(H, num_iterations=5, hidden_dim=64, depth_L=3, base_graph=None, Z=None):
    """
    Create a Custom Variable Message GNN Decoder from a parity-check matrix.
    
    This decoder uses a traditional min-sum algorithm with residual connections
    for variable node updates instead of neural networks.
    
    Args:
        H (torch.Tensor): Parity-check matrix
        num_iterations (int): Number of decoding iterations
        hidden_dim (int): Dimension of hidden representations
        depth_L (int): Depth of residual connections
        base_graph (torch.Tensor, optional): Base graph matrix with shift values
        Z (int, optional): Lifting factor
        
    Returns:
        tuple: (decoder, converter)
            - decoder: CustomVariableMessageGNNDecoder instance
            - converter: TannerToMessageGraph instance
    """
    # Convert Tanner graph to message graph
    converter = TannerToMessageGraph(H)
    
    # Get number of messages
    num_messages = len(converter.messages)
    
    # Get number of message types
    if base_graph is not None and Z is not None:
        # Count unique shift values in base graph (excluding -1)
        unique_shifts = set()
        for i in range(base_graph.shape[0]):
            for j in range(base_graph.shape[1]):
                shift = base_graph[i, j].item()
                if shift >= 0:
                    unique_shifts.add(int(shift))
        num_message_types = len(unique_shifts) if unique_shifts else 1
    else:
        num_message_types = 1
    
    # Create variable index tensor
    variable_index_tensor = create_variable_index_tensor(H, converter)
    
    # Create decoder
    decoder = CustomVariableMessageGNNDecoder(
        num_messages=num_messages,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
        num_message_types=num_message_types,
        depth_L=depth_L
    )
    
    # Set variable index tensor
    decoder.set_variable_index_tensor(variable_index_tensor)
    
    return decoder, converter

def create_variable_index_tensor(H, converter):
    """
    Create a variable index tensor for the CustomVariableMessageGNNDecoder.
    
    Args:
        H (torch.Tensor): Parity check matrix
        converter (TannerToMessageGraph): Converter instance
        
    Returns:
        torch.Tensor: Variable index tensor
    """
    num_checks, num_vars = H.shape
    device = H.device
    
    # Find the maximum number of connections per variable node
    max_connections = max(len(connections) for connections in converter.var_to_messages.values())
    
    # Create the variable index tensor
    variable_index_tensor = -torch.ones((num_vars, max_connections), dtype=torch.long, device=device)
    
    # Fill in the variable index tensor
    for v in range(num_vars):
        connections = converter.var_to_messages[v]
        for i, message_idx in enumerate(connections):
            variable_index_tensor[v, i] = message_idx
    
    return variable_index_tensor

class CustomCheckMessageGNNLayer(MessageGNNLayer):
    """
    A message-centered GNN layer for LDPC decoding using traditional min-sum check update.
    This layer replaces the neural network-based check node update with a traditional min-sum algorithm.
    """
    def __init__(self, num_message_types, hidden_dim, depth=2, dropout=0.0):
        super().__init__(num_message_types, hidden_dim)
        # Add a learnable scaling factor for min-sum algorithm
        self.alpha = nn.Parameter(torch.tensor(0.8))  # Initialize with a reasonable value
        
    def check_layer_update(self, message_features, message_types, check_index_tensor):
        """
        Update check node messages using min-sum algorithm.
        
        Args:
            message_features: Messages from variable nodes to check nodes
            message_types: Types of each message for weight sharing
            check_index_tensor: Tensor mapping messages to check nodes
            
        Returns:
            Updated check-to-variable messages
        """
        print("\n----- Check Layer Update -----")
        print(f"Input message features shape: {message_features.shape}")
        print(f"Check index tensor shape: {check_index_tensor.shape}")
        
        batch_size = message_features.shape[0]
        num_messages = check_index_tensor.shape[0]
        
        # Initialize output messages
        check_to_variable_messages = torch.zeros(batch_size, num_messages, device=message_features.device)
        
        # Process each message
        for msg_idx in range(num_messages):
            check_idx = check_index_tensor[msg_idx, 0].item()
            incoming_msg_indices = check_index_tensor[msg_idx, 1:].long()
            
            # Get incoming messages from variable nodes (excluding the current message)
            valid_indices = incoming_msg_indices[incoming_msg_indices >= 0]
            
            if len(valid_indices) > 0:
                incoming_messages = message_features[:, valid_indices]
                
                # Min-sum algorithm: take the product of signs and minimum of magnitudes
                signs = torch.sign(incoming_messages)
                magnitudes = torch.abs(incoming_messages)
                
                # Product of signs (using multiplication)
                prod_sign = torch.prod(signs, dim=1)
                
                # Find minimum magnitude
                min_magnitude = torch.min(magnitudes, dim=1)[0]
                
                # For each outgoing message, exclude the corresponding incoming message
                for i, idx in enumerate(valid_indices):
                    # Compute sign (excluding current message)
                    other_signs = torch.cat([signs[:, :i], signs[:, i+1:]], dim=1)
                    msg_sign = torch.prod(other_signs, dim=1)
                    
                    # Compute minimum magnitude (excluding current message)
                    other_magnitudes = torch.cat([magnitudes[:, :i], magnitudes[:, i+1:]], dim=1)
                    
                    if other_magnitudes.shape[1] > 0:
                        msg_min_magnitude = torch.min(other_magnitudes, dim=1)[0]
                        
                        # Combine sign and magnitude
                        check_to_variable_messages[:, msg_idx] = msg_sign * msg_min_magnitude
                    else:
                        # If no other messages, set to zero
                        check_to_variable_messages[:, msg_idx] = 0.0
            else:
                # If no incoming messages, set to zero
                check_to_variable_messages[:, msg_idx] = 0.0
        
        print(f"Output check-to-variable messages shape: {check_to_variable_messages.shape}")
        print(f"Check-to-variable messages range: [{check_to_variable_messages.min().item():.4f}, {check_to_variable_messages.max().item():.4f}]")
        print("----- Completed Check Layer Update -----\n")
        
        return check_to_variable_messages
    
    def forward(self, message_features, message_types, variable_adjacency, check_adjacency, 
                variable_index_tensor=None, check_index_tensor=None):
        """
        Perform message updates in the GNN.
        
        Args:
            message_features: Tensor of shape (batch_size, num_messages, hidden_dim)
            message_types: Tensor of shape (num_messages,)
            variable_adjacency: Sparse tensor of shape (num_variables, num_messages)
            check_adjacency: Sparse tensor of shape (num_checks, num_messages)
            variable_index_tensor: Tensor of shape (num_variables, max_variable_degree)
            check_index_tensor: Tensor of shape (num_checks, max_check_degree)
            
        Returns:
            updated_message_features: Tensor of shape (batch_size, num_messages, hidden_dim)
        """
        # Variable-to-check update (using neural network)
        if variable_index_tensor is not None:
            message_features = self.variable_layer_update(
                message_features, message_types, variable_index_tensor
            )
        else:
            message_features = super().variable_layer_update(
                message_features, message_types, variable_adjacency
            )
        
        # Check-to-variable update (using min-sum algorithm)
        if check_index_tensor is not None:
            message_features = self.check_layer_update(
                message_features, message_types, check_index_tensor
            )
        else:
            message_features = super().check_layer_update(
                message_features, message_types, check_adjacency
            )
        
        return message_features


def create_check_index_tensor(H, message_type_map=None):
    """
    Create a check index tensor for the custom check layer.
    
    Args:
        H: Binary parity-check matrix of shape (num_checks, num_variables)
        message_type_map: Optional mapping of message indices to types
        
    Returns:
        check_index_tensor: Tensor of shape (num_checks, max_check_degree)
            Contains indices of messages connected to each check node.
            Padded with -1 for variable-length connections.
    """
    num_checks, num_variables = H.shape
    
    # Convert H to a dense tensor if it's not already
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float)
    
    # Find the maximum check degree
    check_degrees = H.sum(dim=1).int()
    max_check_degree = check_degrees.max().item()
    
    # Initialize the check index tensor with -1 (padding value)
    check_index_tensor = torch.full((num_checks, max_check_degree), -1, dtype=torch.long)
    
    # For each check node
    for c in range(num_checks):
        # Find connected variable nodes
        connected_vars = torch.where(H[c] > 0)[0]
        
        # Map to message indices
        msg_counter = 0
        for v_idx in connected_vars:
            # In the Tanner graph, each edge corresponds to a message
            # We need to map (check, variable) pairs to message indices
            if message_type_map is not None:
                # If a message type map is provided, use it to find the message index
                msg_idx = message_type_map.get((c, v_idx.item()))
                if msg_idx is not None:
                    check_index_tensor[c, msg_counter] = msg_idx
                    msg_counter += 1
            else:
                # Without a message type map, we use a simple mapping
                # This assumes messages are ordered by check nodes first, then variables
                msg_idx = c * num_variables + v_idx.item()
                check_index_tensor[c, msg_counter] = msg_idx
                msg_counter += 1
    
    return check_index_tensor


class CustomMinSumMessageGNNDecoder(MessageGNNDecoder):
    """
    A message-centered GNN LDPC decoder using traditional min-sum algorithm
    for both variable and check node updates.
    """
    def __init__(self, num_messages, num_iterations, hidden_dim, num_message_types=1, depth=2, dropout=0.0):
        super().__init__(num_messages, num_iterations, hidden_dim, num_message_types)
        
        # Create custom variable and check layers
        self.variable_layer = CustomVariableMessageGNNLayer(num_message_types, hidden_dim, depth)
        self.check_layer = CustomCheckMessageGNNLayer(num_message_types, hidden_dim, depth, dropout)
        
        # Replace GNN layers with custom layers
        self.gnn_layers = nn.ModuleList([
            CustomCheckMessageGNNLayer(num_message_types, hidden_dim, depth, dropout)
            for _ in range(num_iterations)
        ])
        
        # Store both variable and check index tensors
        self.variable_index_tensor = None
        self.check_index_tensor = None
    
    def set_variable_index_tensor(self, variable_index_tensor):
        """Set the variable index tensor for efficient message passing."""
        self.variable_index_tensor = variable_index_tensor
    
    def set_check_index_tensor(self, check_index_tensor):
        """Set the check index tensor for efficient message passing."""
        self.check_index_tensor = check_index_tensor
    
    def forward(self, input_llrs, variable_adjacency, check_adjacency, message_types, 
                variable_to_message_mapping, ground_truth=None):
        """
        Forward pass of the Custom Min-Sum Message GNN Decoder.
        
        Args:
            input_llrs: Input LLR values from the channel
            variable_adjacency: Adjacency matrix for variable nodes
            check_adjacency: Adjacency matrix for check nodes
            message_types: Types of each message for weight sharing
            variable_to_message_mapping: Mapping from variable nodes to messages
            ground_truth: Ground truth codeword for training
            
        Returns:
            Decoded codeword probabilities and loss (if ground_truth provided)
        """
        print("\n===== Starting Custom Min-Sum Message GNN Decoder Forward Pass =====")
        print(f"Input LLRs shape: {input_llrs.shape}")
        
        batch_size, num_variables = input_llrs.shape
        device = input_llrs.device
        
        # Initialize messages from variable nodes to check nodes
        variable_to_check_messages = torch.zeros(batch_size, self.num_messages, device=device)
        
        # Initialize messages from check nodes to variable nodes
        check_to_variable_messages = torch.zeros(batch_size, self.num_messages, device=device)
        
        # Iterative decoding
        for iteration in range(self.num_iterations):
            print(f"\n----- Iteration {iteration+1}/{self.num_iterations} -----")
            
            # Variable node update
            variable_to_check_messages = self.variable_layer.variable_layer_update(
                input_llrs, 
                check_to_variable_messages, 
                self.variable_index_tensor,
                iteration
            )
            
            # Check node update
            check_to_variable_messages = self.check_layer.check_layer_update(
                variable_to_check_messages,
                message_types,
                self.check_index_tensor
            )
        
        # Final decision: combine all messages for each variable node
        print("\n----- Final Decision Stage -----")
        
        # Initialize output probabilities
        output_llrs = torch.zeros(batch_size, num_variables, device=device)
        
        print("Using SUM aggregation for mapping message nodes to variable bits")
        
        # For each variable node, sum all incoming messages and the channel LLR
        for var_idx in range(num_variables):
            # Get indices of messages connected to this variable
            msg_indices = variable_to_message_mapping[var_idx]
            valid_indices = msg_indices[msg_indices >= 0]
            
            if len(valid_indices) > 0:
                # Sum all incoming messages
                incoming_messages = check_to_variable_messages[:, valid_indices]
                output_llrs[:, var_idx] = input_llrs[:, var_idx] + torch.sum(incoming_messages, dim=1)
            else:
                # If no incoming messages, just use the channel LLR
                output_llrs[:, var_idx] = input_llrs[:, var_idx]
        
        print(f"Output LLRs shape: {output_llrs.shape}")
        print(f"Output LLRs range: [{output_llrs.min().item():.4f}, {output_llrs.max().item():.4f}]")
        
        # Convert LLRs to probabilities
        output_probs = torch.sigmoid(output_llrs)
        
        print(f"Output probabilities shape: {output_probs.shape}")
        print(f"Output probabilities range: [{output_probs.min().item():.4f}, {output_probs.max().item():.4f}]")
        print("===== Completed Custom Min-Sum Message GNN Decoder Forward Pass =====\n")
        
        # Compute loss if ground truth is provided
        if ground_truth is not None:
            loss = F.binary_cross_entropy(output_probs, ground_truth.float())
            return output_probs, loss
        
        return output_probs


def create_custom_minsum_message_gnn_decoder(H, num_iterations=5, hidden_dim=8, depth=2, dropout=0.0):
    """
    Create a custom min-sum message GNN decoder from a parity-check matrix.
    
    Args:
        H: Binary parity-check matrix of shape (num_checks, num_variables)
        num_iterations: Number of iterations to run the decoder
        hidden_dim: Hidden dimension of the message features
        depth: Depth of the residual connections
        dropout: Dropout probability
        
    Returns:
        decoder: CustomMinSumMessageGNNDecoder instance
        converter: TannerToMessageGraph instance
    """
    # Create a Tanner to message graph converter
    converter = TannerToMessageGraph()
    
    # Convert the parity-check matrix to a message graph
    variable_adjacency, check_adjacency, message_types, variable_to_message_mapping = converter.convert(H)
    
    # Get the number of messages
    num_messages = message_types.shape[0]
    num_message_types = message_types.max().item() + 1
    
    # Create the decoder
    decoder = CustomMinSumMessageGNNDecoder(
        num_messages, num_iterations, hidden_dim, num_message_types, depth, dropout
    )
    
    # Create variable and check index tensors
    variable_index_tensor = create_variable_index_tensor(H, converter.message_type_map)
    check_index_tensor = create_check_index_tensor(H, converter.message_type_map)
    
    # Set the index tensors
    decoder.set_variable_index_tensor(variable_index_tensor)
    decoder.set_check_index_tensor(check_index_tensor)
    
    return decoder, converter 