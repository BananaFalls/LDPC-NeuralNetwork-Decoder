"""
GNN-based LDPC Neural Decoder

This module implements a Graph Neural Network (GNN) based LDPC decoder
with weight sharing across similar edge types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class GNNCheckLayer(nn.Module):
    """
    GNN-based Check Node Layer with weight sharing.
    
    Implements the check node update step using shared weights across similar edge types.
    """
    def __init__(self, num_edge_types=1):
        """
        Initialize the GNN Check Layer.
        
        Args:
            num_edge_types (int): Number of different edge types for weight sharing
        """
        super(GNNCheckLayer, self).__init__()
        
        # Learnable parameters for each edge type
        self.edge_weights = nn.Parameter(torch.ones(num_edge_types))
        self.edge_biases = nn.Parameter(torch.zeros(num_edge_types))
        
        # Activation function parameters (learnable)
        self.alpha = nn.Parameter(torch.ones(1))  # Scaling factor
        self.beta = nn.Parameter(torch.zeros(1))  # Bias term
    
    def forward(self, input_tensor, check_index_tensor, edge_type_tensor=None):
        """
        Perform check node update with weight sharing.
        
        Args:
            input_tensor (torch.Tensor): Input LLRs of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_neighbors)
            edge_type_tensor (torch.Tensor, optional): Edge type indices of shape (num_nodes, max_neighbors)
            
        Returns:
            torch.Tensor: Updated check node messages of shape (batch_size, num_nodes)
        """
        batch_size, num_nodes = input_tensor.shape
        num_nodes_mapped, max_neighbors = check_index_tensor.shape
        
        # If edge_type_tensor is not provided, assume all edges are of the same type
        if edge_type_tensor is None:
            edge_type_tensor = torch.zeros_like(check_index_tensor, dtype=torch.long)
        
        # Create a mask for valid indices (ignoring -1)
        valid_mask = check_index_tensor != -1
        
        # Replace -1 with a safe dummy index (num_nodes)
        safe_indices = check_index_tensor.clone()
        safe_indices[~valid_mask] = num_nodes  # Safe index (out of range)
        
        # Extend input_tensor with an extra column of zeros
        input_extended = torch.cat([input_tensor, torch.zeros((batch_size, 1), device=input_tensor.device)], dim=1)
        
        # Expand input tensor to apply all index_tensor rows
        input_expanded = input_extended.unsqueeze(0).expand(num_nodes_mapped, -1, -1)
        
        # Expand index tensor to match batch dimension
        index_expanded = safe_indices.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Gather elements
        selected_values = torch.gather(input_expanded, dim=2, index=index_expanded)
        
        # Apply mask: Set invalid gathered values (from -1) to zero
        selected_values[~valid_mask.unsqueeze(1).expand(-1, batch_size, -1)] = 0
        
        # Apply edge-specific weights and biases
        if edge_type_tensor is not None:
            # Expand edge_type_tensor to match batch dimension
            edge_type_expanded = edge_type_tensor.unsqueeze(1).expand(-1, batch_size, -1)
            
            # Get weights and biases for each edge
            edge_weights_expanded = self.edge_weights[edge_type_expanded]
            edge_biases_expanded = self.edge_biases[edge_type_expanded]
            
            # Apply weights and biases
            selected_values = selected_values * edge_weights_expanded + edge_biases_expanded
            
            # Zero out invalid edges
            selected_values[~valid_mask.unsqueeze(1).expand(-1, batch_size, -1)] = 0
        
        # Apply Min-Sum with learnable parameters
        # Calculate sign product
        sign_product = torch.prod(torch.sign(selected_values + 1e-10), dim=2)
        
        # Find minimum absolute value (excluding zeros)
        abs_values = torch.abs(selected_values)
        # Replace zeros with a large value
        abs_values[abs_values == 0] = 1e10
        min_abs = torch.min(abs_values, dim=2).values
        
        # Apply learnable scaling and bias
        min_abs = self.alpha * min_abs + self.beta
        
        # Compute Min-Sum result
        min_sum_result = sign_product * min_abs  # Element-wise multiplication
        
        # Transpose to match (batch_size, num_nodes_mapped)
        check_messages = min_sum_result.T
        
        return check_messages


class GNNVariableLayer(nn.Module):
    """
    GNN-based Variable Node Layer with weight sharing.
    
    Implements the variable node update step using shared weights across similar edge types.
    """
    def __init__(self, num_edge_types=1):
        """
        Initialize the GNN Variable Layer.
        
        Args:
            num_edge_types (int): Number of different edge types for weight sharing
        """
        super(GNNVariableLayer, self).__init__()
        
        # Learnable parameters for each edge type
        self.edge_weights = nn.Parameter(torch.ones(num_edge_types))
        self.edge_biases = nn.Parameter(torch.zeros(num_edge_types))
        
        # Learnable parameters for combining messages
        self.combine_weight = nn.Parameter(torch.ones(1))
        self.combine_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, input_llr, check_messages, var_index_tensor, edge_type_tensor=None):
        """
        Perform variable node update with weight sharing.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_messages (torch.Tensor): Messages from check nodes of shape (batch_size, num_nodes)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_neighbors)
            edge_type_tensor (torch.Tensor, optional): Edge type indices of shape (num_nodes, max_neighbors)
            
        Returns:
            torch.Tensor: Updated variable node messages of shape (batch_size, num_nodes)
        """
        batch_size, num_nodes = check_messages.shape
        num_nodes_mapped, max_neighbors = var_index_tensor.shape
        
        # If edge_type_tensor is not provided, assume all edges are of the same type
        if edge_type_tensor is None:
            edge_type_tensor = torch.zeros_like(var_index_tensor, dtype=torch.long)
        
        # Create a mask where valid indices are True (ignoring -1 values)
        valid_mask = var_index_tensor != -1
        
        # Replace -1 with a safe dummy index (num_nodes) which we will zero out later
        safe_indices = var_index_tensor.clone()
        safe_indices[~valid_mask] = num_nodes  # Use an out-of-range index
        
        # Extend check_messages with a row of zeros at index num_nodes
        extended_check_messages = torch.cat([
            check_messages,
            torch.zeros((batch_size, 1), device=check_messages.device)
        ], dim=1)  # Shape: (batch_size, num_nodes + 1)
        
        # Expand check messages and indices
        check_messages_expanded = extended_check_messages.unsqueeze(0).expand(num_nodes_mapped, -1, -1)
        index_expanded = safe_indices.unsqueeze(1).expand(-1, batch_size, -1)
        
        # Gather messages using safe indices
        gathered_messages = torch.gather(check_messages_expanded, dim=2, index=index_expanded)
        
        # Apply edge-specific weights and biases
        if edge_type_tensor is not None:
            # Expand edge_type_tensor to match batch dimension
            edge_type_expanded = edge_type_tensor.unsqueeze(1).expand(-1, batch_size, -1)
            
            # Get weights and biases for each edge
            edge_weights_expanded = self.edge_weights[edge_type_expanded]
            edge_biases_expanded = self.edge_biases[edge_type_expanded]
            
            # Apply weights and biases
            gathered_messages = gathered_messages * edge_weights_expanded + edge_biases_expanded
        
        # Apply mask: Set invalid gathered values to 0
        gathered_messages[~valid_mask.unsqueeze(1).expand(-1, batch_size, -1)] = 0
        
        # Sum valid messages
        summed_messages = torch.sum(gathered_messages, dim=2)  # (num_nodes_mapped, batch_size)
        
        # Transpose to match (batch_size, num_nodes)
        summed_messages = summed_messages.T  # (batch_size, num_nodes_mapped)
        
        # Apply learnable combination
        summed_messages = self.combine_weight * summed_messages + self.combine_bias
        
        # Compute updated variable messages
        variable_messages = input_llr + summed_messages  # Add the LLR to the sum of messages
        
        return variable_messages


class GNNResidualLayer(nn.Module):
    """
    GNN-based Residual Layer with weight sharing.
    
    Implements residual connections between variable node layers with shared weights.
    """
    def __init__(self, num_nodes, depth_L=2, num_edge_types=1):
        """
        Initialize the GNN Residual Layer.
        
        Args:
            num_nodes (int): Number of nodes in the Tanner graph
            depth_L (int): Depth of residual connections
            num_edge_types (int): Number of different edge types for weight sharing
        """
        super(GNNResidualLayer, self).__init__()
        
        # Trainable weights for channel reliability
        self.w_ch = nn.Parameter(torch.ones(1))  # Shared weight for channel LLR
        
        # Trainable weights for residual connections (shared across nodes)
        self.w_res = nn.Parameter(torch.ones(depth_L))  # Residual connection weights
        
        # Edge type specific weights
        self.edge_type_weights = nn.Parameter(torch.ones(num_edge_types))
    
    def forward(self, input_llr, check_messages, prev_var_messages, edge_type_tensor=None):
        """
        Apply residual connections to variable node messages with weight sharing.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_messages (torch.Tensor): Messages from check nodes of shape (batch_size, num_nodes)
            prev_var_messages (list): List of previous variable node messages, each of shape (batch_size, num_nodes)
            edge_type_tensor (torch.Tensor, optional): Edge type indices of shape (num_nodes,)
            
        Returns:
            torch.Tensor: Updated variable node messages with residual connections of shape (batch_size, num_nodes)
        """
        # Apply shared channel weight
        weighted_llr = input_llr * self.w_ch
        
        # Apply edge type specific weights to check messages if provided
        if edge_type_tensor is not None:
            # Expand edge_type_tensor to match batch dimension
            batch_size = check_messages.shape[0]
            edge_weights = self.edge_type_weights[edge_type_tensor]
            edge_weights_expanded = edge_weights.unsqueeze(0).expand(batch_size, -1)
            
            # Apply weights
            weighted_check_messages = check_messages * edge_weights_expanded
        else:
            weighted_check_messages = check_messages
        
        # Add check node messages
        result = weighted_llr + weighted_check_messages
        
        # Add residual connections from previous iterations with shared weights
        for i, prev_messages in enumerate(prev_var_messages):
            if i < len(self.w_res):
                result = result + self.w_res[i] * prev_messages
        
        return result


class GNNOutputLayer(nn.Module):
    """
    GNN-based Output Layer with weight sharing.
    
    Maps the final LLRs to soft-bit values and computes the loss with shared weights.
    """
    def __init__(self):
        """Initialize the GNN Output Layer."""
        super(GNNOutputLayer, self).__init__()
        
        # Learnable parameters for combining final and input LLRs
        self.combine_weight = nn.Parameter(torch.ones(1))
        self.combine_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, final_llr, input_llr, ground_truth=None):
        """
        Compute soft-bit values and optionally the loss with weight sharing.
        
        Args:
            final_llr (torch.Tensor): Final LLRs from the decoder of shape (batch_size, num_nodes)
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_nodes)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_nodes)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        # Combine final LLR with input LLR using learnable parameters
        combined_llr = final_llr + self.combine_weight * input_llr + self.combine_bias
        
        # Apply sigmoid to get soft bit values
        soft_bits = torch.sigmoid(combined_llr)
        
        # Compute loss if ground truth is provided
        if ground_truth is not None:
            # Binary cross-entropy loss
            loss = F.binary_cross_entropy(soft_bits, ground_truth, reduction='none')
            
            # Apply max function over the loss vector (for FER minimization)
            max_loss = torch.max(loss, dim=1).values
            
            return soft_bits, max_loss
        
        return soft_bits, None


class GNNLDPCDecoder(nn.Module):
    """
    GNN-based LDPC Decoder with weight sharing.
    
    Implements a Graph Neural Network approach to LDPC decoding with weight sharing
    across similar edge types.
    """
    def __init__(self, num_nodes, num_iterations=5, depth_L=2, num_edge_types=1):
        """
        Initialize the GNN LDPC Decoder.
        
        Args:
            num_nodes (int): Number of nodes in the Tanner graph
            num_iterations (int): Number of decoding iterations
            depth_L (int): Depth of residual connections
            num_edge_types (int): Number of different edge types for weight sharing
        """
        super(GNNLDPCDecoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_iterations = num_iterations
        self.depth_L = depth_L
        self.num_edge_types = num_edge_types
        
        # Initialize GNN layers with weight sharing
        self.check_layer = GNNCheckLayer(num_edge_types)
        self.variable_layer = GNNVariableLayer(num_edge_types)
        self.residual_layer = GNNResidualLayer(num_nodes, depth_L, num_edge_types)
        self.output_layer = GNNOutputLayer()
    
    def forward(self, input_llr, check_index_tensor, var_index_tensor, ground_truth=None, 
                check_edge_types=None, var_edge_types=None, node_edge_types=None):
        """
        Forward pass of the GNN LDPC Decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_nodes)
            check_edge_types (torch.Tensor, optional): Edge type indices for check nodes of shape (num_nodes, max_check_neighbors)
            var_edge_types (torch.Tensor, optional): Edge type indices for variable nodes of shape (num_nodes, max_var_neighbors)
            node_edge_types (torch.Tensor, optional): Edge type indices for nodes of shape (num_nodes,)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_nodes)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        batch_size = input_llr.shape[0]
        
        # Initialize variable node messages with input LLRs
        var_messages = input_llr
        
        # Queue to store previous variable node messages for residual connections
        prev_var_messages = deque(maxlen=self.depth_L)
        
        # Iterative decoding
        for _ in range(self.num_iterations):
            # Store current variable messages for residual connections
            prev_var_messages.appendleft(var_messages)
            
            # Check node update with weight sharing
            check_messages = self.check_layer(var_messages, check_index_tensor, check_edge_types)
            
            # Variable node update with weight sharing
            var_messages = self.variable_layer(
                input_llr, 
                check_messages, 
                var_index_tensor,
                var_edge_types
            )
            
            # Apply residual connections with weight sharing
            var_messages = self.residual_layer(
                input_llr, 
                check_messages, 
                list(prev_var_messages),
                node_edge_types
            )
        
        # Final output mapping with weight sharing
        soft_bits, loss = self.output_layer(var_messages, input_llr, ground_truth)
        
        return soft_bits, loss
    
    def decode(self, input_llr, check_index_tensor, var_index_tensor, 
               check_edge_types=None, var_edge_types=None, node_edge_types=None):
        """
        Decode input LLRs to hard bit decisions using the GNN decoder.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_check_neighbors)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_var_neighbors)
            check_edge_types (torch.Tensor, optional): Edge type indices for check nodes of shape (num_nodes, max_check_neighbors)
            var_edge_types (torch.Tensor, optional): Edge type indices for variable nodes of shape (num_nodes, max_var_neighbors)
            node_edge_types (torch.Tensor, optional): Edge type indices for nodes of shape (num_nodes,)
            
        Returns:
            torch.Tensor: Hard bit decisions (0 or 1) of shape (batch_size, num_nodes)
        """
        # Get soft bit values
        soft_bits, _ = self.forward(
            input_llr, 
            check_index_tensor, 
            var_index_tensor, 
            check_edge_types=check_edge_types,
            var_edge_types=var_edge_types,
            node_edge_types=node_edge_types
        )
        
        # Convert to hard decisions
        hard_bits = (soft_bits > 0.5).float()
        
        return hard_bits


class BaseGraphGNNDecoder(nn.Module):
    """
    Base Graph GNN LDPC Decoder with weight sharing based on the base graph structure.
    
    This decoder exploits the structure of the 5G LDPC base graph to implement
    weight sharing across similar edge types.
    """
    def __init__(self, base_graph, Z, num_iterations=5, depth_L=2):
        """
        Initialize the Base Graph GNN LDPC Decoder.
        
        Args:
            base_graph (torch.Tensor): Base graph matrix with shift values
            Z (int): Lifting factor
            num_iterations (int): Number of decoding iterations
            depth_L (int): Depth of residual connections
        """
        super(BaseGraphGNNDecoder, self).__init__()
        
        self.base_graph = base_graph
        self.Z = Z
        self.num_iterations = num_iterations
        self.depth_L = depth_L
        
        # Count the number of unique non-negative shift values in the base graph
        # These will be our edge types
        unique_shifts = set()
        for i in range(base_graph.shape[0]):
            for j in range(base_graph.shape[1]):
                shift = base_graph[i, j].item()
                if shift >= 0:
                    unique_shifts.add(shift)
        
        self.num_edge_types = len(unique_shifts)
        self.shift_to_edge_type = {shift: idx for idx, shift in enumerate(unique_shifts)}
        
        # Create the expanded parity-check matrix
        self.H_expanded = self._expand_base_graph()
        
        # Count number of 1s in H
        num_nodes = self.H_expanded.sum().item()
        
        # Create the GNN decoder with weight sharing
        self.decoder = GNNLDPCDecoder(
            num_nodes=num_nodes,
            num_iterations=num_iterations,
            depth_L=depth_L,
            num_edge_types=self.num_edge_types
        )
        
        # Create edge type tensors
        self.check_edge_types, self.var_edge_types, self.node_edge_types = self._create_edge_type_tensors()
    
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
    
    def _create_edge_type_tensors(self):
        """
        Create edge type tensors based on the base graph structure.
        
        Returns:
            tuple: (check_edge_types, var_edge_types, node_edge_types)
        """
        # This is a simplified implementation
        # In practice, you would need to map each edge in the expanded graph
        # back to its corresponding cell in the base graph
        
        # For now, we'll just create dummy edge type tensors
        # In a real implementation, you would derive these from the base graph
        
        # Assume check_index_tensor and var_index_tensor have the same shape
        max_check_neighbors = 10  # Example value
        max_var_neighbors = 10    # Example value
        num_nodes = self.H_expanded.sum().item()
        
        # Create dummy edge type tensors
        check_edge_types = torch.zeros((num_nodes, max_check_neighbors), dtype=torch.long)
        var_edge_types = torch.zeros((num_nodes, max_var_neighbors), dtype=torch.long)
        node_edge_types = torch.zeros(num_nodes, dtype=torch.long)
        
        # In a real implementation, you would fill these tensors with actual edge types
        # based on the base graph structure
        
        return check_edge_types, var_edge_types, node_edge_types
    
    def forward(self, input_llr, check_index_tensor, var_index_tensor, ground_truth=None):
        """
        Forward pass of the Base Graph GNN LDPC Decoder.
        
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
        return self.decoder(
            input_llr, 
            check_index_tensor, 
            var_index_tensor, 
            ground_truth,
            self.check_edge_types,
            self.var_edge_types,
            self.node_edge_types
        )
    
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
        return self.decoder.decode(
            input_llr, 
            check_index_tensor, 
            var_index_tensor,
            self.check_edge_types,
            self.var_edge_types,
            self.node_edge_types
        ) 