import torch
import torch.nn as nn
import torch.nn.functional as F

class CheckLayer(nn.Module):
    """
    Check Node Layer for LDPC Neural Decoder.
    
    Implements the check node update step in belief propagation using the min-sum algorithm.
    """
    def __init__(self):
        super(CheckLayer, self).__init__()
    
    def forward(self, input_tensor, check_index_tensor):
        """
        Perform check node update using min-sum algorithm.
        
        Args:
            input_tensor (torch.Tensor): Input LLRs of shape (batch_size, num_nodes)
            check_index_tensor (torch.Tensor): Check node index mapping of shape (num_nodes, max_neighbors)
            
        Returns:
            torch.Tensor: Updated check node messages of shape (batch_size, num_nodes)
        """
        batch_size, num_nodes = input_tensor.shape
        num_nodes_mapped, max_neighbors = check_index_tensor.shape
        
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
        
        # Apply Min-Sum
        # Calculate sign product
        sign_product = torch.prod(torch.sign(selected_values + 1e-10), dim=2)
        
        # Find minimum absolute value (excluding zeros)
        abs_values = torch.abs(selected_values)
        # Replace zeros with a large value
        abs_values[abs_values == 0] = 1e10
        min_abs = torch.min(abs_values, dim=2).values
        
        # Compute Min-Sum result
        min_sum_result = sign_product * min_abs  # Element-wise multiplication
        
        # Transpose to match (batch_size, num_nodes_mapped)
        check_messages = min_sum_result.T
        
        return check_messages


class VariableLayer(nn.Module):
    """
    Variable Node Layer for LDPC Neural Decoder.
    
    Implements the variable node update step in belief propagation.
    """
    def __init__(self):
        super(VariableLayer, self).__init__()
    
    def forward(self, input_llr, check_messages, var_index_tensor):
        """
        Perform variable node update.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_messages (torch.Tensor): Messages from check nodes of shape (batch_size, num_nodes)
            var_index_tensor (torch.Tensor): Variable node index mapping of shape (num_nodes, max_neighbors)
            
        Returns:
            torch.Tensor: Updated variable node messages of shape (batch_size, num_nodes)
        """
        batch_size, num_nodes = check_messages.shape
        num_nodes_mapped, max_neighbors = var_index_tensor.shape
        
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
        
        # Apply mask: Set invalid gathered values to 0
        gathered_messages[~valid_mask.unsqueeze(1).expand(-1, batch_size, -1)] = 0
        
        # Sum valid messages
        summed_messages = torch.sum(gathered_messages, dim=2)  # (num_nodes_mapped, batch_size)
        
        # Transpose to match (batch_size, num_nodes)
        summed_messages = summed_messages.T  # (batch_size, num_nodes_mapped)
        
        # Compute updated variable messages
        variable_messages = input_llr + summed_messages  # Add the LLR to the sum of messages
        
        return variable_messages


class ResidualLayer(nn.Module):
    """
    Residual Layer for LDPC Neural Decoder.
    
    Implements residual connections between variable node layers.
    """
    def __init__(self, num_nodes, depth_L=2):
        super(ResidualLayer, self).__init__()
        self.num_nodes = num_nodes
        self.depth_L = depth_L
        
        # Trainable weights for channel reliability and residual connections
        self.w_ch = nn.Parameter(torch.ones(num_nodes))  # Channel LLR weight
        self.w_res = nn.Parameter(torch.ones(depth_L))  # Residual connection weights
    
    def forward(self, input_llr, check_messages, prev_var_messages):
        """
        Apply residual connections to variable node messages.
        
        Args:
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            check_messages (torch.Tensor): Messages from check nodes of shape (batch_size, num_nodes)
            prev_var_messages (list): List of previous variable node messages, each of shape (batch_size, num_nodes)
            
        Returns:
            torch.Tensor: Updated variable node messages with residual connections of shape (batch_size, num_nodes)
        """
        batch_size = input_llr.shape[0]
        
        # Apply channel weights
        weighted_llr = input_llr * self.w_ch.unsqueeze(0)
        
        # Add check node messages
        result = weighted_llr + check_messages
        
        # Add residual connections from previous iterations
        for i, prev_messages in enumerate(prev_var_messages):
            if i < self.depth_L:
                result = result + self.w_res[i] * prev_messages
        
        return result


class OutputLayer(nn.Module):
    """
    Output Layer for LDPC Neural Decoder.
    
    Maps the final LLRs to soft-bit values and computes the loss.
    """
    def __init__(self):
        super(OutputLayer, self).__init__()
    
    def forward(self, final_llr, input_llr, ground_truth=None):
        """
        Compute soft-bit values and optionally the loss.
        
        Args:
            final_llr (torch.Tensor): Final LLRs from the decoder of shape (batch_size, num_nodes)
            input_llr (torch.Tensor): Input LLRs from channel of shape (batch_size, num_nodes)
            ground_truth (torch.Tensor, optional): True bits (0 or 1) of shape (batch_size, num_nodes)
            
        Returns:
            tuple: (soft_bits, loss)
                - soft_bits (torch.Tensor): Soft bit values (probabilities) of shape (batch_size, num_nodes)
                - loss (torch.Tensor, optional): Loss value if ground_truth is provided
        """
        # Combine final LLR with input LLR
        combined_llr = final_llr + input_llr
        
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