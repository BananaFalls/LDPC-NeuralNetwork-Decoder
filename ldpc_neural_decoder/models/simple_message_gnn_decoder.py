import torch

class SimpleMessageGNNDecoder:
    """
    A simplified version of the Message GNN Decoder without neural network components.
    Uses simple message passing with weights set to 1.
    """
    def __init__(self, num_iterations=3):
        self.num_iterations = num_iterations

    def forward(self, input_llr, message_to_var_mapping, message_types, 
                var_to_check_adjacency, check_to_var_adjacency, ground_truth=None):
        """
        Forward pass of the decoder.
        
        Args:
            input_llr (torch.Tensor): Initial LLR values for variable nodes
            message_to_var_mapping (torch.Tensor): Mapping from messages to variable nodes
            message_types (torch.Tensor): Types of messages (not used in simplified version)
            var_to_check_adjacency (torch.Tensor): Variable to check node adjacency
            check_to_var_adjacency (torch.Tensor): Check to variable node adjacency
            ground_truth (torch.Tensor, optional): Ground truth for loss calculation (not used)
            
        Returns:
            torch.Tensor: Decoded probabilities for variable nodes
        """
        batch_size = input_llr.shape[0]
        num_vars = input_llr.shape[1]
        
        # Initialize variable node values with input LLRs
        var_values = input_llr.clone()
        
        # Message passing iterations
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            # Variable to Check messages
            # Each variable node sends its current value to all connected check nodes
            var_to_check_messages = var_values.unsqueeze(-1) * var_to_check_adjacency
            
            # Check to Variable messages
            # Each check node processes messages from connected variable nodes
            # For each check node, multiply all incoming messages
            check_to_var_messages = torch.zeros_like(var_to_check_messages)
            for i in range(batch_size):
                for j in range(var_to_check_messages.shape[1]):
                    # Get messages from variable nodes connected to this check node
                    incoming_messages = var_to_check_messages[i, j]
                    # Multiply all non-zero messages (tanh rule)
                    check_to_var_messages[i, j] = torch.prod(torch.tanh(incoming_messages / 2), dim=0)
            
            # Update variable node values
            # Combine input LLR with all incoming messages
            var_values = input_llr.clone()
            for i in range(batch_size):
                for j in range(num_vars):
                    # Get all messages for this variable node
                    messages = check_to_var_messages[i, j]
                    # Sum all non-zero messages
                    var_values[i, j] += torch.sum(messages)
            
            # Print some debug information
            if iteration == 0:  # Print only first iteration for clarity
                print("\nVariable to Check messages (first few):")
                print(var_to_check_messages[0, :3])
                print("\nCheck to Variable messages (first few):")
                print(check_to_var_messages[0, :3])
                print("\nUpdated variable values (first few):")
                print(var_values[0, :3])
        
        # Convert to probabilities using sigmoid
        decoded_probs = torch.sigmoid(var_values)
        
        return decoded_probs 