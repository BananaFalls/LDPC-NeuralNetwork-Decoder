import torch

class BatchMessageGNNDecoder:
    """
    A batch-compatible version of the Message GNN Decoder without neural network components.
    Uses simple message passing with weights set to 1.
    """
    def __init__(self, H_matrix, num_iterations=3):
        """
        Initialize the decoder with the parity check matrix.
        
        Args:
            H_matrix (torch.Tensor): The parity check matrix
            num_iterations (int): Number of decoding iterations
        """
        self.H = H_matrix
        self.num_iterations = num_iterations
        
        # Create message indices for each node
        self.var_to_messages = [[] for _ in range(self.H.shape[1])]  # Variable node -> message indices
        self.check_to_messages = [[] for _ in range(self.H.shape[0])]  # Check node -> message indices
        
        # Create message mapping
        msg_idx = 0
        for j in range(self.H.shape[0]):  # for each check node
            for i in range(self.H.shape[1]):  # for each variable node
                if self.H[j,i] == 1:
                    self.var_to_messages[i].append(msg_idx)
                    self.check_to_messages[j].append(msg_idx)
                    msg_idx += 1

    def variable_node_update(self, var_values, check_to_var_messages, batch_idx):
        """
        Variable node update: combine channel LLR with incoming check messages.
        Following the equation:
        m_{v,c}^{(i,j)}(t + 1) = m_i + sum_{k in C_i,k!=j} m_{c,v}^{(k,i)}(t + 1)
        """
        print("\n1. Variable-to-Check Message Creation:")
        print(f"Channel LLR values (batch {batch_idx}):\n{var_values[batch_idx]}")
        
        # Initialize output messages
        num_messages = int(self.H.sum().item())  # Total number of 1s in H matrix
        var_to_check_messages = torch.zeros((var_values.shape[0], num_messages), device=var_values.device)

        if check_to_var_messages is None:
            # First iteration: just use channel LLR
            print("First iteration: using only channel LLR")
            for var_idx in range(self.H.shape[1]):  # for each variable node
                # Get all message indices for this variable
                for msg_idx in self.var_to_messages[var_idx]:
                    var_to_check_messages[:, msg_idx] = var_values[:, var_idx]
                    
                    if msg_idx < 3:  # Only show first 3 messages
                        print(f"\nMessage {msg_idx} (from variable {var_idx}):")
                        print(f"Channel LLR: {var_values[batch_idx, var_idx]}")
            return var_to_check_messages
        
        # For each variable node
        for var_idx in range(self.H.shape[1]):
            # Get channel LLR (m_i)
            channel_llr = var_values[:, var_idx].unsqueeze(-1)  # [batch_size, 1]
            
            connected_checks = torch.where(self.H[:, var_idx] == 1)[0]

            # For each connected check node j
            for target_check_idx in connected_checks:
                # Get message index for this check-var pair
                target_msg_idx = next(msg_idx for msg_idx in self.var_to_messages[var_idx]
                                    if msg_idx in self.check_to_messages[target_check_idx])
                
                other_checks = connected_checks[connected_checks != target_check_idx]
                other_msg_indices = []
                for check_idx in other_checks:
                    # Find message index for this check-var pair
                    msg_idx = next(m for m in self.var_to_messages[var_idx]
                                 if m in self.check_to_messages[check_idx])
                    other_msg_indices.append(msg_idx)

                # Get the messages we want to include 
                messages_to_use = check_to_var_messages[:, other_msg_indices]  # [batch_size, num_other_messages]
                # Sum all messages except the target message
                sum_messages = torch.sum(messages_to_use, dim=1, keepdim=True)  # [batch_size, 1]

                # compute final message
                var_to_check_messages[:, target_msg_idx] = channel_llr.squeeze(-1) + sum_messages.squeeze(-1)
                
                if target_msg_idx < 3:  # Only show first 3 messages
                    print(f"\nMessage {target_msg_idx} (from variable {var_idx} to check {target_check_idx}):")
                    print(f"Channel LLR: {channel_llr[batch_idx]}")
                    print(f"Sum of other check messages: {sum_messages[batch_idx]}")
                    print(f"Final message: {var_to_check_messages[batch_idx, target_msg_idx]}")
        
        return var_to_check_messages

    def check_node_update(self, var_to_check_messages, batch_idx):
        """
        Check node update using min-sum approximation or tanh-based update.
        """
        print("\n2. Check-to-Variable Message Creation:")
        check_to_var_messages = torch.zeros_like(var_to_check_messages)
        
        # For each check node j
        for check_idx in range(self.H.shape[0]):
            # Get indices of connected variable nodes (where H[j,i] = 1)
            connected_vars = torch.where(self.H[check_idx] == 1)[0]
            
            # For each connected variable node i
            for target_var_idx in connected_vars:
                # Get message index for this check-var pair
                target_msg_idx = next(msg_idx for msg_idx in self.check_to_messages[check_idx]
                                    if msg_idx in self.var_to_messages[target_var_idx])
                
                # Get messages from OTHER connected variable nodes (excluding target)
                other_vars = connected_vars[connected_vars != target_var_idx]
                other_msg_indices = []
                for var_idx in other_vars:
                    # Find message index for this check-var pair
                    msg_idx = next(m for m in self.check_to_messages[check_idx]
                                 if m in self.var_to_messages[var_idx])
                    other_msg_indices.append(msg_idx)
                
                # Get the messages we want to include
                messages_to_use = var_to_check_messages[:, other_msg_indices]  # [batch_size, num_other_messages]
                
                if target_msg_idx < 3:  # Only show first 3 messages
                    print(f"\nCheck node {check_idx}, computing message to var {target_var_idx}:")
                    print(f"Connected variables: {connected_vars.tolist()}")
                    print(f"Other variables (excluding {target_var_idx}): {other_vars.tolist()}")
                    print(f"Messages used: {messages_to_use[batch_idx]}")
                
                # OPTION 1: Min-sum approximation
                # signs = torch.sign(messages_to_use)  # [batch_size, num_other_messages]
                # sign_product = torch.prod(signs, dim=1)  # [batch_size]
                # min_magnitude = torch.min(torch.abs(messages_to_use), dim=1)[0]  # [batch_size]
                # check_to_var_messages[:, target_msg_idx] = sign_product * min_magnitude
                
                # OPTION 2: Exact computation using tanh
                prod_tanh = torch.prod(torch.tanh(messages_to_use / 2), dim=1)  # [batch_size]
                check_to_var_messages[:, target_msg_idx] = 2 * torch.atanh(prod_tanh)
                
                if target_msg_idx < 3:
                    print(f"Final message: {check_to_var_messages[batch_idx, target_msg_idx]}")
        
        return check_to_var_messages

    def compute_posterior_llr(self, input_llr, check_to_var_messages, batch_idx):
        """
        Compute posterior LLR using:
        l_{D,i}(t + 1) = m_i + sum_{k in C_i} m_{c,v}^{(k,i)}(t + 1)
        """
        print("\n=== Computing Posterior LLR ===")
        posterior_llr = input_llr.clone()
        
        # For each variable node i
        for var_idx in range(self.H.shape[1]):
            # Get channel LLR (m_i)
            channel_llr = input_llr[:, var_idx].unsqueeze(-1)  # [batch_size, 1]
            
            connected_checks = torch.where(self.H[:, var_idx] == 1)[0]
            msg_indices = []
            for check_idx in connected_checks:
                # Find message index for this check-var pair
                msg_idx = next(m for m in self.var_to_messages[var_idx]
                             if m in self.check_to_messages[check_idx])
                msg_indices.append(msg_idx)
            
            # Get all messages to this variable
            messages_to_use = check_to_var_messages[:, msg_indices]  # [batch_size, num_messages]
            # Sum all messages
            sum_messages = torch.sum(messages_to_use, dim=1, keepdim=True)  # [batch_size, 1]
            
            # Compute final posterior
            posterior_llr[:, var_idx] = channel_llr.squeeze(-1) + sum_messages.squeeze(-1)
            
            if var_idx < 3:  # Show details for first 3 variables
                print(f"\nVariable node {var_idx}:")
                print(f"Channel LLR: {channel_llr[batch_idx]}")
                print(f"Sum of check messages: {sum_messages[batch_idx]}")
                print(f"Posterior LLR: {posterior_llr[batch_idx, var_idx]}")
        
        return posterior_llr

    def forward(self, input_llr, ground_truth=None):
        """
        Forward pass of the decoder.
        
        Args:
            input_llr (torch.Tensor): Initial LLR values for variable nodes [batch_size, num_vars]
            ground_truth (torch.Tensor, optional): Ground truth for loss calculation
            
        Returns:
            torch.Tensor: Decoded bits (hard decisions) [batch_size, num_vars]
        """
        batch_size = input_llr.shape[0]
        
        print("\n=== Decoder Configuration ===")
        print(f"H matrix shape: {self.H.shape}")
        print(f"Batch size: {batch_size}")
        print(f"Number of iterations: {self.num_iterations}")
        print(f"Input LLR values (first batch):\n{input_llr[0]}")
        
        # Initialize variable values with input LLRs
        var_values = input_llr.clone()
        check_to_var_messages = None
        
        # Message passing iterations
        for iteration in range(self.num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            
            old_var_values = var_values.clone()
            
            for batch_idx in range(batch_size):
                # 1. Variable Node Update
                var_to_check_messages = self.variable_node_update(
                    var_values,
                    check_to_var_messages,
                    batch_idx
                )
                
                # 2. Check Node Update
                check_to_var_messages = self.check_node_update(
                    var_to_check_messages, 
                    batch_idx
                )
                
                # 3. Update variable values
                var_values = self.compute_posterior_llr(
                    input_llr,
                    check_to_var_messages,
                    batch_idx
                )
            
            print("\n=== Iteration Summary ===")
            print(f"Variable values change (first 3):")
            print(f"Old: {old_var_values[0, :3]}")
            print(f"New: {var_values[0, :3]}")
        
        # Make hard decisions based on final LLR values
        # If l_D,i(t) >= 0, decode as 0; otherwise decode as 1
        decoded_bits = (var_values < 0).float()
        
        print("\n=== Final Results ===")
        print(f"Final LLR values (first batch):\n{var_values[0]}")
        print(f"Decoded bits (first batch):\n{decoded_bits[0]}")
        
        # Verify parity check equations
        for batch_idx in range(batch_size):
            syndrome = torch.matmul(decoded_bits[batch_idx], self.H.t()) % 2
            is_valid = torch.all(syndrome == 0)
            print(f"\nBatch {batch_idx} codeword validity check (c × H^T = 0):")
            print(f"Is valid codeword: {is_valid}")
            if not is_valid:
                print(f"Syndrome: {syndrome}")
        
        return decoded_bits 