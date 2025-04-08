import torch
import torch.nn as nn

class ResidualWeightSharingDecoder(nn.Module):
    """
    A neural LDPC decoder with weight sharing and residual connections.
    The parity check matrix H consists of square blocks (circulants),
    where weights are shared across blocks corresponding to the same base matrix element.
    Residual connections are added to combine outputs from previous iterations.
    """
    def __init__(self, base_matrix, expansion_factor, num_iterations=5, residual_depth=2):
        """
        Initialize the decoder with the base matrix and expansion factor.
        
        Args:
            base_matrix (torch.Tensor): Base parity check matrix Hb
            expansion_factor (int): Size of circulant blocks (expansion factor)
            num_iterations (int): Number of decoding iterations
            residual_depth (int): Number of previous iterations to consider in residual connections
        """
        super().__init__()
        self.Hb = base_matrix
        self.z = expansion_factor  # circulant size
        self.num_iterations = num_iterations
        self.residual_depth = residual_depth
        
        # Expand base matrix to full H matrix using circulants
        self.H = self._expand_base_matrix()
        
        # Create message indices for expanded matrix
        self._create_message_indices()
        
        # Create shared weights based on base matrix
        self._create_shared_weights()
        
        # Neural network for message processing (shared across all messages)
        self.message_processor = nn.Sequential(
            nn.Linear(1, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        # Residual connection weights
        num_messages = int(self.H.sum().item())
        self.residual_weights = nn.ParameterDict()
        for t in range(1, residual_depth + 1):
            self.residual_weights[f"w_{t}"] = nn.Parameter(torch.ones(num_messages))
    
    def _create_circulant(self, shift):
        """
        Create a circulant matrix (shifted identity matrix) of size z×z
        Args:
            shift: If -1, returns zero matrix
                  If 0, returns identity matrix
                  If >0, returns shifted identity matrix
        """
        if shift == -1:  # No connection
            return torch.zeros(self.z, self.z)
        
        circulant = torch.zeros(self.z, self.z)
        for i in range(self.z):
            j = (i + shift) % self.z  # For shift=0, this gives identity matrix
            circulant[i, j] = 1
        return circulant
    
    def _expand_base_matrix(self):
        """
        Expand base matrix into full H matrix using circulants
        """
        rows, cols = self.Hb.shape
        H = torch.zeros(rows * self.z, cols * self.z)
        
        for i in range(rows):
            for j in range(cols):
                shift = int(self.Hb[i, j])
                start_row = i * self.z
                start_col = j * self.z
                H[start_row:start_row + self.z, 
                  start_col:start_col + self.z] = self._create_circulant(shift)
        
        return H
    
    def _create_message_indices(self):
        """
        Create message indices for the expanded matrix
        """
        self.var_to_messages = [[] for _ in range(self.H.shape[1])]
        self.check_to_messages = [[] for _ in range(self.H.shape[0])]
        self.msg_to_base_pos = {}  # Maps message index to base matrix position
        
        msg_idx = 0
        for i in range(self.H.shape[0]):
            for j in range(self.H.shape[1]):
                if self.H[i, j] == 1:
                    self.var_to_messages[j].append(msg_idx)
                    self.check_to_messages[i].append(msg_idx)
                    
                    # Store base matrix position for this message
                    base_row = i // self.z
                    base_col = j // self.z
                    self.msg_to_base_pos[msg_idx] = (base_row, base_col)
                    
                    msg_idx += 1
    
    def _create_shared_weights(self):
        """
        Create shared weights based on base matrix structure
        """
        # Create one set of weights per non-zero element in base matrix
        self.shared_var_to_check_weights = nn.ParameterDict()
        self.shared_check_to_var_weights = nn.ParameterDict()
        
        for base_row in range(self.Hb.shape[0]):
            for base_col in range(self.Hb.shape[1]):
                if self.Hb[base_row, base_col] != -1:  # Create weights for any non-zero connection
                    key = f"{base_row}_{base_col}"
                    self.shared_var_to_check_weights[key] = nn.Parameter(torch.ones(1))
                    self.shared_check_to_var_weights[key] = nn.Parameter(torch.ones(1))
    
    def _get_weight(self, msg_idx, check_to_var=True):
        """
        Get the shared weight for a message based on its position in the base matrix
        """
        base_row, base_col = self.msg_to_base_pos[msg_idx]
        key = f"{base_row}_{base_col}"
        
        if check_to_var:
            return self.shared_check_to_var_weights[key]
        else:
            return self.shared_var_to_check_weights[key]
    
    def variable_node_update(self, var_values, check_to_var_messages, prev_var_messages, iteration, batch_idx):
        """
        Variable node update with shared weights and residual connections
        """        
        num_messages = len(sum(self.var_to_messages, []))
        var_to_check_messages = torch.zeros((var_values.shape[0], num_messages), 
                                          device=var_values.device)
        
        if check_to_var_messages is None:
            # First iteration: just use channel LLR
            for var_idx in range(self.H.shape[1]):
                for msg_idx in self.var_to_messages[var_idx]:
                    # Get shared weight for this block
                    weight = self._get_weight(msg_idx, check_to_var=False)
                    var_to_check_messages[:, msg_idx] = var_values[:, var_idx] * weight
            
            # print(f"[variable_node_update] var_to_check_messages: {var_to_check_messages}")
            return var_to_check_messages
        
        # Regular update with residual connections
        for var_idx in range(self.H.shape[1]):
            channel_llr = var_values[:, var_idx].unsqueeze(-1)  # [batch_size, 1]
            
            connected_checks = torch.where(self.H[:, var_idx] == 1)[0]
            for target_check_idx in connected_checks:
                target_msg_idx = next(msg_idx for msg_idx in self.var_to_messages[var_idx]
                                    if msg_idx in self.check_to_messages[target_check_idx])
                
                # Get messages from other checks in same block
                other_checks = connected_checks[connected_checks != target_check_idx]
                other_msg_indices = []
                for check_idx in other_checks:
                    msg_idx = next(m for m in self.var_to_messages[var_idx]
                                 if m in self.check_to_messages[check_idx])
                    other_msg_indices.append(msg_idx)
                
                # Current iteration messages
                if other_msg_indices:
                    messages = check_to_var_messages[:, other_msg_indices]  # [batch_size, num_messages]
                    weights = torch.stack([self._get_weight(idx) for idx in other_msg_indices])  # [num_messages]
                    weights = weights.view(1, -1)  # [1, num_messages] for broadcasting
                    weighted_sum = torch.sum(messages * weights, dim=1)  # [batch_size]
                else:
                    weighted_sum = torch.zeros_like(channel_llr.squeeze(-1))  # [batch_size]
                
                # Add residual connections from previous iterations
                residual_sum = torch.zeros_like(weighted_sum)  # [batch_size]
                for t in range(1, min(iteration + 1, self.residual_depth + 1)):
                    if t <= len(prev_var_messages):
                        prev_messages = prev_var_messages[-t][:, target_msg_idx]  # [batch_size]
                        residual_weight = self.residual_weights[f"w_{t}"][target_msg_idx]  # scalar
                        residual_sum += prev_messages * residual_weight
                
                # Combine current and residual messages
                message = channel_llr.squeeze(-1) + weighted_sum + residual_sum  # [batch_size]
                weight = self._get_weight(target_msg_idx, False)  # scalar
                message = message * weight
                message = self.message_processor(message.unsqueeze(-1)).squeeze(-1)  # [batch_size]
                var_to_check_messages[:, target_msg_idx] = message
        
        return var_to_check_messages
    
    def check_node_update(self, var_to_check_messages, batch_idx):
        """
        Check node update with shared weights
        """        
        check_to_var_messages = torch.zeros_like(var_to_check_messages)
        
        for check_idx in range(self.H.shape[0]):
            connected_vars = torch.where(self.H[check_idx] == 1)[0]
            
            for target_var_idx in connected_vars:
                target_msg_idx = next(msg_idx for msg_idx in self.check_to_messages[check_idx]
                                    if msg_idx in self.var_to_messages[target_var_idx])
                
                other_vars = connected_vars[connected_vars != target_var_idx]
                other_msg_indices = []
                for var_idx in other_vars:
                    msg_idx = next(m for m in self.check_to_messages[check_idx]
                                 if m in self.var_to_messages[var_idx])
                    other_msg_indices.append(msg_idx)
                
                if other_msg_indices:
                    messages = var_to_check_messages[:, other_msg_indices]  # [batch_size, num_messages]
                    weights = torch.stack([self._get_weight(idx, False) for idx in other_msg_indices])  # [num_messages]
                    
                    # Reshape weights for proper broadcasting
                    weights = weights.view(1, -1)  # [1, num_messages]
                    
                    # Min-sum approximation with shared weights
                    weighted_messages = messages * weights  # [batch_size, num_messages]
                    signs = torch.sign(weighted_messages)
                    sign_product = torch.prod(signs, dim=1)
                    min_magnitude = torch.min(torch.abs(weighted_messages), dim=1)[0]
                    
                    message = sign_product * min_magnitude
                else:
                    message = torch.zeros(var_to_check_messages.shape[0], 
                                        device=var_to_check_messages.device)
                
                # Apply shared weight for this block
                weight = self._get_weight(target_msg_idx)
                message = message * weight
                message = self.message_processor(message.unsqueeze(-1)).squeeze(-1)
                check_to_var_messages[:, target_msg_idx] = message
        
        # print(f"[check_node_update] check_to_var_messages: {check_to_var_messages}")
        return check_to_var_messages
    
    def forward(self, input_llr):
        """
        Forward pass with shared weights and residual connections
        """
        batch_size = input_llr.shape[0]
        var_values = input_llr.clone()
        check_to_var_messages = None
        
        # Store variable node messages for residual connections
        prev_var_messages = []
        
        print("\n=== Decoder Configuration ===")
        print(f"Base matrix shape: {self.Hb.shape}")
        print(f"Expansion factor (z): {self.z}")
        print(f"Full H matrix shape: {self.H.shape}")
        print(f"Number of iterations: {self.num_iterations}")
        print(f"Residual depth: {self.residual_depth}")
        
        for iteration in range(self.num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{self.num_iterations} ===")
            
            for batch_idx in range(batch_size):
                # Variable node update with residual connections
                var_to_check_messages = self.variable_node_update(
                    var_values,
                    check_to_var_messages,
                    prev_var_messages,
                    iteration,
                    batch_idx
                )
                
                # Store variable node messages for residual connections
                prev_var_messages.append(var_to_check_messages.clone())
                if len(prev_var_messages) > self.residual_depth:
                    prev_var_messages.pop(0)  # Remove oldest messages
                
                # Check node update
                check_to_var_messages = self.check_node_update(
                    var_to_check_messages, 
                    batch_idx
                )
                
                # Update variable values using posterior LLR
                var_values = self._compute_posterior_llr(
                    input_llr,
                    check_to_var_messages,
                    batch_idx
                )
        
        # Make hard decisions
        decoded_bits = (var_values < 0).float()
        return decoded_bits
    
    def _compute_posterior_llr(self, input_llr, check_to_var_messages, batch_idx):
        """
        Compute posterior LLR (Log-Likelihood Ratio) with shared weights.
        This function combines channel LLRs with messages from check nodes to compute
        the final LLR for each variable node.
        
        Args:
            input_llr: Initial channel LLRs [batch_size, num_variables]
            check_to_var_messages: Messages from check nodes to variable nodes
            batch_idx: Current batch index
        
        Returns:
            posterior_llr: Updated LLRs after combining channel and check node messages
        """
        # Initialize posterior LLRs with channel LLRs
        posterior_llr = input_llr.clone()
        
        # Iterate through each variable node
        for var_idx in range(self.H.shape[1]):
            # Get channel LLR for current variable node
            channel_llr = input_llr[:, var_idx]
            
            # Find all check nodes connected to current variable node
            connected_checks = torch.where(self.H[:, var_idx] == 1)[0]
            
            # Collect message indices for all connected check nodes
            msg_indices = []
            for check_idx in connected_checks:
                # Find the message index that connects this check node to the variable node
                msg_idx = next(m for m in self.var_to_messages[var_idx]
                             if m in self.check_to_messages[check_idx])
                msg_indices.append(msg_idx)
            
            # If there are connected check nodes
            if msg_indices:
                # Get messages from all connected check nodes
                messages = check_to_var_messages[:, msg_indices]
                
                # Get shared weights for all messages
                weights = torch.stack([self._get_weight(idx) for idx in msg_indices]).T
                
                # Compute weighted sum of messages
                weighted_sum = torch.sum(messages * weights, dim=1)
                
                # Update posterior LLR by combining channel LLR with weighted sum
                posterior_llr[:, var_idx] = channel_llr + weighted_sum
        
        return posterior_llr 