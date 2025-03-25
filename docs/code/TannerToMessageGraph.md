# TannerToMessageGraph Class Documentation

## Overview

The `TannerToMessageGraph` class is a utility class that converts a traditional Tanner graph (represented by a parity-check matrix) into a message-centered graph where messages become nodes. This transformation is a key component of the Message-Centered GNN LDPC Decoder architecture.

## Class Definition

```python
class TannerToMessageGraph:
    """
    Converter from a Tanner graph to a message-centered graph.
    
    This class converts a parity-check matrix (representing a Tanner graph) to a message-centered graph
    where messages (edges in the Tanner graph) become nodes in the new graph.
    """
```

## Constructor Parameters

```python
def __init__(self, H):
```

- `H` (torch.Tensor or np.ndarray): Parity-check matrix representing the LDPC code

## Initialization Process

1. Convert numpy array to torch tensor if needed:
   ```python
   if isinstance(H, np.ndarray):
       H = torch.from_numpy(H).float()
   ```

2. Store parity-check matrix and dimensions:
   ```python
   self.H = H
   self.m, self.n = H.shape  # m = number of check nodes, n = number of variable nodes
   ```

3. Create variable-to-check and check-to-variable mappings:
   ```python
   self.var_to_checks = {}   # var_idx -> list of check_idx
   self.check_to_vars = {}   # check_idx -> list of var_idx
   
   # Fill mappings based on non-zero entries in H
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
   ```

4. Create message nodes and mappings:
   ```python
   self.messages = []               # list of (check_idx, var_idx) tuples
   self.message_to_var = {}         # message_idx -> var_idx
   self.message_to_check = {}       # message_idx -> check_idx
   self.var_to_messages = {}        # var_idx -> list of message_idx
   self.check_to_messages = {}      # check_idx -> list of message_idx
   
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
   ```

5. Store the number of messages and create default message types:
   ```python
   self.num_messages = len(self.messages)
   self.message_types = torch.zeros(self.num_messages, dtype=torch.long)
   ```

6. Create adjacency matrices for message passing:
   ```python
   self.create_adjacency_matrices()
   ```

## Method: create_adjacency_matrices

```python
def create_adjacency_matrices(self):
```

Creates two adjacency matrices for message passing in the message-centered graph:

1. `var_to_check_adjacency`: For messages connected through the same variable node
2. `check_to_var_adjacency`: For messages connected through the same check node

### Process Flow

1. Initialize adjacency matrices:
   ```python
   self.var_to_check_adjacency = torch.zeros((self.num_messages, self.num_messages))
   self.check_to_var_adjacency = torch.zeros((self.num_messages, self.num_messages))
   ```

2. Set up var-to-check adjacency (messages connected through same variable):
   ```python
   for var_idx in range(self.n):
       messages_index = self.var_to_messages[var_idx]
       for i, msg1 in enumerate(messages_index):
           for j, msg2 in enumerate(messages_index):
               if i != j:  # Don't connect message to itself
                   self.var_to_check_adjacency[msg1, msg2] = 1.0
   ```

   This means that message with index `msg1` is connected to the message with index `msg2` through the same variable.

3. Set up check-to-var adjacency (messages connected through same check):
   ```python
   for check_idx in range(self.m):
       messages_index = self.check_to_messages[check_idx]
       for i, msg1 in enumerate(messages_index):
           for j, msg2 in enumerate(messages_index):
               if i != j:  # Don't connect message to itself
                   self.check_to_var_adjacency[msg1, msg2] = 1.0
   ```

   This means that message with index `msg1` is connected to the message with index `msg2` through the same check.

## Getter Methods

### get_message_to_var_mapping

```python
def get_message_to_var_mapping(self):
    """
    Get a tensor mapping each message to its variable node.
    
    Returns:
        torch.Tensor: Tensor with shape (num_messages,) where each entry is the variable index
    """
```

Creates and returns a tensor mapping each message to its variable node.

### get_message_to_check_mapping

```python
def get_message_to_check_mapping(self):
    """
    Get a tensor mapping each message to its check node.
    
    Returns:
        torch.Tensor: Tensor with shape (num_messages,) where each entry is the check index
    """
```

Creates and returns a tensor mapping each message to its check node.

### get_var_to_messages_mapping

```python
def get_var_to_messages_mapping(self):
    """
    Get a packed tensor mapping each variable to its connected messages.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Variable-to-message indices and a tensor indicating where each variable's list starts
    """
```

Creates and returns a packed representation of the variable-to-message mapping.

### get_check_to_messages_mapping

```python
def get_check_to_messages_mapping(self):
    """
    Get a packed tensor mapping each check to its connected messages.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Check-to-message indices and a tensor indicating where each check's list starts
    """
```

Creates and returns a packed representation of the check-to-message mapping.

## Typical Usage

The `TannerToMessageGraph` class is typically used in conjunction with the `MessageGNNDecoder` class to set up the message-centered graph:

```python
# Load LDPC code
H = load_ldpc_code('path/to/ldpc_code.txt')

# Create converter
converter = TannerToMessageGraph(H)

# Access key properties
num_messages = converter.num_messages
message_to_var_mapping = converter.get_message_to_var_mapping()
var_to_check_adjacency = converter.var_to_check_adjacency
check_to_var_adjacency = converter.check_to_var_adjacency
```

## Key Features

1. **Graph Transformation**: Converts a traditional Tanner graph to a message-centered graph.
   
2. **Message Mappings**: Maintains mappings between messages, variables, and checks.
   
3. **Adjacency Matrices**: Creates adjacency matrices for efficient message passing.
   
4. **Message Types**: Supports message type assignment for weight sharing.

## Graph Theory Interpretation

The `TannerToMessageGraph` implements a graph transformation where:

1. Edges in the original Tanner graph become nodes in the new graph
2. Nodes connected to the same variable or check node in the original graph become connected in the new graph
3. The resulting graph structure captures the message passing operations in belief propagation
4. The adjacency matrices provide an efficient way to implement message passing 