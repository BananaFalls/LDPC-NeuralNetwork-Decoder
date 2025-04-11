# Utility Functions Documentation

This document provides detailed information about the utility functions in the `message_gnn_decoder.py` file.

## safe_index Function

```python
def safe_index(tensor, idx):
    """
    Helper function for safe indexing when some indices might be -1 (indicating padding)
    """
```

### Parameters

- `tensor` (torch.Tensor): The tensor to index into
- `idx` (torch.Tensor): The indices, potentially containing -1 values

### Behavior

1. Creates a mask for valid indices:
   ```python
   mask = idx != -1
   ```

2. Makes a clone of the indices:
   ```python
   safe_idx = idx.clone()
   ```

3. Sets invalid indices to 0 temporarily:
   ```python
   safe_idx[~mask] = 0
   ```

4. Indexes the tensor with the safe indices:
   ```python
   result = tensor[safe_idx]
   ```

5. Returns the result.

### Purpose

This function is used to safely index a tensor when some indices might be invalid (marked as -1). This is particularly useful when dealing with variable-length sequences or padding.

## create_message_gnn_decoder Function

```python
def create_message_gnn_decoder(H, base_graph=None, lifting_factor=None, num_iterations=5, hidden_dim=64, num_of_residual_layers=2):
    """
    Create a Message GNN Decoder based on a parity-check matrix and optionally a base graph.
    """
```

### Parameters

- `H` (torch.Tensor or np.ndarray): Parity-check matrix
- `base_graph` (torch.Tensor or np.ndarray, optional): Base graph for weight sharing
- `lifting_factor` (int, optional): Lifting factor used to create H from base_graph
- `num_iterations` (int, optional): Number of decoding iterations
- `hidden_dim` (int, optional): Hidden dimension for message features
- `num_of_residual_layers` (int, optional): Number of previous layers to use for residual connections

### Process Flow

1. Convert numpy arrays to torch tensors if needed:
   ```python
   if isinstance(H, np.ndarray):
       H = torch.from_numpy(H).float()
   
   if base_graph is not None and isinstance(base_graph, np.ndarray):
       base_graph = torch.from_numpy(base_graph).float()
   ```

2. Convert Tanner graph to message graph:
   ```python
   converter = TannerToMessageGraph(H)
   ```

3. Get the number of message nodes:
   ```python
   num_messages = converter.num_messages
   ```

4. Determine message types based on the base graph (if provided):
   ```python
   num_message_types = 1  # Default: all messages have the same type
   
   if base_graph is not None and lifting_factor is not None:
       # Create message type mapping based on the base graph
       message_types = create_message_type_mapping(H, base_graph, lifting_factor)
       num_message_types = int(message_types.max().item()) + 1
       
       # Store message types in the converter for easy access
       converter.message_types = message_types
   ```

5. Create the decoder with appropriate parameters:
   ```python
   decoder = MessageGNNDecoder(
       num_messages=num_messages,
       num_iterations=num_iterations,
       hidden_dim=hidden_dim,
       num_message_types=num_message_types,
       num_of_residual_layers=num_of_residual_layers
   )
   ```

6. Return the decoder and converter:
   ```python
   return decoder, converter
   ```

### Purpose

This function provides a convenient way to create a Message GNN Decoder based on a parity-check matrix, optionally with weight sharing based on a base graph.

## create_message_type_mapping Function

```python
def create_message_type_mapping(H, base_graph, lifting_factor):
    """
    Create a mapping of message types based on the base graph structure.
    Messages corresponding to the same edge in the base graph will have the same type.
    """
```

### Parameters

- `H` (torch.Tensor): Expanded parity-check matrix
- `base_graph` (torch.Tensor): Base graph
- `lifting_factor` (int): Lifting factor used to expand the base graph

### Process Flow

1. Convert numpy arrays to torch tensors if needed:
   ```python
   if isinstance(H, np.ndarray):
       H = torch.from_numpy(H).float()
   
   if isinstance(base_graph, np.ndarray):
       base_graph = torch.from_numpy(base_graph).float()
   ```

2. Get dimensions of the matrices:
   ```python
   base_m, base_n = base_graph.shape  # Dimensions of base graph
   m, n = H.shape  # Dimensions of expanded graph
   ```

3. Check if dimensions are consistent with lifting:
   ```python
   if m != base_m * lifting_factor or n != base_n * lifting_factor:
       raise ValueError(f"Expanded matrix dimensions {m}x{n} don't match expected dimensions from base graph {base_m * lifting_factor}x{base_n * lifting_factor}")
   ```

4. Create converter for the expanded graph:
   ```python
   converter = TannerToMessageGraph(H)
   num_messages = converter.num_messages
   ```

5. Initialize message types:
   ```python
   message_types = torch.zeros(num_messages, dtype=torch.long)
   ```

6. Create mapping from expanded graph to base graph:
   ```python
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
   ```

7. Return the message types:
   ```python
   return message_types
   ```

### Purpose

This function creates a mapping of message types based on the base graph structure, enabling weight sharing across messages that correspond to the same edge in the base graph.

## create_variable_index_tensor Function

```python
def create_variable_index_tensor(H, converter):
    """
    Create a tensor mapping variable nodes to message nodes.
    """
```

### Parameters

- `H` (torch.Tensor): Parity-check matrix
- `converter` (TannerToMessageGraph): Converter object

### Process Flow

1. Get the number of variables and device:
   ```python
   num_variables = H.shape[1]
   device = H.device
   ```

2. Find max number of messages per variable:
   ```python
   max_messages_per_var = max(len(msgs) for var_idx, msgs in converter.var_to_messages.items())
   ```

3. Create variable-to-message mapping tensor:
   ```python
   variable_index_tensor = torch.full((num_variables, max_messages_per_var), -1, dtype=torch.long, device=device)
   ```

4. Fill the tensor:
   ```python
   for var_idx, msg_indices in converter.var_to_messages.items():
       for i, msg_idx in enumerate(msg_indices):
           variable_index_tensor[var_idx, i] = msg_idx
   ```

5. Return the variable index tensor:
   ```python
   return variable_index_tensor
   ```

### Purpose

This function creates a tensor that maps variable nodes to their associated message nodes, with padding (-1) for variables with fewer messages than the maximum.

## create_check_index_tensor Function

```python
def create_check_index_tensor(H, message_type_map=None):
    """
    Create a tensor mapping check nodes to message nodes.
    """
```

### Parameters

- `H` (torch.Tensor): Parity-check matrix
- `message_type_map` (dict, optional): Mapping from messages to types

### Process Flow

1. Get dimensions and device:
   ```python
   num_checks, num_variables = H.shape
   device = H.device
   ```

2. Create check-to-variable mapping:
   ```python
   check_to_var = [[] for _ in range(num_checks)]
   
   for check_idx in range(num_checks):
       for var_idx in range(num_variables):
           if H[check_idx, var_idx] != 0:
               check_to_var[check_idx].append(var_idx)
   ```

3. Find max number of variables per check:
   ```python
   max_vars_per_check = max(len(vars) for vars in check_to_var)
   ```

4. Create check index tensor:
   ```python
   check_index_tensor = torch.full((num_checks, max_vars_per_check), -1, dtype=torch.long, device=device)
   ```

5. Fill the tensor:
   ```python
   for check_idx, var_indices in enumerate(check_to_var):
       for i, var_idx in enumerate(var_indices):
           check_index_tensor[check_idx, i] = var_idx
   ```

6. Return the check index tensor:
   ```python
   return check_index_tensor
   ```

### Purpose

This function creates a tensor that maps check nodes to their associated variable nodes, with padding (-1) for checks with fewer variables than the maximum. 