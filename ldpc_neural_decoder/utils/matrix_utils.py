"""
Matrix Utility Functions for LDPC Decoding

This module provides utility functions for working with LDPC parity-check matrices,
including functions to create mapping tensors for check and variable nodes.
"""

import torch
import numpy as np


def get_LLR_indexes(H_to_LLR_mapping_T):
    """
    Creates mapping tensors for check and variable node operations.
    
    For each LLR index, finds all other LLR indices that share the same row (check node)
    or column (variable node) in the parity-check matrix.
    
    Args:
        H_to_LLR_mapping_T (torch.Tensor): Transposed mapping matrix (num_vars, num_checks)
                                          with LLR indices at positions where H_T == 1
    
    Returns:
        tuple: (check_indices_matrix, var_indices_matrix)
            - check_indices_matrix: 2D tensor where each row corresponds to an LLR index,
              and columns contain indices of other LLRs in the same check node.
            - var_indices_matrix: 2D tensor where each row corresponds to an LLR index,
              and columns contain indices of other LLRs in the same variable node.
    """
    num_ones = (H_to_LLR_mapping_T >= 0).sum().item()  # Number of valid LLR indices

    # Dictionary to store shared indices for each LLR index
    check_indices_dict = {i: [] for i in range(num_ones)}
    var_indices_dict = {i: [] for i in range(num_ones)}

    # Obtain Check Layer LLR Indexes
    for row in range(H_to_LLR_mapping_T.shape[0]):
        # Get valid indices (ignore -1)
        check_indices = H_to_LLR_mapping_T[row][H_to_LLR_mapping_T[row] != -1]

        # Map each LLR index to other LLR indices in the same row
        for idx in check_indices:
            check_indices_dict[idx.item()] = [j.item() for j in check_indices if j != idx]

    # Obtain Variable Layer LLR Indexes
    for row in range(H_to_LLR_mapping_T.T.shape[0]):
        # Get valid indices (ignore -1)
        var_indices = H_to_LLR_mapping_T.T[row][H_to_LLR_mapping_T.T[row] != -1]

        # Map each LLR index to other LLR indices in the same row
        for idx in var_indices:
            var_indices_dict[idx.item()] = [j.item() for j in var_indices if j != idx]

    # Convert dictionary to a padded 2D tensor
    check_max_neighbors = max(len(v) for v in check_indices_dict.values())  # Find max list length
    var_max_neighbours = max(len(v) for v in var_indices_dict.values())  # Find max list length

    check_indices_matrix = torch.full((num_ones, check_max_neighbors), -1, dtype=torch.long)  # Initialize with -1
    var_indices_matrix = torch.full((num_ones, var_max_neighbours), -1, dtype=torch.long)  # Initialize with -1

    for i, neighbors in check_indices_dict.items():
        check_indices_matrix[i, :len(neighbors)] = torch.tensor(neighbors)

    for i, neighbors in var_indices_dict.items():
        var_indices_matrix[i, :len(neighbors)] = torch.tensor(neighbors)

    return check_indices_matrix, var_indices_matrix


def create_LLR_mapping(H_T):
    """
    Creates a mapping from the parity-check matrix to LLR indices.
    
    Args:
        H_T (torch.Tensor): Transposed parity-check matrix (num_vars, num_checks)
    
    Returns:
        tuple: (H_to_LLR_mapping_T, check_LLR_matrix, var_LLR_matrix, output_index_tensor)
            - H_to_LLR_mapping_T: Mapping from H_T positions to LLR indices
            - check_LLR_matrix: Check node LLR index mapping
            - var_LLR_matrix: Variable node LLR index mapping
            - output_index_tensor: Mapping from LLR indices to variable nodes
    """
    # Find indices where H_T == 1
    row_indices, col_indices = (H_T == 1).nonzero(as_tuple=True)
    
    # Create a mapping from (row, col) in H_T to LLR index
    H_to_LLR_mapping = torch.full_like(H_T, -1, dtype=torch.long)  # Initialize with -1
    
    # Assign LLR indices in natural order
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        H_to_LLR_mapping[row, col] = i  # Map H index to LLR index
    
    # Transpose the mapping matrix
    H_to_LLR_mapping_T = H_to_LLR_mapping.T  # Transpose
    
    # Compute the shared LLR tensor
    check_LLR_matrix, var_LLR_matrix = get_LLR_indexes(H_to_LLR_mapping_T)
    
    # Create output index tensor (maps LLR indices to variable nodes)
    output_index_tensor = torch.tensor([row_indices], dtype=torch.int64)
    
    return H_to_LLR_mapping_T, check_LLR_matrix, var_LLR_matrix, output_index_tensor


def expand_base_matrix(base_matrix, Z):
    """
    Expands a base matrix using the lifting factor Z.
    
    Args:
        base_matrix (torch.Tensor): Base matrix with shift values
        Z (int): Lifting factor
    
    Returns:
        torch.Tensor: Expanded parity-check matrix
    """
    rows, cols = base_matrix.shape
    H_expanded = torch.zeros((rows * Z, cols * Z), dtype=torch.float32)

    for i in range(rows):
        for j in range(cols):
            shift = base_matrix[i, j].item()
            if shift == -1:
                # Zero Block (ZxZ Zero Matrix)
                H_expanded[i * Z:(i + 1) * Z, j * Z:(j + 1) * Z] = torch.zeros((Z, Z))
            else:
                # Circulant Shifted Identity Matrix
                I_Z = torch.eye(Z)  # Identity Matrix of size ZxZ
                I_Z = torch.roll(I_Z, shifts=int(shift), dims=1)  # Shift columns by "shift" value
                H_expanded[i * Z:(i + 1) * Z, j * Z:(j + 1) * Z] = I_Z

    return H_expanded 