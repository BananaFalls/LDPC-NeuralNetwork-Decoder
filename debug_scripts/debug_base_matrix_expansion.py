import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the project root to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.utils.ldpc_utils import load_base_matrix, expand_base_matrix

def display_matrix(matrix, title, max_size=10):
    """
    Display a matrix with a heatmap
    If matrix is too large, show only a max_size x max_size section
    """
    plt.figure(figsize=(10, 8))
    
    # If matrix is large, show only a portion
    if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
        display_matrix = matrix[:max_size, :max_size].numpy()
        title = f"{title} (showing {max_size}x{max_size} section)"
    else:
        display_matrix = matrix.numpy()
    
    plt.imshow(display_matrix, cmap='Blues')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    # Add grid lines
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.close()

def debug_expansion(file_path, Z=4):
    """
    Load a base matrix, expand it, and visualize the process
    """
    print(f"Loading base matrix from {file_path}")
    
    # Load the base matrix
    base_matrix = load_base_matrix(file_path)
    
    # Print base matrix properties
    print(f"Base matrix shape: {base_matrix.shape}")
    print(f"Base matrix values range from {base_matrix.min().item()} to {base_matrix.max().item()}")
    
    # Count frequency of each value
    unique_values, counts = torch.unique(base_matrix, return_counts=True)
    print("\nFrequency of values in base matrix:")
    for val, count in zip(unique_values.tolist(), counts.tolist()):
        print(f"  Value {val}: {count} occurrences ({count / base_matrix.numel() * 100:.2f}%)")
    
    # Expand the base matrix
    print(f"\nExpanding base matrix with lifting factor Z={Z}")
    expanded_matrix = expand_base_matrix(base_matrix, Z)
    
    # Print expanded matrix properties
    print(f"Expanded matrix shape: {expanded_matrix.shape}")
    print(f"Expanded matrix values are: {torch.unique(expanded_matrix).tolist()}")
    
    # Calculate density
    base_density = (base_matrix >= 0).float().mean().item()
    expanded_density = (expanded_matrix > 0).float().mean().item()
    print(f"\nBase matrix density: {base_density:.6f} ({(base_matrix >= 0).sum().item()} non-zero entries)")
    print(f"Expanded matrix density: {expanded_density:.6f} ({(expanded_matrix > 0).sum().item()} non-zero entries)")
    
    # Visualize matrices
    display_matrix(base_matrix, "Base Matrix")
    display_matrix(expanded_matrix, "Expanded Matrix")
    
    # Let's also visualize a small section to show the expansion pattern
    # Take a 3x3 section of the base matrix and its corresponding expanded section
    if base_matrix.shape[0] >= 3 and base_matrix.shape[1] >= 3:
        base_section = base_matrix[:3, :3]
        expanded_section = expanded_matrix[:3*Z, :3*Z]
        
        print("\nBase matrix section (3x3):")
        print(base_section)
        
        print(f"\nExpanded matrix section ({3*Z}x{3*Z}):")
        for i in range(3):
            for j in range(3):
                block = expanded_section[i*Z:(i+1)*Z, j*Z:(j+1)*Z]
                print(f"Block [{i},{j}] (base value: {base_section[i,j].item()}):")
                print(block)
                print()
        
        display_matrix(base_section, "Base Matrix Section (3x3)")
        display_matrix(expanded_section, "Expanded Matrix Section")

def main():
    # Path to 5G NR LDPC base matrix
    file_path = os.path.join(project_root, "5G LDPC CODES", "NR_2_0_4.txt")
    print(f"Using absolute path: {file_path}")
    
    # Debug expansion with default lifting factor of 4
    debug_expansion(file_path, Z=4)
    
    print("\nComplete! Matrix visualizations have been saved as PNG files.")

if __name__ == "__main__":
    main() 