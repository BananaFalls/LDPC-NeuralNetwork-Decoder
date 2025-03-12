"""
Convert Jupyter notebook to modular Python files.

This script extracts code from the Jupyter notebook and organizes it into the modular structure.
"""

import json
import os
import re
import shutil
import sys

def extract_code_cells(notebook_path):
    """Extract code cells from a Jupyter notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_cells.append(''.join(cell['source']))
    
    return code_cells

def create_directory_structure():
    """Create the directory structure if it doesn't exist."""
    directories = [
        'data',
        'models/saved_models',
        'training',
        'utils',
        'visualization',
        'results'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join('ldpc_neural_decoder', directory), exist_ok=True)

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python convert_notebook.py <path_to_notebook>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    
    # Extract code cells
    code_cells = extract_code_cells(notebook_path)
    
    # Create directory structure
    create_directory_structure()
    
    # Copy the notebook to the data directory
    shutil.copy(notebook_path, os.path.join('ldpc_neural_decoder', 'data'))
    
    print(f"Extracted {len(code_cells)} code cells from {notebook_path}")
    print("Directory structure created")
    print("Notebook copied to data directory")
    print("\nTo complete the conversion:")
    print("1. Review the extracted code cells")
    print("2. Organize the code into the appropriate modules")
    print("3. Run the main script to test the implementation")

if __name__ == '__main__':
    main() 