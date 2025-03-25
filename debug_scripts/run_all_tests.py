#!/usr/bin/env python3
"""
Script to run all test cases for the Message-GNN LDPC Decoder.

This script executes all test scripts in sequence and reports overall status.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# List of test scripts to run
TEST_SCRIPTS = [
    "test_variable_gnn_layer.py",
    "test_check_gnn_layer.py",
    "test_tanner_to_message_graph.py",
    "test_message_gnn_decoder.py",
    "test_message_graph_utils.py",
]


def run_test(script_path):
    """
    Run a test script and return success status.
    
    Args:
        script_path (str): Path to the test script
        
    Returns:
        tuple: (success, output) - Boolean indicating success and output string
    """
    print(f"\n{'='*80}")
    print(f"Running test: {os.path.basename(script_path)}")
    print(f"{'='*80}")
    
    try:
        # Run the script and capture output
        start_time = time.time()
        process = subprocess.run(
            [sys.executable, script_path],
            check=False,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        # Check result
        success = process.returncode == 0
        output = process.stdout
        
        # Print output
        print(output)
        
        if process.stderr:
            print("Errors:")
            print(process.stderr)
        
        status = "PASSED" if success else "FAILED"
        print(f"\nTest {status} in {duration:.2f} seconds (Return code: {process.returncode})")
        
        return success, output
        
    except Exception as e:
        print(f"Error running test: {e}")
        return False, str(e)


def main():
    """Main function to run all tests."""
    # Get the directory this script is in
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Output file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(script_dir, f"test_results_{timestamp}.txt")
    
    # Track results
    results = []
    all_outputs = []
    
    # Print header
    print("\n" + "="*80)
    print(f"Running all tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Run each test
    start_time = time.time()
    for script in TEST_SCRIPTS:
        script_path = os.path.join(script_dir, script)
        
        if os.path.exists(script_path):
            success, output = run_test(script_path)
            results.append((script, success))
            all_outputs.append(f"\n\n{'='*40}\n{script}\n{'='*40}\n{output}")
        else:
            print(f"Test script not found: {script_path}")
            results.append((script, False))
            all_outputs.append(f"\n\n{'='*40}\n{script}\n{'='*40}\nSKIPPED - File not found")
    
    total_time = time.time() - start_time
    
    # Summarize results
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    pass_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for script, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{script}: {status}")
    
    print(f"\nPassed {pass_count} out of {total_count} tests ({pass_count/total_count:.1%})")
    print(f"Total time: {total_time:.2f} seconds")
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write(f"TEST RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("SUMMARY:\n")
        for script, success in results:
            status = "PASSED" if success else "FAILED"
            f.write(f"{script}: {status}\n")
        
        f.write(f"\nPassed {pass_count} out of {total_count} tests ({pass_count/total_count:.1%})\n")
        f.write(f"Total time: {total_time:.2f} seconds\n")
        
        f.write("\n\nDETAILED OUTPUT:")
        for output in all_outputs:
            f.write(output)
    
    print(f"\nDetailed results written to: {output_file}")
    
    # Return exit code based on all tests passing
    return 0 if pass_count == total_count else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 