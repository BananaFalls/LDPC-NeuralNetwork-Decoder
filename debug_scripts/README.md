# Debug Scripts for Message-GNN LDPC Decoder

This directory contains test scripts for debugging and validating the functionality of classes in the Message-GNN LDPC Decoder implementation.

## Test Scripts

The following test scripts are available:

- `test_variable_gnn_layer.py`: Tests the `VariableGNNLayer` class
- `test_check_gnn_layer.py`: Tests the `CheckGNNLayer` class
- `test_tanner_to_message_graph.py`: Tests the `TannerToMessageGraph` class
- `test_message_gnn_decoder.py`: Tests the complete `MessageGNNDecoder` class
- `test_message_graph_utils.py`: Tests utility functions for message graph processing

## Running Tests

To run a single test, use:

```bash
python test_script_name.py
```

To run all tests, use:

```bash
python run_all_tests.py
```

This will execute all tests in sequence and provide a summary of the results. A detailed log file will be generated in the format `test_results_YYYYMMDD_HHMMSS.txt`.

## Output Files

Each test generates visualization files to help understand the model's behavior:

- `variable_gnn_layer_test.png`: Visualization of original vs. updated features in the Variable GNN layer
- `check_gnn_layer_test.png`: Visualization of original vs. updated features in the Check GNN layer
- `tanner_graph_test.png`: Visualization of the Tanner graph
- `message_graph_test.png`: Visualization of the message-centered graph
- `message_gnn_decoder_test.png`: Visualization of decoding results
- `tanner_graph_util_test.png`: Visualization of Tanner graph from utility functions
- `message_graph_util_test.png`: Visualization of message graph from utility functions
- `llr_distribution_test.png`: Visualization of LLR distributions
- `llr_snr_comparison_test.png`: Visualization of LLR distributions at different SNR values

## Test Coverage

The tests check the following functionalities:

1. Initialization of model components with various parameters
2. Forward pass through each layer
3. Shape consistency of input and output tensors
4. Message passing between variable and check nodes
5. Graph conversion (Tanner to message-centered)
6. Visualization and utility functions
7. Decoding performance on simple examples 