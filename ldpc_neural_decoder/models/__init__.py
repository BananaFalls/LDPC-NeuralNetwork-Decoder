"""
Neural network models for LDPC decoding.
"""

# Remove the import that's causing the error
# from ldpc_neural_decoder.models.decoder import LDPCNeuralDecoder, TiedNeuralLDPCDecoder
from ldpc_neural_decoder.models.layers import CheckLayer, VariableLayer, ResidualLayer, OutputLayer
from ldpc_neural_decoder.models.traditional_decoders import BeliefPropagationDecoder, MinSumScaledDecoder
# Remove the import for the missing gnn_ldpc_decoder.py file
# from ldpc_neural_decoder.models.gnn_ldpc_decoder import (
#     GNNLDPCDecoder, 
#     BaseGraphGNNDecoder,
#     GNNCheckLayer,
#     GNNVariableLayer,
#     GNNResidualLayer,
#     GNNOutputLayer
# )
from ldpc_neural_decoder.models.message_gnn_decoder import (
    MessageGNNLayer,
    MessageGNNDecoder,
    TannerToMessageGraph,
    create_message_gnn_decoder,
    CustomVariableMessageGNNLayer,
    CustomVariableMessageGNNDecoder,
    create_custom_variable_message_gnn_decoder,
    CustomCheckMessageGNNLayer,
    CustomMinSumMessageGNNDecoder,
    create_check_index_tensor,
    create_custom_minsum_message_gnn_decoder
)

__all__ = [
    # Remove the missing classes
    # 'LDPCNeuralDecoder',
    # 'TiedNeuralLDPCDecoder',
    'CheckLayer',
    'VariableLayer',
    'ResidualLayer',
    'OutputLayer',
    'BeliefPropagationDecoder',
    'MinSumScaledDecoder',
    # Remove the missing classes
    # 'GNNLDPCDecoder',
    # 'BaseGraphGNNDecoder',
    # 'GNNCheckLayer',
    # 'GNNVariableLayer',
    # 'GNNResidualLayer',
    # 'GNNOutputLayer',
    'MessageGNNDecoder',
    'MessageGNNLayer',
    'TannerToMessageGraph',
    'create_message_gnn_decoder',
    'CustomVariableMessageGNNLayer',
    'CustomVariableMessageGNNDecoder',
    'create_custom_variable_message_gnn_decoder',
    'CustomCheckMessageGNNLayer',
    'CustomMinSumMessageGNNDecoder',
    'create_check_index_tensor',
    'create_custom_minsum_message_gnn_decoder'
] 