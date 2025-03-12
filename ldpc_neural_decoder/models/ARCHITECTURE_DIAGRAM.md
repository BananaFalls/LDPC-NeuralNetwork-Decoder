# Architecture Diagrams for LDPC Neural Decoders

This document provides visual representations of the different LDPC decoder architectures, with a focus on the message-centered GNN approach.

## Traditional Tanner Graph vs. Message Graph

### Traditional Tanner Graph
```
    Variable Nodes (V)                 Check Nodes (C)
    
        V1 o                              o C1
           \                            / |
            \                          /  |
             \                        /   |
        V2 o---------------------------o C2
           |  \                      /
           |   \                    /
           |    \                  /
        V3 o-----\----------------o C3
                  \
                   \
                    \
        V4 o---------\------------o C4
                      \
                       \
                        \
        V5 o-------------\--------o C5
                          \
                           \
        V6 o----------------\----o C6
                             \
                              \
        V7 o-------------------\o C7
```

### Message-Centered Graph (Line Graph)
```
                  Messages as Nodes
                  
                M1 o---------o M2
                   \         /
                    \       /
                     \     /
                      \   /
                       \ /
                M3 o----o----o M4
                   \    |    /
                    \   |   /
                     \  |  /
                      \ | /
                M5 o---o-o---o M6
                       |
                       |
                M7 o---o---o M8
                       |
                       |
                M9 o---o---o M10
                       |
                       |
                M11 o--o--o M12
```

In the message-centered approach, each edge in the Tanner graph becomes a node in the message graph. Two message-nodes are connected if they share the same variable or check node in the original Tanner graph.

## Weight Sharing Visualization

### Standard Neural Decoder (Limited Weight Sharing)
```
    V1 [w1,1 w1,2 ... w1,d]
    V2 [w2,1 w2,2 ... w2,d]
    ...
    Vn [wn,1 wn,2 ... wn,d]
    
    C1 [w'1,1 w'1,2 ... w'1,d']
    C2 [w'2,1 w'2,2 ... w'2,d']
    ...
    Cm [w'm,1 w'm,2 ... w'm,d']
```
Each variable and check node has its own set of weights.

### GNN-based Decoder (Edge-Type Weight Sharing)
```
    Edge Type 1: [w1,1 w1,2 ... w1,h]
    Edge Type 2: [w2,1 w2,2 ... w2,h]
    ...
    Edge Type k: [wk,1 wk,2 ... wk,h]
    
    Neural Networks (Shared):
    Variable Update: NN_V(features)
    Check Update: NN_C(features)
```
Weights are shared across edges of the same type.

### Message-Centered GNN (Message-Type Weight Sharing)
```
    Message Type 1: [w1,1 w1,2 ... w1,h]
    Message Type 2: [w2,1 w2,2 ... w2,h]
    ...
    Message Type k: [wk,1 wk,2 ... wk,h]
    
    Neural Networks (Shared across ALL messages):
    Var-to-Check Update: NN_VC(features)
    Check-to-Var Update: NN_CV(features)
```
Weights are shared across messages of the same type, and the same neural networks are used for all message updates.

## Message Passing Visualization

### Traditional Belief Propagation
```
    Variable Node Update:
    
    LLR_v1 --→
              \
    M_c1→v1 ---+--→ M_v1→c2
              /
    M_c3→v1 --→
    
    Check Node Update:
    
    M_v1→c1 --→
              \
    M_v2→c1 ---+--→ M_c1→v3
              /
    M_v4→c1 --→
```

### Message-Centered GNN
```
    Message Node Update:
    
                 ┌─── M_v1→c2
                 │
    M_v1→c1 ─────┼─── M_v1→c3
                 │
                 └─── M_v1→c4
                 
                 ┌─── M_v2→c1
                 │
    M_c1→v1 ─────┼─── M_v3→c1
                 │
                 └─── M_v4→c1
```
In the message-centered approach, each message node aggregates information from other messages that share the same variable or check node.

## Neural Network Architecture

### Message-Centered GNN Layer
```
    Input: Message Features [batch_size, num_messages, hidden_dim]
    
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │  1. Get message type embeddings                       │
    │     type_embeddings = message_type_embeddings[types]  │
    │                                                       │
    │  2. Combine features with type embeddings             │
    │     combined = message_features + type_embeddings     │
    │                                                       │
    │  3. Variable-to-check message update                  │
    │     a. Aggregate from neighbors                       │
    │        agg = var_to_check_adjacency * combined        │
    │     b. Concatenate with current features              │
    │        input = [combined, agg]                        │
    │     c. Apply neural network                           │
    │        var_to_check = NN_VC(input)                    │
    │                                                       │
    │  4. Check-to-variable message update                  │
    │     a. Aggregate from neighbors                       │
    │        agg = check_to_var_adjacency * combined        │
    │     b. Concatenate with current features              │
    │        input = [combined, agg]                        │
    │     c. Apply neural network                           │
    │        check_to_var = NN_CV(input)                    │
    │                                                       │
    │  5. Combine updates (with residual connection)        │
    │     updated = var_to_check + check_to_var             │
    │                                                       │
    └───────────────────────────────────────────────────────┘
    
    Output: Updated Message Features [batch_size, num_messages, hidden_dim]
```

## Complete Decoder Architecture

### Message-Centered GNN Decoder
```
    Input: Channel LLRs [batch_size, num_variables]
    
    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │  1. Initialize message features from LLRs             │
    │     message_llrs = message_to_var_mapping * llrs      │
    │     message_features = input_embedding(message_llrs)  │
    │                                                       │
    │  2. Iterative GNN decoding                            │
    │     for i = 1 to num_iterations:                      │
    │         message_features = gnn_layers[i](            │
    │             message_features,                         │
    │             message_types,                            │
    │             var_to_check_adjacency,                   │
    │             check_to_var_adjacency                    │
    │         )                                             │
    │                                                       │
    │  3. Decode final message features to LLRs             │
    │     final_llrs = decode_messages(message_features)    │
    │                                                       │
    │  4. Aggregate message LLRs to variable nodes          │
    │     variable_llrs = final_llrs * var_to_message_map   │
    │                                                       │
    │  5. Add input LLRs                                    │
    │     combined_llrs = variable_llrs + input_llrs        │
    │                                                       │
    │  6. Convert to soft/hard bits                         │
    │     soft_bits = sigmoid(combined_llrs)                │
    │     hard_bits = (soft_bits > 0.5).float()             │
    │                                                       │
    └───────────────────────────────────────────────────────┘
    
    Output: Decoded Bits [batch_size, num_variables]
```

## Parameter Efficiency Comparison

```
    Parameter Count Comparison (log scale)
    
    Standard Neural  ████████████████████████████████████  ~450 params
    GNN-based        ██████████████████                    ~200 params
    Message-Centered ██████████                            ~100 params
    
    0               100              200              300              400              500
```

The message-centered GNN achieves the highest parameter efficiency while maintaining or improving decoding performance. 