# Message-Centered GNN Architecture Diagrams

## Tanner Graph vs Message-Centered Graph

```mermaid
flowchart LR
    subgraph "Tanner Graph"
        direction TB
        v1[Variable Node 1]
        v2[Variable Node 2]
        v3[Variable Node 3]
        c1[Check Node 1]
        c2[Check Node 2]
        
        v1 --- c1
        v1 --- c2
        v2 --- c1
        v2 --- c2
        v3 --- c2
    end
    
    transform((Transform))
    
    subgraph "Message-Centered Graph"
        direction TB
        m1[Message 1: v1→c1]
        m2[Message 2: v1→c2]
        m3[Message 3: v2→c1]
        m4[Message 4: v2→c2]
        m5[Message 5: v3→c2]
        
        %% Messages sharing variables
        m1 --- m2
        m3 --- m4
        
        %% Messages sharing checks
        m1 --- m3
        m2 --- m4
        m2 --- m5
        m4 --- m5
    end
    
    "Tanner Graph" --> transform
    transform --> "Message-Centered Graph"
```

## Message GNN Decoder Architecture

```mermaid
flowchart TD
    input[Input LLR Values] --> embed[Message Embedding]
    embed --> var1[Variable GNN Layer 1]
    var1 --> check1[Check GNN Layer 1]
    
    check1 --> var2[Variable GNN Layer 2]
    var2 --> check2[Check GNN Layer 2]
    
    %% More iterations (simplified)
    check2 --> dots[...]
    dots --> varN[Variable GNN Layer N]
    varN --> checkN[Check GNN Layer N]
    
    %% Residual connections (simplified)
    var1 -.-> var2
    var1 -.-> var3
    var2 -.-> var3
    
    %% Final output
    checkN --> output[Output Mapping]
    output --> decisions[Hard Decisions]
    
    %% Legend
    subgraph Legend
        direction LR
        normal[Normal Connection]
        residual[Residual Connection]
        normal === dummy1
        residual -.- dummy2
    end
```

## VariableGNNLayer Internal Structure

```mermaid
flowchart LR
    subgraph "VariableGNNLayer"
        direction TB
        
        input_var[Variable Messages]
        input_check[Check Messages]
        input_types[Message Types]
        input_adj[Adjacency Matrix]
        
        type_embed[Type Embeddings]
        
        aggregate[Message Aggregation]
        combine[Feature Combination]
        mlp[Neural Network]
        
        input_var --> type_embed
        input_types --> type_embed
        
        type_embed --> aggregate
        input_adj --> aggregate
        
        aggregate --> combine
        input_check --> combine
        
        combine --> mlp
        
        mlp --> output[Updated Messages]
    end
```

## CheckGNNLayer Internal Structure

```mermaid
flowchart LR
    subgraph "CheckGNNLayer"
        direction TB
        
        input_check[Check Messages]
        input_var[Variable Messages]
        input_types[Message Types]
        input_adj[Adjacency Matrix]
        
        type_embed[Type Embeddings]
        
        aggregate[Message Aggregation]
        combine[Feature Combination]
        mlp[Neural Network]
        
        input_check --> type_embed
        input_types --> type_embed
        
        type_embed --> aggregate
        input_adj --> aggregate
        
        aggregate --> combine
        input_var --> combine
        
        combine --> mlp
        
        mlp --> output[Updated Messages]
    end
```

## Message Passing Process

```mermaid
sequenceDiagram
    participant LLR as Input LLR
    participant Var as Variable Layer
    participant Check as Check Layer
    participant Out as Output Layer
    
    LLR->>Var: Initial LLR values
    
    loop For each iteration
        Var->>Check: Variable-to-Check messages
        Check->>Var: Check-to-Variable messages
    end
    
    Var->>Out: Final messages
    Out->>Out: Combine messages by variable
    Out->>Out: Add original LLR
    Out->>Out: Apply sigmoid
```

## Class Diagram

```mermaid
classDiagram
    class MessageGNNDecoder {
        -num_messages: int
        -num_iterations: int
        -hidden_dim: int
        -num_message_types: int
        -num_of_residual_layers: int
        -var_gnn_layers: nn.ModuleList
        -check_gnn_layers: nn.ModuleList
        -var_layer_norms: nn.ModuleList
        -check_layer_norms: nn.ModuleList
        -output_projection: nn.Linear
        +forward()
        +output_mapping()
        +count_parameters()
    }
    
    class VariableGNNLayer {
        -message_type_embeddings: nn.Parameter
        -var_update: nn.Sequential
        -output_projection: nn.Linear
        +forward()
        +decode_messages()
    }
    
    class CheckGNNLayer {
        -message_type_embeddings: nn.Parameter
        -check_update: nn.Sequential
        -output_projection: nn.Linear
        +forward()
        +decode_messages()
    }
    
    class TannerToMessageGraph {
        -H: torch.Tensor
        -m: int
        -n: int
        -var_to_checks: dict
        -check_to_vars: dict
        -messages: list
        -message_to_var: dict
        -message_to_check: dict
        -var_to_messages: dict
        -check_to_messages: dict
        -num_messages: int
        -message_types: torch.Tensor
        -var_to_check_adjacency: torch.Tensor
        -check_to_var_adjacency: torch.Tensor
        +create_adjacency_matrices()
        +get_message_to_var_mapping()
        +get_message_to_check_mapping()
        +get_var_to_messages_mapping()
        +get_check_to_messages_mapping()
    }
    
    MessageGNNDecoder *-- VariableGNNLayer
    MessageGNNDecoder *-- CheckGNNLayer
``` 