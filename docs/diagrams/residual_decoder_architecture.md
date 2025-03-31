# Residual Decoder Architecture

## Overview
This diagram illustrates the architecture of the residual weight-sharing decoder, showing the flow of information and residual connections.

```mermaid
graph TD
    subgraph Input
        LLR[Channel LLRs]
    end

    subgraph Decoder
        direction TB
        subgraph Iteration 1
            VN1[Variable Node Update]
            CN1[Check Node Update]
            VN1 --> CN1
        end

        subgraph Iteration 2
            VN2[Variable Node Update]
            CN2[Check Node Update]
            VN2 --> CN2
        end

        subgraph Iteration N
            VNn[Variable Node Update]
            CNn[Check Node Update]
            VNn --> CNn
        end
    end

    subgraph Residual Connections
        VN1 -.-> VN2
        VN2 -.-> VNn
        CN1 -.-> CN2
        CN2 -.-> CNn
    end

    subgraph Output
        PROB[Bit Probabilities]
    end

    LLR --> VN1
    CNn --> PROB

    style LLR fill:#f9f,stroke:#333,stroke-width:2px
    style PROB fill:#f9f,stroke:#333,stroke-width:2px
    style VN1 fill:#bbf,stroke:#333,stroke-width:2px
    style CN1 fill:#bbf,stroke:#333,stroke-width:2px
    style VN2 fill:#bbf,stroke:#333,stroke-width:2px
    style CN2 fill:#bbf,stroke:#333,stroke-width:2px
    style VNn fill:#bbf,stroke:#333,stroke-width:2px
    style CNn fill:#bbf,stroke:#333,stroke-width:2px
```

## Components

1. **Input Layer**
   - Channel LLRs (Log-Likelihood Ratios)
   - Initial bit probabilities

2. **Decoder Iterations**
   - Variable Node Updates
   - Check Node Updates
   - Weight sharing between similar message types

3. **Residual Connections**
   - Skip connections between iterations
   - Helps with gradient flow
   - Preserves information across iterations

4. **Output Layer**
   - Final bit probabilities
   - Decoded bits

## Message Flow

1. **Forward Pass**
   - LLRs → Variable Nodes → Check Nodes → ... (repeat)
   - Residual connections add previous iteration outputs

2. **Backward Pass**
   - Gradients flow through both direct and residual paths
   - Improved gradient flow due to skip connections

## Key Features

1. **Weight Sharing**
   - Parameters shared based on message types
   - Reduces model complexity
   - Maintains code structure

2. **Residual Depth**
   - Configurable number of previous iterations
   - Balances performance and memory usage

3. **Iteration Count**
   - Configurable number of iterations
   - Early stopping possible
   - Residual connections help convergence 