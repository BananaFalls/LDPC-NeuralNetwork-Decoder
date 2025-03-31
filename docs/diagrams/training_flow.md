# Training Flow Diagram

## Overview
This diagram illustrates the complete training process, from data generation to model evaluation.

```mermaid
graph TD
    subgraph Data Generation
        ZC[Zero Codewords]
        BPSK[BPSK Modulation]
        AWGN[AWGN Noise]
        LLR[LLR Computation]
        ZC --> BPSK
        BPSK --> AWGN
        AWGN --> LLR
    end

    subgraph Training Loop
        direction TB
        subgraph Forward Pass
            FP[Forward Pass]
            SD[Soft Decoding]
            FP --> SD
        end

        subgraph Loss Computation
            BCE[BCE Loss]
            PC[Parity Check Loss]
            CL[Combined Loss]
            BCE --> CL
            PC --> CL
        end

        subgraph Backward Pass
            BP[Backward Pass]
            GU[Gradient Update]
            BP --> GU
        end
    end

    subgraph Evaluation
        direction TB
        subgraph Metrics
            BER[Bit Error Rate]
            FER[Frame Error Rate]
            VL[Validation Loss]
        end

        subgraph Visualization
            BP[BER Plot]
            FP[FER Plot]
        end

        subgraph Checkpointing
            BM[Best Model]
            RC[Regular Checkpoints]
        end
    end

    LLR --> FP
    SD --> BCE
    SD --> PC
    CL --> BP
    GU --> FP
    BP --> BER
    BP --> FER
    BP --> VL
    BER --> BP
    FER --> FP
    VL --> BM
    VL --> RC

    style ZC fill:#f9f,stroke:#333,stroke-width:2px
    style LLR fill:#f9f,stroke:#333,stroke-width:2px
    style FP fill:#bbf,stroke:#333,stroke-width:2px
    style SD fill:#bbf,stroke:#333,stroke-width:2px
    style BCE fill:#bbf,stroke:#333,stroke-width:2px
    style PC fill:#bbf,stroke:#333,stroke-width:2px
    style CL fill:#bbf,stroke:#333,stroke-width:2px
    style BP fill:#bbf,stroke:#333,stroke-width:2px
    style GU fill:#bbf,stroke:#333,stroke-width:2px
    style BER fill:#f9f,stroke:#333,stroke-width:2px
    style FER fill:#f9f,stroke:#333,stroke-width:2px
    style VL fill:#f9f,stroke:#333,stroke-width:2px
    style BM fill:#f9f,stroke:#333,stroke-width:2px
    style RC fill:#f9f,stroke:#333,stroke-width:2px
```

## Components

### 1. Data Generation
- Zero codewords generation
- BPSK modulation
- AWGN noise addition
- LLR computation

### 2. Training Loop
- Forward pass with soft decoding
- Loss computation (BCE + Parity)
- Backward pass
- Gradient updates

### 3. Evaluation
- Metrics computation
- Visualization
- Checkpointing

## Flow Description

1. **Data Generation**:
   - Start with zero codewords
   - Apply BPSK modulation
   - Add AWGN noise
   - Compute LLRs

2. **Training Process**:
   - Forward pass through decoder
   - Soft decoding for better gradients
   - Compute combined loss
   - Backward pass and updates

3. **Evaluation**:
   - Compute BER and FER
   - Track validation loss
   - Generate plots
   - Save checkpoints

## Key Features

1. **Soft Decoding**:
   - Continuous probability values
   - Better gradient flow
   - Improved training stability

2. **Loss Functions**:
   - Binary Cross-Entropy
   - Soft parity check loss
   - Weighted combination

3. **Monitoring**:
   - Real-time metrics
   - Visualization
   - Checkpointing 