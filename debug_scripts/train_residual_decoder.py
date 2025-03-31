import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_neural_decoder.models.residual_weight_sharing_decoder import ResidualWeightSharingDecoder
from ldpc_neural_decoder.utils.training_data_generator import load_training_data
from debug_scripts.test_residual_decoder import load_base_matrix

def compute_loss(decoded_bits, transmitted_bits, H):
    """
    Compute combined loss from binary cross-entropy and soft parity check loss
    """
    # Ensure decoded_bits requires gradients
    decoded_bits = decoded_bits.requires_grad_(True)
    
    # Binary cross-entropy loss
    bce_loss = nn.BCELoss()(decoded_bits, transmitted_bits)
    
    # Parity check loss
    # Convert probabilities to hard decisions
    hard_decisions = (decoded_bits > 0.5).float()
    # Compute syndrome
    syndrome = torch.matmul(hard_decisions, H.T) % 2
    # Parity loss is the fraction of unsatisfied checks
    parity_loss = torch.mean(syndrome.float())
    
    # Combined loss (weighted sum)
    total_loss = bce_loss + 0.1 * parity_loss
    
    return total_loss, bce_loss, parity_loss

def evaluate(model, data_loader, H, device):
    """
    Evaluate model on validation/test set
    """
    model.eval()
    total_loss = 0
    total_ber = 0
    total_fer = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (input_llr, transmitted_bits, _) in enumerate(data_loader):
            input_llr = input_llr.to(device)
            transmitted_bits = transmitted_bits.to(device)
            
            # Forward pass
            decoded_bits = model(input_llr)
            
            # Compute loss
            loss, _, _ = compute_loss(decoded_bits, transmitted_bits, H)
            total_loss += loss.item()
            
            # Compute BER
            errors = torch.sum(decoded_bits.round() != transmitted_bits)
            total_ber += errors.item() / transmitted_bits.numel()
            
            # Compute FER (frame error rate)
            frame_errors = torch.sum(torch.any(decoded_bits.round() != transmitted_bits, dim=1))
            total_fer += frame_errors.item() / transmitted_bits.shape[0]
            
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_ber = total_ber / num_batches
    avg_fer = total_fer / num_batches
    
    return avg_loss, avg_ber, avg_fer

def plot_metrics(train_metrics, val_metrics, save_dir):
    """
    Plot and save training metrics (BER and FER)
    """
    # Create training results directory
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training_results"
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract metrics
    epochs = range(1, len(train_metrics['ber']) + 1)
    
    # Create BER plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['ber'], label='Training BER')
    plt.plot(epochs, val_metrics['ber'], label='Validation BER')
    plt.xlabel('Epoch')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Training and Validation BER')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'ber_plot.png'))
    plt.close()
    
    # Create FER plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['fer'], label='Training FER')
    plt.plot(epochs, val_metrics['fer'], label='Validation FER')
    plt.xlabel('Epoch')
    plt.ylabel('Frame Error Rate (FER)')
    plt.title('Training and Validation FER')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'fer_plot.png'))
    plt.close()

def main():
    # Training settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    learning_rate = 1e-3
    save_interval = 5  # Save checkpoint every N epochs
    
    # Initialize metrics tracking
    train_metrics = {'loss': [], 'ber': [], 'fer': []}
    val_metrics = {'loss': [], 'ber': [], 'fer': []}
    
    # Load base matrix
    base_matrix_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "5G LDPC CODES",
        "NR_2_0_4.txt"
    )
    Hb = load_base_matrix(base_matrix_path)
    
    # Model parameters
    z = 4  # Expansion factor
    num_iterations = 10  # Reduced from 10 to 8 for residual decoder
    residual_depth = 2
    
    # Create model
    model = ResidualWeightSharingDecoder(
        base_matrix=Hb,
        expansion_factor=z,
        num_iterations=num_iterations,
        residual_depth=residual_depth
    ).to(device)
    
    # Load training data
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training_data"
    )
    train_data = load_training_data(os.path.join(data_dir, f"train_data_z{z}.pt"))
    val_data = load_training_data(os.path.join(data_dir, f"val_data_z{z}.pt"))
    
    # Create data loaders
    train_loader = [(llr, bits, syms) for llr, bits, syms in train_data]
    val_loader = [(llr, bits, syms) for llr, bits, syms in val_data]
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,  # Standard momentum value
        weight_decay=1e-2  # L2 regularization
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Create directory for checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "checkpoints",
        f"residual_decoder_z{z}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save training config
    config = {
        "expansion_factor": z,
        "num_iterations": num_iterations,
        "residual_depth": residual_depth,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "device": str(device),
        "base_matrix_shape": list(Hb.shape),
        "timestamp": timestamp
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Training loop
    best_val_loss = float('inf')
    print(f"\n=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_bce = 0
        epoch_parity = 0
        epoch_ber = 0
        epoch_fer = 0
        
        for batch_idx, (input_llr, transmitted_bits, _) in enumerate(train_loader):
            input_llr = input_llr.to(device)
            transmitted_bits = transmitted_bits.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            decoded_bits = model(input_llr)
            
            # Compute loss
            loss, bce, parity = compute_loss(decoded_bits, transmitted_bits, model.H)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses and metrics
            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_parity += parity.item()
            
            # Compute BER and FER
            errors = torch.sum(decoded_bits.round() != transmitted_bits)
            epoch_ber += errors.item() / transmitted_bits.numel()
            frame_errors = torch.sum(torch.any(decoded_bits.round() != transmitted_bits, dim=1))
            epoch_fer += frame_errors.item() / transmitted_bits.shape[0]
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (BCE: {bce.item():.4f}, Parity: {parity.item():.4f})")
        
        # Average training metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_bce = epoch_bce / len(train_loader)
        avg_parity = epoch_parity / len(train_loader)
        avg_ber = epoch_ber / len(train_loader)
        avg_fer = epoch_fer / len(train_loader)
        
        # Store training metrics
        train_metrics['loss'].append(avg_loss)
        train_metrics['ber'].append(avg_ber)
        train_metrics['fer'].append(avg_fer)
        
        # Validation
        val_loss, val_ber, val_fer = evaluate(model, val_loader, model.H, device)
        
        # Store validation metrics
        val_metrics['loss'].append(val_loss)
        val_metrics['ber'].append(val_ber)
        val_metrics['fer'].append(val_fer)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training - Loss: {avg_loss:.4f}, BCE: {avg_bce:.4f}, Parity: {avg_parity:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, BER: {val_ber:.4f}, FER: {val_fer:.4f}")
        
        # Plot metrics after each epoch
        plot_metrics(train_metrics, val_metrics, save_dir)
        
        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_ber': val_ber,
                'val_fer': val_fer
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        
        # Regular checkpoint saving
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_ber': val_ber,
                'val_fer': val_fer
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main() 