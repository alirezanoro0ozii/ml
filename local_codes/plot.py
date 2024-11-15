import torch
import os
import matplotlib.pyplot as plt

def load_metrics_from_checkpoints(checkpoint_dir, num_checkpoints):
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    
    # Iterate through checkpoint files and load metrics
    for epoch in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        if os.path.exists(checkpoint_path):
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path)
            
            # Append loaded metrics to lists
            train_losses.append(checkpoint['train_loss'])
            val_losses.append(checkpoint['val_loss'])
            val_accuracies.append(checkpoint['accuracy'])
            val_f1s.append(checkpoint['f1_score']['lvl_1'])  # Change 'lvl_1' as needed
        
    return train_losses, val_losses, val_accuracies, val_f1s

def plot_metrics_from_checkpoints(checkpoint_dir, num_checkpoints, val_epoch):
    # Load metrics from .pt files
    train_losses, val_losses, val_accuracies, val_f1s = load_metrics_from_checkpoints(checkpoint_dir, num_checkpoints)
    
    epochs_range = range(1, num_checkpoints + 1, val_epoch)
    
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(range(1, num_checkpoints + 1), train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot validation accuracy and F1 score
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.plot(epochs_range, val_f1s, label='Validation F1 Score (Level 1)')
    plt.xlabel('Epochs')
    plt.ylabel('Metric')
    plt.title('Validation Accuracy and F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage
checkpoint_dir = 'checkpoints'  # Directory where checkpoints are stored
num_checkpoints = 10  # Number of checkpoints to load
val_epoch = 1  # Validation frequency
plot_metrics_from_checkpoints(checkpoint_dir, num_checkpoints, val_epoch)
