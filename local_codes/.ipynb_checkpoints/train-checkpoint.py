import torch
import torch.nn as nn
import torch.nn.functional as F
from data_process import *

def train_step(model, optimizer, da, device, level_weights=None):
    # Zero the gradients
    optimizer.zero_grad()
    
    # Get batch and convert to tensor
    tensor_batch = data_to_tensor_batch(da.get_batch())
    tensor_batch.gpu(device)
    
    predictions = model(tensor_batch.seq_ids)
    labels = tensor_batch.taxes
    
    # Initialize total loss
    total_loss = 0
    level_losses = {}

    # Calculate loss for each level
    for level, pred in predictions.items():
        level_loss = F.cross_entropy(pred, labels[level])
        
        # Apply level weights if provided
        if level_weights and level in level_weights:
            level_loss *= level_weights[level]
            
        level_losses[level] = level_loss.item()  # Store loss value for logging
        total_loss += level_loss
    
    # Backward pass and optimization step
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), level_losses