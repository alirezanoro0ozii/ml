import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from data_process import *


def evaluate(model, da, device, tax_vocab_sizes, level_weights=None, num_val_batches=1):
    model.eval()  # Set the model to evaluation mode
    
    level_losses = {} 
    total_loss = 0

    level_acc = {}
    total_correct = 0
    total_samples = 0

    level_f1 = {}
    total_cms = {}
    
    with torch.no_grad():  # Disable gradient computation for evaluation
        for _ in range(num_val_batches):
            tensor_batch = data_to_tensor_batch(da.get_batch())
            tensor_batch.gpu(device)
            
            predictions = model(tensor_batch.seq_ids)
            labels = tensor_batch.taxes
            
            batch_loss = 0
            batch_level_losses = {}

            # Calculate loss for each level
            for level, pred in predictions.items():
                level_loss = F.cross_entropy(pred, labels[level])
                # Apply level weights if provided
                if level_weights and level in level_weights:
                    level_loss *= level_weights[level]
                
                batch_level_losses[level] = level_loss.item()
                batch_loss += level_loss
            
            # Update total loss and level-specific losses
            total_loss += batch_loss.item()
            for level, level_loss_value in batch_level_losses.items():
                if level in level_losses:
                    level_losses[level] += level_loss_value
                else:
                    level_losses[level] = level_loss_value
            
            # Calculate accuracy
            for level, pred in predictions.items():
                predicted_classes = torch.argmax(pred, dim=1)
                level_f1[level] = f1_score(labels[level].cpu(), predicted_classes.cpu(), average='micro')
                correct_predictions = (predicted_classes == labels[level]).sum().item()
                total_correct += correct_predictions
                total_samples += labels[level].size(0)
                if level == "begining root":
                # Generate the confusion matrix
                    total_cms[level] = confusion_matrix(labels[level].cpu(), predicted_classes.cpu(), labels=[_ for _ in range(tax_vocab_sizes[level])])
    
    # Average losses
    val_loss = total_loss / num_val_batches
    level_losses = {level: loss / num_val_batches for level, loss in level_losses.items()}
    
    # Calculate overall accuracy
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    model.train()  # Set the model back to training mode
    return val_loss, level_losses, accuracy, level_f1, total_cms