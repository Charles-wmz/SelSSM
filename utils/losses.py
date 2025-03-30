"""
Loss functions for training the GAN model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def content_loss(generated, target, loss_type="L1"):
    """
    Content loss function (L1/L2) for comparing generated and target images.
    
    Args:
        generated: Generated image tensor
        target: Target image tensor
        loss_type: Type of loss to use ("L1" or "L2")
    
    Returns:
        Loss value
    """
    if loss_type == "L1":
        return F.l1_loss(generated, target)
    elif loss_type == "L2":
        return F.mse_loss(generated, target)
    else:
        raise ValueError("Invalid loss type. Choose 'L1' or 'L2'.")

def edge_loss(generated_image, target_image):
    """
    Edge loss based on intensity gradients.
    
    Args:
        generated_image: Generated image tensor
        target_image: Target image tensor
    
    Returns:
        Combined gradient loss in x and y directions
    """
    # Calculate image intensity gradients
    grad_x_real = torch.diff(target_image, dim=-1)
    grad_y_real = torch.diff(target_image, dim=-2)
    
    grad_x_gen = torch.diff(generated_image, dim=-1)
    grad_y_gen = torch.diff(generated_image, dim=-2)
    
    # Calculate L1 loss for gradient differences
    grad_loss_x = nn.L1Loss()(grad_x_gen, grad_x_real)
    grad_loss_y = nn.L1Loss()(grad_y_gen, grad_y_real)
    
    return grad_loss_x + grad_loss_y

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training the selective attention mechanism.
    
    Args:
        margin (float): Margin for negative sample separation
    """
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, selection_weights, positive_weights, negative_weights):
        """
        Forward pass to compute contrastive loss.
        
        Args:
            selection_weights: Dynamic weights for selection matrix [batch_size, channels, height, width]
            positive_weights: Positive sample weights [batch_size, channels, height, width]
            negative_weights: Negative sample weights [batch_size, channels, height, width]
            
        Returns:
            Contrastive loss value
        """
        # Positive sample loss: encourage selection matrix to be close to positive weights
        pos_loss = torch.mean((selection_weights - positive_weights) ** 2)

        # Negative sample loss: push selection matrix away from negative weights
        neg_loss = torch.mean(torch.clamp(self.margin - (selection_weights - negative_weights).abs(), min=0) ** 2)

        return pos_loss + neg_loss 