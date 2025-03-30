"""
Attention mechanisms for the model.
"""

import torch
import torch.nn as nn
from utils.initialization import weights_init_normal

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism that generates attention maps based on channel-wise pooling.
    
    Args:
        kernel_size (int): Size of the convolution kernel for attention map generation
    """
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        
        # Calculate attention through two channels: max pooling and average pooling
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()  # Normalize output to [0, 1]
        
        # Initialize weights
        self.apply(weights_init_normal)
    
    def forward(self, x):
        """
        Forward pass to generate spatial attention map and apply it to input features.
        
        Args:
            x: Input feature map [batch_size, channels, height, width]
            
        Returns:
            Attention-weighted input feature map
        """
        # Calculate average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling
        
        # Concatenate pooled features
        concat_out = torch.cat([avg_out, max_out], dim=1)
        
        # Generate spatial attention map
        attention = self.conv1(concat_out)
        attention = self.sigmoid(attention)  # Output range [0, 1]
        
        # Apply attention to input features
        return x * attention 